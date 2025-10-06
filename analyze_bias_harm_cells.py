# -*- coding: utf-8 -*-
"""
Analyze bias_type → harm within (domain × stakeholder) cells.

Outputs (under ./tables_py/):
- <domain>__<stakeholder>__omnibus.txt                 # omnibus χ² summary for K×H
- <domain>__<stakeholder>__kxH_counts.csv              # bias×harm observed counts (votes)
- <domain>__<stakeholder>__kxH_rowshares.csv           # row-normalized shares (P(h|bias))
- <domain>__<stakeholder>__kxH_stdresid.csv            # standardized residuals (omnibus K×H; NaN where pruned)
- <domain>__<stakeholder>__pairwise.csv                # all bias pairs with χ², p, perm_p, V, w, q
- <domain>__<stakeholder>__PAIR__A__vs__B__perharm.csv # per-harm Fisher p/q, LOR, share diffs
- <domain>__SIGNIFICANT_PAIRS.csv                      # NEW: aggregated domain-specific significant results (q ≤ 0.10)
"""

from __future__ import annotations
import argparse
from pathlib import Path
import itertools
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from statsmodels.graphics.mosaicplot import mosaic


# -------------------- Mosaic plotting --------------------
def plot_mosaic(ctab: pd.DataFrame, stdres_df: pd.DataFrame, out_prefix: Path):
    data = {(i, j): int(v) for (i, j), v in ctab.stack().items()}
    resid_map = stdres_df.stack().to_dict()

    cmap = plt.cm.coolwarm
    norm = mcolors.TwoSlopeNorm(vmin=-3, vcenter=0, vmax=3)
    props = {}
    for key, _ in data.items():
        r = resid_map.get(key, np.nan)
        color = cmap(norm(r)) if np.isfinite(r) else (0.8, 0.8, 0.8, 1.0)
        props[key] = {"color": color}

    fig, ax = plt.subplots()
    mosaic(data, gap=0.01, properties=props, ax=ax)
    ax.set_title(f"Mosaic plot: Bias × Harm\n({out_prefix.stem.replace('__',' – ')})", fontsize=12)
    ax.set_xlabel("Bias type")
    ax.set_ylabel("Harm type")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.7)
    cbar.set_label("Standardized residual (z-score)")
    plt.tight_layout()
    plt.savefig(out_prefix.with_name(out_prefix.name + "__mosaic.png"), dpi=300)
    plt.close()


# -------------------- Paths & schema --------------------
OUT_ROOT = Path("tables_py")
OUT_ROOT.mkdir(parents=True, exist_ok=True)
REQUIRED = {"questionnaire_id", "stakeholder_raw", "bias_type", "domain", "harm", "votes"}


# -------------------- Utilities --------------------
def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    n = p.size
    order = np.argsort(p)
    ranks = np.arange(1, n + 1, dtype=float)
    q_sorted = p[order] * n / ranks
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q = np.empty_like(q_sorted)
    q[order] = q_sorted
    return np.clip(q, 0, 1)


def standardized_residuals(obs: np.ndarray, exp: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return (obs - exp) / np.sqrt(exp)


def cramers_v(chi2: float, n: int, r: int, c: int) -> float:
    if n <= 0: return np.nan
    k = min(r - 1, c - 1)
    if k <= 0: return np.nan
    return float(np.sqrt(chi2 / (n * k)))


def cohens_w_from_obs_exp(obs: np.ndarray, exp: np.ndarray) -> float:
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = np.nansum((obs - exp) ** 2 / exp)
    N = np.nansum(obs)
    return float(np.sqrt(chi2 / N)) if N > 0 else np.nan


def safe_chi2_return_table(table: pd.DataFrame):
    t = table.loc[:, table.sum(axis=0) > 0]
    t = t.loc[t.sum(axis=1) > 0]
    if t.shape[0] < 2 or t.shape[1] < 2:
        return np.nan, np.nan, 0, None, t
    chi2, p, df, exp = chi2_contingency(t.to_numpy(), correction=False)
    return chi2, p, df, exp, t


def per_harm_fisher_2x2(ctab_2xH: pd.DataFrame, add_smooth: float = 0.5) -> pd.DataFrame:
    assert ctab_2xH.shape[0] == 2
    a_name, b_name = ctab_2xH.index.tolist()
    A = int(ctab_2xH.loc[a_name].sum())
    B = int(ctab_2xH.loc[b_name].sum())
    harms = list(ctab_2xH.columns)
    pvals, lors, a_counts, b_counts = [], [], [], []
    for h in harms:
        a = int(ctab_2xH.loc[a_name, h]); b = int(ctab_2xH.loc[b_name, h])
        _, p = fisher_exact([[a, A - a], [b, B - b]], alternative="two-sided")
        a1, b1 = a + add_smooth, b + add_smooth
        A1, B1 = (A - a) + add_smooth, (B - b) + add_smooth
        lor = np.log((a1 / A1) / (b1 / B1))
        pvals.append(p); lors.append(lor); a_counts.append(a); b_counts.append(b)
    qvals = bh_fdr(np.array(pvals))
    return pd.DataFrame({"harm": harms, "a_harm": a_counts, "b_harm": b_counts, "p": pvals, "q": qvals, "lor": lors})


# -------------------- Data prep --------------------
def load_data(in_path: Path) -> pd.DataFrame:
    df = pd.read_csv(in_path)
    missing = REQUIRED.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns in input: {missing}")
    key = ["questionnaire_id", "stakeholder_raw", "bias_type", "domain", "harm"]
    df = df.groupby(key, as_index=False)["votes"].sum()
    if "stakeholder" not in df.columns:
        df["stakeholder"] = df["stakeholder_raw"]
    return df


def filter_cell(df: pd.DataFrame, domain: str, stakeholder: str) -> pd.DataFrame:
    d = df.copy()
    d = d[d["domain"].astype(str).str.lower() == domain.lower()]
    d = d[d["stakeholder"].astype(str).str.lower() == stakeholder.lower()]
    if d.empty:
        raise SystemExit(f"No data for domain='{domain}' & stakeholder='{stakeholder}'")
    return d


def build_kxH(d: pd.DataFrame) -> pd.DataFrame:
    ctab = pd.pivot_table(d, values="votes", index="bias_type", columns="harm", aggfunc="sum", fill_value=0).astype(int)
    ctab = ctab.loc[:, ctab.sum(axis=0) > 0]
    ctab = ctab.loc[ctab.sum(axis=1) > 0]
    return ctab


def row_shares(ctab: pd.DataFrame) -> pd.DataFrame:
    denom = ctab.sum(axis=1).replace(0, np.nan)
    return (ctab.T / denom).T


# -------------------- Main analysis per cell --------------------
def analyze_cell(ctab: pd.DataFrame, out_prefix: Path, perm_n: int = 0):
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    ctab.to_csv(out_prefix.with_name(out_prefix.name + "__kxH_counts.csv"))
    row_shares(ctab).to_csv(out_prefix.with_name(out_prefix.name + "__kxH_rowshares.csv"))

    chi2, p, df, exp, t_pruned = safe_chi2_return_table(ctab)
    stdres_df = pd.DataFrame(np.nan, index=ctab.index, columns=ctab.columns, dtype=float)
    if exp is not None:
        stdres_small = standardized_residuals(t_pruned.to_numpy(), exp)
        stdres_df.loc[t_pruned.index, t_pruned.columns] = stdres_small
    stdres_df.to_csv(out_prefix.with_name(out_prefix.name + "__kxH_stdresid.csv"))

    K, H = ctab.shape
    N = int(ctab.values.sum())
    V = cramers_v(chi2, N, *t_pruned.shape) if np.isfinite(chi2) else np.nan

    with open(out_prefix.with_name(out_prefix.name + "__omnibus.txt"), "w", encoding="utf-8") as f:
        f.write(f"Table: K×H (original) = {K}×{H}; N={N}\nχ²={chi2:.2f}, p={p:.4f}, V={V:.3f}\n")

    rows = ctab.index.tolist()
    pair_rows = []
    for a, b in itertools.combinations(rows, 2):
        ctab_2xH_full = ctab.loc[[a, b]].copy()
        chi2p, pp, dff, exp2, t2 = safe_chi2_return_table(ctab_2xH_full)
        obs2 = t2.to_numpy(); n2 = int(obs2.sum())
        Vp = cramers_v(chi2p, n2, *obs2.shape) if np.isfinite(chi2p) else np.nan
        perharm = per_harm_fisher_2x2(ctab_2xH_full).set_index("harm")
        shares = row_shares(ctab_2xH_full)
        pA, pB = shares.loc[a], shares.loc[b]
        delta = pA - pB
        out_ph = pd.DataFrame({
            "harm": pA.index,
            "share_A": pA.values,
            "share_B": pB.values,
            "share_diff_A_minus_B": delta.values,
            "a_harm": perharm["a_harm"].values,
            "b_harm": perharm["b_harm"].values,
            "p": perharm["p"].values,
            "q": perharm["q"].values,
            "lor": perharm["lor"].values,
        })
        out_ph.to_csv(out_prefix.with_name(out_prefix.name + f"__PAIR__{a}__vs__{b}__perharm.csv"), index=False)

    try:
        plot_mosaic(ctab, stdres_df, out_prefix)
    except Exception as e:
        print(f"[Warning] Mosaic plot failed for {out_prefix.name}: {e}")


# -------------------- COLLECT AND PRINT SIGNIFICANT RESULTS --------------------
def summarize_significant_pairs(out_root: Path, q_threshold: float = 0.10):
    all_files = list(out_root.glob("*__PAIR__*__perharm.csv"))
    if not all_files:
        print("\n[Info] No per-harm files found. Skipping summary.")
        return

    # Group by domain prefix
    domains = sorted({f.name.split("__")[0] for f in all_files})
    print("\n==================== DOMAIN-SPECIFIC SIGNIFICANT RESULTS ====================")
    for domain in domains:
        files = [f for f in all_files if f.name.startswith(domain + "__")]
        rows = []
        for f in files:
            parts = f.stem.split("__")
            stakeholder = parts[1]
            bias_a, bias_b = parts[4], parts[6]
            df = pd.read_csv(f)
            sig = df[df["q"] <= q_threshold]
            for _, r in sig.iterrows():
                rows.append({
                    "Stakeholder": stakeholder,
                    "Bias_A": bias_a,
                    "Bias_B": bias_b,
                    "Harm": r["harm"],
                    "A_share": round(r["share_A"], 3),
                    "B_share": round(r["share_B"], 3),
                    "Δ_share": round(r["share_diff_A_minus_B"], 3),
                    "p": round(r["p"], 4),
                    "q": round(r["q"], 4)
                })

        if not rows:
            print(f"\n[{domain.upper()}] – No significant (q ≤ {q_threshold}) bias–harm pairs found.")
            continue

        df_out = pd.DataFrame(rows)
        df_out = df_out.sort_values(["Stakeholder", "Bias_A", "Bias_B", "q"])
        print(f"\n[{domain.upper()}] Significant pairs (q ≤ {q_threshold}):")
        print(df_out.to_string(index=False))
        df_out.to_csv(out_root / f"{domain}__SIGNIFICANT_PAIRS.csv", index=False)


# -------------------- CLI --------------------
def main():
    ap = argparse.ArgumentParser(description="Analyze bias_type × harm inside (domain × stakeholder) cells.")
    ap.add_argument("--in", dest="in_path", required=True, help="Path to clean_results_human.csv")
    ap.add_argument("--sweep_all", action="store_true", help="Analyze all (domain × stakeholder) cells.")
    args = ap.parse_args()

    df = load_data(Path(args.in_path))

    if args.sweep_all:
        for (domain, who), sub in df.groupby(["domain", "stakeholder"], dropna=False):
            ctab = build_kxH(sub)
            if ctab.shape[0] < 2 or ctab.shape[1] < 2:
                continue
            tag = f"{domain}__{who}".replace(" ", "_")
            out_prefix = OUT_ROOT / tag
            analyze_cell(ctab, out_prefix)
        print(f"\nDone. Results in: {OUT_ROOT.resolve()}")

        # After all analyses, summarize significant pairs
        summarize_significant_pairs(OUT_ROOT, q_threshold=0.10)
        return

    print("Use --sweep_all to process all domain×stakeholder cells at once.")


if __name__ == "__main__":
    main()
