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
"""

from __future__ import annotations
import argparse
from pathlib import Path
import itertools
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.mosaicplot import mosaic
import matplotlib.colors as mcolors

# -------------------- Mosaic plotting --------------------
def plot_mosaic(ctab: pd.DataFrame, stdres_df: pd.DataFrame, out_prefix: Path):
    """
    Create a mosaic plot colored by standardized residuals.
    Red = over-represented, Blue = under-represented.
    Compatible with older statsmodels.mosaic (no `values` kw).
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from statsmodels.graphics.mosaicplot import mosaic
    import matplotlib.colors as mcolors

    # Data for mosaic: dict mapping (bias_type, harm) -> count
    data = {(i, j): int(v) for (i, j), v in ctab.stack().items()}

    # Residuals for color shading (lookup by same (bias, harm) tuple)
    resid_map = stdres_df.stack().to_dict()

    cmap = plt.cm.coolwarm
    norm = mcolors.TwoSlopeNorm(vmin=-3, vcenter=0, vmax=3)

    # Rectangle properties per cell
    props = {}
    for key, _ in data.items():
        r = resid_map.get(key, np.nan)
        color = cmap(norm(r)) if np.isfinite(r) else (0.8, 0.8, 0.8, 1.0)
        props[key] = {"color": color}

    fig, ax = plt.subplots()
    mosaic(data, gap=0.01, properties=props, ax=ax)

    # Titles/labels
    title_tag = out_prefix.stem.replace("__", " – ")
    ax.set_title(f"Mosaic plot: Bias × Harm\n({title_tag})", fontsize=12)
    ax.set_xlabel("Bias type")
    ax.set_ylabel("Harm type")

    # Colorbar legend
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.7)
    cbar.set_label("Standardized residual (z-score)")

    plt.tight_layout()
    out_path = out_prefix.with_name(out_prefix.name + "__mosaic.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved mosaic plot: {out_path}")


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
    # obs and exp must be same shape
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = np.nansum((obs - exp) ** 2 / exp)
    N = np.nansum(obs)
    return float(np.sqrt(chi2 / N)) if N > 0 else np.nan

def safe_chi2_return_table(table: pd.DataFrame):
    """
    Omnibus χ² after dropping all-zero rows/cols.
    Returns: (chi2, p, df, expected, pruned_table)
    """
    t = table.loc[:, table.sum(axis=0) > 0]
    t = t.loc[t.sum(axis=1) > 0]
    if t.shape[0] < 2 or t.shape[1] < 2:
        return np.nan, np.nan, 0, None, t
    chi2, p, df, exp = chi2_contingency(t.to_numpy(), correction=False)
    return chi2, p, df, exp, t

def per_harm_fisher_2x2(ctab_2xH: pd.DataFrame, add_smooth: float = 0.5) -> pd.DataFrame:
    """For pair (A,B), run Fisher per harm and compute smoothed log-odds ratio."""
    assert ctab_2xH.shape[0] == 2
    a_name, b_name = ctab_2xH.index.tolist()
    A = int(ctab_2xH.loc[a_name].sum())
    B = int(ctab_2xH.loc[b_name].sum())
    harms = list(ctab_2xH.columns)
    pvals, lors, a_counts, b_counts = [], [], [], []
    for h in harms:
        a = int(ctab_2xH.loc[a_name, h]); b = int(ctab_2xH.loc[b_name, h])
        _, p = fisher_exact([[a, A - a], [b, B - b]], alternative="two-sided")
        # smoothed LOR (Haldane-Anscombe 0.5)
        a1, b1 = a + add_smooth, b + add_smooth
        A1, B1 = (A - a) + add_smooth, (B - b) + add_smooth
        lor = np.log((a1 / A1) / (b1 / B1))
        pvals.append(p); lors.append(lor); a_counts.append(a); b_counts.append(b)
    qvals = bh_fdr(np.array(pvals))
    return pd.DataFrame({"harm": harms, "a_harm": a_counts, "b_harm": b_counts, "p": pvals, "q": qvals, "lor": lors})

def permutation_pvalue_2byH(df_two_bias_rows: pd.DataFrame, iters: int) -> float:
    """
    Empirical p for a 2×H table by shuffling bias labels within the cell.
    Robust to sparse data: prunes all-zero rows/cols before χ².
    """
    if not iters or iters <= 0:
        return np.nan

    def prune(ct: pd.DataFrame) -> pd.DataFrame:
        # drop harms with zero total and any zero-total bias row
        ct = ct.loc[ct.sum(axis=1) > 0, ct.sum(axis=0) > 0]
        return ct

    # ----- observed χ² on pruned 2×H -----
    ctab_obs = pd.pivot_table(
        df_two_bias_rows, values="votes",
        index="bias_type", columns="harm",
        aggfunc="sum", fill_value=0
    )
    ctab_obs = prune(ctab_obs)
    if ctab_obs.shape[0] < 2 or ctab_obs.shape[1] < 2:
        # Not enough df to compute χ² after pruning
        return np.nan
    chi2_obs, _, _, _ = chi2_contingency(ctab_obs.to_numpy(), correction=False)

    # ----- permutations -----
    labels_orig = df_two_bias_rows["bias_type"].to_numpy().copy()
    rng = np.random.default_rng()
    chi2_null = np.empty(iters, dtype=float)

    for i in range(iters):
        labels = labels_orig.copy()
        rng.shuffle(labels)
        df_two_bias_rows["bias_perm"] = labels

        ctab_perm = pd.pivot_table(
            df_two_bias_rows, values="votes",
            index="bias_perm", columns="harm",
            aggfunc="sum", fill_value=0
        )
        ctab_perm = prune(ctab_perm)
        if ctab_perm.shape[0] < 2 or ctab_perm.shape[1] < 2:
            chi2_null[i] = np.nan
            continue

        chi2_perm, _, _, _ = chi2_contingency(ctab_perm.to_numpy(), correction=False)
        chi2_null[i] = chi2_perm

    chi2_null = chi2_null[np.isfinite(chi2_null)]
    if chi2_null.size == 0:
        return np.nan

    # right-tail p-value
    p_emp = (np.sum(chi2_null >= chi2_obs) + 1.0) / (chi2_null.size + 1.0)
    return float(p_emp)


# -------------------- Data prep --------------------
def load_data(in_path: Path) -> pd.DataFrame:
    df = pd.read_csv(in_path)
    missing = REQUIRED.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns in input: {missing}")
    # aggregate duplicates just in case
    key = ["questionnaire_id", "stakeholder_raw", "bias_type", "domain", "harm"]
    df = df.groupby(key, as_index=False)["votes"].sum()
    # add normalized 'stakeholder' if absent (fallback to stakeholder_raw)
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
    """K×H (bias × harm) votes table for the cell."""
    ctab = pd.pivot_table(
        d, values="votes", index="bias_type", columns="harm", aggfunc="sum", fill_value=0
    ).astype(int)
    # keep harms with any presence and biases with any presence
    ctab = ctab.loc[:, ctab.sum(axis=0) > 0]
    ctab = ctab.loc[ctab.sum(axis=1) > 0]
    return ctab

def row_shares(ctab: pd.DataFrame) -> pd.DataFrame:
    denom = ctab.sum(axis=1).replace(0, np.nan)
    return (ctab.T / denom).T  # rows sum to 1

# -------------------- Main analysis per cell --------------------
def analyze_cell(ctab: pd.DataFrame, out_prefix: Path, perm_n: int = 0):
    """Run omnibus K×H + all pairwise 2×H tests; write CSVs and a short text summary."""
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    # Save K×H observed & shares
    ctab.to_csv(out_prefix.with_name(out_prefix.name + "__kxH_counts.csv"))
    row_shares(ctab).to_csv(out_prefix.with_name(out_prefix.name + "__kxH_rowshares.csv"))

    # Omnibus χ² on K×H (with pruned table)
    chi2, p, df, exp, t_pruned = safe_chi2_return_table(ctab)
    # Standardized residuals aligned back to full K×H: fill NaN outside pruned region
    stdres_df = pd.DataFrame(np.nan, index=ctab.index, columns=ctab.columns, dtype=float)
    if exp is not None:
        stdres_small = standardized_residuals(t_pruned.to_numpy(), exp)
        stdres_df.loc[t_pruned.index, t_pruned.columns] = stdres_small
    stdres_df.to_csv(out_prefix.with_name(out_prefix.name + "__kxH_stdresid.csv"))

    K, H = ctab.shape
    N = int(ctab.values.sum())
    min_exp = np.nanmin(exp) if exp is not None else np.nan
    under5 = int(np.sum(exp < 5)) if exp is not None else np.nan
    V = cramers_v(chi2, N, *t_pruned.shape) if np.isfinite(chi2) else np.nan

    # Write omnibus summary
    with open(out_prefix.with_name(out_prefix.name + "__omnibus.txt"), "w", encoding="utf-8") as f:
        f.write(f"Table: K×H (original) = {K}×{H}; used for χ² = {t_pruned.shape[0]}×{t_pruned.shape[1]} (N={N})\n")
        f.write(f"Omnibus χ² = {chi2:.2f}, df = {df}, p = {p:.4f}, Cramér's V = {V:.3f}\n")
        if exp is not None:
            f.write(f"Cells with expected < 5: {under5} / {t_pruned.shape[0]*t_pruned.shape[1]}, min expected = {min_exp:.2f}\n")

    # Pairwise 2×H tests
    rows = ctab.index.tolist()
    pair_rows = []
    for a, b in itertools.combinations(rows, 2):
        ctab_2xH_full = ctab.loc[[a, b]].copy()
        # χ² on pruned pair table
        chi2p, pp, dff, exp2, t2 = safe_chi2_return_table(ctab_2xH_full)
        obs2 = t2.to_numpy()
        n2 = int(obs2.sum())
        if exp2 is None or dff == 0 or not np.isfinite(chi2p):
            Vp = np.nan; w = np.nan
        else:
            Vp = cramers_v(chi2p, n2, *obs2.shape)
            w = cohens_w_from_obs_exp(obs2, exp2)

        # Optional permutation p (computed on the *full* pair table — OK)
        df_two = ctab_2xH_full.stack().reset_index()
        df_two.columns = ["bias_type", "harm", "votes"]
        perm_p = permutation_pvalue_2byH(df_two, perm_n) if perm_n and perm_n > 0 else np.nan

        pair_rows.append([a, b, chi2p, dff, pp, perm_p, Vp, w, n2])

        # Per-harm 2×2 Fisher + shares + LOR (use full pair table so every harm is reported)
        shares = row_shares(ctab_2xH_full)
        pA = shares.loc[a]; pB = shares.loc[b]
        delta = pA - pB
        perharm = per_harm_fisher_2x2(ctab_2xH_full).set_index("harm")
        out_ph = pd.DataFrame({
            "share_A": pA, "share_B": pB, "share_diff_A_minus_B": delta,
            "a_harm": perharm["a_harm"], "b_harm": perharm["b_harm"],
            "p": perharm["p"], "q": perharm["q"], "lor": perharm["lor"],
        })
        out_ph.to_csv(out_prefix.with_name(out_prefix.name + f"__PAIR__{a}__vs__{b}__perharm.csv"))

    pair_df = pd.DataFrame(pair_rows, columns=[
        "bias_a","bias_b","chi2","df","p_value","perm_p","cramers_v","cohens_w","N"
    ])
    pair_df["q_value"] = bh_fdr(pair_df["p_value"].to_numpy())
    if pair_df["perm_p"].notna().any():
        pair_df = pair_df.sort_values(["perm_p","p_value"])
    else:
        pair_df = pair_df.sort_values("p_value")
    pair_df.to_csv(out_prefix.with_name(out_prefix.name + "__pairwise.csv"), index=False)
    # Create mosaic plot for this cell
    try:
        plot_mosaic(ctab, stdres_df, out_prefix)
    except Exception as e:
        print(f"[Warning] Mosaic plot failed for {out_prefix.name}: {e}")

    # Console summary
    print(f"\n=== {out_prefix.name} ===")
    print(f"K×H (orig) {K}×{H}; χ² used {t_pruned.shape[0]}×{t_pruned.shape[1]} (N={N}) | χ² = {chi2:.2f}, df={df}, p={p:.4f}, V={V:.3f}")
    if exp is not None:
        print(f"Expected<5: {under5}/{t_pruned.shape[0]*t_pruned.shape[1]}, min={min_exp:.2f}")
    print("Top 5 smallest pairwise p-values (classical):")
    print(pair_df[["bias_a","bias_b","p_value","perm_p","cramers_v"]].head(5).to_string(index=False))

# -------------------- CLI --------------------
def main():
    ap = argparse.ArgumentParser(description="Analyze bias_type × harm inside (domain × stakeholder) cells.")
    ap.add_argument("--in", dest="in_path", required=True, help="Path to clean_results_human.csv")
    ap.add_argument("--domain", help="Domain to filter (e.g., hiring). Use with --stakeholder.")
    ap.add_argument("--stakeholder", help="Stakeholder (normalized). Use with --domain.")
    ap.add_argument("--sweep_all", action="store_true", help="Analyze all (domain × stakeholder) cells.")
    ap.add_argument("--perm_n", type=int, default=0, help="Permutation count for pairwise tests (default 0=off)")
    args = ap.parse_args()

    df = load_data(Path(args.in_path))

    if args.sweep_all:
        for (domain, who), sub in df.groupby(["domain","stakeholder"], dropna=False):
            ctab = build_kxH(sub)
            if ctab.shape[0] < 2 or ctab.shape[1] < 2:
                continue
            tag = f"{str(domain).strip()}__{str(who).strip()}".replace(" ","_")
            out_prefix = OUT_ROOT / tag
            analyze_cell(ctab, out_prefix, perm_n=args.perm_n)
        print(f"\nDone. Results in: {OUT_ROOT.resolve()}")
        return

    # single cell
    if not args.domain or not args.stakeholder:
        raise SystemExit("Provide --domain and --stakeholder for a single-cell analysis, or use --sweep_all.")
    sub = filter_cell(df, args.domain, args.stakeholder)
    ctab = build_kxH(sub)
    if ctab.shape[0] < 2 or ctab.shape[1] < 2:
        raise SystemExit("Not enough rows/columns in this cell to test (need ≥2 biases and ≥2 harms).")
    tag = f"{args.domain}__{args.stakeholder}".replace(" ","_")
    out_prefix = OUT_ROOT / tag
    analyze_cell(ctab, out_prefix, perm_n=args.perm_n)
    print(f"\nDone. Results in: {OUT_ROOT.resolve()}")

if __name__ == "__main__":
    main()
