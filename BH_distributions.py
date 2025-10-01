# -*- coding: utf-8 -*-
"""
Compare harm distributions by bias type within a fixed (domain, stakeholder) cell.

Two workflows
-------------
1) Single pair:
   python compare_bias_rows.py --domain hiring --stakeholder_raw "applicant" \
       --bias_a representation --bias_b algorithmic --perm_n 5000

2) Sweep all bias pairs (with omnibus chi-square and BH q-values):
   python compare_bias_rows.py --domain hiring --stakeholder_raw "applicant" \
       --sweep_all --perm_n 5000

Inputs
------
data/clean_results_human.csv
  (expects: questionnaire_id, stakeholder_raw, bias_type, domain, harm, votes)
  'stakeholder' (normalized) is optional; use --stakeholder if you have it.

Outputs (tables/)
-----------------
Single pair mode:
- compare__<domain>__<who>__<bias_a>_vs_<bias_b>__contingency.csv
- compare__<domain>__<who>__<bias_a>_vs_<bias_b>__residuals.csv
- compare__<domain>__<who>__<bias_a>_vs_<bias_b>__summary.txt

Sweep mode:
- sweep__<domain>__<who>__omnibus.txt
- sweep__<domain>__<who>__pairwise.csv  (all pairs with chi2, df, p, perm_p, V, w, BH q)
"""

from __future__ import annotations

import argparse
from pathlib import Path
import itertools
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

try:
    from tqdm import trange
except Exception:
    trange = range  # fallback if tqdm not installed

# -------------------- Config --------------------

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "clean_results_human.csv"
OUT_DIR = ROOT / "tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

REQUIRED_MIN = {
    "questionnaire_id", "stakeholder_raw",
    "bias_type", "domain", "harm", "votes"
}
# 'stakeholder' (normalized) is optional

# -------------------- Helpers --------------------

def drop_all_zero_rows_cols(ctab: pd.DataFrame) -> pd.DataFrame:
    """Drop harms (cols) and bias rows with total == 0."""
    c = ctab.loc[:, ctab.sum(axis=0) > 0]
    r = c.loc[c.sum(axis=1) > 0]
    return r

def safe_chi2(ctab: pd.DataFrame):
    """Run chi2_contingency after dropping all-zero rows/cols. Returns (chi2,p,df,expected) or (np.nan,...)."""
    ctab2 = drop_all_zero_rows_cols(ctab)
    if ctab2.shape[0] < 2 or ctab2.shape[1] < 2:
        # Not enough degrees of freedom after pruning
        return np.nan, np.nan, 0, None
    return chi2_contingency(ctab2.to_numpy(), correction=False)

def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini–Hochberg FDR q-values."""
    pvals = np.asarray(pvals, dtype=float)
    n = pvals.size
    order = np.argsort(pvals)
    ranks = np.arange(1, n + 1, dtype=float)
    q_sorted = pvals[order] * n / ranks
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q = np.empty_like(q_sorted)
    q[order] = q_sorted
    return np.minimum(q, 1.0)

def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = REQUIRED_MIN.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns in input: {missing}")
    key_cols = ["questionnaire_id", "stakeholder_raw", "bias_type", "domain", "harm"]
    df = df.groupby(key_cols, as_index=False)["votes"].sum()
    # keep optional 'stakeholder' if present
    if "stakeholder" not in df.columns:
        df["stakeholder"] = np.nan  # placeholder for API symmetry
    return df

def filter_cell(
    df: pd.DataFrame,
    domain: str,
    stakeholder: str | None,
    stakeholder_raw: str | None,
) -> tuple[pd.DataFrame, str]:
    d = df.copy()
    d = d[d["domain"].str.lower() == domain.lower()]
    if stakeholder is not None and "stakeholder" in d.columns and d["stakeholder"].notna().any():
        d = d[d["stakeholder"].astype(str).str.lower() == stakeholder.lower()]
        who_label = f"stakeholder-{stakeholder}"
    elif stakeholder_raw is not None:
        d = d[d["stakeholder_raw"].str.lower() == stakeholder_raw.lower()]
        who_label = f"stakeholder_raw-{stakeholder_raw}"
    else:
        raise SystemExit("Provide either --stakeholder OR --stakeholder_raw")
    if d.empty:
        raise SystemExit("No rows after filtering. Check domain/stakeholder names.")
    return d, who_label

def build_k_by_h(d: pd.DataFrame) -> pd.DataFrame:
    """Return bias×harm table (K×H) using votes."""
    ctab = pd.pivot_table(
        d, values="votes", index="bias_type", columns="harm", aggfunc="sum", fill_value=0
    ).astype(int)
    # drop empty columns if any
    ctab = ctab.loc[:, ctab.sum(axis=0) > 0]
    return ctab

def build_two_by_k_from_kxH(ctab_kh: pd.DataFrame, bias_a: str, bias_b: str) -> pd.DataFrame:
    rows = {bias_a.lower(): bias_a, bias_b.lower(): bias_b}
    idx_low = [i.lower() for i in ctab_kh.index]
    if bias_a.lower() not in idx_low or bias_b.lower() not in idx_low:
        have = ", ".join(ctab_kh.index.tolist())
        raise SystemExit(f"Requested biases not both present. Available = [{have}]")
    sel = [ctab_kh.index[idx_low.index(bias_a.lower())],
           ctab_kh.index[idx_low.index(bias_b.lower())]]
    out = ctab_kh.loc[sel].copy()
    out.index = [bias_a, bias_b]

    # NEW: drop harm columns that are zero for BOTH rows in this pair
    out = out.loc[:, out.sum(axis=0) > 0]

    return out


def cramers_v_from_chi2(chi2: float, n: int, r: int, c: int) -> float:
    if n <= 0:
        return np.nan
    k = min(r - 1, c - 1)
    if k <= 0:
        return np.nan
    return float(np.sqrt(chi2 / (n * k)))

def cohens_w_from_obs_exp(obs: np.ndarray, exp: np.ndarray) -> float:
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = np.nansum((obs - exp) ** 2 / exp)
    N = np.nansum(obs)
    return float(np.sqrt(chi2 / N)) if N > 0 else np.nan

def standardized_residuals(obs: np.ndarray, exp: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return (obs - exp) / np.sqrt(exp)

def permutation_pvalue_2byH(d_two_bias_rows: pd.DataFrame, iters: int) -> float:
    """
    Empirical two-sided p for 2×H table by shuffling *bias_type* labels within this filtered cell.
    d_two_bias_rows: rows restricted to the two target biases already.
    """
    if iters is None or iters <= 0:
        return np.nan

    ctab_obs = pd.pivot_table(
        d_two_bias_rows, values="votes", index="bias_type", columns="harm",
        aggfunc="sum", fill_value=0
    )
    obs = ctab_obs.to_numpy()
    chi2_obs, _, _, _ = chi2_contingency(obs, correction=False)

    # pre-extract labels
    labels = d_two_bias_rows["bias_type"].values.copy()
    chi2_surr = np.zeros(iters, dtype=float)
    for i in trange(iters):
        np.random.shuffle(labels)
        d_two_bias_rows["bias_perm"] = labels
        ctab_perm = pd.pivot_table(
            d_two_bias_rows, values="votes", index="bias_perm", columns="harm",
            aggfunc="sum", fill_value=0
        )
        # ensure both rows present
        if ctab_perm.shape[0] != 2:
            chi2_surr[i] = np.nan
            continue
        m = ctab_perm.to_numpy()
        chi2_perm, _, _, _ = chi2_contingency(m, correction=False)
        chi2_surr[i] = chi2_perm

    chi2_surr = chi2_surr[np.isfinite(chi2_surr)]
    if chi2_surr.size == 0:
        return np.nan
    p_emp = (np.sum(chi2_surr >= chi2_obs) + 1.0) / (chi2_surr.size + 1.0)
    return float(p_emp)

# -------------------- Workflows --------------------

def run_single_pair(ctab_2xH: pd.DataFrame, out_base: Path, perm_n: int):
    obs = ctab_2xH.to_numpy()
    r, c = obs.shape
    N = int(obs.sum())

    chi2, p, dof, exp = safe_chi2(ctab_2xH)
    if exp is None or dof == 0 or np.isnan(chi2):
        V = np.nan
        w = np.nan
        std_resid = np.full_like(obs, np.nan, dtype=float)
    else:
        V = cramers_v_from_chi2(chi2, N, r, c)
        w = cohens_w_from_obs_exp(obs, exp)
        std_resid = standardized_residuals(obs, exp)

    # shares + diffs
    shares = (ctab_2xH.div(ctab_2xH.sum(axis=1), axis=0)).replace([np.inf, -np.inf], np.nan)
    share_diff = shares.iloc[0] - shares.iloc[1]

    # permutation p
    # reconstruct a narrow df with only the two biases for shuffling
    df_two = ctab_2xH.stack().reset_index()
    df_two.columns = ["bias_type", "harm", "votes"]
    p_perm = permutation_pvalue_2byH(df_two, perm_n) if perm_n and perm_n > 0 else np.nan

    # write
    cont_path = out_base.with_suffix("")  # we'll add endings
    out_cont = Path(str(cont_path) + "__contingency.csv")
    out_resi = Path(str(cont_path) + "__residuals.csv")
    out_sum  = Path(str(cont_path) + "__summary.txt")

    ctab_save = ctab_2xH.copy()
    ctab_save.loc[f"{ctab_2xH.index[0]}__share"] = shares.iloc[0].values
    ctab_save.loc[f"{ctab_2xH.index[1]}__share"] = shares.iloc[1].values
    ctab_save.loc["share_diff__a_minus_b"] = share_diff.values
    ctab_save.to_csv(out_cont)

    resid_df = pd.DataFrame(std_resid, index=ctab_2xH.index, columns=ctab_2xH.columns)
    resid_df.to_csv(out_resi)

    with open(out_sum, "w", encoding="utf-8") as f:
        f.write(f"Table shape: {r}×{c} (N={N})\n")
        f.write(f"Chi-square: χ² = {chi2:.2f}, df = {dof}, p = {p:.4f}\n")
        f.write(f"Cramér's V: {V:.3f}\n")
        f.write(f"Cohen's w:  {w:.3f}\n")
        if np.isfinite(p_perm):
            f.write(f"Permutation p-value (N={perm_n}): {p_perm:.4f}\n")
        f.write("\nTop harms by share difference (row1 - row2):\n")
        for h, v in share_diff.sort_values(ascending=False).items():
            f.write(f"  {h}: {v:+.3f}\n")

    return chi2, p, dof, V, w, p_perm

def run_sweep_all(d: pd.DataFrame, domain: str, who_label: str, perm_n: int):
    """
    1) Omnibus chi-square on K×H
    2) All pairwise 2×H tests + BH correction
    """
    ctab_kh = build_k_by_h(d)
    if ctab_kh.shape[0] < 2:
        raise SystemExit("Fewer than two bias types present after filtering.")
    K, H = ctab_kh.shape
    N = int(ctab_kh.to_numpy().sum())

    # Omnibus
    chi2_omni, p_omni, df_omni, _ = safe_chi2(ctab_kh)

    # Save omnibus summary
    omni_fname = OUT_DIR / f"sweep__{domain}__{who_label.replace(' ','_')}__omnibus.txt"
    with open(omni_fname, "w", encoding="utf-8") as f:
        f.write(f"Omnibus χ² over all bias rows (K×H):\n")
        f.write(f"Table shape: {K}×{H} (N={N})\n")
        f.write(f"Chi-square: χ² = {chi2_omni:.2f}, df = {df_omni}, p = {p_omni:.4f}\n")

    # Pairwise sweep
    rows = ctab_kh.index.tolist()
    results = []
    for a, b in itertools.combinations(rows, 2):
        ctab_2xH = build_two_by_k_from_kxH(ctab_kh, a, b)
        obs = ctab_2xH.to_numpy()
        r, c = obs.shape
        n = int(obs.sum())
        chi2, p, df, exp = safe_chi2(ctab_2xH)
        if exp is None or df == 0 or np.isnan(chi2):
            V = np.nan
            w = np.nan
        else:
            V = cramers_v_from_chi2(chi2, n, r, c)
            w = cohens_w_from_obs_exp(obs, exp)
        # permutation p
        df_two = ctab_2xH.stack().reset_index()
        df_two.columns = ["bias_type", "harm", "votes"]
        perm_p = permutation_pvalue_2byH(df_two, perm_n) if perm_n and perm_n > 0 else np.nan
        results.append([a, b, chi2, df, p, perm_p, V, w, n])

    res_df = pd.DataFrame(results, columns=[
        "bias_a", "bias_b", "chi2", "df", "p_value", "perm_p", "cramers_v", "cohens_w", "N"
    ])
    # BH correction on classical p-values (you can also do on perm_p if you prefer)
    res_df["q_value"] = bh_fdr(res_df["p_value"].values)
    # sort by perm_p then p
    if res_df["perm_p"].notna().any():
        res_df = res_df.sort_values(["perm_p", "p_value"])
    else:
        res_df = res_df.sort_values("p_value")

    sweep_fname = OUT_DIR / f"sweep__{domain}__{who_label.replace(' ','_')}__pairwise.csv"
    res_df.to_csv(sweep_fname, index=False)

    return (chi2_omni, p_omni, df_omni), sweep_fname, omni_fname

# -------------------- Main --------------------

def main():
    ap = argparse.ArgumentParser(description="Compare harm distributions by bias type within a fixed (domain, stakeholder) cell.")
    ap.add_argument("--in", dest="in_path", default=str(DATA_PATH), help="Input CSV path")
    ap.add_argument("--domain", required=True, help='Domain to filter, e.g., "hiring"')
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--stakeholder", help='Stakeholder (normalized), e.g., "applicant"')
    group.add_argument("--stakeholder_raw", help='Stakeholder_raw (exact header), e.g., "applicant group"')

    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--sweep_all", action="store_true", help="Test omnibus + all pairwise bias comparisons")
    mode.add_argument("--bias_a", help='First bias type, e.g., "representation"')

    ap.add_argument("--bias_b", help='Second bias type, e.g., "algorithmic" (required when not using --sweep_all)')
    ap.add_argument("--perm_n", type=int, default=0, help="Number of permutations for empirical p (default: 0 = skip)")
    args = ap.parse_args()

    df = load_data(Path(args.in_path))
    sub, who_label = filter_cell(
        df, domain=args.domain,
        stakeholder=args.stakeholder,
        stakeholder_raw=args.stakeholder_raw,
    )

    if args.sweep_all:
        # Omnibus + all pairwise with BH
        (chi2_omni, p_omni, df_omni), sweep_path, omni_path = run_sweep_all(sub, args.domain, who_label, args.perm_n)
        print("\n=== Omnibus (all bias rows) ===")
        print(f"Domain: {args.domain} | {who_label}")
        print(f"χ² = {chi2_omni:.2f} | df = {df_omni} | p = {p_omni:.4f}")
        print(f"Saved omnibus: {omni_path}")
        print(f"Saved pairwise sweep: {sweep_path}\n")
    else:
        if not args.bias_a or not args.bias_b:
            raise SystemExit("Provide --bias_a and --bias_b for single-pair mode.")
        ctab_kh = build_k_by_h(sub)
        ctab_2xH = build_two_by_k_from_kxH(ctab_kh, args.bias_a, args.bias_b)

        safe_who = who_label.replace(" ", "_")
        base = OUT_DIR / f"compare__{args.domain}__{safe_who}__{args.bias_a}_vs_{args.bias_b}"
        chi2, p, dof, V, w, p_perm = run_single_pair(ctab_2xH, base, args.perm_n)

        print("\n=== Row-to-row harm-distribution comparison ===")
        print(f"Domain: {args.domain} | {who_label}")
        print(f"Bias A vs B: {args.bias_a}  vs  {args.bias_b}")
        r, c = ctab_2xH.shape
        N = int(ctab_2xH.to_numpy().sum())
        print(f"Table shape: {r}×{c} (N={N})")
        print(f"χ² = {chi2:.2f} | df = {dof} | p = {p:.4f} | Cramér's V = {V:.3f} | Cohen's w = {w:.3f}")
        if np.isfinite(p_perm):
            print(f"Permutation p (N={args.perm_n}) = {p_perm:.4f}")
        print(f"Saved under: {base}__*\n")

if __name__ == "__main__":
    main()

# to run: python BH_distributions.py --domain hiring --stakeholder_raw "applicant" --sweep_all --perm_n 5000
# change "applicant" for the stakeholder you want to check, or domain "hiring", respectively
