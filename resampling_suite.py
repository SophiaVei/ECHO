# -*- coding: utf-8 -*-
"""
Resampling suite (Permutation + Bootstrap) for FATES/ECHO

Pairs tested (both methods):
  - stakeholder_raw × harm
  - domain × harm
  - bias_type × harm
  - stakeholder_raw × bias_type
  - domain × bias_type

Outputs (tables/resampling/):
  - perm__<v1>__<v2>__dist.csv          (null distribution: chi2, V)
  - boot__<v1>__<v2>__dist.csv          (bootstrap distribution: chi2, V)
  - summary__<v1>__<v2>.csv             (observed stats, perm p-values, boot CIs)
  - ALL_RESULTS_summary.csv             (one row per pair, easy to paste in paper)

CLI examples:
  python resampling_suite.py
  python resampling_suite.py --n 5000 --seed 13
  python resampling_suite.py --n 2000 --domain hiring
  python resampling_suite.py --stakeholder "applicant group"
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

# ------------------ Config ------------------

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "clean_results_human.csv"
OUT_DIR = ROOT / "tables" / "resampling"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BLOCK_KEYS = ["questionnaire_id", "stakeholder_raw", "bias_type", "domain"]
HARM_KEYS  = BLOCK_KEYS + ["harm"]

PAIRS = [
    ("stakeholder_raw", "harm"),
    ("domain",          "harm"),
    ("bias_type",       "harm"),
    ("stakeholder_raw", "bias_type"),
    ("domain",          "bias_type"),
]

# ------------------ Data ------------------

def load_clean(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    needed = {"questionnaire_id","stakeholder_raw","bias_type","domain","harm","votes"}
    miss = needed.difference(df.columns)
    if miss: raise ValueError(f"Missing columns: {miss}")
    # aggregate duplicates at (block × harm)
    df = df.groupby(HARM_KEYS, as_index=False)["votes"].sum()
    return df

def apply_filters(df: pd.DataFrame, domain: str|None, stakeholder: str|None, bias_type: str|None) -> pd.DataFrame:
    out = df.copy()
    if domain:
        out = out[out["domain"].str.lower() == domain.lower()]
    if stakeholder:
        out = out[out["stakeholder_raw"].str.lower() == stakeholder.lower()]
    if bias_type:
        out = out[out["bias_type"].str.lower() == bias_type.lower()]
    return out

# ------------------ Stats ------------------

def contingency(df: pd.DataFrame, v1: str, v2: str) -> pd.DataFrame:
    """v1×v2 crosstab of vote counts."""
    ct = pd.pivot_table(df, values="votes", index=v1, columns=v2, aggfunc="sum", fill_value=0)
    # drop empty rows/cols just in case
    ct = ct.loc[ct.sum(axis=1) > 0, ct.sum(axis=0) > 0]
    return ct

def chi2_and_v(ct: pd.DataFrame):
    """Return chi2, dof, V."""
    obs = ct.to_numpy()
    chi2, p, dof, expected = chi2_contingency(obs, correction=False)
    n = obs.sum()
    r, c = obs.shape
    v = np.sqrt(chi2 / (n * (min(r - 1, c - 1) if min(r,c) > 1 else np.nan)))
    return chi2, dof, v

# ------------------ Permutation ------------------

def permute_labels(df: pd.DataFrame, var_to_shuffle: str, rng: np.random.Generator) -> pd.DataFrame:
    """Shuffle labels of one variable across rows (votes kept)."""
    out = df.copy()
    out[var_to_shuffle] = np.array(out[var_to_shuffle].values)[rng.permutation(len(out))]
    return out

def permutation_test(df: pd.DataFrame, v1: str, v2: str, n: int, seed: int|None):
    rng = np.random.default_rng(seed)
    ct_obs = contingency(df, v1, v2)
    chi2_obs, dof, v_obs = chi2_and_v(ct_obs)
    # choose which side to shuffle (the one with more unique values gives better mixing)
    shuffle_side = v1 if df[v1].nunique() >= df[v2].nunique() else v2

    chi2_null = np.empty(n, dtype=float)
    v_null    = np.empty(n, dtype=float)
    for i in range(n):
        df_p = permute_labels(df, shuffle_side, rng)
        ct_p = contingency(df_p, v1, v2)
        chi2_p, _, v_p = chi2_and_v(ct_p)
        chi2_null[i] = chi2_p
        v_null[i]    = v_p

    # one-sided (greater) p for chi2 and V
    p_chi2 = (np.sum(chi2_null >= chi2_obs) + 1.0) / (n + 1.0)
    p_v    = (np.sum(v_null    >= v_obs)    + 1.0) / (n + 1.0)

    dist = pd.DataFrame({"chi2_null": chi2_null, "V_null": v_null})
    return chi2_obs, dof, v_obs, p_chi2, p_v, dist, shuffle_side

# ------------------ Bootstrap ------------------

def bootstrap_resample(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Resample rows with replacement; keep same number of rows (weighted by 'votes')."""
    idx = rng.integers(0, len(df), size=len(df))
    return df.iloc[idx].copy()

def bootstrap_test(df: pd.DataFrame, v1: str, v2: str, n: int, seed: int|None, ci: float = 0.95):
    rng = np.random.default_rng(seed)
    ct_obs = contingency(df, v1, v2)
    chi2_obs, dof, v_obs = chi2_and_v(ct_obs)

    chi2_boot = np.empty(n, dtype=float)
    v_boot    = np.empty(n, dtype=float)
    for i in range(n):
        df_b = bootstrap_resample(df, rng)
        ct_b = contingency(df_b, v1, v2)
        chi2_b, _, v_b = chi2_and_v(ct_b)
        chi2_boot[i] = chi2_b
        v_boot[i]    = v_b

    alpha = 1.0 - ci
    v_lo, v_hi = np.quantile(v_boot, [alpha/2, 1 - alpha/2])
    chi2_lo, chi2_hi = np.quantile(chi2_boot, [alpha/2, 1 - alpha/2])

    dist = pd.DataFrame({"chi2_boot": chi2_boot, "V_boot": v_boot})
    return chi2_obs, dof, v_obs, (v_lo, v_hi), (chi2_lo, chi2_hi), dist

# ------------------ Runner ------------------

def run_all(df: pd.DataFrame, n: int, seed: int|None, tag: str):
    rows = []
    for v1, v2 in PAIRS:
        # Permutation
        chi2_o, dof, v_o, p_chi2, p_v, perm_dist, shuffled = permutation_test(df, v1, v2, n=n, seed=seed)
        perm_path = OUT_DIR / f"perm__{v1}__{v2}__dist{tag}.csv"
        perm_dist.to_csv(perm_path, index=False)

        # Bootstrap
        chi2_o_b, dof_b, v_o_b, (v_lo, v_hi), (chi2_lo, chi2_hi), boot_dist = bootstrap_test(df, v1, v2, n=n, seed=seed)
        assert abs(chi2_o_b - chi2_o) < 1e-6 and dof_b == dof, "Observed stats mismatch; check data flow."
        boot_path = OUT_DIR / f"boot__{v1}__{v2}__dist{tag}.csv"
        boot_dist.to_csv(boot_path, index=False)

        # Pair summary
        summary = pd.DataFrame([{
            "pair": f"{v1} × {v2}",
            "dof": int(dof),
            "chi2_obs": chi2_o,
            "V_obs": v_o,
            "perm_p_chi2": p_chi2,
            "perm_p_V": p_v,
            "boot_V_lo": v_lo,
            "boot_V_hi": v_hi,
            "boot_chi2_lo": chi2_lo,
            "boot_chi2_hi": chi2_hi,
            "perm_shuffled_side": shuffled,
            "perm_dist_csv": str(perm_path),
            "boot_dist_csv": str(boot_path),
        }])
        sum_path = OUT_DIR / f"summary__{v1}__{v2}{tag}.csv"
        summary.to_csv(sum_path, index=False)
        rows.append(summary)

        # Console print for quick reading
        print(f"\n=== {v1} × {v2} ===")
        print(f"Observed: χ² = {chi2_o:.2f} (dof={dof}), V = {v_o:.3f}")
        print(f"Permutation p-values:  p_chi2 = {p_chi2:.3f},  p_V = {p_v:.3f} (shuffled: {shuffled})")
        print(f"Bootstrap 95% CI (V): [{v_lo:.3f}, {v_hi:.3f}]")
        print(f"Saved: {perm_path.name}, {boot_path.name}, {sum_path.name}")

    all_summary = pd.concat(rows, ignore_index=True)
    all_path = OUT_DIR / f"ALL_RESULTS_summary{tag}.csv"
    all_summary.to_csv(all_path, index=False)
    print(f"\n==> Wrote consolidated summary: {all_path}")

# ------------------ Main ------------------

def main():
    ap = argparse.ArgumentParser(description="Run permutation + bootstrap for all variable pairs.")
    ap.add_argument("--in", dest="in_path", default=str(DATA_PATH), help="Input CSV (clean_results_human.csv)")
    ap.add_argument("--n", type=int, default=1000, help="Number of resamples (default: 1000)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    # Optional filters
    ap.add_argument("--domain", default=None)
    ap.add_argument("--stakeholder", default=None)  # stakeholder_raw
    ap.add_argument("--bias_type", default=None)
    args = ap.parse_args()

    df = load_clean(Path(args.in_path))
    df = apply_filters(df, domain=args.domain, stakeholder=args.stakeholder, bias_type=args.bias_type)

    tag_bits = []
    if args.domain:      tag_bits.append(f"__domain-{args.domain}")
    if args.stakeholder: tag_bits.append(f"__stakeholder-{args.stakeholder.replace(' ','_')}")
    if args.bias_type:   tag_bits.append(f"__bias-{args.bias_type}")
    tag = "".join(tag_bits)

    run_all(df, n=args.n, seed=args.seed, tag=tag)

if __name__ == "__main__":
    main()
