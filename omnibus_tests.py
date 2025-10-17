# -*- coding: utf-8 -*-
"""
Omnibus χ² tests per (domain × stakeholder_raw) to assess whether the
distribution of harms differs across bias types BEFORE pairwise testing.

INPUT
- --in:  path to data/clean_results_human.csv

OUTPUT
- --out: CSV with one row per (domain × stakeholder_raw):
         domain, stakeholder_raw, n_bias, n_harms, n_votes, chi2, dof, p,
         cramers_v, min_expected, pct_expected_lt5, small_counts_flag

NOTES
- Uses χ² test of homogeneity (scipy.stats.chi2_contingency) on the
  bias×harm contingency table of votes (after dropping all-zero columns/rows).
- Flags potentially unstable results when expected counts are small.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

REQUIRED = {"questionnaire_id", "stakeholder_raw", "bias_type", "domain", "harm", "votes"}

def load_data(in_path: Path) -> pd.DataFrame:
    df = pd.read_csv(in_path)
    missing = REQUIRED.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns in input: {missing}")
    key = ["questionnaire_id", "stakeholder_raw", "bias_type", "domain", "harm"]
    df = df.groupby(key, as_index=False)["votes"].sum()
    # normalize casing/whitespace
    df["bias_type"] = df["bias_type"].str.strip().str.lower()
    df["domain"] = df["domain"].str.strip()
    df["stakeholder_raw"] = df["stakeholder_raw"].str.strip()
    df["harm"] = df["harm"].str.strip()
    return df

def build_ctab(sub: pd.DataFrame) -> pd.DataFrame:
    """Bias × harm table of votes with all-zero rows/cols dropped."""
    ctab = pd.pivot_table(
        sub, values="votes", index="bias_type", columns="harm",
        aggfunc="sum", fill_value=0
    ).astype(int)
    ctab = ctab.loc[:, ctab.sum(axis=0) > 0]       # drop empty harms
    ctab = ctab.loc[ctab.sum(axis=1) > 0, :]       # drop empty biases
    return ctab

def cramers_v(chi2: float, n: int, r: int, c: int) -> float:
    """Cramér’s V for r×c contingency table."""
    if n <= 0 or r < 2 or c < 2:
        return np.nan
    return float(np.sqrt(chi2 / (n * (min(r, c) - 1))))

def analyze(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (domain, who), sub in df.groupby(["domain", "stakeholder_raw"], dropna=False):
        ctab = build_ctab(sub)
        if ctab.shape[0] < 2 or ctab.shape[1] < 2:
            # need at least 2 biases and 2 harms to run omnibus χ²
            continue

        obs = ctab.values
        n_votes = int(obs.sum())
        chi2, p, dof, expected = chi2_contingency(obs, correction=False)
        r, c = obs.shape
        v = cramers_v(chi2, n_votes, r, c)

        min_exp = float(expected.min())
        pct_lt5 = float((expected < 5).sum() / expected.size) if expected.size else np.nan
        small_flag = bool(min_exp < 1 or pct_lt5 > 0.2)  # common rules of thumb

        rows.append({
            "domain": domain,
            "stakeholder_raw": who,
            "n_bias": r,
            "n_harms": c,
            "n_votes": n_votes,
            "chi2": chi2,
            "dof": dof,
            "p": p,
            "cramers_v": v,
            "min_expected": min_exp,
            "pct_expected_lt5": pct_lt5,
            "small_counts_flag": small_flag,
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["domain", "stakeholder_raw"], kind="mergesort").reset_index(drop=True)
    return out

def main():
    ap = argparse.ArgumentParser(description="Omnibus χ² tests per (domain × stakeholder_raw).")
    ap.add_argument("--in", dest="in_path", default="data/clean_results_human_plusllm.csv",
                    help="Input CSV path (clean_results_human.csv)")
    ap.add_argument("--out", dest="out_path", default="tables_py/omnibus_domain_stakeholder.csv",
                    help="Output CSV path for omnibus results")
    args = ap.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = load_data(in_path)
    res = analyze(df)
    if res.empty:
        print("No valid (domain × stakeholder) cells with ≥2 biases and ≥2 harms. Nothing to write.")
        return

    res.to_csv(out_path, index=False)
    print(f"Wrote omnibus χ² results to: {out_path.resolve()}")
    print("\nAll omnibus results:")
    # print ALL rows to console
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 200):
        print(res.to_string(index=False))

if __name__ == "__main__":
    main()
