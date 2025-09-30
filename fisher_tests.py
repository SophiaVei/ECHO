# -*- coding: utf-8 -*-
"""
Fisher's Exact Tests on FATES (clean_results_human.csv)

What it does
------------
- Builds 2×2 tables like:
      A == a_level    |  A != a_level
  B == b_level            a                 c
  B != b_level            b                 d
  and runs Fisher's exact test on [[a, b], [c, d]].

- Works in two modes:
  1) Single test: specify --a, --a_level, --b, --b_level
  2) Sweep: fix A (and its focal level) and sweep Fisher across *all* levels
     of B (or vice-versa). Includes Benjamini–Hochberg FDR correction.

Inputs
------
- data/clean_results_human.csv  (expects columns: questionnaire_id, stakeholder_raw,
  bias_type, domain, harm, votes ...)

Examples
--------
# Single test: "representation" bias vs "privacy violation" harm
python fisher_tests.py --a bias_type --a_level representation --b harm --b_level "privacy violation"

# Sweep across all harms for one bias type
python fisher_tests.py --a bias_type --a_level representation --b harm --sweep b

# Sweep across all bias types for one harm
python fisher_tests.py --a harm --a_level "privacy violation" --b bias_type --sweep b

# Add filters (domain/stakeholder/bias_type)
python fisher_tests.py --a bias_type --a_level measurement --b harm --sweep b --domain hiring
python fisher_tests.py --a stakeholder_raw --a_level "applicant group" --b harm --sweep b

Outputs
-------
- Prints tidy summaries to console
- Writes CSV to tables/ with counts (a,b,c,d), odds ratio, p-value, and BH q-value when sweeping
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

# -------------------- Config --------------------

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "clean_results_human.csv"
OUT_DIR = ROOT / "tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BLOCK_KEYS = ["questionnaire_id", "stakeholder_raw", "bias_type", "domain"]
HARM_KEYS = BLOCK_KEYS + ["harm"]

# -------------------- Data --------------------

def load_clean(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    needed = {"questionnaire_id","stakeholder_raw","bias_type","domain","harm","votes"}
    missing = needed.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Aggregate duplicates at (block × harm) level to be safe
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

# -------------------- 2×2 builder --------------------

def two_by_two_counts(df: pd.DataFrame, a: str, a_level: str, b: str, b_level: str) -> tuple[int,int,int,int]:
    """Return (a, b, c, d) counts based on vote totals."""
    if a not in df.columns or b not in df.columns:
        raise ValueError(f"Columns not found: {a}, {b}")

    # cast to lower for stable matching
    a_level_low = a_level.lower()
    b_level_low = b_level.lower()

    A_is = df[a].str.lower() == a_level_low
    B_is = df[b].str.lower() == b_level_low

    a_ct = int(df[A_is & B_is]["votes"].sum())                # A==a_level & B==b_level
    b_ct = int(df[A_is & ~B_is]["votes"].sum())               # A==a_level & B!=b_level
    c_ct = int(df[~A_is & B_is]["votes"].sum())               # A!=a_level & B==b_level
    d_ct = int(df[~A_is & ~B_is]["votes"].sum())              # A!=a_level & B!=b_level
    return a_ct, b_ct, c_ct, d_ct

def fisher_test_from_counts(a:int,b:int,c:int,d:int, alternative:str="two-sided"):
    """Run Fisher's exact test; returns odds_ratio, p."""
    table = [[a, b], [c, d]]
    orat, pval = fisher_exact(table, alternative=alternative)
    return orat, pval

# -------------------- Utilities --------------------

def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini–Hochberg FDR (q-values) for a list/array of p-values."""
    pvals = np.asarray(pvals, dtype=float)
    n = pvals.size
    order = np.argsort(pvals)
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(1, n+1)
    q = pvals * n / ranks
    # enforce monotonicity
    q_sorted = np.minimum.accumulate(q[order][::-1])[::-1]
    q_out = np.empty_like(q_sorted)
    q_out[order] = q_sorted
    return np.minimum(q_out, 1.0)

# -------------------- Workflows --------------------

def run_single(df: pd.DataFrame, a: str, a_level: str, b: str, b_level: str, alternative: str):
    a_ct, b_ct, c_ct, d_ct = two_by_two_counts(df, a, a_level, b, b_level)
    orat, pval = fisher_test_from_counts(a_ct, b_ct, c_ct, d_ct, alternative=alternative)

    print("\n=== Fisher's Exact Test (single 2×2) ===")
    print(f"A: {a} == '{a_level}'   vs   A != '{a_level}'")
    print(f"B: {b} == '{b_level}'   vs   B != '{b_level}'")
    print("2×2 table (votes):")
    print(f"               {b}=={b_level:<20}  {b}!={b_level}")
    print(f"A=={a_level:<14} {a_ct:>8}               {b_ct:>8}")
    print(f"A!={a_level:<14} {c_ct:>8}               {d_ct:>8}")
    print(f"Odds ratio: {orat:.4g}   |   p ({alternative}) = {pval:.4g}")

def run_sweep(df: pd.DataFrame, a: str, a_level: str, b: str, alt: str, sweep_side: str, outfile: Path):
    """
    Sweep Fisher across all levels on the chosen side.
    sweep_side ∈ {'b','a'} meaning: vary B levels (holding A fixed), or vary A levels (holding B fixed).
    """
    results = []
    if sweep_side.lower() == "b":
        levels = sorted(df[b].dropna().unique().tolist())
        for lev in levels:
            a_ct, b_ct, c_ct, d_ct = two_by_two_counts(df, a, a_level, b, str(lev))
            orat, p = fisher_test_from_counts(a_ct, b_ct, c_ct, d_ct, alternative=alt)
            results.append((str(lev), a_ct, b_ct, c_ct, d_ct, orat, p))
        sweep_label = f"{b} levels"
    elif sweep_side.lower() == "a":
        levels = sorted(df[a].dropna().unique().tolist())
        for lev in levels:
            a_ct, b_ct, c_ct, d_ct = two_by_two_counts(df, a, str(lev), b, a_level)  # reuse a_level as B's focal level
            orat, p = fisher_test_from_counts(a_ct, b_ct, c_ct, d_ct, alternative=alt)
            results.append((str(lev), a_ct, b_ct, c_ct, d_ct, orat, p))
        sweep_label = f"{a} levels"
    else:
        raise ValueError("--sweep must be 'a' or 'b'")

    df_out = pd.DataFrame(results, columns=[
        "level", "a_ct", "b_ct", "c_ct", "d_ct", "odds_ratio", "p_value"
    ])
    df_out["q_value"] = bh_fdr(df_out["p_value"].values)

    df_out.to_csv(outfile, index=False)
    print(f"\n=== Fisher sweep over {sweep_label} ===")
    print(f"Fixed A focal: {a} == '{a_level}'   (sweeping levels of {b} if --sweep b; else sweeping {a})")
    print(df_out.sort_values("p_value").head(12).to_string(index=False))
    print(f"\nSaved: {outfile}")

# -------------------- Main --------------------

def main():
    ap = argparse.ArgumentParser(description="Run Fisher's Exact Tests on FATES.")
    ap.add_argument("--in", dest="in_path", default=str(DATA_PATH), help="Input CSV (clean_results_human.csv)")
    ap.add_argument("--a", required=True, help="First variable (e.g., bias_type, stakeholder_raw, harm, domain)")
    ap.add_argument("--a_level", required=True, help="Focal level for variable A")
    ap.add_argument("--b", required=True, help="Second variable")
    ap.add_argument("--b_level", default=None, help="Focal level for variable B (required for single test; omitted for --sweep b)")
    ap.add_argument("--sweep", choices=["a","b"], default=None, help="Sweep Fisher across all levels of this side")
    ap.add_argument("--alternative", choices=["two-sided","less","greater"], default="two-sided", help="Alternative hypothesis")
    # Optional filters to restrict the dataset before building counts
    ap.add_argument("--domain", default=None)
    ap.add_argument("--stakeholder", default=None)  # stakeholder_raw
    ap.add_argument("--bias_type", default=None)

    args = ap.parse_args()

    df = load_clean(Path(args.in_path))
    df = apply_filters(df, domain=args.domain, stakeholder=args.stakeholder, bias_type=args.bias_type)

    if args.sweep is None:
        if not args.b_level:
            raise SystemExit("For a single test, you must provide --b_level.")
        run_single(df, args.a, args.a_level, args.b, args.b_level, args.alternative)
    else:
        # Compose output filename
        bits = [f"fisher__A-{args.a}_eq_{args.a_level}__sweep-{args.sweep}_over_{args.b}"]
        if args.domain:      bits.append(f"domain-{args.domain}")
        if args.stakeholder: bits.append(f"stakeholder-{args.stakeholder.replace(' ','_')}")
        if args.bias_type:   bits.append(f"bias-{args.bias_type}")
        outfile = OUT_DIR / ("__".join(bits) + ".csv")
        run_sweep(df, args.a, args.a_level, args.b, args.alternative, args.sweep, outfile)

if __name__ == "__main__":
    main()
