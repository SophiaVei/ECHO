# -*- coding: utf-8 -*-
"""
Multinomial logistic regression of harm ~ bias_type, per (domain × stakeholder_raw).

Now with *automatic collapsing of rare harms* per cell to stabilize estimates.

What it does
------------
For each (domain, stakeholder_raw) cell (or a single pair you specify),
it fits:

    harm_class ~ C(bias_type, Treatment(reference=bias_ref))

- Dependent variable: harm (categorical), possibly with a collapsed 'other (collapsed)' level
- Predictor: bias_type (categorical, with explicit baseline)
- Vote counts are used as frequency weights by expanding rows.

New (sparse-handling)
---------------------
Per cell, harms are collapsed into 'other (collapsed)' if they:
  - have total votes < --min-harm-count  (default: 10), OR
  - have vote share < --min-harm-share   (optional), OR
  - (if --require-bias-overlap) appear under <2 distinct bias types.

We avoid choosing the 'other (collapsed)' as harm baseline if a non-other harm exists.

Outputs (one CSV per analyzed cell)
-----------------------------------
Rows for each (non-baseline harm, bias_type indicator):
  domain, stakeholder_raw, n_votes, harm, harm_ref, bias_type, bias_ref,
  OR, ci_low, ci_high, coef, se, z, p, stars, converged, note (if any)

Also writes a mapping CSV listing which harms (if any) were collapsed.

Stars (two-sided, z-scale):
  **** : |z| ≥ 3.29  (p < .001)
  ***  : |z| ≥ 2.58  (p < .01)
  **   : |z| ≥ 1.96  (p < .05)
  *    : |z| ≥ 1.645 (p < .10)

Examples
--------
# Run for ALL (domain × stakeholder_raw) cells
python multinomial_logistic_regression.py --in data/clean_results_human_plusllm.csv --out out_mlogit

# Single cell with custom baselines and different collapse thresholds
python multinomial_logistic_regression.py --in data/clean_results_human_plusllm.csv \
  --domain "hiring" --stakeholder "developer" \
  --bias-ref "algorithmic" --harm-ref "alienation" \
  --min-harm-count 8 --require-bias-overlap \
  --out out_mlogit
"""

from __future__ import annotations
import argparse
from pathlib import Path
import re
import unicodedata
import warnings
import math

import numpy as np
import pandas as pd
import statsmodels.api as sm
from patsy import dmatrix

# ------------------------- Canonicalization -------------------------
def _canon(s: str) -> str:
    """NFKC normalize, unify hyphens, collapse whitespace, lowercase + strip."""
    s = unicodedata.normalize("NFKC", str(s))
    s = re.sub(r"[\u2010\u2011\u2012\u2013\u2014\u2212]", "-", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip().lower()

REQUIRED_COLS = {
    "questionnaire_id",
    "stakeholder_raw",
    "bias_type",
    "domain",
    "harm",
    "votes",
}

OTHER_LABEL = "other (collapsed)"

# ------------------------- Helpers -------------------------
def stars_from_z(z: float) -> str:
    a = abs(z)
    if a >= 3.29: return "****"  # p < .001
    if a >= 2.58: return "***"   # p < .01
    if a >= 1.96: return "**"    # p < .05
    if a >= 1.645: return "*"    # p < .10
    return ""

def p_from_z(z: float) -> float:
    # two-sided p using std normal via erf; p = 2*(1-Phi(|z|)) = 1 - erf(|z|/sqrt(2))
    return 1.0 - math.erf(abs(z) / math.sqrt(2.0))

def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = REQUIRED_COLS.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Canonicalize matching columns; keep original stakeholder_raw for display
    df["bias_type"] = df["bias_type"].map(_canon)
    df["domain"] = df["domain"].map(_canon)
    df["harm"] = df["harm"].map(_canon)
    df["__stakeholder_canon"] = df["stakeholder_raw"].map(_canon)

    # Aggregate duplicates just in case
    key = ["__stakeholder_canon", "stakeholder_raw", "bias_type", "domain", "harm"]
    df = df.groupby(key, as_index=False)["votes"].sum()
    # Drop non-positive votes
    df = df.loc[df["votes"] > 0].copy()
    return df

def expand_by_votes(df: pd.DataFrame) -> pd.DataFrame:
    """Expand rows by 'votes' as frequency weights."""
    if df.empty:
        return df
    v = pd.to_numeric(df["votes"], errors="coerce").fillna(0).astype(int)
    df = df.loc[v > 0].copy()
    if df.empty:
        return df.drop(columns=["votes"], errors="ignore")
    reps = v.to_numpy()
    df_expanded = df.loc[df.index.repeat(reps)].copy()
    df_expanded.drop(columns=["votes"], inplace=True, errors="ignore")
    return df_expanded

def pick_default_baselines(df_cell: pd.DataFrame, bias_ref: str | None, harm_ref: str | None) -> tuple[str, str]:
    """
    Defaults:
      - bias_ref: most frequent bias_type (by votes) in the cell
      - harm_ref: most frequent harm (by votes) in the cell
    """
    if bias_ref is None:
        bias_ref = df_cell.groupby("bias_type")["votes"].sum().sort_values(ascending=False).index[0]
    if harm_ref is None:
        harm_ref = df_cell.groupby("harm")["votes"].sum().sort_values(ascending=False).index[0]
    return bias_ref, harm_ref

def collapse_rare_harms(
    df_cell: pd.DataFrame,
    min_harm_count: int = 10,
    min_harm_share: float | None = None,
    require_bias_overlap: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """
    Collapse rare/sparse harm levels into OTHER_LABEL for a single cell.

    Rules (OR-combined):
      - total votes of harm < min_harm_count
      - if min_harm_share is given: (votes_of_harm / total_votes) < min_harm_share
      - if require_bias_overlap: harm must appear under >=2 distinct bias_type levels; else collapse.

    Returns:
      (df_collapsed, mapping_dict) where mapping_dict maps original_harm -> collapsed_label (or itself if kept).
    """
    df = df_cell.copy()
    total_votes = df["votes"].sum()
    # totals per harm
    harm_totals = df.groupby("harm")["votes"].sum()
    harm_bias_n = df.groupby("harm")["bias_type"].nunique()

    to_collapse = set()
    for h in harm_totals.index:
        rules = []
        if harm_totals[h] < min_harm_count:
            rules.append("count")
        if min_harm_share is not None and total_votes > 0:
            if (harm_totals[h] / total_votes) < float(min_harm_share):
                rules.append("share")
        if require_bias_overlap and harm_bias_n.get(h, 0) < 2:
            rules.append("overlap")
        if rules:
            to_collapse.add(h)

    mapping = {h: (OTHER_LABEL if h in to_collapse else h) for h in harm_totals.index}
    df["harm"] = df["harm"].map(mapping)

    # Re-aggregate after relabeling
    key = ["__stakeholder_canon", "stakeholder_raw", "bias_type", "domain", "harm"]
    df = df.groupby(key, as_index=False)["votes"].sum()

    return df, mapping

def fit_cell_mnlogit(df_cell: pd.DataFrame, bias_ref: str, harm_ref: str):
    """
    Fit MNLogit for a single (domain × stakeholder) cell using numeric endog codes.
    Returns: (result_df, converged, note)
    """
    note = ""

    # Keep needed cols, then expand by votes
    dfw = expand_by_votes(df_cell[["bias_type", "harm", "votes"]].copy())
    if dfw.empty:
        return pd.DataFrame(), False, "Empty after expansion."

    # Must have at least 2 levels in both
    if dfw["bias_type"].nunique() < 2 or dfw["harm"].nunique() < 2:
        return pd.DataFrame(), False, "Needs ≥2 levels in both bias_type and harm."

    # Enforce category order so baselines are first
    bias_levels = [bias_ref] + sorted([b for b in dfw["bias_type"].unique() if b != bias_ref])
    harm_levels = [harm_ref] + sorted([h for h in dfw["harm"].unique() if h != harm_ref])

    dfw["bias_type"] = pd.Categorical(dfw["bias_type"], categories=bias_levels, ordered=True)
    dfw["harm"]      = pd.Categorical(dfw["harm"], categories=harm_levels, ordered=True)

    # Endog: integer class codes with harm_ref == 0
    y = dfw["harm"].cat.codes.values  # 0..K-1

    # Exog: intercept + bias indicators with bias_ref as baseline
    X = dmatrix(f"1 + C(bias_type, Treatment(reference='{bias_ref}'))",
                data=dfw, return_type="dataframe")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = sm.MNLogit(y, X)
            fit = model.fit(method="newton", disp=False, maxiter=200)
            converged = bool(getattr(fit, "mle_retvals", {}).get("converged", True))
    except Exception as e:
        return pd.DataFrame(), False, f"Fit failed: {e}"

    # Extract arrays → DataFrames with nice labels
    harm_nonref = [h for h in harm_levels[1:]]  # columns correspond to non-baseline harms
    params_df = pd.DataFrame(fit.params, index=X.columns, columns=harm_nonref)
    bse_df    = pd.DataFrame(fit.bse,    index=X.columns, columns=harm_nonref)

    # --- Tidy into long table for bias indicators (exclude intercept) ---
    rows = []
    bias_col_regex = re.compile(r"^C\(bias_type.*\)\[T\.(.+?)\]$")  # robust column matcher

    for harm_lvl in harm_nonref:
        for col in X.columns:
            if col.lower().startswith("intercept"):
                continue
            m = bias_col_regex.search(col)
            if not m:
                continue
            bias_lvl = m.group(1)

            coef = float(params_df.loc[col, harm_lvl])
            se   = float(bse_df.loc[col, harm_lvl])
            z    = coef / se if se > 0 else np.nan
            p    = p_from_z(z) if np.isfinite(z) else np.nan

            # 95% CI on OR via normal approx
            lo_coef = coef - 1.96 * se
            hi_coef = coef + 1.96 * se

            rows.append(dict(
                harm=harm_lvl,
                harm_ref=harm_ref,
                bias_type=bias_lvl,
                bias_ref=bias_ref,
                coef=round(coef, 6),
                se=round(se, 6),
                z=round(z, 6) if np.isfinite(z) else np.nan,
                p=round(p, 6) if np.isfinite(p) else np.nan,
                OR=round(math.exp(coef), 6),
                ci_low=round(math.exp(lo_coef), 6),
                ci_high=round(math.exp(hi_coef), 6),
                stars=stars_from_z(z) if np.isfinite(z) else "",
            ))

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["harm", "p", "bias_type"]).reset_index(drop=True)
    return out, converged, note

def analyze_all(
    df: pd.DataFrame,
    out_dir: Path,
    only_domain: str | None,
    only_stakeholder: str | None,
    bias_ref: str | None,
    harm_ref: str | None,
    min_harm_count: int,
    min_harm_share: float | None,
    require_bias_overlap: bool,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Which cells to run
    if only_domain and only_stakeholder:
        groups = [(_canon(only_domain), _canon(only_stakeholder))]
    else:
        groups = sorted(df.groupby(["domain", "__stakeholder_canon"]).size().index.tolist())

    any_written = False
    for domain, st_canon in groups:
        sub = df[(df["domain"] == domain) & (df["__stakeholder_canon"] == st_canon)].copy()
        if sub.empty:
            continue

        # Pick display stakeholder exactly as appears most often for that canon
        st_display = sub.groupby("stakeholder_raw")["votes"].sum().idxmax()

        # ---- Collapse rare harms per-cell ----
        sub_collapsed, mapping = collapse_rare_harms(
            sub,
            min_harm_count=min_harm_count,
            min_harm_share=min_harm_share,
            require_bias_overlap=require_bias_overlap,
        )

        # If after collapsing we have <3 harm levels, skip (too degenerate for MNLogit)
        n_harms = sub_collapsed["harm"].nunique()
        if n_harms < 3:
            base = f"mlogit__{domain}__{_canon(st_display)}"
            out_csv = out_dir / f"{base}.csv"
            stub = pd.DataFrame([{
                "domain": domain,
                "stakeholder_raw": st_display,
                "n_votes": int(sub["votes"].sum()),
                "bias_ref": bias_ref or "(auto)",
                "harm_ref": harm_ref or "(auto)",
                "converged": False,
                "note": f"Skipped: only {n_harms} harm levels after collapsing."
            }])
            stub.to_csv(out_csv, index=False)
            # write mapping file too
            map_df = pd.DataFrame({"original_harm": list(mapping.keys()),
                                   "mapped_to": [mapping[h] for h in mapping.keys()]})
            map_df.to_csv(out_dir / f"{base}__collapse_map.csv", index=False)
            print(f"[skip] {base}: only {n_harms} harm levels after collapsing; wrote stub + map.")
            any_written = True
            continue

        # pick baselines (if not provided); avoid OTHER_LABEL as harm_ref if possible
        b_ref, h_ref = pick_default_baselines(sub_collapsed, bias_ref, harm_ref)
        if h_ref == OTHER_LABEL:
            # choose most frequent non-other harm if available
            non_other = (sub_collapsed[sub_collapsed["harm"] != OTHER_LABEL]
                         .groupby("harm")["votes"].sum()
                         .sort_values(ascending=False))
            if len(non_other) > 0:
                h_ref = non_other.index[0]

        # Fit
        result_df, converged, note = fit_cell_mnlogit(sub_collapsed, b_ref, h_ref)

        # Attach metadata columns
        n_votes = int(sub_collapsed["votes"].sum())
        if result_df.empty:
            result_df = pd.DataFrame([{
                "domain": domain,
                "stakeholder_raw": st_display,
                "n_votes": n_votes,
                "bias_ref": b_ref,
                "harm_ref": h_ref,
                "converged": bool(converged),
                "note": note if note else "No estimable coefficients."
            }])
        else:
            result_df.insert(0, "stakeholder_raw", st_display)
            result_df.insert(0, "domain", domain)
            result_df.insert(2, "n_votes", n_votes)
            result_df["converged"] = bool(converged)
            if note:
                result_df["note"] = note

        # Save results
        base = f"mlogit__{domain}__{_canon(st_display)}"
        out_csv = out_dir / f"{base}.csv"
        result_df.to_csv(out_csv, index=False)
        print(f"[ok] Wrote MNL results: {out_csv}")

        # Save collapse mapping for transparency
        map_df = pd.DataFrame({"original_harm": list(mapping.keys()),
                               "mapped_to": [mapping[h] for h in mapping.keys()]})
        map_df.to_csv(out_dir / f"{base}__collapse_map.csv", index=False)
        any_written = True

    if not any_written:
        print("No (domain × stakeholder) cells found/matched. Nothing written.")

# ------------------------- CLI -------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Multinomial logistic regression: harm ~ bias_type per (domain × stakeholder) with rare-harm collapsing."
    )
    ap.add_argument("--in", dest="in_path", default="data/clean_results_human_plusllm.csv",
                    help="Input CSV (must include questionnaire_id, stakeholder_raw, bias_type, domain, harm, votes).")
    ap.add_argument("--out", dest="out_dir", default="out_mlogit",
                    help="Output directory for CSVs.")
    ap.add_argument("--domain", help="(Optional) run only this domain (case-insensitive).")
    ap.add_argument("--stakeholder", help="(Optional) run only this stakeholder_raw (case-insensitive).")
    ap.add_argument("--bias-ref", dest="bias_ref", default=None,
                    help="(Optional) baseline bias_type (case-insensitive). Defaults to most frequent in the cell.")
    ap.add_argument("--harm-ref", dest="harm_ref", default=None,
                    help="(Optional) baseline harm (case-insensitive). Defaults to most frequent non-'other' in the cell.")
    # new collapse controls
    ap.add_argument("--min-harm-count", dest="min_harm_count", type=int, default=10,
                    help="Collapse harms with total votes < this count (per cell). Default: 10")
    ap.add_argument("--min-harm-share", dest="min_harm_share", type=float, default=None,
                    help="Additionally collapse harms with share < this fraction (e.g., 0.03). Optional.")
    ap.add_argument("--require-bias-overlap", dest="require_bias_overlap", action="store_true",
                    help="If set, also collapse harms that appear under <2 distinct bias types in the cell.")
    args = ap.parse_args()

    csv_path = Path(args.in_path)
    out_dir  = Path(args.out_dir)

    df = load_data(csv_path)

    only_domain = _canon(args.domain) if args.domain else None
    only_stakeholder = _canon(args.stakeholder) if args.stakeholder else None
    bias_ref = _canon(args.bias_ref) if args.bias_ref else None
    harm_ref = _canon(args.harm_ref) if args.harm_ref else None

    analyze_all(
        df=df,
        out_dir=out_dir,
        only_domain=only_domain,
        only_stakeholder=only_stakeholder,
        bias_ref=bias_ref,
        harm_ref=harm_ref,
        min_harm_count=int(args.min_harm_count),
        min_harm_share=float(args.min_harm_share) if args.min_harm_share is not None else None,
        require_bias_overlap=bool(args.require_bias_overlap),
    )

if __name__ == "__main__":
    main()
