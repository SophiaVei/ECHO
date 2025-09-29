# -*- coding: utf-8 -*-
"""
Make Correspondence Analysis (CA) biplots from clean_results_human.csv.

Examples:
  python correspondence_analysis.py
  python correspondence_analysis.py --rows bias_type --cols harm
  python correspondence_analysis.py --rows stakeholder_raw --cols harm
  python correspondence_analysis.py --rows bias_type --cols harm --domain hiring
  python correspondence_analysis.py --rows bias_type --cols harm --top_rows 5 --top_cols 12

Output:
  figures/CA_<rows>_by_<cols>__<optional-filters>.png
  (Also prints inertia / variance explained, chi², and Cramér’s V)
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# NOTE: install once:  pip install prince
import prince

# -------------------- Config --------------------

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "clean_results_human.csv"
FIG_DIR = ROOT / "figures"

BG_COLOR = "#ffffff"
FG_COLOR = "#2a2a2a"
GRID_COLOR = "#e7e7e7"

plt.rcParams.update({
    "figure.dpi": 160,
    "figure.facecolor": BG_COLOR,
    "axes.facecolor": BG_COLOR,
    "savefig.facecolor": BG_COLOR,
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "axes.edgecolor": FG_COLOR,
    "axes.labelcolor": FG_COLOR,
    "xtick.color": FG_COLOR,
    "ytick.color": FG_COLOR,
    "grid.color": GRID_COLOR,
    "legend.frameon": False,
})

BLOCK_KEYS = ["questionnaire_id", "stakeholder_raw", "bias_type", "domain"]
HARM_KEYS  = BLOCK_KEYS + ["harm"]

# -------------------- Data utils --------------------

def load_clean(path: Path) -> pd.DataFrame:
    """Load and (re)aggregate to ensure one row per (block × harm) with vote counts."""
    df = pd.read_csv(path)
    need = {"bias_type","harm","votes","stakeholder_raw","domain","questionnaire_id"}
    miss = need.difference(df.columns)
    if miss:
        raise ValueError(f"Missing columns in input: {miss}")

    # Aggregate duplicates then attach block totals if needed later
    agg = df.groupby(HARM_KEYS, as_index=False)["votes"].sum()
    return agg

def filter_df(df: pd.DataFrame, domain: str | None, stakeholder: str | None, bias_type: str | None) -> pd.DataFrame:
    out = df.copy()
    if domain:      out = out[out["domain"].str.lower() == domain.lower()]
    if stakeholder: out = out[out["stakeholder_raw"].str.lower() == stakeholder.lower()]
    if bias_type:   out = out[out["bias_type"].str.lower() == bias_type.lower()]
    return out

def make_crosstab(df: pd.DataFrame, rows: str, cols: str) -> pd.DataFrame:
    """Contingency table of summed vote counts."""
    if rows not in df.columns or cols not in df.columns:
        raise ValueError(f"Columns not found: {rows}, {cols}")
    ct = pd.pivot_table(df, values="votes", index=rows, columns=cols, aggfunc="sum", fill_value=0)
    # Drop empty rows/cols if any
    ct = ct.loc[ct.sum(axis=1) > 0, ct.sum(axis=0) > 0]
    return ct

def trim_table(ct: pd.DataFrame, top_rows: int | None, top_cols: int | None) -> pd.DataFrame:
    """Optionally keep only the most frequent rows/cols to keep plots readable."""
    keep_rows = ct.sum(axis=1).sort_values(ascending=False)
    keep_cols = ct.sum(axis=0).sort_values(ascending=False)
    if top_rows is not None:
        keep_rows = keep_rows.head(top_rows)
    if top_cols is not None:
        keep_cols = keep_cols.head(top_cols)
    return ct.loc[keep_rows.index, keep_cols.index]

# -------------------- Stats helpers --------------------

def chisq_cramers_v(ct: pd.DataFrame):
    """Chi-square test + Cramér's V for the contingency table."""
    # expected = (row_sum * col_sum) / total
    obs = ct.to_numpy()
    n = obs.sum()
    row_sums = obs.sum(axis=1, keepdims=True)
    col_sums = obs.sum(axis=0, keepdims=True)
    expected = row_sums @ col_sums / n
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = ((obs - expected) ** 2 / expected).sum()
    r, c = obs.shape
    phi2 = chi2 / n
    k = min(r-1, c-1)
    v = np.sqrt(phi2 / k) if k > 0 else np.nan
    df = (r - 1) * (c - 1)
    return chi2, df, v

# -------------------- Plot --------------------

def plot_ca(ct: pd.DataFrame, title: str, outfile: Path):
    """Run CA and draw a modern biplot with bias types and harms."""
    ca = prince.CA(n_components=2, random_state=42)
    ca = ca.fit(ct)

    row_coords = ca.row_coordinates(ct)
    col_coords = ca.column_coordinates(ct)

    # --- variance / inertia ---
    try:
        eig = np.asarray(ca.eigenvalues_).astype(float).ravel()
    except Exception:
        sv = np.asarray(ca.singular_values_).astype(float).ravel()
        eig = sv ** 2

    total_inertia = np.sum(eig) if eig.size else np.nan
    explained = eig / total_inertia if total_inertia and np.isfinite(total_inertia) else np.array([])

    d1_pct = (explained[0] * 100) if explained.size > 0 else np.nan
    d2_pct = (explained[1] * 100) if explained.size > 1 else np.nan

    # --- modern plot ---
    fig, ax = plt.subplots(figsize=(9, 9))

    # Bias types (rows) - blue circles
    ax.scatter(row_coords[0], row_coords[1], s=80, c="#1f77b4", alpha=0.8, label="Bias Types")
    for i, lbl in enumerate(row_coords.index):
        ax.text(row_coords.iloc[i, 0], row_coords.iloc[i, 1], str(lbl),
                fontsize=10, ha="center", va="center", color="#1f77b4")

    # Harms (columns) - orange squares
    ax.scatter(col_coords[0], col_coords[1], s=80, c="#ff7f0e", marker="s", alpha=0.8, label="Harms")
    for i, lbl in enumerate(col_coords.index):
        ax.text(col_coords.iloc[i, 0], col_coords.iloc[i, 1], str(lbl),
                fontsize=9, ha="center", va="center", color="#ff7f0e")

    # Axes lines
    ax.axhline(0, color="#999999", lw=0.8, ls="--")
    ax.axvline(0, color="#999999", lw=0.8, ls="--")

    # Labels
    ax.set_xlabel(f"Dimension 1 ({d1_pct:.1f}% inertia)")
    ax.set_ylabel(f"Dimension 2 ({d2_pct:.1f}% inertia)")
    ax.set_title(title, fontsize=14, weight="bold")

    # Modern grid
    ax.grid(True, which="both", ls="--", lw=0.5, alpha=0.4)

    # Remove outer box (spines) for a modern look
    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(False)

    # Clean legend (no box)
    ax.legend(frameon=False, loc="best")

    # Save
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outfile, bbox_inches="tight", dpi=300)
    plt.close(fig)


    # Console summary
    chi2, df, v = chisq_cramers_v(ct)
    print(f"[CA] Table shape: {ct.shape[0]} rows × {ct.shape[1]} cols")
    print(f"[CA] Total inertia: {total_inertia:.4f} | Dim1: {d1_pct:.2f}% | Dim2: {d2_pct:.2f}%")
    print(f"[Assoc] Chi²: {chi2:.2f} (df={df}) | Cramér’s V: {v:.3f}")
    return d1_pct, d2_pct, v


# -------------------- Main --------------------

def main():
    parser = argparse.ArgumentParser(description="Correspondence Analysis (CA) biplots.")
    parser.add_argument("--in", dest="in_path", default=str(DATA_PATH), help="Input CSV (clean_results_human.csv)")
    parser.add_argument("--rows", default="bias_type", help="Row variable (e.g., bias_type, stakeholder_raw, domain)")
    parser.add_argument("--cols", default="harm", help="Column variable (e.g., harm)")
    parser.add_argument("--domain", default=None, help="Optional filter: domain (hiring/diagnosis)")
    parser.add_argument("--stakeholder", default=None, help='Optional filter: stakeholder_raw (e.g., "applicant group")')
    parser.add_argument("--bias_type", default=None, help="Optional filter: bias_type (representation, ...)")
    parser.add_argument("--top_rows", type=int, default=None, help="Keep top-N rows by total count")
    parser.add_argument("--top_cols", type=int, default=None, help="Keep top-N cols by total count")
    args = parser.parse_args()

    df = load_clean(Path(args.in_path))
    df = filter_df(df, domain=args.domain, stakeholder=args.stakeholder, bias_type=args.bias_type)

    ct = make_crosstab(df, rows=args.rows, cols=args.cols)
    ct = trim_table(ct, top_rows=args.top_rows, top_cols=args.top_cols)

    # Compose title & filename
    bits = [f"CA: {args.rows} × {args.cols}"]
    fname_bits = [f"CA_{args.rows}_by_{args.cols}"]
    if args.domain:
        bits.append(f"Domain={args.domain}")
        fname_bits.append(f"domain-{args.domain}")
    if args.stakeholder:
        bits.append(f"Stakeholder={args.stakeholder}")
        fname_bits.append(f"stakeholder-{args.stakeholder.replace(' ','_')}")
    if args.bias_type:
        bits.append(f"Bias={args.bias_type}")
        fname_bits.append(f"bias-{args.bias_type}")
    title = " | ".join(bits)
    outfile = FIG_DIR / ( "__".join(fname_bits) + ".png" )

    plot_ca(ct, title=title, outfile=outfile)
    print(f"Saved: {outfile}")

if __name__ == "__main__":
    main()
