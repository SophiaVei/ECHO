# -*- coding: utf-8 -*-
"""
Adjusted residuals (post-hoc) for bias × harm, per (stakeholder_raw, domain).

What it does
------------
Given your `clean_results_human.csv`, this script filters to a single
(stakeholder_raw, domain), builds the contingency table of counts
(votes) for bias_type × harm, and computes *adjusted residuals* for
each cell (Haberman residuals).

Interpretation
--------------
- Residuals are ~N(0,1) under independence.
- |z| > 1.96  ⇒ p < 0.05
- |z| > 2.58  ⇒ p < 0.01
- |z| > 3.29  ⇒ p < 0.001
- Positive  z ⇒ harm is over-represented for that bias.
- Negative  z ⇒ under-represented.

Outputs
-------
1) CSV with counts, expected, adjusted residual, p-value, and stars.
2) Heatmap (PNG/PDF/EPS) with a diverging colormap centered at 0 and
   significance stars on each cell.

Example
-------
python adjusted_residuals_bias_harm.py \
  --in data/clean_results_human.csv \
  --stakeholder "developer" \
  --domain "hiring" \
  --out out_residuals

Notes
-----
- Robust string canonicalization is used to avoid silent mismatches
  from Unicode hyphens / odd whitespace.
- If a row or column sums to zero, it is dropped before computation.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import math
import re
import unicodedata

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# ------------------------- Preferred orders (match radar script) -------------------------
PREFERRED_HARMS_ORDER = [
    "opportunity loss", "economic loss",
    "alienation", "increased labor", "service or benefit loss",
    "loss of agency or control", "technology-facilitated violence",
    "diminished health and well-being", "privacy violation",
    "stereotyping", "demeaning", "erasure", "group alienation",
    "denying self-identity", "reifying categories",
]

BIAS_PLOT_ORDER = ["representation", "measurement", "algorithmic", "evaluation", "deployment"]


# ------------------------- Canonicalization -------------------------
def _canon(s: str) -> str:
    """
    Canonicalize labels:
    - NFKC normalize
    - Map Unicode hyphen-like chars to ASCII '-'
    - Collapse all whitespace to single ASCII space
    - Lowercase + strip
    """
    s = unicodedata.normalize("NFKC", str(s))
    s = re.sub(r"[\u2010\u2011\u2012\u2013\u2014\u2212]", "-", s)  # hyphen-like → '-'
    s = re.sub(r"\s+", " ", s)
    return s.strip().lower()


# ------------------------- Data I/O -------------------------
REQUIRED_COLS = {
    "questionnaire_id",
    "stakeholder_raw",
    "bias_type",
    "domain",
    "harm",
    "votes",
}

def load_and_filter(
    csv_path: Path,
    stakeholder_raw: str,
    domain: str,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    missing = REQUIRED_COLS.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Canonicalize matching keys for robust filtering
    df["bias_type"]       = df["bias_type"].map(_canon)
    df["domain"]          = df["domain"].map(_canon)
    df["harm"]            = df["harm"].map(_canon)

    # Keep user's original stakeholder_raw for display, but use a canon column to match
    df["__stakeholder_canon"] = df["stakeholder_raw"].map(_canon)

    want_st = _canon(stakeholder_raw)
    want_dm = _canon(domain)

    df = df.loc[(df["__stakeholder_canon"] == want_st) & (df["domain"] == want_dm)].copy()
    if df.empty:
        raise ValueError(
            f"No rows found for stakeholder='{stakeholder_raw}' and domain='{domain}'."
        )

    # Aggregate votes across questionnaire_id (and any other duplicates)
    key = ["bias_type", "harm"]
    df = df.groupby(key, as_index=False)["votes"].sum()

    # Drop zeros early (won't affect expected counts)
    df = df.loc[df["votes"] > 0].copy()

    return df


# ------------------------- Stats -------------------------
def adjusted_residuals_from_counts(ctab: pd.DataFrame) -> pd.DataFrame:
    """
    Compute adjusted (Haberman) residuals for a bias × harm contingency table.

    Z_ij = (O_ij - E_ij) / sqrt(E_ij * (1 - r_i/N) * (1 - c_j/N))

    Where:
        O_ij: observed count
        E_ij: expected = (row_i * col_j) / N
        r_i: row sum
        c_j: column sum
        N: total count
    """
    # Drop degenerate rows/cols (sum=0)
    ctab = ctab.loc[ctab.sum(axis=1) > 0, ctab.sum(axis=0) > 0]

    # Convert to float for safe division
    O = ctab.astype(float).values
    r = O.sum(axis=1, keepdims=True)
    c = O.sum(axis=0, keepdims=True)
    N = O.sum()

    if N <= 0 or O.size == 0:
        raise ValueError("Contingency table is empty after filtering.")

    # Expected counts under independence
    E = (r @ c) / N

    # Adjusted residual denominator
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = np.sqrt(E * (1.0 - r / N) * (1.0 - c / N))
        Z = (O - E) / denom

    # Handle any numerical issues (0/0 etc.)
    Z = np.where(np.isfinite(Z), Z, 0.0)

    # Wrap back into DataFrame, with same index/columns
    z_df = pd.DataFrame(Z, index=ctab.index, columns=ctab.columns)

    # Also return E (expected) for reporting
    e_df = pd.DataFrame(E, index=ctab.index, columns=ctab.columns)

    return z_df, e_df


def normal_two_sided_p(z: np.ndarray) -> np.ndarray:
    """Two-sided p-values from |Z| using standard normal."""
    from math import erf, sqrt
    # Φ(|z|) = 0.5 * (1 + erf(|z| / sqrt(2)))
    # p = 2 * (1 - Φ(|z|)) = 1 - erf(|z| / sqrt(2))
    absz = np.abs(z)
    return 1.0 - erf(absz / math.sqrt(2.0))


def stars_from_z(z: float) -> str:
    """
    Two-sided thresholds:
      p < .10  ⇒ |z| ≥ 1.645  → *
      p < .05  ⇒ |z| ≥ 1.96   → **
      p < .01  ⇒ |z| ≥ 2.58   → ***
      p < .001 ⇒ |z| ≥ 3.29   → ****
    """
    a = abs(z)
    if a >= 3.29:
        return "****"  # p < 0.001
    if a >= 2.58:
        return "***"   # p < 0.01
    if a >= 1.96:
        return "**"    # p < 0.05
    if a >= 1.645:
        return "*"     # p < 0.10
    return ""


# ------------------------- Plotting -------------------------
def set_pub_style():
    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "font.size": 12,
        "font.family": "DejaVu Sans",
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#222222",
        "text.color": "#222222",
        "xtick.color": "#222222",
        "ytick.color": "#222222",
        "grid.color": "#E0E0E0",
        "legend.frameon": False,
    })


def heatmap_with_stars(
    Z: pd.DataFrame,
    counts: pd.DataFrame,
    out_base: Path,
    title: str,
    vmax: float | None = None,
):
    """
    Heatmap of positive adjusted residuals (z > 0) only.
    - Fixed cell size across all plots (consistent square size).
    - Figure size adapts to matrix shape + text, so nothing is cut.
    """
    set_pub_style()

    # ----- Ordering & positive-only Z -----
    # ----- Ordering to match radar script -----
    # Columns (harms): preferred fixed order when present, then extras by total count desc
    present_harms = list(counts.columns)
    preferred_present = [h for h in PREFERRED_HARMS_ORDER if h in present_harms]
    extras_harms = [h for h in present_harms if h not in PREFERRED_HARMS_ORDER]
    if extras_harms:
        extras_harms = list(counts[extras_harms].sum(axis=0).sort_values(ascending=False).index)
    col_order = preferred_present + extras_harms

    # Rows (biases): fixed list when present, then any extras by total count desc
    present_biases = list(counts.index)
    fixed_biases_present = [b for b in BIAS_PLOT_ORDER if b in present_biases]
    extras_biases = [b for b in present_biases if b not in BIAS_PLOT_ORDER]
    if extras_biases:
        extras_biases = list(counts.loc[extras_biases].sum(axis=1).sort_values(ascending=False).index)
    row_order = fixed_biases_present + extras_biases

    # Positive-only Z in that order
    Zp = Z.loc[row_order, col_order].copy().clip(lower=0.0)

    nrows, ncols = Zp.shape

    # ----- Handle case: no positive cells -----
    pos_vals = Zp.values[np.isfinite(Zp.values) & (Zp.values > 0)]
    no_pos = pos_vals.size == 0

    # ----- Consistent scaling for heatmap color -----
    if not no_pos:
        if vmax is None:
            vmax = float(np.nanpercentile(pos_vals, 99))
            vmax = max(2.58, round(vmax, 2))
    else:
        vmax = 1.0  # dummy for empty display

    from matplotlib.colors import LinearSegmentedColormap
    white_red = LinearSegmentedColormap.from_list("white_red", ["#FFFFFF", "#D7301F"])

    # ===== Fixed cell size (inches) — tweak if you want larger/smaller squares =====
    CELL_W_IN = 0.5
    CELL_H_IN = 0.5

    heat_w_in = max(1e-6, ncols * CELL_W_IN)
    heat_h_in = max(1e-6, nrows * CELL_H_IN)

    # ===== Margins (inches) to avoid clipping =====
    # Left/right allow for y labels and some padding; bottom allows for rotated x labels; top for title.
    LEFT_IN   = 1.25
    RIGHT_IN  = 0.60
    BOTTOM_IN = 1.50
    TOP_IN    = 1.00

    # Colorbar size
    CBAR_W_IN = 0.28
    CBAR_PAD_IN = 0.10

    # Total figure size (heatmap + cbar + margins)
    fig_w_in = LEFT_IN + heat_w_in + CBAR_PAD_IN + CBAR_W_IN + RIGHT_IN
    fig_h_in = TOP_IN + heat_h_in + BOTTOM_IN

    # Use constrained_layout so Matplotlib can fine-tune spacing;
    # we’ll still give enough room so nothing gets cut.
    fig = plt.figure(figsize=(fig_w_in, fig_h_in), constrained_layout=True)
    from matplotlib import gridspec
    gs = gridspec.GridSpec(
        ncols=2, nrows=1, figure=fig,
        width_ratios=[heat_w_in, CBAR_W_IN],
        wspace=CBAR_PAD_IN / max(1e-6, (heat_w_in + CBAR_W_IN))
    )

    ax = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])

    # ----- Draw heatmap -----
    if no_pos:
        im = ax.imshow(np.zeros_like(Zp.values), cmap="Greys", vmin=0, vmax=1, aspect="equal")
    else:
        im = ax.imshow(Zp.values, cmap=white_red, vmin=0.0, vmax=vmax, aspect="equal")

    # Tick labels
    ax.set_xticks(np.arange(ncols))
    ax.set_xticklabels(col_order, rotation=40, ha="right")
    ax.set_yticks(np.arange(nrows))
    ax.set_yticklabels(row_order)

    # Grid-like look (optional): thin separators around cells
    ax.set_xticks(np.arange(-0.5, ncols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, nrows, 1), minor=True)
    ax.grid(which="minor", linestyle="-", linewidth=0.4, color="#EEEEEE")
    ax.tick_params(which="minor", bottom=False, left=False)

    # Stars only for significant positive z
    if not no_pos:
        for i in range(nrows):
            for j in range(ncols):
                z = float(Zp.iat[i, j])
                if z > 0:
                    s = stars_from_z(z)
                    if s:
                        ax.text(j, i, s, ha="center", va="center", fontsize=11, color="black")

    # Colorbar
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Adjusted residual Z>0")

    # Title
    ax.set_title(title, fontweight="bold", pad=10)

    # Ensure full matrix area maps to data bounds
    ax.set_xlim(-0.5, ncols - 0.5)
    ax.set_ylim(nrows - 0.5, -0.5)

    # Final save — bbox_inches='tight' as a safeguard (constrained_layout already helps)
    for ext in [".png", ".pdf", ".eps"]:
        fig.savefig(out_base.with_suffix(ext), bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)



# ------------------------- Main -------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Adjusted residuals for bias × harm per (stakeholder_raw, domain)."
    )
    ap.add_argument("--in", dest="in_path", default="data/clean_results_human_plusllm.csv")
    ap.add_argument("--stakeholder", required=True, help="stakeholder_raw value (case-insensitive).")
    ap.add_argument("--domain", required=True, help="domain value (case-insensitive).")
    ap.add_argument("--out", dest="out_dir", default="out_residuals", help="Output directory.")
    ap.add_argument("--name", default="", help="Optional suffix for file names.")
    args = ap.parse_args()

    csv_path = Path(args.in_path)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load & filter
    df = load_and_filter(csv_path, stakeholder_raw=args.stakeholder, domain=args.domain)

    # Build contingency table (bias × harm) of votes
    ctab = pd.pivot_table(
        df, values="votes", index="bias_type", columns="harm", aggfunc="sum", fill_value=0
    ).astype(int)

    if ctab.values.sum() == 0:
        raise ValueError("All counts are zero after filtering. Nothing to analyze.")

    # Compute adjusted residuals
    Z, E = adjusted_residuals_from_counts(ctab)

    # Prepare long-form results with p-values and stars
    recs = []
    for bi in Z.index:
        for ha in Z.columns:
            o = int(ctab.loc[bi, ha]) if bi in ctab.index and ha in ctab.columns else 0
            e = float(E.loc[bi, ha])
            z = float(Z.loc[bi, ha])
            # two-sided p from |Z| (std normal)
            from math import erf, sqrt
            p = 2.0 * (1.0 - 0.5 * (1.0 + erf(abs(z) / math.sqrt(2.0))))
            recs.append(
                dict(
                    bias_type=bi,
                    harm=ha,
                    observed=o,
                    expected=round(e, 3),
                    z=round(z, 3),
                    p=round(p, 6),
                    stars=stars_from_z(z),
                )
            )
    out_df = pd.DataFrame(recs).sort_values(by=["z"], key=lambda s: s.abs(), ascending=False)

    # Save CSV
    suffix = f"__{args.name}" if args.name else ""
    base = f"{_canon(args.domain)}__{_canon(args.stakeholder)}{suffix}"
    out_csv = out_dir / f"adjusted_residuals__{base}.csv"
    out_df.to_csv(out_csv, index=False)

    # Heatmap
    title = f"{args.domain} — {args.stakeholder}\nBias × Harm adjusted residuals"
    out_base = out_dir / f"heatmap__adjusted_residuals__{base}"
    heatmap_with_stars(Z, ctab, out_base, title=title)

    # Console summary
    print(f"[ok] Wrote CSV: {out_csv}")
    print(f"[ok] Wrote heatmap: {out_base.with_suffix('.png')}")
    print()
    print("Top cells by |Z|:")
    print(out_df.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
