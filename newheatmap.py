# -*- coding: utf-8 -*-
"""
Custom Stakeholder × Harm heatmaps with user-specified x-axis order and tick colors.
No plot titles. Bigger fonts, bigger cells, and x-axis labels pushed down (and optionally staggered)
to avoid overlaps.
Creates three figures:
  1) MAIN: all stakeholders EXCEPT "applicant group" and "patient group"
  2) APPLICANTS: ONLY "applicant group" and "patient group" and ONLY the last six harms
  3) MAIN_NO_GROUPHARMS: same rows as MAIN, but x-axis EXCLUDES the 6 group-harm categories
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from textwrap import wrap
import matplotlib.ticker as mtick

# ---------- I/O ----------
ROOT = Path(__file__).resolve().parent
DEFAULT_IN = ROOT / "data" / "clean_results_human.csv"
FIG_DIR = ROOT / "figures"
OUT_MAIN = "10_heatmap_stakeholder_by_harm__custom__MAIN.png"
OUT_APPL = "10_heatmap_stakeholder_by_harm__custom__APPLICANTS.png"
OUT_MAIN_NO_GROUPHARMS = "10_heatmap_stakeholder_by_harm__custom__MAIN_NO_GROUPHARMS.png"

# ---------- Sizing/Styling Controls (tweak here) ----------
WIDTH_PER_COL  = 2.10   # default inches per x-axis column
HEIGHT_PER_ROW = 1.45   # default inches per y-axis row
BOTTOM_MARGIN  = 0.58   # a bit less so labels sit closer overall
ROTATION_DEG   = 52     # x-label rotation angle
X_PAD_PX       = 18     # <<< closer to the axis (was 34)
STAGGER_EVERY_OTHER = True  # set False to disable staggering

# Font sizes
FS_BASE         = 18
FS_TICK         = 22    # x & y tick label size
FS_ANNOT        = 23    # in-cell percentages
FS_COLORBAR_TK  = 18
FS_COLORBAR_LAB = 20

# ---------- Manual line breaks for very long x labels ----------
LABEL_BREAKS = {
    "service or benefit loss": "service or\nbenefit loss",
    "loss of agency or control": "loss of agency or\ncontrol",
    "technology-facilitated violence": "technology-\nfacilitated\nviolence",
    "diminished health and well-being": "diminished health\nand well-being",
}
def pretty_label(lbl: str) -> str:
    if lbl in LABEL_BREAKS:
        return LABEL_BREAKS[lbl]
    return "\n".join(wrap(lbl, width=12))  # fallback wrap

# ---------- Matplotlib style ----------
BG = "#ffffff"
FG = "#2a2a2a"
GRID = "#e7e7e7"
plt.rcParams.update({
    "figure.dpi": 150,
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "savefig.facecolor": BG,
    "font.size": FS_BASE,
    "axes.edgecolor": FG,
    "axes.labelcolor": FG,
    "xtick.color": FG,
    "ytick.color": FG,
    "xtick.labelsize": FS_TICK,
    "ytick.labelsize": FS_TICK,
    "grid.color": GRID,
    "legend.frameon": False,
})

# ---------- Desired orders ----------
ORDER_DESIRED = [
    "opportunity loss", "economic loss",
    "alienation", "increased labor", "service or benefit loss",
    "loss of agency or control", "technology-facilitated violence", "diminished health and well-being", "privacy violation",
    "stereotyping", "demeaning", "erasure", "group alienation", "denying self-identity", "reifying categories",
]
ORDER_APPLICANTS = [
    "stereotyping", "demeaning", "erasure",
    "group alienation", "denying self-identity", "reifying categories",
]

# ---------- Tick group colors ----------
GROUPS = [
    ["opportunity loss", "economic loss"],
    ["alienation", "increased labor", "service or benefit loss"],
    ["loss of agency or control", "technology-facilitated violence", "diminished health and well-being", "privacy violation"],
    ["stereotyping", "demeaning", "erasure", "group alienation", "denying self-identity", "reifying categories"],
]
GROUP_COLORS = ["#E41A1C", "#377EB8", "#4DAF4A", "#984EA3"]

# ---------- Alias map ----------
ALIASES = {
    "oportunity loss": "opportunity loss",
    "opportunity losses": "opportunity loss",
    "economic losses": "economic loss",
    "service/benefit loss": "service or benefit loss",
    "service or benefit losses": "service or benefit loss",
    "loss of agency": "loss of agency or control",
    "loss of control": "loss of agency or control",
    "tech facilitated violence": "technology-facilitated violence",
    "technology facilitated violence": "technology-facilitated violence",
    "deck facilitated violence": "technology-facilitated violence",
    "diminished health and well being": "diminished health and well-being",
    "privacy violations": "privacy violation",
    "demeaning erasiing": "demeaning",
    "erasiing": "erasure",
    "erasing": "erasure",
    "refying categories": "reifying categories",
    "denying self identity": "denying self-identity",
}

def canon(label: str) -> str:
    l = (label or "").strip().lower()
    return ALIASES.get(l, label)

def ensure_figdir():
    Path(FIG_DIR).mkdir(parents=True, exist_ok=True)

# ---------- Data wrangling ----------
def load_clean(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    expected = {
        "questionnaire_id","stakeholder","is_group_level","bias_type","domain",
        "harm_family","harm","votes","total_votes","vote_share","stakeholder_raw"
    }
    missing = expected.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns in input: {missing}")
    block_keys = ["questionnaire_id", "stakeholder_raw", "bias_type", "domain"]
    harm_keys = block_keys + ["harm"]
    agg = df.groupby(harm_keys, as_index=False)["votes"].sum()
    totals = agg.groupby(block_keys, as_index=False)["votes"].sum().rename(columns={"votes": "total_votes"})
    df2 = agg.merge(totals, on=block_keys, how="left")
    df2["vote_share"] = df2["votes"] / df2["total_votes"]
    df2["harm"] = df2["harm"].apply(canon)
    return df2

def apply_filters(df, domain=None, bias_type=None, stakeholder=None):
    out = df.copy()
    if domain:
        out = out[out["domain"].str.lower() == domain.lower()].copy()
    if bias_type:
        out = out[out["bias_type"].str.lower() == bias_type.lower()].copy()
    if stakeholder:
        out = out[out["stakeholder_raw"].str.lower() == stakeholder.lower()].copy()
    return out

def build_matrix(df, col_order):
    if df.empty:
        return np.zeros((0,0)), [], []
    d = df.copy()
    d["weighted_share"] = d["vote_share"] * d["total_votes"]
    mat = (d.groupby(["stakeholder_raw","harm"], as_index=False)
             .agg(weighted_share=("weighted_share","sum"), total=("total_votes","sum")))
    mat["share"] = mat["weighted_share"] / mat["total"]
    all_stakeholders = sorted(mat["stakeholder_raw"].unique().tolist())
    pivot = mat.pivot(index="stakeholder_raw", columns="harm", values="share").reindex(index=all_stakeholders)
    for c in col_order:
        if c not in pivot.columns:
            pivot[c] = 0.0
    pivot = pivot[col_order].fillna(0.0)
    return pivot.to_numpy(), pivot.index.tolist(), col_order

# ---------- Plotting ----------
def draw_heatmap(
    grid,
    rows,
    cols,
    out_fname,
    vmax_override=None,
    *,
    width_per_col: float | None = None,
    height_per_row: float | None = None,
):
    """Draw heatmap. Optionally override width/height scaling per plot."""
    if grid.size == 0:
        print(f"[{out_fname}] No data to plot after filtering.")
        return

    # Figure size scales with matrix size (with optional per-plot overrides)
    width_per_col = WIDTH_PER_COL if width_per_col is None else width_per_col
    height_per_row = HEIGHT_PER_ROW if height_per_row is None else height_per_row

    width  = max(22, width_per_col  * len(cols))
    height = max(11, height_per_row * len(rows))
    fig, ax = plt.subplots(figsize=(width, height), constrained_layout=False)

    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("white_to_lav", ["#ffffff", "#b2a0df"])
    vmax = float(vmax_override) if vmax_override is not None else max(1e-6, float(grid.max()))

    im = ax.imshow(grid, cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest", aspect="auto")

    # Y ticks: stakeholder names
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(rows, fontsize=FS_TICK)
    ax.tick_params(axis="y", pad=8)

    # X ticks: pretty labels, rotation, color by groups (closer to plot)
    xlabels = [pretty_label(c) for c in cols]
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(xlabels, ha="right", rotation=ROTATION_DEG, rotation_mode="anchor", fontsize=FS_TICK)
    ax.tick_params(axis="x", pad=X_PAD_PX)

    # Optional staggering of every other label (nudged UP slightly)
    if STAGGER_EVERY_OTHER:
        for i, lbl in enumerate(ax.get_xticklabels()):
            x, y = lbl.get_position()
            if i % 2 == 1:
                lbl.set_position((x, y + 0.02))  # small upward nudge
            try:
                lbl.set_linespacing(1.15)
            except Exception:
                pass

    # Color x tick labels by group
    color_by_label = {}
    for color, group in zip(GROUP_COLORS, GROUPS):
        for lab in group:
            color_by_label[lab] = color
    for tick, col_label in zip(ax.get_xticklabels(), cols):
        tick.set_color(color_by_label.get(col_label, FG))

    # Minor grid
    ax.set_xticks(np.arange(-.5, len(cols), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(rows), 1), minor=True)
    ax.grid(which="minor", color="#f1f1f1", linewidth=0.9)
    ax.tick_params(which="both", length=0)

    # In-cell annotations — big, not bold
    for i in range(len(rows)):
        for j in range(len(cols)):
            ax.text(j, i, f"{grid[i, j]*100:.0f}%", ha="center", va="center",
                    color="#222222", fontsize=FS_ANNOT, fontweight="normal")

    # Colorbar — force 1 decimal across ALL plots
    cbar = fig.colorbar(im, ax=ax, fraction=0.055, pad=0.035, label="Share")
    cbar.ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
    cbar.update_ticks()
    cbar.ax.tick_params(labelsize=FS_COLORBAR_TK)
    cbar.ax.yaxis.label.set_size(FS_COLORBAR_LAB)

    # No titles; extra bottom room for labels
    plt.subplots_adjust(left=0.22, right=0.97, top=0.92, bottom=BOTTOM_MARGIN)

    ensure_figdir()
    out_path = Path(FIG_DIR) / out_fname
    fig.savefig(out_path, bbox_inches="tight", dpi=600)
    plt.close(fig)
    print(f"Saved: {out_path}")

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", default=str(DEFAULT_IN))
    ap.add_argument("--domain", default=None)
    ap.add_argument("--bias_type", default=None)
    ap.add_argument("--stakeholder", default=None)
    args = ap.parse_args()

    df = load_clean(Path(args.in_path))
    df = apply_filters(df, args.domain, args.bias_type, args.stakeholder)

    grid_full, rows_full, cols_full = build_matrix(df, ORDER_DESIRED)
    row_to_idx = {r: i for i, r in enumerate(rows_full)}

    # MAIN (exclude applicant/patient groups)
    drop_names = {"applicant group", "patient group"}
    keep_rows_main = [r for r in rows_full if r.lower() not in drop_names]
    grid_main = grid_full[[row_to_idx[r] for r in keep_rows_main]] if keep_rows_main else np.zeros((0, len(cols_full)))

    # APPLICANTS (just applicant & patient group; only 6 harms)
    keep_rows_appl = [r for r in rows_full if r.lower() in drop_names]
    cols_appl = ORDER_APPLICANTS
    grid_allcols, rows_all, _ = build_matrix(df, ORDER_DESIRED)
    if keep_rows_appl:
        idx_appl_rows = [rows_all.index(r) for r in keep_rows_appl if r in rows_all]
        col_idx_map = {c: i for i, c in enumerate(ORDER_DESIRED)}
        idx_appl_cols = [col_idx_map[c] for c in cols_appl]
        grid_appl = grid_allcols[np.ix_(idx_appl_rows, idx_appl_cols)]
    else:
        grid_appl = np.zeros((0, len(cols_appl)))

    # MAIN_NO_GROUPHARMS (same rows as MAIN, drop the 6 representational harms)
    cols_main_no_group = [c for c in ORDER_DESIRED if c not in ORDER_APPLICANTS]
    col_idx_map_full = {c: i for i, c in enumerate(ORDER_DESIRED)}
    idx_cols_main_no_group = [col_idx_map_full[c] for c in cols_main_no_group]
    grid_main_no_groupharms = grid_main[:, idx_cols_main_no_group] if grid_main.size else np.zeros((0, len(cols_main_no_group)))

    # Shared color scale across all figures
    vmax_global = max((float(g.max()) for g in (grid_main, grid_appl, grid_main_no_groupharms) if g.size), default=1e-6)

    if grid_main.size:
        draw_heatmap(grid_main, keep_rows_main, cols_full, OUT_MAIN, vmax_override=vmax_global)

    if grid_appl.size:
        draw_heatmap(grid_appl, keep_rows_appl, cols_appl, OUT_APPL, vmax_override=vmax_global)

    if grid_main_no_groupharms.size:
        # Keep your previous taller cells here if desired; numbers below were from last step
        HEIGHT_PER_ROW_NG = 3.0
        WIDTH_PER_COL_NG  = 2.30
        draw_heatmap(
            grid_main_no_groupharms,
            keep_rows_main,
            cols_main_no_group,
            OUT_MAIN_NO_GROUPHARMS,
            vmax_override=vmax_global,
            height_per_row=HEIGHT_PER_ROW_NG,
            width_per_col=WIDTH_PER_COL_NG,
        )

if __name__ == "__main__":
    main()
