# -*- coding: utf-8 -*-
"""
Per-domain collage images with ALL-bias radars for each stakeholder.
- One output figure *per domain* (PNG, PDF, EPS).
- Polar grid behind polygons; no radial % labels (rings remain).
- Reliable horizontal gaps via spacer columns (--col_gap).
- Optional axis expansion to enlarge each radar circle (--ax_expand).
- Stakeholder titles drawn ABOVE each radar, aligned per row, height tunable via --row_title_offset.
- Legend on the right, vertically centered.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import math
import re
import textwrap
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# -------------------- I/O --------------------
DATA_DEFAULT = Path("data/clean_results_human.csv")
OUT_DIR_DEFAULT = Path("radars_pub")
REQUIRED = {"questionnaire_id", "stakeholder_raw", "bias_type", "domain", "harm", "votes"}

# -------------------- Base sizes (scaled later) --------------------
RADIAL_TICK_FONTSIZE = 13
ANGULAR_TICK_FONTSIZE = 13
TITLE_FONTSIZE = 14
SECTION_TITLE_FONTSIZE = 17
LEGEND_FONTSIZE = 13

# will be set in main()
RADIAL_TICK_FS = ANGULAR_TICK_FS = TITLE_FS = SECTION_TITLE_FS = LEGEND_FS = None
RADAR_LW = GRID_LW = None

# -------------------- Style --------------------
def set_pub_style(fs: float, grid_lw: float):
    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "font.size": 13 * fs,
        "font.family": "DejaVu Sans",
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#222222",
        "text.color": "#222222",
        "xtick.color": "#222222",
        "ytick.color": "#222222",
        "xtick.labelsize": 13 * fs,
        "ytick.labelsize": 13 * fs,
        "grid.color": "#E0E0E0",
        "grid.linestyle": "-",
        "grid.linewidth": grid_lw,
        "legend.frameon": False,
    })

def expand_polar_ax(ax, scale=1.08):
    """Expand/shrink a polar axes within its subplot cell by a scale factor."""
    box = ax.get_position()
    cx = box.x0 + box.width / 2
    cy = box.y0 + box.height / 2
    new_w = box.width * scale
    new_h = box.height * scale
    x0 = max(0.0, cx - new_w / 2)
    y0 = max(0.0, cy - new_h / 2)
    x1 = min(1.0, cx + new_w / 2)
    y1 = min(1.0, cy + new_h / 2)
    ax.set_position([x0, y0, x1 - x0, y1 - y0])

BIAS_COLORS = {
    "algorithmic":      "#72B7B2",
    "deployment":       "#B39DDB",
    "evaluation":       "#F28E8E",
    "measurement":      "#FFD6A5",
    "representation":   "#90CAF9",
}
FALLBACK = ["#cce6e0", "#b2a0df", "#f9acac", "#ffdabb", "#a7c7e7", "#c6e2a9", "#f5c2e7", "#c7c7c7"]

# -------------------- Utils --------------------
def sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9\-_.]+", "_", str(name).strip().lower())

def wrap_labels(labels, width=16):
    return ["\n".join(textwrap.wrap(str(x), width=width, break_long_words=True, replace_whitespace=False))
            for x in labels]

def percent_ticks_auto_10(rmax, step=0.10, cap=0.60):
    top = min(cap, (np.ceil(max(rmax, step) / step) * step))
    return np.arange(step, top + 1e-9, step), float(top)

def radar_angles(n):
    ang = np.linspace(0, 2*np.pi, n, endpoint=False)
    return np.r_[ang, ang[0]]

def build_ctab(df_cell: pd.DataFrame) -> pd.DataFrame:
    ctab = pd.pivot_table(df_cell, values="votes", index="bias_type", columns="harm",
                          aggfunc="sum", fill_value=0).astype(int)
    ctab = ctab.loc[:, ctab.sum(axis=0) > 0]
    ctab.index = [str(x).strip().lower() for x in ctab.index]
    return ctab

def row_shares(ctab: pd.DataFrame) -> pd.DataFrame:
    denom = ctab.sum(axis=1).replace(0, np.nan)
    return (ctab.T / denom).T

def order_harms(ctab: pd.DataFrame) -> list[str]:
    return list(ctab.sum(axis=0).sort_values(ascending=False).index)

def color_for_bias(name: str, i: int) -> str:
    return BIAS_COLORS.get(str(name).lower(), FALLBACK[i % len(FALLBACK)])

# -------------------- Drawing --------------------
def draw_all_radar_on_ax(ax, ctab: pd.DataFrame, title: str, max_labels: int | None):
    """Draw a stakeholder's ALL-bias radar on the given polar axis (no title drawn here)."""
    if ctab.empty or ctab.shape[0] < 1 or ctab.shape[1] < 3:
        return False

    harms = order_harms(ctab)
    if max_labels and max_labels > 0:
        harms = harms[:max_labels]

    shares = row_shares(ctab)[harms]
    ang = radar_angles(len(harms))
    xticks = wrap_labels(harms, width=14)

    data_max = float(np.nanmax(shares.values))
    r_target = max(0.20, data_max * 1.08)
    yticks, ytop = percent_ticks_auto_10(r_target, step=0.10, cap=0.60)
    ax.set_ylim(0, ytop)

    ax.set_axisbelow(True)
    ax.grid(True, alpha=0.85, linewidth=GRID_LW)

    for i, b in enumerate(shares.index):
        vals = np.r_[shares.loc[b].values, shares.loc[b].values[0]]
        col = color_for_bias(b, i)
        ax.plot(ang, vals, lw=RADAR_LW, color=col, zorder=3)
        ax.fill(ang, vals, color=col, alpha=0.20, zorder=2)

    # keep rings, hide numeric radial labels
    ax.set_yticks(yticks)
    ax.set_yticklabels([])

    ax.set_xticks(ang[:-1])
    ax.set_xticklabels(xticks, fontsize=ANGULAR_TICK_FS)

    # normalize radial label padding so circles don’t push titles unevenly
    ax.set_rlabel_position(0)
    return True

# -------------------- Data --------------------
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = REQUIRED.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    key = ["questionnaire_id", "stakeholder_raw", "bias_type", "domain", "harm"]
    df = df.groupby(key, as_index=False)["votes"].sum()
    df["bias_type"] = df["bias_type"].str.strip().str.lower()
    df["domain"] = df["domain"].str.strip().str.lower()
    df["stakeholder_raw"] = df["stakeholder_raw"].str.strip()
    return df

# -------------------- Main --------------------
def main():
    ap = argparse.ArgumentParser(description="Per-domain collage(s) with ALL-bias radars.")
    ap.add_argument("--in", dest="in_path", default=str(DATA_DEFAULT))
    ap.add_argument("--out", dest="out_dir", default=str(OUT_DIR_DEFAULT))
    ap.add_argument("--cols", type=int, default=4, help="Number of subplot columns per domain.")
    ap.add_argument("--min_harms", type=int, default=3, help="Skip stakeholders with < min_harms nonzero harms")
    ap.add_argument("--top_harms", type=int, default=0, help="If >0, limit axes to top-N harms by pooled freq")
    ap.add_argument("--domains", default="", help="Optional: comma-separated domains to include (case-insensitive).")

    # scaling knobs
    ap.add_argument("--font_scale", type=float, default=1.35, help="Scale all text.")
    ap.add_argument("--line_scale", type=float, default=1.25, help="Scale line widths.")
    ap.add_argument("--fig_scale",  type=float, default=1.0,  help="Scale overall figure size (inches).")
    # horizontal gap and axis expansion
    ap.add_argument("--col_gap", type=float, default=1.3, help="Relative width of spacer columns between plots.")
    ap.add_argument("--ax_expand", type=float, default=1.45, help="Scale each polar axes within its grid cell.")
    # NEW: stakeholder title height
    ap.add_argument("--row_title_offset", type=float, default=0.10,
                    help="Vertical offset for stakeholder titles (figure coords). Try 0.026–0.040.")

    args = ap.parse_args()

    # sizes from CLI
    FS = args.font_scale
    LS = args.line_scale
    FIGS = args.fig_scale

    global RADIAL_TICK_FS, ANGULAR_TICK_FS, TITLE_FS, SECTION_TITLE_FS, LEGEND_FS, RADAR_LW, GRID_LW
    RADIAL_TICK_FS   = int(RADIAL_TICK_FONTSIZE * FS)
    ANGULAR_TICK_FS  = int(ANGULAR_TICK_FONTSIZE * FS)
    TITLE_FS         = int(TITLE_FONTSIZE * FS)
    SECTION_TITLE_FS = int(SECTION_TITLE_FONTSIZE * FS)
    LEGEND_FS        = int(LEGEND_FONTSIZE * FS)

    RADAR_LW = 2.2 * LS
    GRID_LW  = 0.8 * LS

    set_pub_style(FS, GRID_LW)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    df = load_data(Path(args.in_path))

    # group by domain
    by_domain: dict[str, list[tuple[str, pd.DataFrame]]] = {}
    for (domain, who), sub in df.groupby(["domain", "stakeholder_raw"], dropna=False):
        ctab = build_ctab(sub)
        if ctab.shape[0] < 2 or ctab.shape[1] < args.min_harms:
            continue
        by_domain.setdefault(str(domain), []).append((str(who), ctab))

    # choose domains
    domains_arg = [d.strip().lower() for d in args.domains.split(",") if d.strip()] if args.domains else []
    all_domains = [d for d in sorted(by_domain.keys()) if by_domain.get(d)]
    domain_order = [d for d in domains_arg if d in by_domain and by_domain[d]] if domains_arg else all_domains
    if not domain_order:
        print("[collage] No domains with valid panels. Nothing to draw.")
        return

    legend_handles = [Line2D([0], [0], color=col, lw=3, label=lab) for lab, col in BIAS_COLORS.items()]
    cols = max(1, args.cols)

    # ---- make one figure per domain ----
    for domain in domain_order:
        panels = sorted(by_domain[domain], key=lambda x: x[0])
        n = len(panels)
        if n == 0:
            print(f"[skip] {domain}: no stakeholders passed min_harms={args.min_harms}")
            continue

        rows = math.ceil(n / cols)

        # horizontal spacer columns
        ncols_eff = cols * 2 - 1
        width_ratios = []
        for c in range(cols):
            width_ratios.append(1.0)
            if c < cols - 1:
                width_ratios.append(args.col_gap)

        # figure & gridspec (reserve right margin for legend)
        fig_w = 5.6 * cols * FIGS
        fig_h = 4.6 * rows * FIGS
        fig = plt.figure(figsize=(fig_w, fig_h))
        right_margin = 0.86
        gs = fig.add_gridspec(
            nrows=rows, ncols=ncols_eff,
            left=0.02, right=right_margin, top=0.985, bottom=0.05,
            hspace=0.22, wspace=0.06,
            width_ratios=width_ratios
        )

        # place panels
        row_axes = [[] for _ in range(rows)]  # collect (ax, who, bbox) per row
        y_min, y_max = 1.0, 0.0

        for i, (who, ctab) in enumerate(panels):
            r = i // cols
            c = i % cols
            ax = fig.add_subplot(gs[r, c * 2], polar=True)
            expand_polar_ax(ax, scale=args.ax_expand)
            draw_all_radar_on_ax(ax, ctab, title=who,
                                 max_labels=(args.top_harms if args.top_harms > 0 else None))
            bb = ax.get_position()
            y_min = min(y_min, bb.y0)
            y_max = max(y_max, bb.y1)
            row_axes[r].append((ax, who, bb))

        # aligned stakeholder titles per row (same y for the whole row)
        row_title_offset = args.row_title_offset  # figure coords
        for r in range(rows):
            if not row_axes[r]:
                continue
            y_line = max(bb.y1 for _, _, bb in row_axes[r]) + row_title_offset
            for ax, who, bb in row_axes[r]:
                x_center = (bb.x0 + bb.x1) / 2.0
                fig.text(x_center, y_line, who,
                         fontsize=TITLE_FS, fontweight="bold",
                         ha="center", va="bottom")

        # domain title (tight, top-left, comfortably high)
        fig.text(0.02, min(y_max + 0.30, 1.30), f"{domain} — Radars Across Stakeholders",
                 fontsize=SECTION_TITLE_FS, fontweight="bold", ha="left", va="bottom")

        # legend on the right, vertically centered
        y_anchor = (y_min + y_max) / 2.0
        fig.legend(
            handles=legend_handles,
            labels=[h.get_label() for h in legend_handles],
            loc="center left",
            bbox_to_anchor=(0.93, y_anchor),
            fontsize=LEGEND_FS,
            frameon=False
        )

        # save
        out_base = Path(out_dir) / f"{sanitize(domain)}__ALL_BIASES__COLLAGE"
        fig.savefig(out_base.with_suffix(".png"), dpi=600)
        fig.savefig(out_base.with_suffix(".pdf"))
        fig.savefig(out_base.with_suffix(".eps"))
        plt.close(fig)

        print(f"[collage] wrote {out_base.with_suffix('.png')} ({n} panels)")

if __name__ == "__main__":
    main()
