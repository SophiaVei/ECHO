# -*- coding: utf-8 -*-
"""
Per-domain collage images with ALL-bias radars for each stakeholder.
- Collage per domain (PNG/PDF/EPS) + singles per stakeholder.
- Harms drawn CLOCKWISE in a fixed preferred order (when present).
- Legend order fixed: Representation, Measurement, Algorithmic, Evaluation, Deployment.
- Single-figure title placed much higher (configurable via --single_title_y).
- Single plots show 10% radial tick labels at a discrete angle (45°) to avoid overlap;
  collage plots keep rings without numeric labels.
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

DATA_DEFAULT = Path("data/clean_results_human.csv")
OUT_DIR_DEFAULT = Path("radars_pub")
REQUIRED = {"questionnaire_id", "stakeholder_raw", "bias_type", "domain", "harm", "votes"}

RADIAL_TICK_FONTSIZE = 13
ANGULAR_TICK_FONTSIZE = 13
TITLE_FONTSIZE = 14
SECTION_TITLE_FONTSIZE = 17
LEGEND_FONTSIZE = 13

RADIAL_TICK_FS = ANGULAR_TICK_FS = TITLE_FS = SECTION_TITLE_FS = LEGEND_FS = None
RADAR_LW = GRID_LW = None

PREFERRED_HARMS_ORDER = [
    "opportunity loss", "economic loss",
    "alienation", "increased labor", "service or benefit loss",
    "loss of agency or control", "technology-facilitated violence",
    "diminished health and well-being", "privacy violation",
    "stereotyping", "demeaning", "erasure", "group alienation",
    "denying self-identity", "reifying categories",
]

BIAS_COLORS = {
    "algorithmic":      "#72B7B2",
    "deployment":       "#B39DDB",
    "evaluation":       "#F28E8E",
    "measurement":      "#FFD6A5",
    "representation":   "#90CAF9",
}
BIAS_PLOT_ORDER = ["representation", "measurement", "algorithmic", "evaluation", "deployment"]
BIAS_LABELS = {
    "representation": "Representation",
    "measurement": "Measurement",
    "algorithmic": "Algorithmic",
    "evaluation": "Evaluation",
    "deployment": "Deployment",
}
FALLBACK = ["#cce6e0", "#b2a0df", "#f9acac", "#ffdabb", "#a7c7e7", "#c6e2a9", "#f5c2e7", "#c7c7c7"]

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
    ctab = pd.pivot_table(
        df_cell,
        values="votes",
        index="bias_type",
        columns="harm",
        aggfunc="sum",
        fill_value=0
    ).astype(int)
    ctab = ctab.loc[:, ctab.sum(axis=0) > 0]
    ctab.index = [str(x).strip().lower() for x in ctab.index]
    ctab.columns = [str(c).strip().lower() for c in ctab.columns]
    return ctab

def row_shares(ctab: pd.DataFrame) -> pd.DataFrame:
    denom = ctab.sum(axis=1).replace(0, np.nan)
    return (ctab.T / denom).T

def order_harms_fixed(ctab: pd.DataFrame) -> list[str]:
    present = list(ctab.columns)
    preferred_present = [h for h in PREFERRED_HARMS_ORDER if h in present]
    extras = [h for h in present if h not in PREFERRED_HARMS_ORDER]
    if extras:
        sums = ctab[extras].sum(axis=0).sort_values(ascending=False)
        extras = list(sums.index)
    return preferred_present + extras

def color_for_bias(name: str, i: int) -> str:
    return BIAS_COLORS.get(str(name).lower(), FALLBACK[i % len(FALLBACK)])

# --------- Drawing ---------
def draw_all_radar_on_ax(
    ax,
    ctab: pd.DataFrame,
    title: str,
    max_labels: int | None,
    show_radial_labels: bool = False,
    rlabel_angle_deg: float = 0.0,
    radial_tick_fs: int | None = None,
):
    """
    Draw an ALL-bias radar.
    - show_radial_labels: when True, show 10% radial tick labels (percent); else hide labels.
    - rlabel_angle_deg: angle for radial label placement (e.g., 45 for discrete placement).
    - radial_tick_fs: override radial tick fontsize (used for smaller singles).
    """
    if ctab.empty or ctab.shape[0] < 1 or ctab.shape[1] < 3:
        return False

    harms = order_harms_fixed(ctab)
    if max_labels and max_labels > 0:
        harms = harms[:max_labels]

    shares = row_shares(ctab)[harms]

    ax.set_theta_direction(-1)  # clockwise

    ang = radar_angles(len(harms))
    xticks = wrap_labels(harms, width=14)

    data_max = float(np.nanmax(shares.values))
    r_target = max(0.20, data_max * 1.08)
    yticks, ytop = percent_ticks_auto_10(r_target, step=0.10, cap=0.60)
    ax.set_ylim(0, ytop)

    ax.set_axisbelow(True)
    ax.grid(True, alpha=0.85, linewidth=GRID_LW)

    bias_order = [b for b in BIAS_PLOT_ORDER if b in shares.index]
    shares = shares.reindex(index=bias_order)

    for i, b in enumerate(shares.index):
        vals = np.r_[shares.loc[b].values, shares.loc[b].values[0]]
        col = color_for_bias(b, i)
        ax.plot(ang, vals, lw=RADAR_LW, color=col, zorder=3)
        ax.fill(ang, vals, color=col, alpha=0.20, zorder=2)

    # radial ticks
    ax.set_yticks(yticks)
    if show_radial_labels:
        ax.set_rlabel_position(rlabel_angle_deg)
        fs = radial_tick_fs if radial_tick_fs is not None else RADIAL_TICK_FS
        ax.set_yticklabels([f"{int(t*100)}%" for t in yticks], fontsize=fs)
        ax.tick_params(axis="y", pad=6)
    else:
        ax.set_yticklabels([])

    ax.set_xticks(ang[:-1])
    ax.set_xticklabels(xticks, fontsize=ANGULAR_TICK_FS)

    if not show_radial_labels:
        ax.set_rlabel_position(0)
    return True

def save_single_radar(out_dir: Path, domain: str, who: str, ctab: pd.DataFrame,
                      fig_scale: float, ax_expand: float, title_y: float,
                      single_radial_tick_fs: int):
    fig_w = 5.6 * fig_scale
    fig_h = 4.6 * fig_scale
    fig = plt.figure(figsize=(fig_w, fig_h))

    ax = fig.add_subplot(111, polar=True)
    expand_polar_ax(ax, scale=ax_expand)

    # Show percentages at 45° in singles, with smaller font
    draw_all_radar_on_ax(
        ax, ctab, title=who, max_labels=None,
        show_radial_labels=True, rlabel_angle_deg=45,
        radial_tick_fs=single_radial_tick_fs
    )

    fig.text(0.02, title_y, f"{domain} — {who}",
             fontsize=SECTION_TITLE_FS, fontweight="bold",
             ha="left", va="bottom")

    # Legend pushed further right
    legend_handles = []
    for key in BIAS_PLOT_ORDER:
        if key in BIAS_COLORS:
            legend_handles.append(Line2D([0], [0], color=BIAS_COLORS[key], lw=3, label=BIAS_LABELS[key]))

    bb = ax.get_position()
    y_anchor = (bb.y0 + bb.y1) / 2.0
    fig.legend(handles=legend_handles,
               labels=[h.get_label() for h in legend_handles],
               loc="center left",
               bbox_to_anchor=(1.10, y_anchor),
               fontsize=LEGEND_FS, frameon=False)

    singles_dir = Path(out_dir) / "singles"
    singles_dir.mkdir(parents=True, exist_ok=True)
    out_base = singles_dir / f"{sanitize(domain)}__{sanitize(who)}__ALL_BIASES"

    fig.savefig(out_base.with_suffix(".png"), dpi=600)
    fig.savefig(out_base.with_suffix(".pdf"))
    fig.savefig(out_base.with_suffix(".eps"))
    plt.close(fig)

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
    df["harm"] = df["harm"].str.strip().str.lower()
    return df

def main():
    ap = argparse.ArgumentParser(description="Per-domain collage(s) with ALL-bias radars.")
    ap.add_argument("--in", dest="in_path", default=str(DATA_DEFAULT))
    ap.add_argument("--out", dest="out_dir", default=str(OUT_DIR_DEFAULT))
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--min_harms", type=int, default=3)
    ap.add_argument("--top_harms", type=int, default=0)
    ap.add_argument("--domains", default="")
    ap.add_argument("--font_scale", type=float, default=1.35)
    ap.add_argument("--line_scale", type=float, default=1.25)
    ap.add_argument("--fig_scale",  type=float, default=1.0)
    ap.add_argument("--col_gap", type=float, default=1.3)
    ap.add_argument("--ax_expand", type=float, default=1.45)
    ap.add_argument("--row_title_offset", type=float, default=0.10)
    ap.add_argument("--single_title_y", type=float, default=1.14)
    # NEW: scale for smaller radial percentage labels in singles
    ap.add_argument("--single_radial_tick_scale", type=float, default=0.8)

    args = ap.parse_args()

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

    # compute single-only smaller fs
    SINGLE_RADIAL_TICK_FS = max(7, int(RADIAL_TICK_FS * args.single_radial_tick_scale))

    set_pub_style(FS, GRID_LW)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    df = load_data(Path(args.in_path))

    by_domain: dict[str, list[tuple[str, pd.DataFrame]]] = {}
    for (domain, who), sub in df.groupby(["domain", "stakeholder_raw"], dropna=False):
        ctab = build_ctab(sub)
        if ctab.shape[0] < 2 or ctab.shape[1] < args.min_harms:
            continue
        by_domain.setdefault(str(domain), []).append((str(who), ctab))

    domains_arg = [d.strip().lower() for d in args.domains.split(",") if d.strip()] if args.domains else []
    all_domains = [d for d in sorted(by_domain.keys()) if by_domain.get(d)]
    domain_order = [d for d in domains_arg if d in by_domain and by_domain[d]] if domains_arg else all_domains
    if not domain_order:
        print("[collage] No domains with valid panels. Nothing to draw.")
        return

    legend_handles = []
    for key in BIAS_PLOT_ORDER:
        if key in BIAS_COLORS:
            legend_handles.append(Line2D([0], [0], color=BIAS_COLORS[key], lw=3, label=BIAS_LABELS[key]))

    cols = max(1, args.cols)

    for domain in domain_order:
        panels = sorted(by_domain[domain], key=lambda x: x[0])
        n = len(panels)
        if n == 0:
            print(f"[skip] {domain}: no stakeholders passed min_harms={args.min_harms}")
            continue

        rows = math.ceil(n / cols)

        ncols_eff = cols * 2 - 1
        width_ratios = []
        for c in range(cols):
            width_ratios.append(1.0)
            if c < cols - 1:
                width_ratios.append(args.col_gap)

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

        row_axes = [[] for _ in range(rows)]
        y_min, y_max = 1.0, 0.0

        for i, (who, ctab) in enumerate(panels):
            r = i // cols
            c = i % cols
            ax = fig.add_subplot(gs[r, c * 2], polar=True)
            expand_polar_ax(ax, scale=args.ax_expand)
            # Collage: keep rings without numbers
            draw_all_radar_on_ax(ax, ctab, title=who,
                                 max_labels=(args.top_harms if args.top_harms > 0 else None),
                                 show_radial_labels=False)
            bb = ax.get_position()
            y_min = min(y_min, bb.y0)
            y_max = max(y_max, bb.y1)
            row_axes[r].append((ax, who, bb))

            # Save single figure with high title & smaller percentage labels at 45°
            save_single_radar(out_dir, domain, who, ctab,
                              fig_scale=FIGS, ax_expand=args.ax_expand,
                              title_y=args.single_title_y,
                              single_radial_tick_fs=SINGLE_RADIAL_TICK_FS)

        row_title_offset = args.row_title_offset
        for r in range(rows):
            if not row_axes[r]:
                continue
            y_line = max(bb.y1 for _, _, bb in row_axes[r]) + row_title_offset
            for ax, who, bb in row_axes[r]:
                plt.gcf().text((bb.x0 + bb.x1) / 2.0, y_line, who,
                               fontsize=TITLE_FS, fontweight="bold",
                               ha="center", va="bottom")

        plt.gcf().text(0.02, min(y_max + 0.30, 1.30), f"{domain} — Radars Across Stakeholders",
                       fontsize=SECTION_TITLE_FS, fontweight="bold", ha="left", va="bottom")

        y_anchor = (y_min + y_max) / 2.0
        fig.legend(handles=legend_handles,
                   labels=[h.get_label() for h in legend_handles],
                   loc="center left",
                   bbox_to_anchor=(0.93, y_anchor),
                   fontsize=LEGEND_FS, frameon=False)

        out_base = Path(out_dir) / f"{sanitize(domain)}__ALL_BIASES__COLLAGE"
        fig.savefig(out_base.with_suffix(".png"), dpi=600)
        fig.savefig(out_base.with_suffix(".pdf"))
        fig.savefig(out_base.with_suffix(".eps"))
        plt.close(fig)

        print(f"[collage] wrote {out_base.with_suffix('.png')} ({n} panels)")

if __name__ == "__main__":
    main()
