# make_bias_harm_radars_pub.py
# -*- coding: utf-8 -*-
"""
ALL-BIASES radar plots per (domain × stakeholder_raw), plus ONE collage **per domain**.

OUTPUTS (under ./radars_pub/):
- <domain>__<stakeholder_raw>__ALL_BIASES__radar.png / .pdf
- <domain>__ALL_BIASES__COLLAGE.png / .pdf   (one per domain; single legend, 4 columns)

Notes
- Row-normalized shares P(h|bias) as radius.
- Radial ticks every 10%; smaller tick labels; fixed bias colors + single legend on each collage.
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

# -------------------- Paths & schema --------------------
DATA_DEFAULT = Path("data/clean_results_human.csv")
OUT_ROOT = Path("radars_pub")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

REQUIRED = {"questionnaire_id", "stakeholder_raw", "bias_type", "domain", "harm", "votes"}

# -------------------- Theme --------------------
def set_pub_style():
    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "font.size": 10.0,
        "font.family": "DejaVu Sans",
        "axes.edgecolor": "#444444",
        "axes.labelcolor": "#222222",
        "text.color": "#222222",
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "grid.color": "#E6E6E6",
        "grid.linestyle": "-",
        "grid.linewidth": 0.6,
        "legend.frameon": False,
    })

# Fixed bias colors (consistent across all plots)
BIAS_COLORS = {
    "algorithmic":      "#72B7B2",
    "deployment":       "#B39DDB",
    "evaluation":       "#F28E8E",
    "measurement":      "#FFD6A5",
    "representation":   "#90CAF9",
}
# Fallback palette (rarely used)
CUSTOM_PALETTE = ["#cce6e0", "#b2a0df", "#f9acac", "#ffdabb", "#a7c7e7", "#c6e2a9", "#f5c2e7", "#c7c7c7"]

# -------------------- Utilities --------------------
def sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9\-_.]+", "_", str(name).strip().lower())

def wrap_labels(labels, width=16):
    return ["\n".join(textwrap.wrap(str(x), width=width, break_long_words=True, replace_whitespace=False)) for x in labels]

def percent_ticks_auto_10(rmax, step=0.10, hard_cap=0.60):
    top = min(hard_cap, (np.ceil(max(rmax, step) / step) * step))
    ticks = np.arange(step, top + 1e-9, step)
    return ticks, float(top)

def build_kxH(df_cell: pd.DataFrame) -> pd.DataFrame:
    ctab = pd.pivot_table(
        df_cell, values="votes", index="bias_type", columns="harm", aggfunc="sum", fill_value=0
    ).astype(int)
    ctab = ctab.loc[:, ctab.sum(axis=0) > 0]
    ctab.index = [str(x).strip().lower() for x in ctab.index]
    return ctab

def row_shares(ctab: pd.DataFrame) -> pd.DataFrame:
    denom = ctab.sum(axis=1).replace(0, np.nan)
    return (ctab.T / denom).T

def order_harms(ctab: pd.DataFrame, by: str = "pooled") -> list[str]:
    if by == "pooled":
        return list(ctab.sum(axis=0).sort_values(ascending=False).index)
    return list(ctab.columns)

def radar_angles(n_axes: int):
    angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False)
    return np.concatenate([angles, [angles[0]]])

def save_figure(fig: mpl.figure.Figure, out_base: Path):
    fig.savefig(out_base.with_suffix(".png"), dpi=600)
    fig.savefig(out_base.with_suffix(".pdf"))
    plt.close(fig)

# -------------------- Plotters (ALL only) --------------------
def plot_all_bias_radar(ctab: pd.DataFrame, title: str, out_base: Path,
                        max_labels: int | None = None):
    """Standalone ALL_BIASES figure with legend."""
    if ctab.empty or ctab.shape[0] < 1 or ctab.shape[1] < 3:
        return
    harms_order = order_harms(ctab, by="pooled")
    if max_labels is not None and max_labels > 0:
        harms_order = harms_order[:max_labels]

    shares = row_shares(ctab)[harms_order]
    angles = radar_angles(len(harms_order))
    xticklabels = wrap_labels(harms_order, width=16)

    def color_for_bias(b: str, i: int) -> str:
        return BIAS_COLORS.get(str(b).lower(), CUSTOM_PALETTE[i % len(CUSTOM_PALETTE)])

    fig = plt.figure(figsize=(6.2, 6.2))
    ax = plt.subplot(111, polar=True)

    data_max = float(np.nanmax(shares.values))
    rmax_target = max(0.20, data_max * 1.08)
    yticks, ytop = percent_ticks_auto_10(rmax_target, step=0.10, hard_cap=0.60)
    ax.set_ylim(0, ytop)

    for i, bias in enumerate(shares.index):
        vals = np.r_[shares.loc[bias].values, shares.loc[bias].values[0]]
        col = color_for_bias(bias, i)
        ax.plot(angles, vals, linewidth=2.1, color=col, label=bias, zorder=3)
        ax.fill(angles, vals, color=col, alpha=0.18, zorder=2)

    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{int(t*100)}%" for t in yticks], fontsize=8)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(xticklabels, fontsize=9)

    ax.set_title(title, pad=12, loc="center", fontsize=11, fontweight="bold")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.02),
              borderaxespad=0.0, handlelength=1.6)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    save_figure(fig, out_base)

def draw_all_bias_radar_on_ax(ax, ctab: pd.DataFrame, title: str, max_labels: int | None = None):
    """Same ALL plot but draws on provided axes (no legend) for collages."""
    if ctab.empty or ctab.shape[0] < 1 or ctab.shape[1] < 3:
        return False

    harms_order = order_harms(ctab, by="pooled")
    if max_labels is not None and max_labels > 0:
        harms_order = harms_order[:max_labels]

    shares = row_shares(ctab)[harms_order]
    angles = radar_angles(len(harms_order))
    xticklabels = wrap_labels(harms_order, width=14)

    def color_for_bias(b: str, i: int) -> str:
        return BIAS_COLORS.get(str(b).lower(), CUSTOM_PALETTE[i % len(CUSTOM_PALETTE)])

    data_max = float(np.nanmax(shares.values))
    rmax_target = max(0.20, data_max * 1.08)
    yticks, ytop = percent_ticks_auto_10(rmax_target, step=0.10, hard_cap=0.60)
    ax.set_ylim(0, ytop)

    for i, bias in enumerate(shares.index):
        vals = np.r_[shares.loc[bias].values, shares.loc[bias].values[0]]
        col = color_for_bias(bias, i)
        ax.plot(angles, vals, linewidth=1.8, color=col, zorder=3)
        ax.fill(angles, vals, color=col, alpha=0.18, zorder=2)

    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{int(t*100)}%" for t in yticks], fontsize=7)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(xticklabels, fontsize=7)
    ax.set_title(title, pad=8, fontsize=9, fontweight="bold")
    ax.grid(alpha=0.25)
    return True

# -------------------- Data helpers --------------------
def load_data(in_path: Path) -> pd.DataFrame:
    df = pd.read_csv(in_path)
    missing = REQUIRED.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns in input: {missing}")
    key = ["questionnaire_id", "stakeholder_raw", "bias_type", "domain", "harm"]
    df = df.groupby(key, as_index=False)["votes"].sum()
    df["bias_type"] = df["bias_type"].str.strip().str.lower()
    df["domain"] = df["domain"].str.strip().str.lower()
    df["stakeholder_raw"] = df["stakeholder_raw"].str.strip()
    return df

def each_cell(df: pd.DataFrame):
    for (domain, who), sub in df.groupby(["domain", "stakeholder_raw"], dropna=False):
        yield str(domain), str(who), sub

def ensure_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

# -------------------- Main --------------------
def main():
    set_pub_style()
    ap = argparse.ArgumentParser(description="ALL-BIASES radars per (domain × stakeholder_raw) + one collage per domain.")
    ap.add_argument("--in", dest="in_path", default=str(DATA_DEFAULT),
                    help="Input CSV path (clean_results_human.csv)")
    ap.add_argument("--out", dest="out_dir", default=str(OUT_ROOT),
                    help="Output directory (default: ./radars_pub)")
    ap.add_argument("--min_harms", type=int, default=5,
                    help="Minimum number of nonzero harms to plot an ALL-BIASES radar (default: 5)")
    ap.add_argument("--top_harms", type=int, default=0,
                    help="If >0, limit radar axes to top-N harms by pooled frequency")
    ap.add_argument("--cols", type=int, default=4,
                    help="Columns for each collage grid (default: 4)")
    ap.add_argument("--max_panels_per_domain", type=int, default=200,
                    help="Max panels to include per-domain collage (default: 200)")
    args = ap.parse_args()

    in_path = Path(args.in_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_data(in_path)

    # {domain: list of {who, ctab}}
    panels_by_domain: dict[str, list[dict]] = {}

    for domain, who, sub in each_cell(df):
        cell_tag = f"{sanitize(domain)}__{sanitize(who)}"
        ctab = build_kxH(sub)
        if ctab.shape[0] < 2 or ctab.shape[1] < args.min_harms:
            continue

        # Standalone ALL plot
        out_all_base = out_dir / f"{cell_tag}__ALL_BIASES__radar"
        ensure_dir(out_all_base)
        title_all = f"{domain} × {who}"
        plot_all_bias_radar(
            ctab,
            title_all,
            out_all_base,
            max_labels=(args.top_harms if args.top_harms > 0 else None)
        )

        # Stash for domain collage
        panels_by_domain.setdefault(domain, []).append({"who": who, "ctab": ctab})

    # -------- Collage per domain (single legend, 4 columns default) --------
    for domain, panels in panels_by_domain.items():
        if not panels:
            continue
        # stable order by stakeholder
        panels = sorted(panels, key=lambda p: p["who"])[: args.max_panels_per_domain]

        n = len(panels)
        cols = max(1, args.cols)
        rows = math.ceil(n / cols)

        fig = plt.figure(figsize=(4.8 * cols, 4.6 * rows))
        fig.suptitle(f"{domain} — radars across stakeholders",
                     fontsize=14, fontweight="bold", y=0.995)

        for i, p in enumerate(panels, start=1):
            ax = fig.add_subplot(rows, cols, i, polar=True)
            title = f"{p['who']}"
            draw_all_bias_radar_on_ax(
                ax,
                p["ctab"],
                title=title,
                max_labels=(args.top_harms if args.top_harms > 0 else None)
            )

        legend_handles = [Line2D([0], [0], color=col, lw=3, label=lab)
                          for lab, col in BIAS_COLORS.items()]
        fig.legend(handles=legend_handles, loc="upper right", bbox_to_anchor=(0.995, 0.995))

        fig.tight_layout(rect=[0, 0, 1, 0.95])
        out_base = out_dir / f"{sanitize(domain)}__ALL_BIASES__COLLAGE"
        save_figure(fig, out_base)

    print(f"Done. Plots and per-domain collages written under: {out_dir.resolve()}")

if __name__ == "__main__":
    main()
