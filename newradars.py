# make_all_collages_by_domain.py
# -*- coding: utf-8 -*-
"""
Build TWO collage figures: one per domain (e.g., 'hiring' and 'diagnosis'),
each showing ALL-BIASES radar plots for every stakeholder in that domain.

Outputs (under ./radars_pub):
  <domain>__ALL_BIASES__COLLAGE.png / .pdf

Notes
- Uses row-normalized shares P(h|bias) as radius.
- Fixed bias colors, one shared legend per collage.
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

# -------------------- Style --------------------
def set_pub_style():
    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "font.size": 13,               # global base font size
        "font.family": "DejaVu Sans",
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#222222",
        "text.color": "#222222",
        "xtick.color": "#222222",
        "ytick.color": "#222222",
        "xtick.labelsize": 13,         # larger tick labels
        "ytick.labelsize": 13,
        "grid.color": "#E0E0E0",
        "grid.linestyle": "-",
        "grid.linewidth": 0.8,
        "legend.frameon": False,
    })

BIAS_COLORS = {
    "algorithmic":      "#72B7B2",
    "deployment":       "#B39DDB",
    "evaluation":       "#F28E8E",
    "measurement":      "#FFD6A5",
    "representation":   "#90CAF9",
}
FALLBACK = ["#cce6e0", "#b2a0df", "#f9acac", "#ffdabb", "#a7c7e7", "#c6e2a9", "#f5c2e7", "#c7c7c7"]

# Tunables for radar text sizes
RADIAL_TICK_FONTSIZE = 13
ANGULAR_TICK_FONTSIZE = 13
TITLE_FONTSIZE = 14
SUPERTITLE_FONTSIZE = 17
LEGEND_FONTSIZE = 13

# -------------------- Utils --------------------
def sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9\-_.]+", "_", str(name).strip().lower())

def wrap_labels(labels, width=16):
    return ["\n".join(textwrap.wrap(str(x), width=width, break_long_words=True, replace_whitespace=False)) for x in labels]

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

def save_figure(fig: mpl.figure.Figure, base: Path):
    fig.savefig(base.with_suffix(".png"), dpi=600)
    fig.savefig(base.with_suffix(".pdf"))
    plt.close(fig)

# -------------------- Drawing --------------------
def draw_all_radar_on_ax(ax, ctab: pd.DataFrame, title: str, max_labels: int | None):
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

    for i, b in enumerate(shares.index):
        vals = np.r_[shares.loc[b].values, shares.loc[b].values[0]]
        col = color_for_bias(b, i)
        ax.plot(ang, vals, lw=2.2, color=col, zorder=3)
        ax.fill(ang, vals, color=col, alpha=0.20, zorder=2)

    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{int(t*100)}%" for t in yticks], fontsize=RADIAL_TICK_FONTSIZE)
    ax.set_xticks(ang[:-1])
    ax.set_xticklabels(xticks, fontsize=ANGULAR_TICK_FONTSIZE)

    ax.set_title(title, pad=12, fontsize=TITLE_FONTSIZE, fontweight="bold")
    ax.grid(alpha=0.35)
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
    set_pub_style()
    ap = argparse.ArgumentParser(description="Two collages: one per domain with ALL-bias radars for all stakeholders.")
    ap.add_argument("--in", dest="in_path", default=str(DATA_DEFAULT))
    ap.add_argument("--out", dest="out_dir", default=str(OUT_DIR_DEFAULT))
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--min_harms", type=int, default=3, help="Skip stakeholders with < min_harms nonzero harms")
    ap.add_argument("--top_harms", type=int, default=0, help="If >0, limit axes to top-N harms by pooled freq")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    df = load_data(Path(args.in_path))

    # group by domain, collect stakeholder panels
    by_domain: dict[str, list[tuple[str, pd.DataFrame]]] = {}
    for (domain, who), sub in df.groupby(["domain", "stakeholder_raw"], dropna=False):
        ctab = build_ctab(sub)
        if ctab.shape[0] < 2 or ctab.shape[1] < args.min_harms:
            continue
        by_domain.setdefault(str(domain), []).append((str(who), ctab))

    # build one collage per domain
    for domain, panels in by_domain.items():
        if not panels:
            print(f"[skip] {domain}: no stakeholders passed min_harms={args.min_harms}")
            continue
        panels.sort(key=lambda x: x[0])  # sort alphabetically

        n = len(panels)
        cols = max(1, args.cols)
        rows = math.ceil(n / cols)

        fig = plt.figure(figsize=(5.5 * cols, 5.0 * rows))  # enlarge figure size
        fig.suptitle(f"{domain} — Radars Across Stakeholders", fontsize=SUPERTITLE_FONTSIZE, fontweight="bold", y=0.995)

        for i, (who, ctab) in enumerate(panels, start=1):
            ax = fig.add_subplot(rows, cols, i, polar=True)
            draw_all_radar_on_ax(ax, ctab, title=who, max_labels=(args.top_harms if args.top_harms > 0 else None))

        # single legend
        legend_handles = [Line2D([0], [0], color=col, lw=3, label=lab) for lab, col in BIAS_COLORS.items()]
        fig.legend(handles=legend_handles, loc="upper right", bbox_to_anchor=(0.995, 0.995), fontsize=LEGEND_FONTSIZE)

        fig.tight_layout(rect=[0, 0, 1, 0.94])  # leave more room for larger text
        out_base = out_dir / f"{sanitize(domain)}__ALL_BIASES__COLLAGE"
        save_figure(fig, out_base)
        print(f"[collage] wrote {out_base.with_suffix('.png')} ({n} panels)")

if __name__ == "__main__":
    main()
