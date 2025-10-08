# -*- coding: utf-8 -*-
"""
Create insight-driven, publication-ready plots from clean_results_human.csv.

Outputs (figures/):
  01_overall_harm_shares.png
  02_stacked_shares_by_stakeholder.png
  03_heatmap_harms_by_bias_type__<stakeholder>.png   (for top-K or requested stakeholder)
  04_top_harms_by_stakeholder_small_multiples.png
  05_domain_splits__<stakeholder>.png                (for top-K or requested stakeholder)
  06_respondents_by_questionnaire.png                (<- people, assuming single-choice per block)
  07_responses_by_stakeholder.png
  08_responses_by_bias_type.png
  09_responses_by_domain.png
  10_heatmap_stakeholder_by_harm.png
  11_stacked_shares_by_domain.png
  12_heatmap_domain_by_harm.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from textwrap import wrap
from matplotlib.colors import LinearSegmentedColormap

# ------------------------ Config ------------------------

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "clean_results_human.csv"
FIG_DIR = ROOT / "figures"

PALETTE = [
    "#cce6e0", "#b2a0df", "#f9acac", "#ffdabb",
    "#7fc8f8", "#c4f1be", "#f6d186", "#a0d2eb", "#ffb3c1", "#cab8ff"
]

BG_COLOR = "#ffffff"
FG_COLOR = "#2a2a2a"
GRID_COLOR = "#e7e7e7"

plt.rcParams.update({
    "figure.dpi": 120,
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
    "grid.linestyle": "-",
    "grid.alpha": 0.5,
    "legend.frameon": False,
})

# One brand color for all heatmaps (you can change this hex anytime)
HEAT_COLOR = "#b2a0df"  # lavender from your palette

def cmap_white_to(hex_color: str) -> LinearSegmentedColormap:
    """Two-color colormap from white to the given color."""
    return LinearSegmentedColormap.from_list("white_to_brand", ["#ffffff", hex_color])

# ------------------------ Data utils ------------------------

BLOCK_KEYS = ["questionnaire_id", "stakeholder_raw", "bias_type", "domain"]
HARM_KEYS = BLOCK_KEYS + ["harm"]

def load_clean(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    expected = {
        "questionnaire_id","stakeholder","is_group_level","bias_type","domain",
        "harm_family","harm","votes","total_votes","vote_share","stakeholder_raw"
    }
    missing = expected.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns in input: {missing}")

    # Aggregate duplicates at (block + harm) then recompute shares per block
    agg = df.groupby(HARM_KEYS, as_index=False)["votes"].sum()
    totals = agg.groupby(BLOCK_KEYS, as_index=False)["votes"].sum().rename(columns={"votes":"total_votes"})
    df2 = agg.merge(totals, on=BLOCK_KEYS, how="left")
    df2["vote_share"] = df2["votes"] / df2["total_votes"]
    return df2

def apply_filters(df: pd.DataFrame, domain: str | None, bias_type: str | None, stakeholder: str | None) -> pd.DataFrame:
    out = df.copy()
    if domain:
        out = out[out["domain"].str.lower() == domain.lower()].copy()
    if bias_type:
        out = out[out["bias_type"].str.lower() == bias_type.lower()].copy()
    if stakeholder:
        out = out[out["stakeholder_raw"].str.lower() == stakeholder.lower()].copy()
    return out

def ensure_figdir():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

def color_cycle(n: int) -> list[str]:
    if n <= len(PALETTE): return PALETTE[:n]
    reps = int(np.ceil(n / len(PALETTE)))
    return (PALETTE * reps)[:n]

def wrap_labels(labels, width=18):
    return ["\n".join(wrap(str(x), width)) for x in labels]

def get_levels(full_df: pd.DataFrame) -> dict[str, list[str]]:
    """Global levels for axes so heatmaps always show all categories (even if zero)."""
    levels = {
        "bias_type": sorted(full_df["bias_type"].dropna().unique().tolist()),
        "harm": sorted(full_df["harm"].dropna().unique().tolist()),
        "stakeholder_raw": sorted(full_df["stakeholder_raw"].dropna().unique().tolist()),
        "domain": sorted(full_df["domain"].dropna().unique().tolist()),
    }
    return levels

# ------------------------ Plot style helpers (readable heatmaps) ------------------------

def _heatmap_label_params(n_rows: int, n_cols: int):
    """Return sensible (wrap_width, xfontsize, yfontsize, rotation, margins) based on grid size."""
    # Wrap width scales with number of columns (more cols => narrower wrap)
    if n_cols <= 12:
        wrap_w, xfs = 18, 11
        bottom = 0.18
        rotation = 30
    elif n_cols <= 20:
        wrap_w, xfs = 16, 10
        bottom = 0.22
        rotation = 35
    elif n_cols <= 35:
        wrap_w, xfs = 14, 9
        bottom = 0.28
        rotation = 40
    else:
        wrap_w, xfs = 12, 8
        bottom = 0.34
        rotation = 45

    # Y-axis font size and left margin scale with rows
    if n_rows <= 12:
        yfs = 11
        left = 0.15
    elif n_rows <= 25:
        yfs = 10
        left = 0.20
    elif n_rows <= 40:
        yfs = 9
        left = 0.25
    else:
        yfs = 8
        left = 0.30

    return wrap_w, xfs, yfs, rotation, left, bottom

def _heatmap_figsize(n_rows: int, n_cols: int, min_w=8, min_h=4.5):
    """Compute a figure size big enough to keep tick labels readable."""
    # Each column gets ~0.45 in width; each row gets ~0.45 in height
    w = max(min_w, 0.45 * n_cols)
    h = max(min_h, 0.45 * n_rows)
    return (w, h)

def _style_heatmap_axes(ax, rows, cols):
    """Apply readable ticks, wrapping, rotation, and tick params."""
    n_rows, n_cols = len(rows), len(cols)
    wrap_w, xfs, yfs, rotation, left, bottom = _heatmap_label_params(n_rows, n_cols)

    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(rows, fontsize=yfs)

    xlabels = wrap_labels(cols, wrap_w)
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(xlabels, rotation=rotation, ha="right", fontsize=xfs)

    # Gridlines to aid reading across/along labels without clutter
    ax.set_xticks(np.arange(-.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="#f1f1f1", linewidth=0.6)
    ax.tick_params(which="both", length=0)

    # Leave space for labels/colorbar
    plt.subplots_adjust(left=left, bottom=bottom, right=0.95, top=0.90)

# ------------------------ Insight plots ------------------------

def plot_stacked_shares_by_domain(df: pd.DataFrame, title_suffix: str = ""):
    """Stacked bar chart: domains on Y, stacked harms as shares."""
    if df.empty: return
    dff = df.copy()
    dff["weighted_share"] = dff["vote_share"] * dff["total_votes"]
    shares = (dff.groupby(["domain","harm"], as_index=False)
                .agg(weighted_share=("weighted_share","sum"), total=("total_votes","sum")))
    shares["share"] = shares["weighted_share"] / shares["total"]

    domains = shares["domain"].drop_duplicates().tolist()
    pivot = shares.pivot(index="domain", columns="harm", values="share").fillna(0.0)
    pivot = pivot.loc[domains]

    fig, ax = plt.subplots(figsize=(10, max(4.5, 0.45*len(pivot)+2)))
    bottoms = np.zeros(len(pivot)); colors = color_cycle(pivot.shape[1])
    for i, harm in enumerate(pivot.columns):
        vals = pivot[harm].values
        ax.barh(range(len(pivot)), vals, left=bottoms, label=harm, color=colors[i], edgecolor=FG_COLOR)
        bottoms += vals
    ax.set_yticks(range(len(pivot))); ax.set_yticklabels(pivot.index.tolist())
    ax.set_xlabel("Share of votes (stacked by harm)")
    ttl = "Harm Share Distribution by Domain"
    if title_suffix: ttl += f" — {title_suffix}"
    ax.set_title(ttl); ax.legend(ncol=2, bbox_to_anchor=(1.02, 1), loc="upper left"); ax.grid(axis="x")
    fig.tight_layout(); fig.savefig(FIG_DIR / "11_stacked_shares_by_domain.png", bbox_inches="tight"); plt.close(fig)

def plot_heatmap_domain_by_harm(df: pd.DataFrame, LEVELS: dict, title_suffix: str = ""):
    """Domain (rows) × Harm (cols) heatmap with all categories shown (zeros if absent)."""
    if df.empty: return
    d = df.copy()
    d["weighted_share"] = d["vote_share"] * d["total_votes"]
    mat = (d.groupby(["domain","harm"], as_index=False)
             .agg(weighted_share=("weighted_share","sum"), total=("total_votes","sum")))
    mat["share"] = mat["weighted_share"] / mat["total"]

    pivot = (mat.pivot(index="domain", columns="harm", values="share")
                .reindex(index=LEVELS["domain"], columns=LEVELS["harm"])
                .fillna(0.0))

    rows = pivot.index.tolist()
    cols = pivot.columns.tolist()
    grid = pivot.to_numpy()

    fig, ax = plt.subplots(figsize=_heatmap_figsize(len(rows), len(cols)), constrained_layout=False)
    cmap = cmap_white_to(HEAT_COLOR)
    vmax = max(1e-6, float(grid.max()))
    im = ax.imshow(grid, aspect="auto", cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest")

    _style_heatmap_axes(ax, rows, cols)

    # optional in-cell annotations; keep small to avoid clutter
    ann_fs = 8 if len(rows) > 20 or len(cols) > 20 else 9
    for i in range(len(rows)):
        for j in range(len(cols)):
            ax.text(j, i, f"{grid[i,j]*100:.0f}%", ha="center", va="center", color="#333333", fontsize=ann_fs)

    ttl = "Harm Shares: Domain × Harm"
    if title_suffix: ttl += f" — {title_suffix}"
    ax.set_title(ttl)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="Share")
    cbar.ax.tick_params(labelsize=9)

    fig.savefig(FIG_DIR / "12_heatmap_domain_by_harm.png", bbox_inches="tight"); plt.close(fig)

def plot_respondents_by_questionnaire(df: pd.DataFrame):
    """
    Assumes single-choice per block: total_votes == # respondents in that block.
    Sums across blocks (stakeholder_raw × bias_type × domain) inside each questionnaire_id.
    """
    blocks = df.groupby(BLOCK_KEYS, as_index=False)["total_votes"].first()
    q = (blocks.groupby("questionnaire_id", as_index=False)["total_votes"]
               .sum().rename(columns={"total_votes":"n_respondents"}))
    q = q.sort_values("questionnaire_id")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(q["questionnaire_id"].astype(str), q["n_respondents"], color=color_cycle(len(q)))
    ax.set_xlabel("questionnaire_id"); ax.set_ylabel("# respondents"); ax.set_title("Respondents by Questionnaire")
    ax.grid(axis="y"); fig.tight_layout(); fig.savefig(FIG_DIR / "06_respondents_by_questionnaire.png", bbox_inches="tight"); plt.close(fig)

def plot_overall_harm_shares(df: pd.DataFrame, title_suffix: str = ""):
    if df.empty: return
    harm_shares = (df.groupby("harm", as_index=False)
                     .agg(votes=("votes","sum"), total=("total_votes","sum")))
    harm_shares["share"] = harm_shares["votes"] / harm_shares["total"]
    harm_shares = harm_shares.sort_values("share", ascending=True)

    labels = wrap_labels(harm_shares["harm"].tolist())
    vals = harm_shares["share"].values
    fig, ax = plt.subplots(figsize=(14, max(4, 0.4*len(labels)+2)))
    ax.barh(range(len(labels)), vals, color=color_cycle(len(labels)), edgecolor=FG_COLOR)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.set_xlabel("Overall vote share")
    ttl = "Overall Harm Importance (Weighted Share)"
    if title_suffix: ttl += f" — {title_suffix}"
    ax.set_title(ttl); ax.grid(axis="x")
    for i, v in enumerate(vals):
        ax.text(v + 0.005, i, f"{v:.0%}", va="center", ha="left", color=FG_COLOR)
    fig.tight_layout(); fig.savefig(FIG_DIR / "01_overall_harm_shares.png", bbox_inches="tight"); plt.close(fig)

def plot_stacked_shares_by_stakeholder(df: pd.DataFrame, top_n_stakeholders: int = 10, title_suffix: str = ""):
    if df.empty: return
    st_totals = (df.groupby("stakeholder_raw", as_index=False)["total_votes"].sum()
                   .sort_values("total_votes", ascending=False)
                   .head(top_n_stakeholders))
    top_st = set(st_totals["stakeholder_raw"])
    dff = df[df["stakeholder_raw"].isin(top_st)].copy()
    dff["weighted_share"] = dff["vote_share"] * dff["total_votes"]
    shares = (dff.groupby(["stakeholder_raw","harm"], as_index=False)
                 .agg(weighted_share=("weighted_share","sum"), total=("total_votes","sum")))
    shares["share"] = shares["weighted_share"] / shares["total"]

    stakeholders = shares["stakeholder_raw"].drop_duplicates().tolist()
    pivot = shares.pivot(index="stakeholder_raw", columns="harm", values="share").fillna(0.0)
    pivot = pivot.loc[stakeholders]

    fig, ax = plt.subplots(figsize=(10, max(5, 0.45*len(pivot)+2)))
    bottoms = np.zeros(len(pivot))
    colors = color_cycle(pivot.shape[1])
    for i, harm in enumerate(pivot.columns):
        vals = pivot[harm].values
        ax.barh(range(len(pivot)), vals, left=bottoms, label=harm, color=colors[i], edgecolor=FG_COLOR)
        bottoms += vals
    ax.set_yticks(range(len(pivot))); ax.set_yticklabels(wrap_labels(pivot.index.tolist()))
    ax.set_xlabel("Share of votes (stacked by harm)")
    ttl = f"Harm Share Distribution by Stakeholder (Top {len(pivot)})"
    if title_suffix: ttl += f" — {title_suffix}"
    ax.set_title(ttl); ax.legend(ncol=2, bbox_to_anchor=(1.02, 1), loc="upper left"); ax.grid(axis="x")
    fig.tight_layout(); fig.savefig(FIG_DIR / "02_stacked_shares_by_stakeholder.png", bbox_inches="tight"); plt.close(fig)

def plot_heatmap_harms_by_bias_type(df: pd.DataFrame, stakeholder: str, LEVELS: dict, title_suffix: str = ""):
    """Bias (rows) × Harm (cols) heatmap for a stakeholder, with all biases & harms shown."""
    dff = df[df["stakeholder_raw"].str.lower() == stakeholder.lower()].copy()
    if dff.empty: return
    dff["weighted_share"] = dff["vote_share"] * dff["total_votes"]
    mat = (dff.groupby(["bias_type","harm"], as_index=False)
              .agg(weighted_share=("weighted_share","sum"), total=("total_votes","sum")))
    mat["share"] = mat["weighted_share"] / mat["total"]

    pivot = (mat.pivot(index="bias_type", columns="harm", values="share")
                .reindex(index=LEVELS["bias_type"], columns=LEVELS["harm"])
                .fillna(0.0))

    rows = pivot.index.tolist()
    cols = pivot.columns.tolist()
    grid = pivot.to_numpy()

    fig, ax = plt.subplots(figsize=_heatmap_figsize(len(rows), len(cols)), constrained_layout=False)
    cmap = cmap_white_to(HEAT_COLOR)
    vmax = max(1e-6, float(grid.max()))
    im = ax.imshow(grid, aspect="auto", cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest")

    _style_heatmap_axes(ax, rows, cols)

    ann_fs = 8 if len(rows) > 20 or len(cols) > 20 else 9
    for i in range(len(rows)):
        for j in range(len(cols)):
            ax.text(j, i, f"{grid[i,j]*100:.0f}%", ha="center", va="center", color="#333333", fontsize=ann_fs)

    ttl = f"{stakeholder}: Harm Shares by Bias Type"
    if title_suffix: ttl += f" — {title_suffix}"
    ax.set_title(ttl)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="Share")
    cbar.ax.tick_params(labelsize=9)

    fig.savefig(FIG_DIR / f"03_heatmap_harms_by_bias_type__{stakeholder.replace(' ','_')}.png", bbox_inches="tight"); plt.close(fig)

def plot_top_harms_small_multiples(df: pd.DataFrame, title_suffix: str = "", top_k_harms: int = 6):
    if df.empty: return
    dff = df.copy()
    dff["weighted_share"] = dff["vote_share"] * dff["total_votes"]
    scores = (dff.groupby(["stakeholder_raw","harm"], as_index=False)
                 .agg(weighted_share=("weighted_share","sum"), total=("total_votes","sum")))
    scores["share"] = scores["weighted_share"] / scores["total"]

    stakeholders = scores["stakeholder_raw"].drop_duplicates().tolist()
    n = len(stakeholders); cols = 3; rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, max(6, rows*3.2)))
    axes = np.array(axes).reshape(rows, cols)

    for idx, st in enumerate(stakeholders):
        ax = axes[idx // cols, idx % cols]
        sub = (scores[scores["stakeholder_raw"] == st]
                   .sort_values("share", ascending=False).head(top_k_harms))
        ax.barh(range(len(sub)), sub["share"].values, color=color_cycle(len(sub)), edgecolor=FG_COLOR)
        ax.set_yticks(range(len(sub))); ax.set_yticklabels(wrap_labels(sub["harm"].tolist(), 18))
        ax.invert_yaxis(); ax.set_xlim(0, max(0.35, float(sub["share"].max())*1.15))
        ax.set_title(st); ax.grid(axis="x")
        for i, v in enumerate(sub["share"].values):
            ax.text(v + 0.005, i, f"{v:.0%}", va="center", ha="left", fontsize=9, color=FG_COLOR)

    # turn off any empty panels
    for j in range(idx+1, rows*cols):
        axes[j // cols, j % cols].axis("off")

    fig.suptitle("Top Harms by Stakeholder (Weighted Shares)" + (f" — {title_suffix}" if title_suffix else ""), y=0.98)
    fig.tight_layout(); fig.savefig(FIG_DIR / "04_top_harms_by_stakeholder_small_multiples.png", bbox_inches="tight"); plt.close(fig)

def plot_domain_splits_for_stakeholder(df: pd.DataFrame, stakeholder: str, title_suffix: str = ""):
    dff = df[df["stakeholder_raw"].str.lower() == stakeholder.lower()].copy()
    if dff.empty: return
    dff["weighted_share"] = dff["vote_share"] * dff["total_votes"]
    mat = (dff.groupby(["domain","harm"], as_index=False)
              .agg(weighted_share=("weighted_share","sum"), total=("total_votes","sum")))
    mat["share"] = mat["weighted_share"] / mat["total"]

    domains = mat["domain"].drop_duplicates().tolist()
    pivot = mat.pivot(index="domain", columns="harm", values="share").fillna(0.0)
    pivot = pivot.loc[domains]

    fig, ax = plt.subplots(figsize=(10, max(4.5, 0.5*len(domains))))
    bottoms = np.zeros(len(pivot)); colors = color_cycle(pivot.shape[1])
    for i, harm in enumerate(pivot.columns):
        vals = pivot[harm].values
        ax.barh(range(len(pivot)), vals, left=bottoms, label=harm, color=colors[i], edgecolor=FG_COLOR)
        bottoms += vals
    ax.set_yticks(range(len(pivot))); ax.set_yticklabels(domains)
    ax.set_xlabel("Share of votes (stacked by harm)")
    ttl = f"{stakeholder}: Harm Shares by Domain"
    if title_suffix: ttl += f" — {title_suffix}"
    ax.set_title(ttl); ax.legend(ncol=2, bbox_to_anchor=(1.02, 1), loc="upper left"); ax.grid(axis="x")
    fig.tight_layout(); fig.savefig(FIG_DIR / f"05_domain_splits__{stakeholder.replace(' ','_')}.png", bbox_inches="tight"); plt.close(fig)

# ------- Participation / coverage plots (how many answered etc.) -------

def plot_responses_by_stakeholder(df: pd.DataFrame):
    blocks = df.groupby(BLOCK_KEYS, as_index=False)["total_votes"].first()
    s = blocks.groupby("stakeholder_raw", as_index=False)["total_votes"].sum().sort_values("total_votes", ascending=True)
    fig, ax = plt.subplots(figsize=(8, max(4, 0.4*len(s)+2)))
    ax.barh(range(len(s)), s["total_votes"].values, color=color_cycle(len(s)))
    ax.set_yticks(range(len(s))); ax.set_yticklabels(wrap_labels(s["stakeholder_raw"].tolist()))
    ax.set_xlabel("total votes"); ax.set_title("Responses by Stakeholder")
    ax.grid(axis="x"); fig.tight_layout(); fig.savefig(FIG_DIR / "07_responses_by_stakeholder.png", bbox_inches="tight"); plt.close(fig)

def plot_responses_by_bias_type(df: pd.DataFrame):
    blocks = df.groupby(BLOCK_KEYS, as_index=False)["total_votes"].first()
    b = blocks.groupby("bias_type", as_index=False)["total_votes"].sum().sort_values("total_votes", ascending=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(range(len(b)), b["total_votes"].values, color=color_cycle(len(b)))
    ax.set_yticks(range(len(b))); ax.set_yticklabels(b["bias_type"].tolist())
    ax.set_xlabel("total votes"); ax.set_title("Responses by Bias Type")
    ax.grid(axis="x"); fig.tight_layout(); fig.savefig(FIG_DIR / "08_responses_by_bias_type.png", bbox_inches="tight"); plt.close(fig)

def plot_responses_by_domain(df: pd.DataFrame):
    blocks = df.groupby(BLOCK_KEYS, as_index=False)["total_votes"].first()
    d = blocks.groupby("domain", as_index=False)["total_votes"].sum().sort_values("total_votes", ascending=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(range(len(d)), d["total_votes"].values, color=color_cycle(len(d)))
    ax.set_yticks(range(len(d))); ax.set_yticklabels(d["domain"].tolist())
    ax.set_xlabel("total votes"); ax.set_title("Responses by Domain")
    ax.grid(axis="x"); fig.tight_layout(); fig.savefig(FIG_DIR / "09_responses_by_domain.png", bbox_inches="tight"); plt.close(fig)

def plot_heatmap_stakeholder_by_harm(
    df: pd.DataFrame,
    LEVELS: dict,
    title_suffix: str = "",
    x_scale: float = 0.75  # <-- increase to make the x-axis even wider
):
    """Stakeholder (rows) × Harm (cols) heatmap with all categories shown (zeros if absent)."""
    if df.empty:
        return

    d = df.copy()
    d["weighted_share"] = d["vote_share"] * d["total_votes"]
    mat = (d.groupby(["stakeholder_raw","harm"], as_index=False)
             .agg(weighted_share=("weighted_share","sum"), total=("total_votes","sum")))
    mat["share"] = mat["weighted_share"] / mat["total"]

    pivot = (mat.pivot(index="stakeholder_raw", columns="harm", values="share")
                .reindex(index=LEVELS["stakeholder_raw"], columns=LEVELS["harm"])
                .fillna(0.0))

    rows = pivot.index.tolist()
    cols = pivot.columns.tolist()
    grid = pivot.to_numpy()

    # --- make the x-axis larger by increasing figure width per column ---
    # width grows with number of columns; height scales with rows
    width_inches = max(10, x_scale * len(cols))   # was ~0.45 * len(cols); bumped up with x_scale
    height_inches = max(4.8, 0.45 * len(rows))
    fig, ax = plt.subplots(figsize=(width_inches, height_inches), constrained_layout=False)

    cmap = cmap_white_to(HEAT_COLOR)
    vmax = max(1e-6, float(grid.max()))
    im = ax.imshow(grid, aspect="auto", cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest")

    _style_heatmap_axes(ax, rows, cols)

    ann_fs = 8 if len(rows) > 20 or len(cols) > 20 else 9
    for i in range(len(rows)):
        for j in range(len(cols)):
            ax.text(j, i, f"{grid[i,j]*100:.0f}%", ha="center", va="center", color="#333333", fontsize=ann_fs)

    ttl = "Harm Shares: Stakeholder × Harm"
    if title_suffix: ttl += f" — {title_suffix}"
    ax.set_title(ttl)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="Share")
    cbar.ax.tick_params(labelsize=9)

    fig.savefig(FIG_DIR / "10_heatmap_stakeholder_by_harm.png", bbox_inches="tight")
    plt.close(fig)

# ------------------------ Main ------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate modern figures from clean_results_human.csv")
    parser.add_argument("--in", dest="in_path", default=str(DATA_PATH), help="Input clean CSV (default: data/clean_results_human.csv)")
    parser.add_argument("--domain", default=None, help="Filter to a single domain (e.g., hiring, diagnosis, company)")
    parser.add_argument("--bias_type", default=None, help="Filter to a single bias_type (representation, algorithmic, ...)")
    parser.add_argument("--stakeholder", default=None, help='Filter to single stakeholder_raw (e.g., "applicant group")')
    parser.add_argument("--for_stakeholder", default=None, help='Create heatmap+domain-splits for this stakeholder_raw')
    parser.add_argument("--top_n_stakeholders", type=int, default=10, help="Top N stakeholders for stacked chart (default: 10)")
    parser.add_argument("--top_k_focus", type=int, default=4, help="How many stakeholders to auto-focus (heatmaps/domain splits) if none specified")
    # Kept for CLI stability (not used directly in this version of heatmaps)
    parser.add_argument("--top_m_harms", type=int, default=10, help="(unused) Columns in stakeholder×harm heatmap")
    parser.add_argument("--top_n_domains", type=int, default=None, help="(unused) Top N domains to show in domain heatmap")

    args = parser.parse_args()

    df = load_clean(Path(args.in_path))
    df = apply_filters(df, domain=args.domain, bias_type=args.bias_type, stakeholder=args.stakeholder)
    ensure_figdir()

    suffix_bits = []
    if args.domain: suffix_bits.append(f"Domain: {args.domain}")
    if args.bias_type: suffix_bits.append(f"Bias: {args.bias_type}")
    if args.stakeholder: suffix_bits.append(f"Stakeholder: {args.stakeholder}")
    suffix = " | ".join(suffix_bits)

    LEVELS = get_levels(df)

    # Core figures
    plot_overall_harm_shares(df, title_suffix=suffix)
    plot_stacked_shares_by_stakeholder(df, top_n_stakeholders=args.top_n_stakeholders, title_suffix=suffix)
    plot_top_harms_small_multiples(df, title_suffix=suffix, top_k_harms=6)

    # Domain-focused distributions
    plot_stacked_shares_by_domain(df, title_suffix=suffix)
    plot_heatmap_domain_by_harm(df, LEVELS, title_suffix=suffix)

    # Participation / coverage
    plot_respondents_by_questionnaire(df)
    plot_responses_by_stakeholder(df)
    plot_responses_by_bias_type(df)
    plot_responses_by_domain(df)

    # Overview heatmap (all stakeholders/harms)
    plot_heatmap_stakeholder_by_harm(df, LEVELS, title_suffix=suffix)

    # Focused heatmaps/domain splits
    if args.for_stakeholder:
        targets = [args.for_stakeholder]
    elif args.stakeholder:
        targets = [args.stakeholder]
    else:
        blocks = df.groupby(BLOCK_KEYS, as_index=False)["total_votes"].first()
        topk = (blocks.groupby("stakeholder_raw", as_index=False)["total_votes"].sum()
                  .sort_values("total_votes", ascending=False)
                  .head(args.top_k_focus))
        targets = topk["stakeholder_raw"].tolist()

    for st in targets:
        plot_heatmap_harms_by_bias_type(df, stakeholder=st, LEVELS=LEVELS, title_suffix=suffix)
        plot_domain_splits_for_stakeholder(df, stakeholder=st, title_suffix=suffix)

if __name__ == "__main__":
    main()
