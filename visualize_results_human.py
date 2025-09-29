# -*- coding: utf-8 -*-
"""
Create insight-driven, publication-ready plots from clean_results_human.csv.

Outputs (figures/):
  01_overall_harm_shares.png
  02_stacked_shares_by_stakeholder.png
  03_heatmap_harms_by_bias_type__<stakeholder>.png   (for top-K or requested stakeholder)
  04_top_harms_by_stakeholder_small_multiples.png
  05_domain_splits__<stakeholder>.png                (for top-K or requested stakeholder)
  06_responses_by_questionnaire.png
  07_responses_by_stakeholder.png
  08_responses_by_bias_type.png
  09_responses_by_domain.png
  10_heatmap_stakeholder_by_harm.png

CLI examples:
  python visualize_results_human.py
  python visualize_results_human.py --domain hiring
  python visualize_results_human.py --bias_type representation
  python visualize_results_human.py --stakeholder "applicant group" --domain hiring
  python visualize_results_human.py --top_k_focus 4 --top_n_stakeholders 7
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
        out = out[out["domain"].str.lower() == domain.lower()]
    if bias_type:
        out = out[out["bias_type"].str.lower() == bias_type.lower()]
    if stakeholder:
        out = out[out["stakeholder_raw"].str.lower() == stakeholder.lower()]
    return out

def ensure_figdir():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

def color_cycle(n: int) -> list[str]:
    if n <= len(PALETTE): return PALETTE[:n]
    reps = int(np.ceil(n / len(PALETTE)))
    return (PALETTE * reps)[:n]

def wrap_labels(labels, width=18):
    return ["\n".join(wrap(str(x), width)) for x in labels]

# ------------------------ Insight plots ------------------------

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

def plot_heatmap_harms_by_bias_type(df: pd.DataFrame, stakeholder: str, title_suffix: str = ""):
    dff = df[df["stakeholder_raw"].str.lower() == stakeholder.lower()]
    if dff.empty: return
    dff["weighted_share"] = dff["vote_share"] * dff["total_votes"]
    mat = (dff.groupby(["bias_type","harm"], as_index=False)
              .agg(weighted_share=("weighted_share","sum"), total=("total_votes","sum")))
    mat["share"] = mat["weighted_share"] / mat["total"]

    rows = sorted(mat["bias_type"].unique().tolist())
    cols = sorted(mat["harm"].unique().tolist())
    grid = np.zeros((len(rows), len(cols)))
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            v = mat.loc[(mat["bias_type"] == r) & (mat["harm"] == c), "share"]
            grid[i, j] = float(v.iloc[0]) if len(v) else 0.0

    fig, ax = plt.subplots(figsize=(max(8, 0.45*len(cols)), max(4, 0.5*len(rows))))
    cmap = cmap_white_to(HEAT_COLOR)
    vmax = max(1e-6, float(grid.max()))  # keep 0..max scale; avoids flat color if all zeros
    im = ax.imshow(grid, aspect="auto", cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest")
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(wrap_labels(cols, 16), rotation=45, ha="right")
    for i in range(len(rows)):
        for j in range(len(cols)):
            ax.text(j, i, f"{grid[i,j]*100:.0f}%", ha="center", va="center", color="#333333", fontsize=9)
    ttl = f"{stakeholder}: Harm Shares by Bias Type"
    if title_suffix: ttl += f" — {title_suffix}"
    ax.set_title(ttl); fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Share")
    fig.tight_layout(); fig.savefig(FIG_DIR / f"03_heatmap_harms_by_bias_type__{stakeholder.replace(' ','_')}.png", bbox_inches="tight"); plt.close(fig)

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

    for j in range(idx+1, rows*cols): axes[j // cols, j % cols].axis("off")
    fig.suptitle("Top Harms by Stakeholder (Weighted Shares)" + (f" — {title_suffix}" if title_suffix else ""), y=0.98)
    fig.tight_layout(); fig.savefig(FIG_DIR / "04_top_harms_by_stakeholder_small_multiples.png", bbox_inches="tight"); plt.close(fig)

def plot_domain_splits_for_stakeholder(df: pd.DataFrame, stakeholder: str, title_suffix: str = ""):
    dff = df[df["stakeholder_raw"].str.lower() == stakeholder.lower()]
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

def plot_responses_by_questionnaire(df: pd.DataFrame):
    blocks = df.groupby(BLOCK_KEYS, as_index=False)["total_votes"].first()  # one row per block
    q = blocks.groupby("questionnaire_id", as_index=False)["total_votes"].sum()
    q = q.sort_values("questionnaire_id")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(q["questionnaire_id"].astype(str), q["total_votes"], color=color_cycle(len(q)))
    ax.set_xlabel("questionnaire_id"); ax.set_ylabel("total votes"); ax.set_title("Responses by Questionnaire")
    ax.grid(axis="y"); fig.tight_layout(); fig.savefig(FIG_DIR / "06_responses_by_questionnaire.png", bbox_inches="tight"); plt.close(fig)

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

def plot_heatmap_stakeholder_by_harm(df: pd.DataFrame, top_n_stakeholders: int = 8, top_m_harms: int = 10, title_suffix: str = ""):
    """Stakeholder (rows) × Harm (cols) weighted-share heatmap; trimmed to top-N/M for readability."""
    if df.empty: return
    # top stakeholders by participation
    st_tot = (df.groupby("stakeholder_raw", as_index=False)["total_votes"].sum()
                .sort_values("total_votes", ascending=False).head(top_n_stakeholders))
    keep_st = set(st_tot["stakeholder_raw"])
    d = df[df["stakeholder_raw"].isin(keep_st)].copy()
    d["weighted_share"] = d["vote_share"] * d["total_votes"]
    mat = (d.groupby(["stakeholder_raw","harm"], as_index=False)
             .agg(weighted_share=("weighted_share","sum"), total=("total_votes","sum")))
    mat["share"] = mat["weighted_share"] / mat["total"]

    # take top harms by overall share
    top_h = (mat.groupby("harm", as_index=False)["weighted_share"].sum()
                .sort_values("weighted_share", ascending=False).head(top_m_harms))["harm"].tolist()
    mat = mat[mat["harm"].isin(top_h)]

    rows = st_tot["stakeholder_raw"].tolist()
    cols = sorted(top_h)
    grid = np.zeros((len(rows), len(cols)))
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            v = mat.loc[(mat["stakeholder_raw"] == r) & (mat["harm"] == c), "share"]
            grid[i, j] = float(v.iloc[0]) if len(v) else 0.0

    fig, ax = plt.subplots(figsize=(max(8, 0.45*len(cols)), max(4.5, 0.45*len(rows))))
    cmap = cmap_white_to(HEAT_COLOR)
    vmax = max(1e-6, float(grid.max()))
    im = ax.imshow(grid, aspect="auto", cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest")
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(wrap_labels(cols, 16), rotation=45, ha="right")
    for i in range(len(rows)):
        for j in range(len(cols)):
            ax.text(j, i, f"{grid[i,j]*100:.0f}%", ha="center", va="center", color="#333333", fontsize=9)
    ttl = f"Harm Shares: Stakeholder × Harm"
    if title_suffix: ttl += f" — {title_suffix}"
    ax.set_title(ttl); fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Share")
    fig.tight_layout(); fig.savefig(FIG_DIR / "10_heatmap_stakeholder_by_harm.png", bbox_inches="tight"); plt.close(fig)

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
    parser.add_argument("--top_m_harms", type=int, default=10, help="Columns in stakeholder×harm heatmap (default: 10)")
    args = parser.parse_args()

    df = load_clean(Path(args.in_path))
    df = apply_filters(df, domain=args.domain, bias_type=args.bias_type, stakeholder=args.stakeholder)
    ensure_figdir()

    suffix_bits = []
    if args.domain: suffix_bits.append(f"Domain: {args.domain}")
    if args.bias_type: suffix_bits.append(f"Bias: {args.bias_type}")
    if args.stakeholder: suffix_bits.append(f"Stakeholder: {args.stakeholder}")
    suffix = " | ".join(suffix_bits)

    # Core figures
    plot_overall_harm_shares(df, title_suffix=suffix)
    plot_stacked_shares_by_stakeholder(df, top_n_stakeholders=args.top_n_stakeholders, title_suffix=suffix)
    plot_top_harms_small_multiples(df, title_suffix=suffix, top_k_harms=6)

    # Participation/coverage
    plot_responses_by_questionnaire(df)
    plot_responses_by_stakeholder(df)
    plot_responses_by_bias_type(df)
    plot_responses_by_domain(df)
    plot_heatmap_stakeholder_by_harm(df, top_n_stakeholders=args.top_n_stakeholders, top_m_harms=args.top_m_harms, title_suffix=suffix)

    # Focused heatmaps/domain splits
    targets = []
    if args.for_stakeholder:
        targets = [args.for_stakeholder]
    elif args.stakeholder:
        targets = [args.stakeholder]
    else:
        # auto-pick top-K stakeholders by total votes
        blocks = df.groupby(BLOCK_KEYS, as_index=False)["total_votes"].first()
        topk = (blocks.groupby("stakeholder_raw", as_index=False)["total_votes"].sum()
                  .sort_values("total_votes", ascending=False)
                  .head(args.top_k_focus))
        targets = topk["stakeholder_raw"].tolist()

    for st in targets:
        plot_heatmap_harms_by_bias_type(df, stakeholder=st, title_suffix=suffix)
        plot_domain_splits_for_stakeholder(df, stakeholder=st, title_suffix=suffix)

if __name__ == "__main__":
    main()
