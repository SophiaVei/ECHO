# make_bias_harm_radars_pub.py
# -*- coding: utf-8 -*-
"""
Radar plots of harm profiles per bias type inside each (domain × stakeholder_raw) cell.

OUTPUTS (under ./radars_pub/):
- <domain>__<stakeholder_raw>__ALL_BIASES__radar.png / .pdf
- <domain>__<stakeholder_raw>__PAIR__<A>__vs__<B>__radar.png / .pdf
- <domain>__<stakeholder_raw>__PAIR__<A>__vs__<B>__stats.csv

Notes
- Row-normalized shares P(h|bias) as radius.
- Per-harm Fisher 2×2 with BH-FDR (q<.05) → filled vertex markers (size ∝ |log-odds|).
- Radial ticks every 10%; smaller tick labels; custom soft palette.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import itertools
import re
import textwrap
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact
import matplotlib as mpl
import matplotlib.pyplot as plt

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

# Custom soft palette (first four are exactly as requested)
CUSTOM_PALETTE = [
    "#cce6e0",  # 1
    "#b2a0df",  # 2
    "#f9acac",  # 3
    "#ffdabb",  # 4  (note: single #)
    "#a7c7e7",  # 5  soft blue
    "#c6e2a9",  # 6  soft green
    "#f5c2e7",  # 7  soft pink
    "#c7c7c7",  # 8  neutral gray
]

# -------------------- Utilities --------------------
def sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9\-_.]+", "_", str(name).strip().lower())

def wrap_labels(labels, width=16):
    return ["\n".join(textwrap.wrap(str(x), width=width, break_long_words=True, replace_whitespace=False)) for x in labels]

def percent_ticks_auto_10(rmax, step=0.10, hard_cap=0.60):
    """
    Percent y-ticks every 10% up to the next step above rmax (capped).
    Returns (ticks array, ytop).
    """
    top = min(hard_cap, (np.ceil(max(rmax, step) / step) * step))
    ticks = np.arange(step, top + 1e-9, step)
    return ticks, float(top)

def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    n = p.size
    order = np.argsort(p)
    ranks = np.arange(1, n + 1, dtype=float)
    q_sorted = p[order] * n / ranks
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q = np.empty_like(q_sorted)
    q[order] = q_sorted
    return np.clip(q, 0, 1)

def standardized_residuals(obs: np.ndarray, exp: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return (obs - exp) / np.sqrt(exp)

def safe_chi2_2xH(ctab_2xH: pd.DataFrame):
    t = ctab_2xH.loc[:, ctab_2xH.sum(axis=0) > 0]
    if t.shape[0] < 2 or t.shape[1] < 2:
        return np.nan, np.nan, 0, None
    return chi2_contingency(t.to_numpy(), correction=False)

def per_harm_tests(ctab_2xH: pd.DataFrame, add_smooth: float = 0.5) -> pd.DataFrame:
    assert ctab_2xH.shape[0] == 2
    a_name, b_name = ctab_2xH.index.tolist()
    A = int(ctab_2xH.loc[a_name].sum())
    B = int(ctab_2xH.loc[b_name].sum())
    harms = list(ctab_2xH.columns)
    pvals, lors, a_counts, b_counts = [], [], [], []
    for h in harms:
        a = int(ctab_2xH.loc[a_name, h]); b = int(ctab_2xH.loc[b_name, h])
        _, p = fisher_exact([[a, A - a], [b, B - b]], alternative="two-sided")
        a1, b1 = a + add_smooth, b + add_smooth
        A1, B1 = (A - a) + add_smooth, (B - b) + add_smooth
        lor = np.log((a1 / A1) / (b1 / B1))
        pvals.append(p); lors.append(lor); a_counts.append(a); b_counts.append(b)
    qvals = bh_fdr(np.array(pvals))
    return pd.DataFrame({"harm": harms, "a_harm": a_counts, "b_harm": b_counts, "p": pvals, "q": qvals, "lor": lors})

def build_kxH(df_cell: pd.DataFrame) -> pd.DataFrame:
    ctab = pd.pivot_table(
        df_cell, values="votes", index="bias_type", columns="harm", aggfunc="sum", fill_value=0
    ).astype(int)
    # keep all harms with any presence in the cell
    ctab = ctab.loc[:, ctab.sum(axis=0) > 0]
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

# -------------------- Plotters --------------------
def plot_all_bias_radar(ctab: pd.DataFrame, title: str, subtitle: str, out_base: Path,
                        max_labels: int | None = None):
    if ctab.empty or ctab.shape[0] < 1 or ctab.shape[1] < 3:
        return
    harms_order = order_harms(ctab, by="pooled")
    if max_labels is not None:
        harms_order = harms_order[:max_labels]

    shares = row_shares(ctab)[harms_order]
    angles = radar_angles(len(harms_order))
    xticklabels = wrap_labels(harms_order, width=16)

    colors = {b: CUSTOM_PALETTE[i % len(CUSTOM_PALETTE)] for i, b in enumerate(shares.index)}

    fig = plt.figure(figsize=(6.2, 6.2))
    ax = plt.subplot(111, polar=True)

    # dynamic radial limit (no clipping)
    data_max = float(np.nanmax(shares.values))
    rmax_target = max(0.20, data_max * 1.08)
    yticks, ytop = percent_ticks_auto_10(rmax_target, step=0.10, hard_cap=0.60)
    ax.set_ylim(0, ytop)

    # polygons (no baseline)
    for bias in shares.index:
        vals = np.r_[shares.loc[bias].values, shares.loc[bias].values[0]]
        ax.plot(angles, vals, linewidth=2.1, color=colors[bias], label=bias, zorder=3)
        ax.fill(angles, vals, color=colors[bias], alpha=0.18, zorder=2)

    # percent ticks every 10% and slightly smaller labels
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{int(t*100)}%" for t in yticks], fontsize=8)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(xticklabels, fontsize=9)

    ax.set_title(title + (f"\n{subtitle}" if subtitle else ""), pad=12,
                 loc="center", fontsize=11, fontweight="bold")

    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.02),
              borderaxespad=0.0, handlelength=1.6)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    save_figure(fig, out_base)

def plot_pair_radar(ctab_2xH: pd.DataFrame, title: str, subtitle: str, out_img_base: Path, out_csv: Path,
                    q_alpha: float = 0.05, max_labels: int | None = None, mark_effect_size: bool = True):
    # keep all harms with any presence in the pair
    ctab_2xH = ctab_2xH.loc[:, ctab_2xH.sum(axis=0) > 0]
    if ctab_2xH.shape[1] < 3:
        return

    a_name, b_name = ctab_2xH.index.tolist()
    harms_order = order_harms(ctab_2xH, by="pooled")
    if max_labels is not None:
        harms_order = harms_order[:max_labels]
    ctab_2xH = ctab_2xH[harms_order]

    shares = row_shares(ctab_2xH)
    pA = shares.loc[a_name]; pB = shares.loc[b_name]
    delta = pA - pB

    sig_df = per_harm_tests(ctab_2xH).set_index("harm").loc[harms_order].reset_index()

    chi2, p, df, exp = safe_chi2_2xH(ctab_2xH)
    residA = residB = np.full_like(pA.values, np.nan, dtype=float)
    if exp is not None:
        obs = ctab_2xH.to_numpy()
        std_res = standardized_residuals(obs, exp)
        residA, residB = std_res[0, :], std_res[1, :]

    stats = pd.DataFrame({
        "harm": harms_order,
        "share_A": pA.values,
        "share_B": pB.values,
        "share_diff_A_minus_B": delta.values,
        "lor": sig_df["lor"],
        "p": sig_df["p"],
        "q": sig_df["q"],
        "std_resid_A": residA,
        "std_resid_B": residB,
        "chi2_pair": [chi2]*len(harms_order),
        "chi2_p_value": [p]*len(harms_order),
        "chi2_df": [df]*len(harms_order),
    })
    stats.to_csv(out_csv, index=False)

    angles = radar_angles(len(harms_order))
    xticklabels = wrap_labels(harms_order, width=16)

    # two contrasting colors from the custom palette
    colorA, colorB = CUSTOM_PALETTE[1], CUSTOM_PALETTE[2]

    fig = plt.figure(figsize=(6.2, 6.2))
    ax = plt.subplot(111, polar=True)

    # dynamic radial limit (no clipping)
    data_max = float(np.nanmax(shares.values))
    rmax_target = max(0.20, data_max * 1.08)
    yticks, ytop = percent_ticks_auto_10(rmax_target, step=0.10, hard_cap=0.60)
    ax.set_ylim(0, ytop)

    valsA = np.r_[pA.values, pA.values[0]]
    valsB = np.r_[pB.values, pB.values[0]]

    ax.plot(angles, valsA, linewidth=2.1, color=colorA, label=a_name, zorder=3)
    ax.fill(angles, valsA, color=colorA, alpha=0.22, zorder=2)

    ax.plot(angles, valsB, linewidth=2.1, color=colorB, label=b_name, zorder=3)
    ax.fill(angles, valsB, color=colorB, alpha=0.22, zorder=2)

    # significance markers (q < alpha), neutral small-ish dots
    for i, _ in enumerate(harms_order):
        if sig_df.loc[i, "q"] < q_alpha:
            r = max(valsA[i], valsB[i])
            size = 24 + 60 * min(2.0, abs(sig_df.loc[i, "lor"])) if mark_effect_size else 40
            ax.scatter(angles[i], r, s=size, color="#4d4d4d", zorder=5)

    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{int(t*100)}%" for t in yticks], fontsize=8)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(xticklabels, fontsize=9)

    ax.set_title(title + (f"\n{subtitle}" if subtitle else ""), pad=12,
                 loc="center", fontsize=11, fontweight="bold")

    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.02),
              borderaxespad=0.0, handlelength=1.6)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    save_figure(fig, out_img_base)

# -------------------- Pipeline --------------------
def load_data(in_path: Path) -> pd.DataFrame:
    df = pd.read_csv(in_path)
    missing = REQUIRED.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns in input: {missing}")
    key = ["questionnaire_id", "stakeholder_raw", "bias_type", "domain", "harm"]
    df = df.groupby(key, as_index=False)["votes"].sum()
    return df

def each_cell(df: pd.DataFrame):
    for (domain, who), sub in df.groupby(["domain", "stakeholder_raw"], dropna=False):
        yield str(domain), str(who), sub

def ensure_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

def main():
    set_pub_style()
    ap = argparse.ArgumentParser(description="Radar plots for bias→harm profiles per (domain × stakeholder_raw).")
    ap.add_argument("--in", dest="in_path", default=str(DATA_DEFAULT), help="Input CSV path (clean_results_human.csv)")
    ap.add_argument("--out", dest="out_dir", default=str(OUT_ROOT), help="Output directory (default: ./radars_pub)")
    ap.add_argument("--min_harms", type=int, default=5, help="Minimum number of nonzero harms to plot a radar (default: 5)")
    ap.add_argument("--top_harms", type=int, default=0, help="If >0, limit radar axes to top-N harms by pooled frequency")
    ap.add_argument("--no_effect_size_markers", action="store_true", help="Do not scale significance markers by |LOR|")
    args = ap.parse_args()

    in_path = Path(args.in_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(in_path)

    for domain, who, sub in each_cell(df):
        cell_tag = f"{sanitize(domain)}__{sanitize(who)}"
        ctab = build_kxH(sub)
        if ctab.shape[0] < 2 or ctab.shape[1] < args.min_harms:
            continue

        max_labels = args.top_harms if args.top_harms > 0 else None

        # ALL BIASES
        out_all_base = out_dir / f"{cell_tag}__ALL_BIASES__radar"
        title_all = f"{domain} × {who}"
        subtitle_all = "Harm distributions by bias type"
        ensure_dir(out_all_base)
        plot_all_bias_radar(ctab, title_all, subtitle_all, out_all_base, max_labels=max_labels)

        # Pairwise
        biases = ctab.index.tolist()
        for a, b in itertools.combinations(biases, 2):
            ctab_2xH = ctab.loc[[a, b]].copy()
            if ctab_2xH.sum(axis=0).gt(0).sum() < args.min_harms:
                continue
            out_img_base = out_dir / f"{cell_tag}__PAIR__{sanitize(a)}__vs__{sanitize(b)}__radar"
            out_csv = out_dir / f"{cell_tag}__PAIR__{sanitize(a)}__vs__{sanitize(b)}__stats.csv"
            ensure_dir(out_img_base)
            title = f"{domain} × {who}: {a} vs {b}"
            subtitle = "Row-normalized harm shares; filled markers indicate q < 0.05 (Fisher + BH)"
            plot_pair_radar(
                ctab_2xH,
                title=title,
                subtitle=subtitle,
                out_img_base=out_img_base,
                out_csv=out_csv,
                q_alpha=0.05,
                max_labels=max_labels,
                mark_effect_size=not args.no_effect_size_markers
            )

    print(f"Done. Plots written under: {out_dir.resolve()}")

if __name__ == "__main__":
    main()



# to run: python make_bias_harm_radars.py --in data/clean_results_human.csv --out radars