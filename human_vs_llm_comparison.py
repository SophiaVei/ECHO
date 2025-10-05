# human_vs_llm_comparison.py
# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as mticker

# ---------- Text normalization helpers ----------
def _ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def canon_lower(s: str) -> str:
    return (s or "").strip().lower()

def canon_lower_fix(s: str) -> str:
    s = (s or "").strip()
    s = s.replace("Insitution", "Institution")
    s = re.sub(r"\s+", " ", s)
    return s.lower()

def normalize_harm_label(label: str) -> str:
    s = _ws(label).rstrip(":").rstrip(".")
    replacements = {
        "Erasing": "Erasure",
        "Erasing.": "Erasure",
        "Denying self-identity": "Denying Self-Identity",
        "Reifying categories": "Reifying Categories",
        "Technology-Facilitated Violence": "Technology-Facilitated Violence",
        "Diminished Health and Well-Being": "Diminished Health and Well-Being",
    }
    if s in replacements:
        return replacements[s]
    s_cap = s.title().replace("Well-Being", "Well-Being")
    return replacements.get(s_cap, s)

def normalize_stakeholder(stakeholder: str) -> Tuple[str, bool]:
    s = _ws(stakeholder).replace("group", "Group").replace("Insitution", "Institution")
    is_group = "Group" in s
    base = s.replace(" Group", "")
    return base, is_group

def normalize_bias_type(stage: str) -> str:
    canon = {
        "Representation": "Representation",
        "Algorithmic": "Algorithmic",
        "Measurement": "Measurement",
        "Deployment": "Deployment",
        "Evaluation": "Evaluation",
    }
    s = _ws(stage).title()
    return canon.get(s, s)

def normalize_domain(domain: str) -> str:
    s = _ws(domain)
    s = re.sub(r"[:)\.]+$", "", s)
    fixes = {"Hiring": "Hiring", "Diagnosis": "Diagnosis", "Company": "Company"}
    return fixes.get(s, s)

# ---------- Header regex ----------
CSV_HEADER_RE = re.compile(
    r'^Questionnaire\s+(\d+)\s*\(([^,]+),\s*([^)]+)\)\s*/\s*(.+)$',
    re.IGNORECASE,
)

# ---------- Parser ----------
def parse_human_llm_csv(path: Path) -> pd.DataFrame:
    """
    Parse a file with columns: [label, SUMS, LLM]
      - 'label' holds either a header line or a harm label.
      - 'SUMS' holds the human count (integer, including zeros).
      - 'LLM' is '1' when the LLM selected that harm for this block, else blank.
    Returns a tidy long DataFrame with:
      questionnaire_id, stakeholder, is_group_level, bias_type, domain,
      harm_family, harm, votes, llm_selected, total_votes, vote_share
    """
    df_raw = pd.read_csv(
        path,
        header=None,
        names=["label", "SUMS", "LLM"],
        dtype=str,
        keep_default_na=False,
        encoding="utf-8",
        engine="python",
    )

    rows: List[Dict] = []
    current = None

    for _, row in df_raw.iterrows():
        label_raw = row["label"]
        if not label_raw:
            continue
        label_clean = _ws(label_raw.strip('"'))

        # Header?
        m = CSV_HEADER_RE.match(label_clean)
        if m:
            qid = int(m.group(1))
            stakeholder_raw = m.group(2)
            bias_type_raw = m.group(3)
            domain_raw = m.group(4)

            stakeholder_base, is_group = normalize_stakeholder(stakeholder_raw)
            current = {
                "questionnaire_id": qid,
                "stakeholder_raw": stakeholder_raw.strip(),
                "stakeholder": canon_lower(stakeholder_base),
                "is_group_level": bool(is_group),
                "bias_type": canon_lower(normalize_bias_type(bias_type_raw)),
                "domain": canon_lower(normalize_domain(domain_raw)),
            }
            continue

        if current is None:
            continue

        harm_label = _ws(label_clean)
        if harm_label == "":
            continue

        # Human votes
        sums_raw = _ws(row["SUMS"])
        votes = None
        if sums_raw != "":
            try:
                votes = int(float(sums_raw))
            except Exception:
                votes = None

        # LLM pick
        llm_raw = _ws(row["LLM"])
        llm_sel = 1 if llm_raw == "1" else 0

        harm_norm = canon_lower(normalize_harm_label(harm_label))
        harm_family = "group" if current["is_group_level"] else "individual"

        if harm_norm:
            rows.append({
                **current,
                "harm_family": harm_family,
                "harm": harm_norm,
                "votes": votes if votes is not None else 0,
                "llm_selected": llm_sel,
            })

    df = pd.DataFrame(rows)

    # Totals / shares
    block_keys = ["questionnaire_id", "stakeholder", "is_group_level", "bias_type", "domain", "harm_family"]
    totals = (
        df.groupby(block_keys, as_index=False)["votes"]
          .sum()
          .rename(columns={"votes": "total_votes"})
    )
    df = df.merge(totals, on=block_keys, how="left")
    df["vote_share"] = df["votes"] / df["total_votes"].replace({0: np.nan})

    # ---- NEW: Enforce exactly ONE LLM pick per block ----
    def enforce_single_llm_pick(g: pd.DataFrame) -> pd.DataFrame:
        picks = g.index[g["llm_selected"] == 1].tolist()
        if len(picks) <= 1:
            return g
        # keep the pick that aligns best with humans (highest vote_share), then alphabetical harm
        keep = (
            g.loc[picks].sort_values(["vote_share", "harm"], ascending=[False, True]).index[0]
        )
        g.loc[picks, "llm_selected"] = 0
        g.loc[keep, "llm_selected"] = 1
        return g

    df = df.groupby(block_keys, group_keys=False).apply(enforce_single_llm_pick)

    # Ordering
    df = df.sort_values(
        ["questionnaire_id", "stakeholder", "bias_type", "domain", "harm_family", "harm"]
    ).reset_index(drop=True)

    return df

# ---------- Agreement metrics ----------
BLOCK_KEYS = ["questionnaire_id", "stakeholder", "is_group_level", "bias_type", "domain", "harm_family"]

def pick_block_metrics(block: pd.DataFrame) -> Dict:
    # Human top(s) by votes (allow ties)
    max_votes = block["votes"].max()
    human_top = set(block.loc[block["votes"] == max_votes, "harm"].astype(str))

    # LLM single pick (by construction)
    llm_picks = set(block.loc[block["llm_selected"] == 1, "harm"].astype(str))

    top_hit_any = (len(llm_picks) > 0) and (len(human_top & llm_picks) > 0)

    # Exact 1-vs-1 defined only if humans have a unique top
    exact_defined = (len(human_top) == 1) and (len(llm_picks) == 1)
    exact_acc = np.nan
    if exact_defined:
        exact_acc = float(next(iter(human_top)) == next(iter(llm_picks)))

    # Point-biserial r (human share vs LLM binary)
    r_pb = np.nan
    try:
        y = block["llm_selected"].astype(float).values
        x = block["vote_share"].astype(float).values
        if y.sum() > 0 and y.sum() < len(y) and np.isfinite(x).all():
            r_pb = float(np.corrcoef(x, y)[0, 1])
    except Exception:
        r_pb = np.nan

    return {
        "n_harms": int(len(block)),
        "human_top_card": int(len(human_top)),
        "llm_sel_card": int(len(llm_picks)),
        "top_hit": bool(top_hit_any),
        "exact_defined": bool(exact_defined),
        "exact_acc": exact_acc,
        "point_biserial_r": r_pb,
        "human_top_harms": "; ".join(sorted(human_top)),
        "llm_selected_harms": "; ".join(sorted(llm_picks)),
    }

# ---------- Plot helpers ----------
PALETTE = ["#4C78A8", "#72B7B2", "#E45756", "#F58518"]

def save_barplot(df_plot: pd.DataFrame, xcol: str, ycols: List[str], title: str, fname: Path):
    # ycols is expected to be ["top_hit_rate", "exact_acc_rate"]
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(df_plot))
    width = 0.36

    # Bars = LLM metrics vs humans
    labels_map = {
        "top_hit_rate": "LLM Top-Hit vs Human Top",
        "exact_acc_rate": "LLM Exact Match (unique human top)"
    }
    colors = {"top_hit_rate": "#4C78A8", "exact_acc_rate": "#72B7B2"}

    for i, ycol in enumerate(ycols):
        vals = np.nan_to_num(df_plot[ycol].values.astype(float), nan=0.0)
        ax.bar(x + i*width, vals, width,
               label=labels_map.get(ycol, ycol),
               color=colors.get(ycol, "#4C78A8"),
               edgecolor="none")

    # Right axis = coverage of exact metric
    has_exact = "exact_defined_frac" in df_plot.columns
    if has_exact:
        ax2 = ax.twinx()
        ax2.plot(x + width, df_plot["exact_defined_frac"].values.astype(float),
                 marker="o", linestyle="-", linewidth=1.5, markersize=6)
        ax2.set_ylim(0, 1)
        ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
        ax2.set_ylabel("Exact metric coverage")

    # Cosmetics
    ax.set_xticks(x + width/2)
    ax.set_xticklabels(df_plot[xcol].astype(str).values, rotation=30, ha="right")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.set_ylabel("Agreement (LLM vs humans)")
    ax.set_xlabel(xcol.replace("_", " ").title())
    ax.set_title(title, pad=8)
    leg = ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()

    fig.savefig(fname, dpi=200, facecolor="white")
    plt.close(fig)

# ---------- Main ----------
def main(in_path: Path, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)

    df = parse_human_llm_csv(in_path)
    df.to_csv(outdir / "clean_results_humanandllm_long.csv", index=False, encoding="utf-8")

    # Block-level metrics
    block_rows = []
    for keys, block in df.groupby(BLOCK_KEYS, dropna=False):
        met = pick_block_metrics(block)
        row = dict(zip(BLOCK_KEYS, keys))
        row.update(met)
        block_rows.append(row)
    blocks = pd.DataFrame(block_rows)
    blocks.to_csv(outdir / "block_level_agreement.csv", index=False, encoding="utf-8")

    # Consistency check
    blocks["expected_n"] = blocks.apply(
        lambda r: 7 if r["harm_family"] == "group" else 9, axis=1
    )
    bad = blocks.loc[blocks["n_harms"] != blocks["expected_n"]].copy()
    bad.to_csv(outdir / "consistency_report_blocks_with_wrong_harm_count.csv", index=False, encoding="utf-8")

    # Overall summary (note exact_acc_rate is over blocks where exact_defined is True)
    overall = {
        "blocks": len(blocks),
        "top_hit_rate": blocks["top_hit"].mean(),
        "exact_acc_rate": blocks.loc[blocks["exact_defined"], "exact_acc"].mean(),
        "exact_defined_frac": blocks["exact_defined"].mean(),
        "mean_point_biserial_r": blocks["point_biserial_r"].replace([np.inf, -np.inf], np.nan).mean(),
    }
    pd.DataFrame([overall]).to_csv(outdir / "llm_agreement_summary.csv", index=False, encoding="utf-8")

    # Summaries by facets
    def facet_summary(gcols: List[str], outname: str):
        g = blocks.groupby(gcols, as_index=False).agg(
            blocks=("top_hit", "count"),
            top_hit_rate=("top_hit", "mean"),
            exact_acc_rate=("exact_acc", "mean"),          # NaN where undefined
            exact_defined_frac=("exact_defined", "mean"),
            mean_point_biserial_r=("point_biserial_r", "mean"),
        )
        g.to_csv(outdir / outname, index=False, encoding="utf-8")
        return g

    by_domain = facet_summary(["domain"], "summary_by_domain.csv")
    by_stakeholder = facet_summary(["stakeholder", "is_group_level"], "summary_by_stakeholder.csv")
    by_bias = facet_summary(["bias_type"], "summary_by_bias.csv")

    # Plots
    if len(by_domain):
        save_barplot(by_domain, "domain",
                     ["top_hit_rate", "exact_acc_rate"],
                     "LLM–Human Agreement by Domain",
                     outdir / "agreement_by_domain.png")

    if len(by_stakeholder):
        by_stakeholder["stakeholder_label"] = by_stakeholder.apply(
            lambda r: f"{r['stakeholder']}{' (group)' if r['is_group_level'] else ''}", axis=1)
        by_stakeholder = by_stakeholder.sort_values("stakeholder_label")
        save_barplot(by_stakeholder, "stakeholder_label",
                     ["top_hit_rate", "exact_acc_rate"],
                     "LLM–Human Agreement by Stakeholder",
                     outdir / "agreement_by_stakeholder.png")

    if len(by_bias):
        save_barplot(by_bias, "bias_type",
                     ["top_hit_rate", "exact_acc_rate"],
                     "LLM–Human Agreement by Bias Type",
                     outdir / "agreement_by_bias.png")

    # Confusion matrix (single LLM pick per block; human tiebreak alphabetical if tie)
    human_one, llm_one = [], []
    for _, r in blocks.iterrows():
        h_list = [s for s in r["human_top_harms"].split("; ") if s]
        l_list = [s for s in r["llm_selected_harms"].split("; ") if s]
        if not l_list:
            continue
        l = sorted(l_list)[0]  # single by construction; keep sorted for safety
        if not h_list:
            continue
        # unique human top preferred; else deterministic tiebreak
        h = h_list[0] if len(h_list) == 1 else sorted(h_list)[0]
        human_one.append(h)
        llm_one.append(l)

    if human_one and llm_one:
        all_harms = sorted(set(human_one) | set(llm_one))
        cm = pd.crosstab(
            pd.Series(human_one, name="Human"),
            pd.Series(llm_one, name="LLM"),
            dropna=False
        ).reindex(index=all_harms, columns=all_harms, fill_value=0)

        cmap = LinearSegmentedColormap.from_list("white_purple", ["#ffffff", "#3C2B7A"], N=256)
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(cm.values, aspect="auto", cmap=cmap)

        ax.set_xticks(np.arange(cm.shape[1]) + 0.5, minor=True)
        ax.set_yticks(np.arange(cm.shape[0]) + 0.5, minor=True)
        ax.grid(which="minor", color="#f0f0f0", linewidth=0.8)
        ax.tick_params(which="minor", bottom=False, left=False)

        ax.set_xticks(range(len(cm.columns)))
        ax.set_yticks(range(len(cm.index)))
        ax.set_xticklabels(cm.columns, rotation=45, ha="right")
        ax.set_yticklabels(cm.index)

        ax.set_xlabel("LLM (selected)")
        ax.set_ylabel("Human (top harm)")
        ax.set_title("Confusion Matrix — single LLM pick; human ties broken alphabetically", pad=8)

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Count", rotation=90)

        plt.tight_layout()
        plt.savefig(outdir / "confusion_heatmap.png", dpi=200, facecolor="white")
        plt.close(fig)

    # Per-block winners table
    winners = blocks[BLOCK_KEYS + ["human_top_harms", "llm_selected_harms",
                                   "top_hit", "exact_defined", "exact_acc",
                                   "n_harms", "expected_n"]]
    winners = winners.sort_values(["questionnaire_id", "stakeholder", "bias_type", "domain", "harm_family"])
    winners.to_csv(outdir / "winners_human_vs_llm.csv", index=False, encoding="utf-8")

    # Console summary
    print("\n=== SANITY CHECKS ===")
    print(f"Parsed rows: {len(df)}")
    print(f"Blocks: {len(blocks)}")
    print(f"Blocks with wrong harm count (should be 7 or 9): {len(bad)}")
    print(f"Top-hit agreement (any ties): {overall['top_hit_rate']:.3f}")
    print(f"Exact-defined fraction (unique human top): {overall['exact_defined_frac']:.3f}")
    ea = overall['exact_acc_rate']
    print(f"Exact 1-vs-1 accuracy (over defined blocks): {ea if not np.isnan(ea) else 'NA'}")
    mpr = overall['mean_point_biserial_r']
    print(f"Mean point-biserial r: {mpr if not np.isnan(mpr) else 'NA'}")
    print(f"Wrote outputs to: {outdir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Compare human vs LLM selections in questionnaire CSV.")
    ap.add_argument("--in", dest="in_path", required=True, help="Path to results_humanandllm.csv")
    ap.add_argument("--outdir", dest="outdir", default="data/human_vs_llm_outputs_v2", help="Output directory")
    args = ap.parse_args()
    main(Path(args.in_path), Path(args.outdir))
