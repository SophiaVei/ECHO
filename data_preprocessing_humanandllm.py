# -*- coding: utf-8 -*-
"""
Parse results_humanandllm.csv -> ONE canonical long CSV where:
    votes = human SUMS + (LLM_selected ? 1 : 0)

Input layout (3 columns):
    label, SUMS, LLM
    - label holds either a header line or a harm label
    - SUMS is human votes (integer, can be blank)
    - LLM is '1' when the LLM selected that harm for this block, else blank

Default output (written to data/):
  - clean_results_human_plusllm.csv   (ONE canonical file for analysis)
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

# ------------------------ Normalization helpers ------------------------

def _ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def canon_lower(s: str) -> str:
    return (s or "").strip().lower()

def canon_lower_fix(s: str) -> str:
    """Lowercase + fix common typos + collapse whitespace for final keys."""
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

# ------------------------ Parser ------------------------

CSV_HEADER_RE = re.compile(
    r'^Questionnaire\s+(\d+)\s*\(([^,]+),\s*([^)]+)\)\s*/\s*(.+)$',
    re.IGNORECASE,
)

BLOCK_KEYS = ["questionnaire_id", "stakeholder", "is_group_level", "bias_type", "domain"]

def parse_csv_rows_human_plus_llm(path: Path) -> List[Dict]:
    """
    Read rows from results_humanandllm.csv (label,SUMS,LLM) and emit tidy rows with:
        votes = SUMS (human) + (1 if LLM == '1' else 0)
    """
    rows: List[Dict] = []
    current = None

    with path.open(newline="", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f)
        for r in reader:
            if not r:
                continue

            col0 = _ws((r[0] if len(r) > 0 else "").strip().strip('"'))
            col1 = _ws((r[1] if len(r) > 1 else "").strip().strip('"'))
            col2 = _ws((r[2] if len(r) > 2 else "").strip().strip('"'))

            # skip blank lines
            if not col0 and not col1 and not col2:
                continue

            # Header line starts a new block
            m = CSV_HEADER_RE.match(col0)
            if m:
                qid = int(m.group(1))
                stakeholder_raw = m.group(2)
                bias_type_raw = m.group(3)
                domain_raw = m.group(4)

                stakeholder_base, is_group = normalize_stakeholder(stakeholder_raw)
                current = {
                    "questionnaire_id": qid,
                    "stakeholder_raw": stakeholder_raw.strip(),
                    "stakeholder": stakeholder_base,
                    "is_group_level": is_group,
                    "bias_type": normalize_bias_type(bias_type_raw),
                    "domain": normalize_domain(domain_raw),
                }
                continue

            # Separator like ",0," -> ignore
            if current is None:
                continue

            # Harm rows: label + SUMS (human) + LLM (binary)
            # Accept numeric SUMS (including "0" or empty) and LLM of "1" (else 0)
            # Rule: votes = human_sums + llm_selected
            if col0:
                # valid harm if SUMS present OR LLM present
                has_human = (col1.lstrip("-").isdigit() if col1 != "" else False)
                has_llm = (col2 == "1")
                if has_human or has_llm:
                    harm = normalize_harm_label(col0)
                    human_votes = int(col1) if has_human else 0
                    llm_selected = 1 if has_llm else 0
                    votes = human_votes + llm_selected

                    harm_family = "Group" if current["is_group_level"] else "Individual"
                    rows.append({
                        **current,
                        "harm_family": harm_family,
                        "harm": harm,
                        "votes": votes,
                        # keep these for transparency/debug if you ever need them downstream
                        "human_votes": human_votes,
                        "llm_selected": llm_selected,
                    })

    return rows

# ------------------------ Data shaping ------------------------

def to_dataframe(rows: List[Dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)

    # Compute totals per (combined) votes and join
    totals = (
        df.groupby(BLOCK_KEYS, as_index=False)["votes"]
          .sum()
          .rename(columns={"votes": "total_votes"})
    )
    df = df.merge(totals, on=BLOCK_KEYS, how="left")

    # vote_share (fraction of total per block)
    df["vote_share"] = df["votes"] / df["total_votes"]

    # Order/format columns to match clean_results_human.csv (plus we keep stakeholder_raw)
    col_order = [
        "questionnaire_id",
        "stakeholder",
        "is_group_level",
        "bias_type",
        "domain",
        "harm_family",
        "harm",
        "votes",
        "total_votes",
        "vote_share",
        "stakeholder_raw",
    ]
    df = df[col_order + ["human_votes", "llm_selected"]]  # keep extras at the end for reference

    # Canonicalize text columns
    df["stakeholder_raw"] = df["stakeholder_raw"].apply(canon_lower_fix)
    for col in ["stakeholder", "bias_type", "domain", "harm_family", "harm"]:
        df[col] = df[col].apply(canon_lower)

    # Sort for stability
    df = df.sort_values(
        ["questionnaire_id", "stakeholder", "bias_type", "domain", "harm_family", "harm"]
    ).reset_index(drop=True)

    return df

# ------------------------ Orchestrator ------------------------

def run(in_path: Path | None = None,
        out_long: Path | None = None) -> None:
    root = Path(__file__).resolve().parent
    data_dir = root / "data"

    in_path = in_path or (data_dir / "results_humanandllm.csv")
    out_long = out_long or (data_dir / "clean_results_human_plusllm.csv")

    rows = parse_csv_rows_human_plus_llm(Path(in_path))
    df_long = to_dataframe(rows)
    # Write only the core columns to exactly mirror your original schema (drop debug extras)
    core_cols = [
        "questionnaire_id","stakeholder","is_group_level","bias_type","domain",
        "harm_family","harm","votes","total_votes","vote_share","stakeholder_raw"
    ]
    df_long[core_cols].to_csv(out_long, index=False, encoding="utf-8")
    print(f"Wrote: {out_long}")

# ------------------------ CLI ------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse results_humanandllm.csv into ONE canonical long file with votes = human + LLM."
    )
    parser.add_argument("--in", dest="in_path", default=None,
                        help="Input CSV path (default: data/results_humanandllm.csv)")
    parser.add_argument("--out", dest="out_long", default=None,
                        help="Output long CSV path (default: data/clean_results_human_plusllm.csv)")
    return parser.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    in_path = Path(args.in_path) if args.in_path else None
    out_long = Path(args.out_long) if args.out_long else None
    run(in_path=in_path, out_long=out_long)
