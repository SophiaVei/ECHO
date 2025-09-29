# -*- coding: utf-8 -*-
"""
Parse questionnaire CSV -> one canonical long file (project-relative paths).

Expected layout:
    <project>/
      data/
        results_human.csv
      parse_questionnaires.py

Default output (written to data/):
  - clean_results_human.csv   (ONE canonical file for analysis)

Optional (only if --write-pivots is used):
  - pivot_counts_by_harm.csv
  - pivot_totals_by_questionnaire.csv
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

# ------------------------ Normalization helpers ------------------------

def _ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def canon_lower(s: str) -> str:
    return (s or "").strip().lower()

def canon_lower_fix(s: str) -> str:
    """Lowercase + fix common typos + collapse whitespace for final keys."""
    s = (s or "").strip()
    # fix a known typo before lowercasing
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
    # strip trailing punctuation / parens like "Hiring)" or "Diagnosis:"
    s = _ws(domain)
    s = re.sub(r"[:)\.]+$", "", s)
    fixes = {"Hiring": "Hiring", "Diagnosis": "Diagnosis", "Company": "Company"}
    return fixes.get(s, s)

# ------------------------ Parsers ------------------------

CSV_HEADER_RE = re.compile(
    r'^Questionnaire\s+(\d+)\s*\(([^,]+),\s*([^)]+)\)\s*/\s*(.+)$',
    re.IGNORECASE,
)

def parse_csv_rows(path: Path) -> List[Dict]:
    """
    Parse the CSV (two columns: label, value) and return tidy rows with 'votes'.
    """
    rows: List[Dict] = []
    current = None
    with path.open(newline="", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f)
        for r in reader:
            if not r:
                continue
            col0 = _ws((r[0] or "").strip().strip('"'))
            col1 = _ws((r[1] if len(r) > 1 else "").strip().strip('"'))

            if not col0 and not col1:
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

            # Harm rows (label,value) inside a current header block
            if current is not None and col0 and col1 and col1.lstrip("-").isdigit():
                harm = normalize_harm_label(col0)
                votes = int(col1)
                # derive harm_family from header
                harm_family = "Group" if current["is_group_level"] else "Individual"
                rows.append({
                    **current,
                    "harm_family": harm_family,
                    "harm": harm,
                    "votes": votes,
                })
                continue

            # Other lines (e.g., separators like ",0") are ignored
    return rows

# ------------------------ Data shaping ------------------------

BLOCK_KEYS = ["questionnaire_id", "stakeholder", "is_group_level", "bias_type", "domain"]

def to_dataframe(rows: List[Dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)

    # Compute totals per block and join them
    totals = (
        df.groupby(BLOCK_KEYS, as_index=False)["votes"]
          .sum()
          .rename(columns={"votes": "total_votes"})
    )
    df = df.merge(totals, on=BLOCK_KEYS, how="left")

    # vote share (fraction of total per block)
    df["vote_share"] = df["votes"] / df["total_votes"]

    # Order columns
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
    df = df[col_order].sort_values(
        ["questionnaire_id", "stakeholder", "bias_type", "domain", "harm_family", "harm"]
    ).reset_index(drop=True)

    # Canonicalize ALL text columns to lowercase; also fix stakeholder_raw typos before lowering
    df["stakeholder_raw"] = df["stakeholder_raw"].apply(canon_lower_fix)
    for col in ["stakeholder", "bias_type", "domain", "harm_family", "harm"]:
        df[col] = df[col].apply(canon_lower)

    return df

def make_pivots(df: pd.DataFrame):
    p1 = (
        df.pivot_table(
            index=BLOCK_KEYS,
            columns=["harm_family", "harm"],
            values="votes",
            aggfunc="sum",
            fill_value=0,
        )
        .sort_index()
        .reset_index()
    )
    p2 = (
        df[BLOCK_KEYS + ["total_votes"]]
          .drop_duplicates()
          .sort_values(BLOCK_KEYS)
          .reset_index(drop=True)
    )
    return p1, p2

# ------------------------ Orchestrator (project-relative paths) ------------------------

def run(in_path: Optional[Path] = None,
        out_long: Optional[Path] = None,
        write_pivots: bool = False) -> None:
    root = Path(__file__).resolve().parent
    data_dir = root / "data"

    in_path = in_path or (data_dir / "results_human.csv")
    out_long = out_long or (data_dir / "clean_results_human.csv")
    out_pivot1 = data_dir / "pivot_counts_by_harm.csv"
    out_pivot2 = data_dir / "pivot_totals_by_questionnaire.csv"

    rows = parse_csv_rows(Path(in_path))
    df_long = to_dataframe(rows)
    df_long.to_csv(out_long, index=False, encoding="utf-8")
    print(f"Wrote: {out_long}")

    if write_pivots:
        p1, p2 = make_pivots(df_long)
        p1.to_csv(out_pivot1, index=False, encoding="utf-8")
        p2.to_csv(out_pivot2, index=False, encoding="utf-8")
        print(f"Wrote: {out_pivot1}")
        print(f"Wrote: {out_pivot2}")

# ------------------------ CLI ------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse questionnaire CSV into ONE canonical long file (+ optional pivots).")
    parser.add_argument("--in", dest="in_path", default=None, help="Input CSV path (default: data/results_human.csv)")
    parser.add_argument("--out", dest="out_long", default=None, help="Output long CSV path (default: data/clean_results_human.csv)")
    parser.add_argument("--write-pivots", action="store_true", help="Also write pivot CSVs")
    return parser.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    in_path = Path(args.in_path) if args.in_path else None
    out_long = Path(args.out_long) if args.out_long else None
    run(in_path=in_path, out_long=out_long, write_pivots=args.write_pivots)





"""Vote share: It’s the proportion of votes a specific harm received within its block.

A block = one combination of
questionnaire_id × stakeholder × bias_type × domain × is_group_level.

Example:
Take this row from the final file:

questionnaire_id = 1
stakeholder = Applicant
is_group_level = FALSE
bias_type = Algorithmic
domain = Hiring
harm = Opportunity Loss
votes = 7
total_votes = 33
vote_share = 0.212121...


That means:
In Questionnaire 1, the Applicant (individual) under Algorithmic/Hiring,
33 total harm votes were given across all individual-harm categories.
Of those, 7 were for Opportunity Loss.
So share = 7 ÷ 33 ≈ 0.21 (≈ 21%).

Why keep it?
Makes comparisons easier: we can directly compare proportions across stakeholders, bias types, or domains, even if the total number of votes differs.
Useful for stats/plots: barplots of shares, statistical tests of proportions, etc.
Avoids recalculating later - saves us from dividing every time in analysis scripts
"""