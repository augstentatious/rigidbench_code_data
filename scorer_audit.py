#!/usr/bin/env python3
"""Audit deterministic scorer outputs for RigidBench v3.1."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def result_files(root: Path) -> list[Path]:
    # Restrict to the manuscript's top-level model result folders, excluding
    # archival runs under results/rigidbench_v3/.
    files = sorted({p for p in root.glob("*/*rigidbench_v3_results.jsonl") if p.is_file()})
    if not files:
        raise FileNotFoundError(f"No rigidbench_v3_results.jsonl files under {root}")
    return files


def model_slug(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).parts[0]
    except ValueError:
        return path.parent.name


def normalize_outcome(value: str | None) -> str:
    if not value:
        return "UNKNOWN"
    text = value.strip().upper()
    return {
        "SEMANTIC_SUB": "SEM_SUB",
        "PHONOLOGICAL_SUB": "PHO_SUB",
    }.get(text, text)


def load_rows(root: Path) -> list[dict]:
    rows: list[dict] = []
    for path in result_files(root):
        slug = model_slug(path, root)
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                row["model_slug"] = slug
                row["item_id"] = row.get("item_id") or row.get("triple_id")
                row["outcome"] = normalize_outcome(row.get("outcome") or row.get("error_type"))
                row["completion"] = row.get("completion") or row.get("raw_completion") or ""
                rows.append(row)
    return rows


def contains_expected_name(row: dict) -> bool:
    name = row.get("proper_noun") or ""
    if not name:
        return False
    pattern = rf"(?<![A-Za-z]){re.escape(name)}(?![A-Za-z])"
    return bool(re.search(pattern, row.get("completion", ""), flags=re.IGNORECASE))


def print_case(row: dict) -> None:
    print(f"item_id: {row.get('item_id')}")
    print(f"model: {row.get('model_slug')}")
    print(f"outcome: {row.get('outcome')}")
    print(f"expected_name: {row.get('proper_noun', '')}")
    print(f"matched_word: {row.get('matched_word', '')}")
    print("completion:")
    print(row.get("completion", ""))
    print("-" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default="results", help="Directory containing model result folders")
    parser.add_argument(
        "--max-discrepancies",
        type=int,
        default=0,
        help="Maximum scorer/heuristic discrepancies to print; 0 prints all",
    )
    args = parser.parse_args()

    rows = load_rows(Path(args.results_root))

    print("SEM_SUB cases")
    print("=" * 80)
    for row in rows:
        if row["outcome"] == "SEM_SUB":
            print_case(row)

    print("NOISE cases")
    print("=" * 80)
    for row in rows:
        if row["outcome"] == "NOISE":
            print_case(row)

    agreements = 0
    discrepancies: list[dict] = []
    for row in rows:
        scorer_preserved = row["outcome"] in {"PRES", "ALIAS_OK"}
        heuristic_preserved = contains_expected_name(row)
        if scorer_preserved == heuristic_preserved:
            agreements += 1
        else:
            discrepancies.append(row)

    agreement_rate = agreements / len(rows) if rows else 0.0
    print("Scorer agreement summary")
    print("=" * 80)
    print(f"rows: {len(rows)}")
    print(f"agreement_rate: {agreement_rate:.4f}")
    print(f"discrepancies: {len(discrepancies)}")

    if agreement_rate < 0.99:
        print("FLAG: agreement below 99%; inspect discrepancies below.")
        limit = len(discrepancies) if args.max_discrepancies == 0 else args.max_discrepancies
        for row in discrepancies[:limit]:
            print_case(row)
    else:
        print("PASS: agreement is at least 99%.")


if __name__ == "__main__":
    main()
