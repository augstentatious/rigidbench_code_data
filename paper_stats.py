"""Recompute the paper-level RigidBench v3.1 statistics.

Run from an extracted release directory:

    python paper_stats.py

The script also works from the FINAL paper directory if
``rigidbench_code_and_data.zip`` is present.
"""

from __future__ import annotations

import json
import math
import zipfile
from collections import Counter
from pathlib import Path


MODEL_ORDER = [
    "gpt_55",
    "kimi_k2p6",
    "gemini_25_pro",
    "gemini_25_flash",
    "deepseek_v4",
    "claude_sonnet_46",
    "llama4_scout",
    "gpt_oss_120b",
    "grok_43",
]


def load_rows() -> list[dict]:
    root = Path(".")
    result_files = [root / "results" / name / "rigidbench_v3_results.jsonl" for name in MODEL_ORDER]
    if all(path.exists() for path in result_files):
        rows: list[dict] = []
        for path in result_files:
            rows.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line)
        return rows

    zip_path = root / "rigidbench_code_and_data.zip"
    if not zip_path.exists():
        raise SystemExit("Could not find results/ or rigidbench_code_and_data.zip")

    rows = []
    with zipfile.ZipFile(zip_path) as zf:
        for name in MODEL_ORDER:
            member = f"results\\{name}\\rigidbench_v3_results.jsonl"
            with zf.open(member) as fh:
                rows.extend(json.loads(line.decode("utf-8")) for line in fh if line.strip())
    return rows


def chi_square_pressure(rows: list[dict]) -> tuple[float, float, float, list[list[int]]]:
    table: list[list[int]] = []
    for level in ["low", "mid", "high"]:
        items = [r for r in rows if r.get("pressure_level") == level]
        sem = sum(r.get("error_type") == "SEM_SUB" for r in items)
        table.append([len(items) - sem, sem])

    row_totals = [sum(row) for row in table]
    col_totals = [sum(table[i][j] for i in range(len(table))) for j in range(2)]
    n = sum(row_totals)
    chi2 = 0.0
    for i, row in enumerate(table):
        for j, observed in enumerate(row):
            expected = row_totals[i] * col_totals[j] / n
            chi2 += (observed - expected) ** 2 / expected

    # Survival function for chi-square with df=2.
    p_value = math.exp(-chi2 / 2.0)
    cramers_v = math.sqrt(chi2 / n)
    return chi2, p_value, cramers_v, table


def logistic_odds(rows: list[dict]) -> list[float]:
    try:
        import numpy as np
        from sklearn.linear_model import LogisticRegression
    except ImportError as exc:
        raise SystemExit("Install scikit-learn to recompute logistic odds ratios") from exc

    pressure = {"low": 0, "mid": 1, "high": 2}
    x_rows = []
    y = []
    for row in rows:
        x_rows.append(
            [
                pressure.get(row.get("pressure_level"), 1),
                row.get("semantic_sim_name_to_lure", 0.0) or 0.0,
                row.get("phon_distance_name_to_neighbor", 3) or 3,
            ]
        )
        y.append(1 if row.get("error_type") == "SEM_SUB" else 0)

    model = LogisticRegression(max_iter=1000)
    model.fit(np.array(x_rows), np.array(y))
    return [math.exp(coef) for coef in model.coef_[0]]


def main() -> None:
    rows = load_rows()
    counts = Counter(row.get("error_type") for row in rows)
    print(f"Total rows: {len(rows)}")
    print(f"Outcome counts: {dict(counts)}")

    for level in ["low", "mid", "high"]:
        items = [row for row in rows if row.get("pressure_level") == level]
        sem = sum(row.get("error_type") == "SEM_SUB" for row in items)
        print(f"{level}: {sem}/{len(items)} = {sem / len(items):.4f}")

    chi2, p_value, cramers_v, table = chi_square_pressure(rows)
    print(f"Pressure table [[no_sub, sem_sub], ...]: {table}")
    print(f"chi2={chi2:.2f}, df=2, p={p_value:.2e}, Cramer's V={cramers_v:.2f}")

    odds = logistic_odds(rows)
    print(
        "Regularized logistic odds ratios: "
        f"pressure={odds[0]:.2f}, semantic_similarity={odds[1]:.2f}, "
        f"phonological_distance={odds[2]:.2f}"
    )


if __name__ == "__main__":
    main()
