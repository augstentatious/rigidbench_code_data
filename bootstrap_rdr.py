#!/usr/bin/env python3
"""Bootstrap confidence intervals and relation tables for RigidBench v3.1.

The default input is the paper-root ``results/`` directory, which is the
authoritative result set for the current manuscript. Pass ``--results-root`` to
analyze another checkout.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


MODEL_LABELS = {
    "gpt_55": "GPT-5.5",
    "kimi_k2p6": "Kimi K2-P6",
    "gemini_25_pro": "Gemini 2.5 Pro",
    "gemini_25_flash": "Gemini 2.5 Flash",
    "deepseek_v4": "DeepSeek V4-Pro",
    "claude_sonnet_46": "Claude Sonnet 4.6",
    "llama4_scout": "Llama 4 Scout",
    "gpt_oss_120b": "GPT-OSS 120B",
    "grok_43": "Grok 4.3",
}

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

RELATION_ORDER = [
    ("R1", "identity_neutral", "Identity-neutral"),
    ("R2", "virtue_name", "Virtue name"),
    ("R3", "etymological", "Etymological"),
    ("R4", "kinship", "Kinship"),
    ("R5", "alias", "Alias"),
    ("R6", "role_title", "Role/title"),
    ("R7", "semantic_field", "Semantic field"),
    ("R8", "historical_set", "Historical set"),
]


def result_files(root: Path) -> list[Path]:
    """Return one rigidbench_v3_results file per model directory."""
    # The paper checkout also contains archival run folders under
    # results/rigidbench_v3/. The current manuscript aggregate is exactly one
    # result file under each top-level model folder, so restrict discovery to
    # root/<model>/*rigidbench_v3_results.jsonl.
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
    aliases = {
        "SEMANTIC_SUB": "SEM_SUB",
        "SEMANTIC_SUBSTITUTION": "SEM_SUB",
        "PHONOLOGICAL_SUB": "PHO_SUB",
        "PHONOLOGICAL_SUBSTITUTION": "PHO_SUB",
    }
    return aliases.get(text, text)


def load_results(root: Path) -> list[dict]:
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


def percentile(values: Iterable[float], q: float) -> float:
    vals = sorted(v for v in values if not math.isnan(v))
    if not vals:
        return float("nan")
    pos = (len(vals) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return vals[lo]
    frac = pos - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


def fmt2(value: float) -> str:
    return f"{value:.2f}"


def fmt3(value: float) -> str:
    return f"{value:.3f}"


def clopper_pearson_all_successes(n: int, confidence: float = 0.95) -> float:
    """One-sided exact lower bound for n successes in n Bernoulli trials."""
    if n <= 0:
        return float("nan")
    alpha = 1.0 - confidence
    return alpha ** (1.0 / n)


def bootstrap_by_item(
    rows: list[dict],
    resamples: int,
    seed: int,
) -> dict[str, tuple[float, float]]:
    by_item: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_item[row["item_id"]].append(row)

    item_ids = sorted(by_item)
    rng = random.Random(seed)
    rdr_values: list[float] = []
    ssr_values: list[float] = []

    for _ in range(resamples):
        sem = 0
        pho = 0
        total = 0
        for item_id in (rng.choice(item_ids) for _ in item_ids):
            cluster = by_item[item_id]
            total += len(cluster)
            for row in cluster:
                if row["outcome"] == "SEM_SUB":
                    sem += 1
                elif row["outcome"] == "PHO_SUB":
                    pho += 1
        rdr_values.append(sem / (sem + pho) if sem + pho else float("nan"))
        ssr_values.append(sem / total if total else float("nan"))

    return {
        "rdr": (percentile(rdr_values, 0.025), percentile(rdr_values, 0.975)),
        "ssr": (percentile(ssr_values, 0.025), percentile(ssr_values, 0.975)),
    }


def write_rdr_ci_table(path: Path, stats: dict[str, float | tuple[float, float]]) -> None:
    rdr_ci = stats["rdr_bootstrap_ci"]
    ssr_ci = stats["ssr_bootstrap_ci"]
    lines = [
        "% Auto-generated by bootstrap_rdr.py",
        "\\begin{tabular}{@{}lcccc@{}}",
        "  \\toprule",
        "  \\textbf{Metric} & \\textbf{Count} & \\textbf{Estimate} & \\textbf{Exact CI} & \\textbf{Bootstrap CI} \\\\",
        "  \\midrule",
        (
            f"  Pooled RDR & {stats['sem_sub']}/{stats['sem_sub'] + stats['pho_sub']} "
            f"& {fmt2(stats['rdr'])} "
            f"& [{fmt2(stats['cp_lower'])}, 1.00] "
            f"& [{fmt2(rdr_ci[0])}, {fmt2(rdr_ci[1])}] \\\\"
        ),
        (
            f"  Pooled SSR & {stats['sem_sub']}/{stats['total']} "
            f"& {fmt2(stats['ssr'])} "
            f"& -- "
            f"& [{fmt2(ssr_ci[0])}, {fmt2(ssr_ci[1])}] \\\\"
        ),
        "  \\bottomrule",
        "\\end{tabular}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_per_model_table(path: Path, rows: list[dict]) -> None:
    by_model: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_model[row["model_slug"]].append(row)

    ordered = [m for m in MODEL_ORDER if m in by_model] + sorted(set(by_model) - set(MODEL_ORDER))
    lines = [
        "% Auto-generated by bootstrap_rdr.py",
        "\\begin{tabular}{@{}lrrrr@{}}",
        "  \\toprule",
        "  \\textbf{Model} & \\textbf{N} & \\textbf{SEM} & \\textbf{PHO} & \\textbf{RDR} \\\\",
        "  \\midrule",
    ]
    for slug in ordered:
        model_rows = by_model[slug]
        sem = sum(1 for row in model_rows if row["outcome"] == "SEM_SUB")
        pho = sum(1 for row in model_rows if row["outcome"] == "PHO_SUB")
        rdr = sem / (sem + pho) if sem + pho else float("nan")
        rdr_text = fmt2(rdr) if not math.isnan(rdr) else "--"
        label = MODEL_LABELS.get(slug, slug.replace("_", "\\_"))
        lines.append(f"  {label} & {len(model_rows)} & {sem} & {pho} & {rdr_text} \\\\")
    lines.extend(["  \\bottomrule", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_relation_table(path: Path, rows: list[dict]) -> None:
    by_relation: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_relation[row.get("primary_relation", "unknown")].append(row)

    lines = [
        "% Auto-generated by bootstrap_rdr.py",
        "\\begin{tabular}{@{}llrrrr@{}}",
        "  \\toprule",
        "  \\textbf{ID} & \\textbf{Relation type} & \\textbf{N} & \\textbf{SEM} & \\textbf{PHO} & \\textbf{SSR} \\\\",
        "  \\midrule",
    ]
    for code, relation, label in RELATION_ORDER:
        rel_rows = by_relation.get(relation, [])
        sem = sum(1 for row in rel_rows if row["outcome"] == "SEM_SUB")
        pho = sum(1 for row in rel_rows if row["outcome"] == "PHO_SUB")
        ssr = sem / len(rel_rows) if rel_rows else float("nan")
        ssr_text = fmt3(ssr) if not math.isnan(ssr) else "--"
        lines.append(f"  {code} & {label} & {len(rel_rows)} & {sem} & {pho} & {ssr_text} \\\\")
    lines.extend(["  \\bottomrule", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default="results", help="Directory containing model result folders")
    parser.add_argument("--out-dir", default="tables", help="Directory for generated LaTeX tables")
    parser.add_argument("--resamples", type=int, default=10_000, help="Bootstrap resamples")
    parser.add_argument("--seed", type=int, default=20260507, help="Bootstrap RNG seed")
    args = parser.parse_args()

    root = Path(args.results_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_results(root)
    counts = Counter(row["outcome"] for row in rows)
    sem = counts["SEM_SUB"]
    pho = counts["PHO_SUB"]
    rdr = sem / (sem + pho) if sem + pho else float("nan")
    ssr = sem / len(rows) if rows else float("nan")
    cp_lower = clopper_pearson_all_successes(sem + pho)
    bootstrap_ci = bootstrap_by_item(rows, args.resamples, args.seed)

    stats = {
        "total": len(rows),
        "sem_sub": sem,
        "pho_sub": pho,
        "rdr": rdr,
        "ssr": ssr,
        "cp_lower": cp_lower,
        "rdr_bootstrap_ci": bootstrap_ci["rdr"],
        "ssr_bootstrap_ci": bootstrap_ci["ssr"],
    }

    write_rdr_ci_table(out_dir / "rdr_ci.tex", stats)
    write_per_model_table(out_dir / "per_model_rdr.tex", rows)
    write_relation_table(out_dir / "per_relation_ssr.tex", rows)

    print(f"Loaded {len(rows)} rows from {root}")
    print(f"Outcome counts: {dict(sorted(counts.items()))}")
    print(
        "Pooled RDR = "
        f"{fmt2(rdr)} [95\\% CI: {fmt2(cp_lower)}, 1.00]"
    )
    print(
        "Item-level bootstrap RDR 95% CI: "
        f"[{fmt2(bootstrap_ci['rdr'][0])}, {fmt2(bootstrap_ci['rdr'][1])}]"
    )
    print(
        "Item-level bootstrap SSR 95% CI: "
        f"[{fmt2(bootstrap_ci['ssr'][0])}, {fmt2(bootstrap_ci['ssr'][1])}]"
    )
    print(f"Wrote {out_dir / 'rdr_ci.tex'}")
    print(f"Wrote {out_dir / 'per_model_rdr.tex'}")
    print(f"Wrote {out_dir / 'per_relation_ssr.tex'}")


if __name__ == "__main__":
    main()
