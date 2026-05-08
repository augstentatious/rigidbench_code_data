#!/usr/bin/env python3
"""Analyze RigidBench v2 and v3.1 results.

Reads the JSONL output from run_all.py and produces:
1. Aggregate error type distribution (the key result)
2. Per-pressure-level breakdown
3. Per-model comparison
4. Chi-squared test: semantic_sub vs phonological_sub
5. Logistic regression: P(semantic_sub) ~ pressure_level + phon_distance + semantic_sim
6. Exportable tables (CSV, LaTeX)

v3.1 analysis (--v3 flag):
7. IPR / RDR aggregate metrics per model
8. Per-family, per-pressure-level, per-relation breakdowns
9. Dose-response monotonicity (Spearman)
10. Family D clarify/abstain analysis
11. LaTeX macro export

Usage:
    python analyze_results.py --input results/rigidbench_v2_results.jsonl
    python analyze_results.py --v3 --input results/
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import re
import sys
from collections import Counter, defaultdict
from typing import Any

# ---------------------------------------------------------------------------
# Load results
# ---------------------------------------------------------------------------


def load_results(path: str) -> list[dict[str, Any]]:
    results = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------


def aggregate_summary(results: list[dict]) -> dict[str, Any]:
    """Compute the headline numbers."""
    total = len(results)
    counts = Counter(r["error_type"] for r in results)

    preserved = counts.get("preserved", 0)
    semantic = counts.get("semantic_sub", 0)
    phonological = counts.get("phonological_sub", 0)
    other = counts.get("other", 0)

    # The key ratio: among failures, what fraction are semantic vs phonological?
    failures = semantic + phonological + other
    sem_ratio = semantic / failures if failures > 0 else 0
    phon_ratio = phonological / failures if failures > 0 else 0

    return {
        "total_trials": total,
        "preserved": preserved,
        "preserved_pct": preserved / total * 100 if total else 0,
        "semantic_sub": semantic,
        "semantic_sub_pct": semantic / total * 100 if total else 0,
        "phonological_sub": phonological,
        "phonological_sub_pct": phonological / total * 100 if total else 0,
        "other": other,
        "other_pct": other / total * 100 if total else 0,
        "total_failures": failures,
        "semantic_share_of_failures": sem_ratio * 100,
        "phonological_share_of_failures": phon_ratio * 100,
        "asymmetry_ratio": sem_ratio / phon_ratio if phon_ratio > 0 else float("inf"),
    }


def per_pressure_breakdown(results: list[dict]) -> dict[str, dict]:
    """Break down by pressure level."""
    by_level = defaultdict(list)
    for r in results:
        by_level[r["pressure_level"]].append(r)

    return {level: aggregate_summary(rs) for level, rs in sorted(by_level.items())}


def per_model_breakdown(results: list[dict]) -> dict[str, dict]:
    """Break down by model."""
    by_model = defaultdict(list)
    for r in results:
        by_model[r["model"]].append(r)

    return {model: aggregate_summary(rs) for model, rs in sorted(by_model.items())}


def per_triple_breakdown(results: list[dict]) -> dict[str, dict]:
    """Break down by triple ID."""
    by_triple = defaultdict(list)
    for r in results:
        by_triple[r["triple_id"]].append(r)

    out = {}
    for tid, rs in sorted(by_triple.items()):
        name = rs[0]["proper_noun"]
        lure = rs[0]["semantic_lure"]
        neighbor = rs[0]["phonological_neighbor"]
        summary = aggregate_summary(rs)
        summary["proper_noun"] = name
        summary["semantic_lure"] = lure
        summary["phonological_neighbor"] = neighbor
        out[tid] = summary
    return out


def chi_squared_test(results: list[dict]) -> dict[str, float] | None:
    """Chi-squared test: are semantic subs significantly > phonological subs?"""
    try:
        from scipy.stats import chisquare, binomtest
    except ImportError:
        print("  [scipy not installed — skipping chi-squared test]")
        return None

    semantic = sum(1 for r in results if r["error_type"] == "semantic_sub")
    phonological = sum(1 for r in results if r["error_type"] == "phonological_sub")

    if semantic + phonological == 0:
        return {"note": "no substitutions observed"}

    # Null hypothesis: semantic and phonological substitutions are equally likely
    total_subs = semantic + phonological
    chi2, p_chi = chisquare([semantic, phonological], [total_subs / 2, total_subs / 2])

    # Also do a binomial test (more appropriate for 2 categories)
    binom = binomtest(semantic, total_subs, 0.5, alternative="greater")

    return {
        "semantic_subs": semantic,
        "phonological_subs": phonological,
        "chi2_statistic": round(chi2, 4),
        "chi2_p_value": round(p_chi, 6),
        "binomial_p_value": round(binom.pvalue, 6),
        "significant_at_05": p_chi < 0.05,
        "significant_at_01": p_chi < 0.01,
    }


def pressure_dose_response(results: list[dict]) -> dict[str, Any]:
    """Test whether substitution rate increases with pressure."""
    pressure_order = {"low": 0, "mid": 1, "high": 2}
    by_level = per_pressure_breakdown(results)

    levels = []
    sub_rates = []
    for level in ["low", "mid", "high"]:
        if level in by_level:
            levels.append(level)
            sub_rates.append(by_level[level]["semantic_sub_pct"])

    monotonic = all(sub_rates[i] <= sub_rates[i + 1] for i in range(len(sub_rates) - 1))

    return {
        "levels": levels,
        "semantic_sub_rates": sub_rates,
        "monotonically_increasing": monotonic,
    }


def etymological_effect(results: list[dict]) -> dict[str, Any]:
    """Compare substitution rates for etymologically linked vs unlinked pairs."""
    linked = [r for r in results if r.get("etymological_link", False)]
    unlinked = [r for r in results if not r.get("etymological_link", False)]

    return {
        "etymologically_linked": aggregate_summary(linked) if linked else None,
        "not_linked": aggregate_summary(unlinked) if unlinked else None,
    }


def baseline_frequency_control(results: list[dict]) -> dict[str, Any]:
    """Compare pressured substitution rates against baseline (no-pressure)
    rates to control for unigram frequency confounds.

    If the model substitutes 'Grace' for 'Karis' 80% of the time under high
    pressure BUT also emits 'Grace' 60% of the time in the neutral baseline,
    then only 20pp of the effect is attributable to semantic pressure.  This
    analysis computes that delta.
    """
    baseline = [r for r in results if r.get("pressure_level") == "baseline"]
    pressured = [r for r in results if r.get("pressure_level") != "baseline"]

    if not baseline:
        return {"note": "no baseline data — run with --include-baseline"}

    baseline_summary = aggregate_summary(baseline)
    pressured_summary = aggregate_summary(pressured)

    # Per-triple baseline vs pressured comparison
    from collections import defaultdict

    baseline_by_triple = defaultdict(list)
    pressured_by_triple = defaultdict(list)
    for r in baseline:
        baseline_by_triple[r["triple_id"]].append(r)
    for r in pressured:
        pressured_by_triple[r["triple_id"]].append(r)

    per_triple = {}
    for tid in sorted(
        set(list(baseline_by_triple.keys()) + list(pressured_by_triple.keys()))
    ):
        b = (
            aggregate_summary(baseline_by_triple[tid])
            if baseline_by_triple[tid]
            else None
        )
        p = (
            aggregate_summary(pressured_by_triple[tid])
            if pressured_by_triple[tid]
            else None
        )
        delta_sem = None
        if b and p:
            delta_sem = p["semantic_sub_pct"] - b["semantic_sub_pct"]
        per_triple[tid] = {
            "baseline_sem_pct": b["semantic_sub_pct"] if b else None,
            "pressured_sem_pct": p["semantic_sub_pct"] if p else None,
            "delta_sem_pp": round(delta_sem, 1) if delta_sem is not None else None,
        }

    return {
        "baseline_aggregate": baseline_summary,
        "pressured_aggregate": pressured_summary,
        "delta_semantic_sub_pp": round(
            pressured_summary["semantic_sub_pct"]
            - baseline_summary["semantic_sub_pct"],
            1,
        ),
        "per_triple": per_triple,
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def print_report(results: list[dict]) -> None:
    """Print the full analysis report."""

    print("\n" + "=" * 72)
    print("  RIGIDBENCH v2 — RESULTS ANALYSIS")
    print("  Phonological-Semantic Asymmetry in Name Substitution")
    print("=" * 72)

    # --- Aggregate ---
    agg = aggregate_summary(results)
    print(f"\n  Total trials: {agg['total_trials']}")
    print(f"\n  ┌─────────────────────────────────────────────────────┐")
    print(f"  │  OVERALL ERROR TYPE DISTRIBUTION                    │")
    print(f"  ├─────────────────────────────────────────────────────┤")
    print(
        f"  │  Preserved identity:    {agg['preserved']:4d}  ({agg['preserved_pct']:5.1f}%)          │"
    )
    print(
        f"  │  Semantic substitution: {agg['semantic_sub']:4d}  ({agg['semantic_sub_pct']:5.1f}%)  ★ KEY   │"
    )
    print(
        f"  │  Phonological sub:      {agg['phonological_sub']:4d}  ({agg['phonological_sub_pct']:5.1f}%)          │"
    )
    print(
        f"  │  Other:                 {agg['other']:4d}  ({agg['other_pct']:5.1f}%)          │"
    )
    print(f"  ├─────────────────────────────────────────────────────┤")
    print(f"  │  Among failures:                                     │")
    print(
        f"  │    Semantic share:  {agg['semantic_share_of_failures']:5.1f}%                       │"
    )
    print(
        f"  │    Phonological:    {agg['phonological_share_of_failures']:5.1f}%                       │"
    )
    print(
        f"  │    Asymmetry ratio: {agg['asymmetry_ratio']:5.1f}× (sem/phon)            │"
    )
    print(f"  └─────────────────────────────────────────────────────┘")

    # --- Per model ---
    by_model = per_model_breakdown(results)
    if len(by_model) > 1:
        print(f"\n  {'─' * 72}")
        print(f"  PER-MODEL BREAKDOWN")
        print(f"  {'─' * 72}")
        for model, summary in by_model.items():
            print(f"\n  {model}:")
            print(
                f"    Preserved: {summary['preserved_pct']:5.1f}%  |  "
                f"Semantic: {summary['semantic_sub_pct']:5.1f}%  |  "
                f"Phonological: {summary['phonological_sub_pct']:5.1f}%  |  "
                f"Other: {summary['other_pct']:5.1f}%"
            )
            if summary["total_failures"] > 0:
                print(
                    f"    Failure asymmetry: {summary['asymmetry_ratio']:.1f}× semantic/phonological"
                )

    # --- Per pressure level ---
    print(f"\n  {'─' * 72}")
    print(f"  PRESSURE DOSE-RESPONSE")
    print(f"  {'─' * 72}")
    dose = pressure_dose_response(results)
    for level, rate in zip(dose["levels"], dose["semantic_sub_rates"]):
        bar = "█" * int(rate)
        print(f"    {level:4s}: semantic_sub_rate = {rate:5.1f}% {bar}")
    print(
        f"    Monotonically increasing: {'YES ✓' if dose['monotonically_increasing'] else 'NO ✗'}"
    )

    # --- Chi-squared ---
    print(f"\n  {'─' * 72}")
    print(f"  STATISTICAL TESTS")
    print(f"  {'─' * 72}")
    chi = chi_squared_test(results)
    if chi and "note" not in chi:
        print(f"    H0: semantic and phonological substitutions equally likely")
        print(f"    Semantic subs:     {chi['semantic_subs']}")
        print(f"    Phonological subs: {chi['phonological_subs']}")
        print(f"    χ² = {chi['chi2_statistic']},  p = {chi['chi2_p_value']}")
        print(
            f"    Binomial p = {chi['binomial_p_value']} (one-sided: semantic > phonological)"
        )
        sig = (
            "YES ★"
            if chi["significant_at_01"]
            else ("yes" if chi["significant_at_05"] else "no")
        )
        print(f"    Significant: {sig}")
    elif chi:
        print(f"    {chi['note']}")

    # --- Etymological effect ---
    print(f"\n  {'─' * 72}")
    print(f"  ETYMOLOGICAL LINK EFFECT")
    print(f"  {'─' * 72}")
    etym = etymological_effect(results)
    for label, summary in etym.items():
        if summary:
            print(
                f"    {label}: semantic_sub={summary['semantic_sub_pct']:.1f}%, "
                f"preserved={summary['preserved_pct']:.1f}%"
            )

    # --- Baseline frequency control ---
    bfc = baseline_frequency_control(results)
    if "note" not in bfc:
        print(f"\n  {'─' * 72}")
        print(f"  BASELINE FREQUENCY CONTROL")
        print(f"  {'─' * 72}")
        print(
            f"    Baseline sem_sub rate: {bfc['baseline_aggregate']['semantic_sub_pct']:5.1f}%"
        )
        print(
            f"    Pressured sem_sub rate: {bfc['pressured_aggregate']['semantic_sub_pct']:5.1f}%"
        )
        print(f"    Delta (pressure effect): {bfc['delta_semantic_sub_pp']:+.1f} pp")
        print(f"\n    Per-triple deltas:")
        for tid, info in bfc["per_triple"].items():
            if info["delta_sem_pp"] is not None:
                arrow = (
                    "↑"
                    if info["delta_sem_pp"] > 0
                    else ("↓" if info["delta_sem_pp"] < 0 else "=")
                )
                print(
                    f"      {tid}: baseline={info['baseline_sem_pct']:5.1f}%  "
                    f"pressured={info['pressured_sem_pct']:5.1f}%  "
                    f"Δ={info['delta_sem_pp']:+.1f}pp {arrow}"
                )

    # --- Per triple ---
    print(f"\n  {'─' * 72}")
    print(f"  PER-TRIPLE BREAKDOWN")
    print(f"  {'─' * 72}")
    by_triple = per_triple_breakdown(results)
    print(
        f"  {'ID':<8} {'Name':<12} {'Lure':<12} {'PhonNbr':<12} {'Pres%':>6} {'Sem%':>6} {'Pho%':>6}"
    )
    for tid, summary in by_triple.items():
        print(
            f"  {tid:<8} {summary['proper_noun']:<12} {summary['semantic_lure']:<12} "
            f"{summary['phonological_neighbor']:<12} {summary['preserved_pct']:>5.0f}% "
            f"{summary['semantic_sub_pct']:>5.0f}% {summary['phonological_sub_pct']:>5.0f}%"
        )

    # --- Failed items (most interesting) ---
    print(f"\n  {'─' * 72}")
    print(f"  INDIVIDUAL FAILURES (all non-preserved completions)")
    print(f"  {'─' * 72}")
    failures = [r for r in results if r["error_type"] != "preserved"]
    if failures:
        for r in failures:
            # Support both old "first_word" and new "matched_word" field names
            matched = r.get("matched_word", r.get("first_word", "???"))
            print(
                f"    {r['triple_id']} @ {r['pressure_level']:4s} | "
                f"{r['proper_noun']:12s} → {matched:12s} "
                f"[{r['error_type']:15s}] "
                f"model={r['model']}"
            )
            print(f"      raw: {r['raw_completion'][:80]}")
    else:
        print("    No failures observed (all identities preserved).")

    print(f"\n{'=' * 72}")
    print(f"  END OF ANALYSIS")
    print(f"{'=' * 72}\n")


def export_csv(results: list[dict], path: pathlib.Path) -> None:
    """Export results as CSV."""
    import csv

    fieldnames = [
        "triple_id",
        "model",
        "pressure_level",
        "proper_noun",
        "semantic_lure",
        "phonological_neighbor",
        "matched_word",
        "error_type",
        "raw_completion",
        "phon_distance_name_to_lure",
        "phon_distance_name_to_neighbor",
        "semantic_sim_name_to_lure",
        "semantic_sim_name_to_neighbor",
        "etymological_link",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    print(f"  CSV exported to {path}")


def export_summary_json(results: list[dict], path: pathlib.Path) -> None:
    """Export structured summary as JSON."""
    summary = {
        "aggregate": aggregate_summary(results),
        "per_pressure": per_pressure_breakdown(results),
        "per_model": per_model_breakdown(results),
        "per_triple": per_triple_breakdown(results),
        "chi_squared": chi_squared_test(results),
        "dose_response": pressure_dose_response(results),
        "etymological": etymological_effect(results),
        "baseline_frequency_control": baseline_frequency_control(results),
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"  Summary JSON exported to {path}")


# ---------------------------------------------------------------------------
# v3.1 analysis functions
# ---------------------------------------------------------------------------


def load_v3_results(path: str) -> list[dict[str, Any]]:
    """Load v3.1 JSONL results.

    Each line is a JSON object with fields: item_id (or triple_id), family,
    model, outcome (or error_type), score, primary_relation,
    pressure_operator, pressure_level, prompt_turn_count, completion
    (or raw_completion), timestamp.
    """
    results: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            # Normalise field names — run_all.py writes "triple_id" and
            # "error_type" but the spec uses "item_id" and "outcome".
            if "item_id" not in row and "triple_id" in row:
                row["item_id"] = row["triple_id"]
            if "outcome" not in row and "error_type" in row:
                row["outcome"] = row["error_type"]
            if "completion" not in row and "raw_completion" in row:
                row["completion"] = row["raw_completion"]
            results.append(row)
    return results


def v3_aggregate_metrics(
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute per-model aggregate metrics for v3.1 results.

    Returns a dict keyed by model name with:
      - ipr: identity preservation rate (mean score)
      - rdr: referential descriptivism ratio
      - per_family: {family: mean_score}
      - per_pressure: {pressure_level: mean_score}
      - per_relation: {primary_relation: mean_score}
    """
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in results:
        by_model[r["model"]].append(r)

    out: dict[str, Any] = {}
    for model, rows in sorted(by_model.items()):
        # IPR = mean score
        scores = [r.get("score", 0.0) for r in rows]
        ipr = sum(scores) / len(scores) if scores else 0.0

        # RDR = SEM_SUB / (SEM_SUB + PHO_SUB)
        # Computed only over errors, not preservations (matches paper §3.5)
        sem = sum(1 for r in rows if r["outcome"] == "SEM_SUB")
        pho = sum(1 for r in rows if r["outcome"] == "PHO_SUB")
        pres = sum(1 for r in rows if r["outcome"] == "PRES")
        alias = sum(1 for r in rows if r["outcome"] == "ALIAS_OK")
        denom = sem + pho
        rdr = sem / denom if denom > 0 else float("nan")

        # Per-family breakdown
        fam_scores: dict[str, list[float]] = defaultdict(list)
        for r in rows:
            fam_scores[r.get("family", "unknown")].append(r.get("score", 0.0))
        per_family = {f: sum(s) / len(s) for f, s in sorted(fam_scores.items())}

        # Per-pressure-level breakdown
        pres_scores: dict[str, list[float]] = defaultdict(list)
        for r in rows:
            pres_scores[r.get("pressure_level", "unknown")].append(r.get("score", 0.0))
        per_pressure = {p: sum(s) / len(s) for p, s in sorted(pres_scores.items())}

        # Per-relation breakdown
        rel_scores: dict[str, list[float]] = defaultdict(list)
        for r in rows:
            rel = r.get("primary_relation", "unknown")
            if rel:
                rel_scores[rel].append(r.get("score", 0.0))
        per_relation = {rel: sum(s) / len(s) for rel, s in sorted(rel_scores.items())}

        out[model] = {
            "n": len(rows),
            "ipr": ipr,
            "rdr": rdr,
            "sem_sub": sem,
            "pho_sub": pho,
            "pres": pres,
            "alias_ok": alias,
            "per_family": per_family,
            "per_pressure": per_pressure,
            "per_relation": per_relation,
        }

    return out


def v3_dose_response(
    results: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """For each model, compute mean score at each pressure level and test
    monotonicity via Spearman correlation of pressure level vs error rate.

    Returns {model: {levels, mean_scores, error_rates, spearman_rho,
    spearman_p, monotonic}}.
    """
    pressure_rank = {"low": 0, "mid": 1, "high": 2}

    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in results:
        by_model[r["model"]].append(r)

    out: dict[str, dict[str, Any]] = {}
    for model, rows in sorted(by_model.items()):
        level_scores: dict[str, list[float]] = defaultdict(list)
        for r in rows:
            pl = r.get("pressure_level", "")
            if pl in pressure_rank:
                level_scores[pl].append(r.get("score", 0.0))

        levels: list[str] = []
        mean_scores: list[float] = []
        error_rates: list[float] = []
        for lvl in ["low", "mid", "high"]:
            if lvl in level_scores:
                scores = level_scores[lvl]
                ms = sum(scores) / len(scores) if scores else 0.0
                er = 1.0 - ms  # error rate = 1 - mean score
                levels.append(lvl)
                mean_scores.append(ms)
                error_rates.append(er)

        # Spearman correlation: rank of pressure level vs error rate
        spearman_rho: float | None = None
        spearman_p: float | None = None
        monotonic = False
        if len(levels) >= 3:
            try:
                from scipy.stats import spearmanr

                ranks = [pressure_rank[l] for l in levels]
                rho, p = spearmanr(ranks, error_rates)
                spearman_rho = round(float(rho), 4)
                spearman_p = round(float(p), 6)
            except ImportError:
                # Manual Spearman for 3 points (exact ranks 0,1,2)
                n = len(levels)
                ranks = [float(pressure_rank[l]) for l in levels]
                mean_r = sum(ranks) / n
                mean_e = sum(error_rates) / n
                cov = sum(
                    (ranks[i] - mean_r) * (error_rates[i] - mean_e) for i in range(n)
                )
                std_r = math.sqrt(sum((ranks[i] - mean_r) ** 2 for i in range(n)))
                std_e = math.sqrt(sum((error_rates[i] - mean_e) ** 2 for i in range(n)))
                if std_r > 0 and std_e > 0:
                    spearman_rho = round(cov / (std_r * std_e), 4)
                else:
                    spearman_rho = 0.0
                spearman_p = None  # can't compute without scipy

            monotonic = all(
                error_rates[i] <= error_rates[i + 1]
                for i in range(len(error_rates) - 1)
            )

        out[model] = {
            "levels": levels,
            "mean_scores": mean_scores,
            "error_rates": error_rates,
            "spearman_rho": spearman_rho,
            "spearman_p": spearman_p,
            "monotonic": monotonic,
        }

    return out


def v3_family_d_analysis(
    results: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """For Family D (clarify_abstain) items, compute clarify rate and abstain
    rate per model.

    Returns {model: {n, clarify, abstain, clarify_rate, abstain_rate,
    combined_rate, mean_score}}.
    """
    fam_d = [r for r in results if r.get("family") == "clarify_abstain"]

    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in fam_d:
        by_model[r["model"]].append(r)

    out: dict[str, dict[str, Any]] = {}
    for model, rows in sorted(by_model.items()):
        n = len(rows)
        clarify = sum(1 for r in rows if r["outcome"] == "CLARIFY")
        abstain = sum(1 for r in rows if r["outcome"] == "ABSTAIN")
        scores = [r.get("score", 0.0) for r in rows]
        out[model] = {
            "n": n,
            "clarify": clarify,
            "abstain": abstain,
            "clarify_rate": clarify / n if n > 0 else 0.0,
            "abstain_rate": abstain / n if n > 0 else 0.0,
            "combined_rate": (clarify + abstain) / n if n > 0 else 0.0,
            "mean_score": sum(scores) / n if n > 0 else 0.0,
        }

    return out


def v3_print_summary(results: list[dict[str, Any]]) -> None:
    """Print a formatted summary table for v3.1 results."""
    metrics = v3_aggregate_metrics(results)
    dose = v3_dose_response(results)
    fam_d = v3_family_d_analysis(results)

    print("\n" + "=" * 72)
    print("  RIGIDBENCH v3.1 — RESULTS ANALYSIS")
    print("  Relational Invariance Under Pressure")
    print("=" * 72)

    print(f"\n  Total results: {len(results)}")

    # --- Per-model headline ---
    print(f"\n  {'─' * 72}")
    print(
        f"  {'Model':<30s} {'N':>5s} {'IPR':>7s} {'RDR':>7s} "
        f"{'SEM':>5s} {'PHO':>5s} {'PRES':>5s}"
    )
    print(f"  {'─' * 72}")
    for model, m in metrics.items():
        rdr_str = f"{m['rdr']:.3f}" if not math.isnan(m["rdr"]) else "  N/A"
        print(
            f"  {model:<30s} {m['n']:5d} {m['ipr']:7.3f} {rdr_str:>7s} "
            f"{m['sem_sub']:5d} {m['pho_sub']:5d} {m['pres']:5d}"
        )

    # --- Per-family breakdown ---
    all_families = sorted({r.get("family", "unknown") for r in results})
    if all_families:
        print(f"\n  {'─' * 72}")
        print("  PER-FAMILY MEAN SCORE")
        print(f"  {'─' * 72}")
        header = f"  {'Model':<30s}"
        for fam in all_families:
            short = fam[:12]
            header += f" {short:>12s}"
        print(header)
        for model, m in metrics.items():
            line = f"  {model:<30s}"
            for fam in all_families:
                val = m["per_family"].get(fam)
                line += f" {val:12.3f}" if val is not None else f" {'—':>12s}"
            print(line)

    # --- Per-pressure-level breakdown ---
    print(f"\n  {'─' * 72}")
    print("  PER-PRESSURE-LEVEL MEAN SCORE")
    print(f"  {'─' * 72}")
    for model, m in metrics.items():
        print(f"  {model}:")
        for pl in ["low", "mid", "high"]:
            val = m["per_pressure"].get(pl)
            if val is not None:
                bar = "█" * int(val * 40)
                print(f"    {pl:4s}: {val:.3f} {bar}")

    # --- Per-relation breakdown ---
    all_relations = sorted({r.get("primary_relation", "") for r in results} - {""})
    if all_relations:
        print(f"\n  {'─' * 72}")
        print("  PER-RELATION MEAN SCORE")
        print(f"  {'─' * 72}")
        header = f"  {'Model':<25s}"
        for rel in all_relations:
            header += f" {rel:>12s}"
        print(header)
        for model, m in metrics.items():
            line = f"  {model:<25s}"
            for rel in all_relations:
                val = m["per_relation"].get(rel)
                line += f" {val:12.3f}" if val is not None else f" {'—':>12s}"
            print(line)

    # --- Dose-response ---
    print(f"\n  {'─' * 72}")
    print("  DOSE-RESPONSE (error rate by pressure level)")
    print(f"  {'─' * 72}")
    for model, d in dose.items():
        print(f"  {model}:")
        for lvl, er in zip(d["levels"], d["error_rates"]):
            bar = "█" * int(er * 40)
            print(f"    {lvl:4s}: error_rate = {er:.3f} {bar}")
        if d["spearman_rho"] is not None:
            p_str = f", p = {d['spearman_p']}" if d["spearman_p"] is not None else ""
            print(f"    Spearman rho = {d['spearman_rho']}{p_str}")
        mono = "YES" if d["monotonic"] else "NO"
        print(f"    Monotonically increasing error: {mono}")

    # --- Family D ---
    if fam_d:
        print(f"\n  {'─' * 72}")
        print("  FAMILY D (clarify_abstain) ANALYSIS")
        print(f"  {'─' * 72}")
        print(
            f"  {'Model':<30s} {'N':>4s} {'CLR':>5s} {'ABS':>5s} "
            f"{'CLR%':>7s} {'ABS%':>7s} {'Score':>7s}"
        )
        for model, d in fam_d.items():
            print(
                f"  {model:<30s} {d['n']:4d} {d['clarify']:5d} {d['abstain']:5d} "
                f"{d['clarify_rate'] * 100:6.1f}% {d['abstain_rate'] * 100:6.1f}% "
                f"{d['mean_score']:7.3f}"
            )

    print(f"\n{'=' * 72}")
    print(f"  END OF v3.1 ANALYSIS")
    print(f"{'=' * 72}\n")


def _latex_safe_model_name(model: str) -> str:
    """Convert a model name to a LaTeX-safe macro suffix.

    Generates SHORT tags that match the paper's results.tex macro names.
    E.g. 'claude-sonnet-4-20250514' -> 'Claude',
         'gpt-4o' -> 'GPTFourO',
         'gemini-2.5-flash-preview-04-17' -> 'Gemini'.
    """
    name = model.lower()
    # Short tags matching results.tex macro conventions.
    # Order matters: more specific patterns first.
    replacements = [
        (r"gpt[-_]?4o", "GPTFourO"),
        (r"gpt[-_]?4", "GPTFour"),
        (r"gpt[-_]?3\.?5", "GPTThreeFive"),
        (r"claude.*opus", "ClaudeOpus"),
        (r"claude.*sonnet.*4", "Claude"),  # paper: \ResClaude*
        (r"claude.*sonnet.*3[._-]5", "ClaudeSonnetThreeFive"),
        (r"claude.*sonnet", "ClaudeSonnet"),
        (r"claude.*haiku", "ClaudeHaiku"),
        (r"gemini.*2[._-]?5.*flash", "Gemini"),  # paper: \ResGemini*
        (r"gemini.*2[._-]?5.*pro", "GeminiPro"),
        (r"gemini.*2.*flash", "GeminiTwoFlash"),
        (r"gemini.*pro", "GeminiPro"),
        (r"gemini.*flash", "GeminiFlash"),
        (r"o4[-_]?mini", "OFourMini"),
        (r"o3[-_]?mini", "OThreeMini"),
        (r"o3", "OThree"),
        (r"o1[-_]?mini", "OOneMini"),
        (r"o1", "OOne"),
    ]
    for pattern, replacement in replacements:
        if re.search(pattern, name):
            return replacement

    # Fallback: CamelCase from alphanumeric parts
    parts = re.findall(r"[a-zA-Z0-9]+", model)
    return "".join(p.capitalize() for p in parts)


def _compute_ssr_per_pressure(
    results: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    """Compute SSR (semantic substitution rate) per model per pressure level.

    SSR_level = count(SEM_SUB at level) / count(all items at level).
    """
    by_model: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for r in results:
        model = r["model"]
        pl = r.get("pressure_level", "")
        outcome = r.get("outcome", "")
        if pl and outcome:
            by_model[model][pl].append(outcome)

    out: dict[str, dict[str, float]] = {}
    for model, levels in sorted(by_model.items()):
        out[model] = {}
        for pl in ["low", "mid", "high"]:
            if pl in levels:
                outcomes = levels[pl]
                sem = sum(1 for o in outcomes if o == "SEM_SUB")
                out[model][pl] = sem / len(outcomes) if outcomes else 0.0
    return out


def _compute_rsr_per_relation(
    results: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    """Compute RSR (relation-specific SSR) per model per relation type.

    RSR_relation = count(SEM_SUB for relation) / count(all items for relation).
    """
    # Map relation family names to R1-R8 codes
    _rel_to_code: dict[str, str] = {
        "identity": "One",
        "virtue_name": "Two",
        "etymological": "Three",
        "kinship": "Four",
        "alias": "Five",
        "role_title": "Six",
        "semantic_field": "Seven",
        "historical_set": "Eight",
    }

    by_model: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for r in results:
        model = r["model"]
        rel = r.get("primary_relation", "")
        outcome = r.get("outcome", "")
        if rel and outcome:
            by_model[model][rel].append(outcome)

    out: dict[str, dict[str, float]] = {}
    for model, rels in sorted(by_model.items()):
        out[model] = {}
        for rel_name, code in _rel_to_code.items():
            if rel_name in rels:
                outcomes = rels[rel_name]
                sem = sum(1 for o in outcomes if o == "SEM_SUB")
                out[model][code] = sem / len(outcomes) if outcomes else 0.0
    return out


def _compute_relation_vulnerability_spearman(
    rsr_per_model: dict[str, dict[str, float]],
) -> dict[str, float]:
    """Compute Spearman ρ between predicted vulnerability ordering and observed RSR.

    Predicted ordering (most to least vulnerable):
        R2 > R3 > R7 > R6 > R8 > R4 > R5 > R1

    We assign predicted vulnerability scores (higher = more vulnerable):
        R2=8, R3=7, R7=6, R6=5, R8=4, R4=3, R5=2, R1=1

    Then correlate these scores with observed RSR values per model.
    This is the correlation reported in Table 4 of the paper.
    """
    # Code names used in _compute_rsr_per_relation output
    predicted_vulnerability: dict[str, float] = {
        "One": 1.0,  # R1 Identity
        "Two": 8.0,  # R2 Virtue name
        "Three": 7.0,  # R3 Etymological
        "Four": 3.0,  # R4 Kinship
        "Five": 2.0,  # R5 Alias
        "Six": 5.0,  # R6 Role/title
        "Seven": 6.0,  # R7 Semantic field
        "Eight": 4.0,  # R8 Historical set
    }

    def _spearman_manual(x: list[float], y: list[float]) -> float:
        """Compute Spearman rank correlation without scipy."""
        n = len(x)
        if n < 3:
            return float("nan")

        def _rank(vals: list[float]) -> list[float]:
            indexed = sorted(enumerate(vals), key=lambda t: -t[1])
            ranks = [0.0] * n
            i = 0
            while i < len(indexed):
                j = i
                while j < len(indexed) and indexed[j][1] == indexed[i][1]:
                    j += 1
                avg_rank = (i + 1 + j) / 2.0
                for k in range(i, j):
                    ranks[indexed[k][0]] = avg_rank
                i = j
            return ranks

        rx = _rank(x)
        ry = _rank(y)
        d_sq = sum((a - b) ** 2 for a, b in zip(rx, ry))
        return 1.0 - 6.0 * d_sq / (n * (n**2 - 1))

    out: dict[str, float] = {}
    for model, rsr_vals in rsr_per_model.items():
        # Use all 8 relations; missing ones default to 0.0 RSR
        codes = sorted(predicted_vulnerability.keys())
        pred = [predicted_vulnerability[c] for c in codes]
        obs = [rsr_vals.get(c, 0.0) for c in codes]

        try:
            from scipy.stats import spearmanr

            rho, _ = spearmanr(pred, obs)
            out[model] = round(float(rho), 3)
        except ImportError:
            out[model] = round(_spearman_manual(pred, obs), 3)

    return out


def v3_export_latex_macros(
    results: list[dict[str, Any]],
    path: str,
) -> None:
    """Write a results_v3.tex file with LaTeX macros for paper integration.

    Uses \\def so auto-generated values override the \\ResPending
    defaults in results.tex (\\def works regardless of prior definition).
    Generates all macro families expected by the
    paper tables: core metrics, SSR/PSR rates, SSR-per-pressure (dose-
    response), RSR-per-relation, Spearman rho, Family D rates, hypothesis
    status, and frequency-control deltas.
    """
    metrics = v3_aggregate_metrics(results)
    dose = v3_dose_response(results)
    fam_d = v3_family_d_analysis(results)
    ssr_pressure = _compute_ssr_per_pressure(results)
    rsr_relation = _compute_rsr_per_relation(results)
    rel_vuln_spearman = _compute_relation_vulnerability_spearman(rsr_relation)

    def _def(name: str, val: str) -> str:
        r"""Emit \def\MacroName{value} — works whether or not the macro exists."""
        return f"\\def\\{name}{{{val}}}"

    lines: list[str] = []
    lines.append("% Auto-generated by analyze_results.py --v3")
    lines.append("% Overrides \\ResPending defaults from results.tex via \\def")
    lines.append("")

    for model, m in metrics.items():
        tag = _latex_safe_model_name(model)
        n = m["n"]

        # --- Core metrics ---
        lines.append(f"% --- {model} (tag: {tag}) ---")
        lines.append(_def(f"Res{tag}IPR", f"{m['ipr']:.3f}"))
        rdr_val = f"{m['rdr']:.3f}" if not math.isnan(m["rdr"]) else "N/A"
        lines.append(_def(f"Res{tag}RDR", rdr_val))

        # SSR and PSR as rates (paper tables expect these)
        ssr_rate = m["sem_sub"] / n if n > 0 else 0.0
        psr_rate = m["pho_sub"] / n if n > 0 else 0.0
        lines.append(_def(f"Res{tag}SSR", f"{ssr_rate:.3f}"))
        lines.append(_def(f"Res{tag}PSR", f"{psr_rate:.3f}"))

        # --- SSR per pressure level (dose-response table) ---
        if model in ssr_pressure:
            for pl in ["low", "mid", "high"]:
                if pl in ssr_pressure[model]:
                    pl_tag = pl.capitalize()
                    val = ssr_pressure[model][pl]
                    lines.append(_def(f"Res{tag}SSR{pl_tag}", f"{val:.3f}"))

        # --- RSR per relation (relation vulnerability table) ---
        if model in rsr_relation:
            for code, val in rsr_relation[model].items():
                lines.append(_def(f"Res{tag}RSR{code}", f"{val:.3f}"))

        # --- Relation-vulnerability Spearman (Table 4 in paper) ---
        if model in rel_vuln_spearman:
            rho_val = rel_vuln_spearman[model]
            if not math.isnan(rho_val):
                lines.append(_def(f"Res{tag}RSRSpearman", str(rho_val)))

        # --- Dose-response Spearman (separate macro, not used in Table 4) ---
        if model in dose:
            d = dose[model]
            if d["spearman_rho"] is not None:
                lines.append(_def(f"Res{tag}DoseSpearman", str(d["spearman_rho"])))

        # --- Family D ---
        if model in fam_d:
            fd = fam_d[model]
            lines.append(_def(f"Res{tag}ClarifyRate", f"{fd['clarify_rate']:.3f}"))
            lines.append(_def(f"Res{tag}AbstainRate", f"{fd['abstain_rate']:.3f}"))

        lines.append("")

    # --- Hypothesis status ---
    # H1: RDR >> 0.5 for all models
    all_rdr = [m["rdr"] for m in metrics.values() if not math.isnan(m["rdr"])]
    if all_rdr:
        h1 = (
            "\\textbf{Confirmed}"
            if all(r > 0.7 for r in all_rdr)
            else (
                "\\textbf{Partially confirmed}"
                if all(r > 0.5 for r in all_rdr)
                else "\\textbf{Not confirmed}"
            )
        )
        lines.append(_def("ResHOneStatus", h1))

    # H2: SSR monotonically increases with pressure
    if dose:
        all_monotonic = all(d.get("monotonic", False) for d in dose.values())
        h2 = "\\textbf{Confirmed}" if all_monotonic else "\\textbf{Not confirmed}"
        lines.append(_def("ResHTwoStatus", h2))

    # H3/H4 require more complex analysis; leave for manual override
    lines.append("")

    # --- Global stats ---
    lines.append("% --- Global ---")
    lines.append(_def("ResTotalResults", str(len(results))))
    n_models = len(metrics)
    lines.append(_def("ResTotalModels", str(n_models)))
    families = sorted({r.get("family", "") for r in results} - {""})
    lines.append(_def("ResTotalFamilies", str(len(families))))

    out_path = pathlib.Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"  LaTeX macros exported to {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze RigidBench v2 and v3.1 results"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to results JSONL (v2) or output directory (v3)",
    )
    parser.add_argument("--csv", type=str, default=None, help="Export CSV to this path")
    parser.add_argument(
        "--summary-json", type=str, default=None, help="Export summary JSON"
    )
    parser.add_argument(
        "--v3",
        action="store_true",
        help="Analyse v3.1 results.  --input should be the output directory "
        "containing rigidbench_v3_results.jsonl",
    )
    parser.add_argument(
        "--v3-latex",
        type=str,
        default=None,
        help="(v3 only) Export LaTeX macros to this path",
    )
    args = parser.parse_args()

    # --- v3.1 mode ---
    if args.v3:
        input_path = pathlib.Path(args.input)
        if input_path.is_dir():
            v3_path = input_path / "rigidbench_v3_results.jsonl"
        else:
            v3_path = input_path

        if not v3_path.exists():
            print(f"ERROR: v3.1 results file not found: {v3_path}", file=sys.stderr)
            sys.exit(1)

        results = load_v3_results(str(v3_path))
        print(f"Loaded {len(results)} v3.1 results from {v3_path}")

        v3_print_summary(results)

        # LaTeX export
        latex_path = args.v3_latex
        if latex_path is None:
            latex_path = str(v3_path.parent / "results_v3.tex")
        v3_export_latex_macros(results, latex_path)

        # Summary JSON
        if args.summary_json:
            summary = {
                "aggregate": v3_aggregate_metrics(results),
                "dose_response": v3_dose_response(results),
                "family_d": v3_family_d_analysis(results),
            }
            out = pathlib.Path(args.summary_json)
            with open(out, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, default=str)
            print(f"  Summary JSON exported to {out}")

        return

    # --- Legacy v2 mode ---
    results = load_results(args.input)
    print(f"Loaded {len(results)} results from {args.input}")

    print_report(results)

    if args.csv:
        export_csv(results, pathlib.Path(args.csv))

    if args.summary_json:
        export_summary_json(results, pathlib.Path(args.summary_json))
    else:
        # Default: export summary JSON next to input
        default_json = pathlib.Path(args.input).parent / "analysis_summary.json"
        export_summary_json(results, default_json)


if __name__ == "__main__":
    main()
