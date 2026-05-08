#!/usr/bin/env python3
"""Logistic robustness models for RigidBench v3.1.

The requested mixed-effects fit is attempted with statsmodels. If statsmodels
is unavailable or the mixed fit does not converge cleanly, the script falls
back to a standard logistic regression with model fixed effects and
item-cluster robust sandwich standard errors.
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from collections import defaultdict
from pathlib import Path


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


def result_files(root: Path) -> list[Path]:
    # Restrict to the nine top-level model result folders. The checkout also
    # contains archival runs under results/rigidbench_v3/ that are not part of
    # the manuscript aggregate.
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
                rows.append(
                    {
                        "y": 1.0
                        if normalize_outcome(row.get("outcome") or row.get("error_type")) == "SEM_SUB"
                        else 0.0,
                        "pressure": {"low": 0.0, "mid": 1.0, "high": 2.0}.get(
                            row.get("pressure_level"), 0.0
                        ),
                        "semantic_sim": float(row.get("semantic_sim_name_to_lure") or 0.0),
                        "phon_distance": float(row.get("phon_distance_name_to_neighbor") or 0.0),
                        "model": slug,
                        "item_id": row.get("item_id") or row.get("triple_id"),
                    }
                )
    return rows


def fit_statsmodels_mixed(rows: list[dict]) -> tuple[str, list[dict]] | None:
    try:
        import pandas as pd
        from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
    except ImportError:
        return None

    df = pd.DataFrame(rows)
    model = BinomialBayesMixedGLM.from_formula(
        "y ~ pressure + semantic_sim + phon_distance",
        {"model": "0 + C(model)", "item": "0 + C(item_id)"},
        df,
        vcp_p=0.5,
        fe_p=5.0,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = model.fit_map(minim_opts={"maxiter": 1000})

    warning_text = " ".join(str(w.message) for w in caught)
    if "did not converge" in warning_text.lower():
        return None

    output: list[dict] = []
    for name, coef, se in zip(model.exog_names, result.fe_mean, result.fe_sd):
        if name == "Intercept":
            continue
        output.append(summary_row(name, float(coef), float(se)))
    return "mixed-effects logistic regression (random intercepts for model and item)", output


def fit_fixed_effects_clustered(rows: list[dict]) -> tuple[str, list[dict]]:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("NumPy is required for the fallback logistic regression") from exc

    models = sorted({row["model"] for row in rows})
    columns = ["Intercept", "pressure", "semantic_sim", "phon_distance"] + [
        f"model:{m}" for m in models[1:]
    ]

    x_rows: list[list[float]] = []
    y_vals: list[float] = []
    clusters: list[str] = []
    for row in rows:
        x_row = [
            1.0,
            row["pressure"],
            row["semantic_sim"],
            row["phon_distance"],
        ]
        x_row.extend(1.0 if row["model"] == model else 0.0 for model in models[1:])
        x_rows.append(x_row)
        y_vals.append(row["y"])
        clusters.append(row["item_id"])

    x = np.asarray(x_rows, dtype=float)
    y = np.asarray(y_vals, dtype=float)
    beta = np.zeros(x.shape[1], dtype=float)

    def log_likelihood(beta_vec):
        eta_vec = x @ beta_vec
        return float(np.sum(y * eta_vec - np.logaddexp(0.0, eta_vec)))

    for _ in range(100):
        eta = np.clip(x @ beta, -35.0, 35.0)
        p = 1.0 / (1.0 + np.exp(-eta))
        weights = p * (1.0 - p)
        gradient = x.T @ (y - p)
        hessian = x.T @ (x * weights[:, None])
        step = np.linalg.solve(hessian + np.eye(hessian.shape[0]) * 1e-10, gradient)

        old_ll = log_likelihood(beta)
        next_beta = beta + step
        scale = 1.0
        while log_likelihood(next_beta) < old_ll and scale > 1e-6:
            scale *= 0.5
            next_beta = beta + scale * step

        if np.max(np.abs(next_beta - beta)) < 1e-8:
            beta = next_beta
            break
        beta = next_beta

    eta = np.clip(x @ beta, -35.0, 35.0)
    p = 1.0 / (1.0 + np.exp(-eta))
    weights = p * (1.0 - p)
    hessian = x.T @ (x * weights[:, None])
    bread = np.linalg.pinv(hessian)

    by_cluster: dict[str, list[int]] = defaultdict(list)
    for idx, cluster in enumerate(clusters):
        by_cluster[cluster].append(idx)

    meat = np.zeros_like(hessian)
    for indices in by_cluster.values():
        idx = np.asarray(indices)
        score = x[idx].T @ (y[idx] - p[idx])
        meat += np.outer(score, score)

    n_clusters = len(by_cluster)
    n_obs = len(rows)
    n_params = x.shape[1]
    correction = (n_clusters / (n_clusters - 1.0)) * ((n_obs - 1.0) / (n_obs - n_params))
    covariance = bread @ meat @ bread * correction

    output = []
    for name in ["pressure", "semantic_sim", "phon_distance"]:
        idx = columns.index(name)
        se = math.sqrt(max(float(covariance[idx, idx]), 0.0))
        output.append(summary_row(name, float(beta[idx]), se))

    return "logistic regression with model fixed effects and item-cluster robust SEs", output


def summary_row(name: str, coef: float, se: float) -> dict:
    lower = coef - 1.96 * se
    upper = coef + 1.96 * se
    return {
        "name": name,
        "coef": coef,
        "se": se,
        "or": math.exp(coef),
        "lower": math.exp(lower),
        "upper": math.exp(upper),
    }


def display_name(name: str) -> str:
    return {
        "pressure": "Pressure level",
        "semantic_sim": "Semantic similarity",
        "phon_distance": "Phonological distance",
    }.get(name, name)


def write_table(path: Path, method: str, rows: list[dict]) -> None:
    lines = [
        "% Auto-generated by mixed_effects.py",
        f"% Method: {method}",
        "\\begin{tabular}{@{}lccc@{}}",
        "  \\toprule",
        "  \\textbf{Predictor} & \\textbf{OR} & \\textbf{95\\% CI} & \\textbf{SE} \\\\",
        "  \\midrule",
    ]
    for row in rows:
        lines.append(
            f"  {display_name(row['name'])} & {row['or']:.2f} "
            f"& [{row['lower']:.2f}, {row['upper']:.2f}] "
            f"& {row['se']:.2f} \\\\"
        )
    lines.extend(["  \\bottomrule", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default="results", help="Directory containing model result folders")
    parser.add_argument("--out-dir", default="tables", help="Directory for generated LaTeX tables")
    args = parser.parse_args()

    rows = load_rows(Path(args.results_root))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fit = fit_statsmodels_mixed(rows)
    if fit is None:
        fit = fit_fixed_effects_clustered(rows)
    method, summaries = fit

    table_path = out_dir / "mixed_effects_logit.tex"
    write_table(table_path, method, summaries)

    print(f"Loaded {len(rows)} rows from {args.results_root}")
    print(f"Method: {method}")
    for row in summaries:
        print(
            f"{display_name(row['name'])}: OR = {row['or']:.2f} "
            f"[{row['lower']:.2f}, {row['upper']:.2f}], SE = {row['se']:.2f}"
        )
    print(f"Wrote {table_path}")


if __name__ == "__main__":
    main()
