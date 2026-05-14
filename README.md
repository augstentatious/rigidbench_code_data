# RigidBench v3.1

Benchmark and evaluation harness for **referential invariance under semantic pressure** in large language models.

This repository contains the code, benchmark items, and pre-computed model outputs for the RigidBench v3.1 paper artifact. Reproducing the reported aggregate statistics does **not** require API keys; API credentials are only needed when running the benchmark against a new model.

RigidBench tests whether language models preserve referent identity when surrounding context is semantically loaded in favor of a different entity. It produces two key metrics:

- **SSR** (semantic substitution rate): fraction of completions where the model substitutes a semantically related name
- **RDR** (referential descriptivism ratio): SSR / (SSR + PSR), where PSR is the phonological substitution rate

## Structure

```
rigidbench/
├── run_all.py           # evaluation harness
├── analyze_results.py   # per-run analysis script
├── paper_stats.py       # recomputes aggregate paper statistics
├── requirements.txt
└── results/             # pre-computed results for all 9 model runs reported in the paper
    ├── gpt_55/
    ├── kimi_k2p6/
    ├── gemini_25_pro/
    ├── gemini_25_flash/
    ├── deepseek_v4/
    ├── claude_sonnet_46/
    ├── llama4_scout/
    ├── gpt_oss_120b/
    └── grok_43/
```

Each `results/<model>/rigidbench_v3_results.jsonl` contains all 140 benchmark items with the model's raw completions and outcome classifications.

## Reproducing the paper results

```bash
pip install -r requirements.txt

# Recompute the aggregate statistics reported in the paper
python paper_stats.py

# Generate confidence intervals, per-model RDR, and per-relation SSR tables
python bootstrap_rdr.py

# Run the regression robustness check
python mixed_effects.py

# Audit scorer edge cases
python scorer_audit.py --max-discrepancies 25

# Analyze a single model's results
python analyze_results.py --v3 --input results/kimi_k2p6

# Analyze all nine model runs
for d in results/*/; do
    python analyze_results.py --v3 --input "$d"
done
```

## Running on a new model

```bash
# OpenAI-compatible endpoint (Groq, Fireworks, OpenRouter, etc.)
export OPENAI_API_KEY="..."
python run_all.py --model openai/meta-llama/llama-3-70b --base-url https://api.groq.com/openai/v1

# Anthropic
export ANTHROPIC_API_KEY="..."
python run_all.py --model claude-3-5-sonnet-20241022

# Google Gemini
export GOOGLE_API_KEY="..."
python run_all.py --model gemini-2.0-flash
```

Results are written to `results_<model_slug>/rigidbench_v3_results.jsonl`.

## Benchmark structure

140 items across 5 task families:

| Family | N | Task |
|---|---|---|
| A: Completion under pressure | 90 | Single-turn completion; semantic pressure via context |
| B: Multi-turn persistence | 20 | Referent established in prior turns |
| C: Summary compression | 15 | Lossy summarization task |
| D: Clarify/abstain | 10 | Genuinely ambiguous; correct response is to ask |
| E: Entity set competition | 5 | Multiple competing entities in context |

8 semantic relation types (R1-R8) spanning virtue names, kinship, role/title, semantic field, historical set, alias, etymological link, and identity-neutral names.

## Outcome categories

| Code | Description |
|---|---|
| `PRES` | Canonical referent preserved |
| `SEM_SUB` | Semantic substitution (lure entity) |
| `PHO_SUB` | Phonological substitution (neighbor name) |
| `ENT_CONF` | Entity confusion (Family E) |
| `CLARIFY` | Model requests clarification |
| `ABSTAIN` | Model declines to answer |

Classification uses deterministic regex matching against a registered answer key. No LLM judge is used.

## License

CC BY 4.0
