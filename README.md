# Tool-Lab

This repository now contains both the paper draft and a runnable Tool-Lab experiment framework.

The Python runner is built around the paper's core design decisions:

- Config-driven experiments so the same runner can be reused across domains.
- Two interface modes:
	`scrolling` for Lau-style dynamic information boards and `fixed` for Payne-style matrices.
- Provider adapters for OpenAI, Anthropic, and Google, plus a `mock` provider for local smoke tests.
- Five replications by default for each model-condition cell, with aggregated behavior written automatically.

## Install

```bash
python -m pip install -e .
```

## Run an experiment

Smoke test with the bundled mock model:

```bash
tool-lab run --config experiments/political_general_election.yaml
```

Run the same experiment with a real provider and keep the Lau-style scrolling interface:

```bash
tool-lab run \
	--config experiments/political_general_election.yaml \
	--provider openai \
	--model gpt-5.4
```

Switch the same task into a Payne-style fixed matrix at runtime:

```bash
tool-lab run \
	--config experiments/political_general_election.yaml \
	--provider anthropic \
	--model claude-sonnet-4-5 \
	--matrix-mode fixed
```

Run the bundled fixed-matrix consumer task with Google:

```bash
tool-lab run \
	--config experiments/consumer_choice_fixed_matrix.yaml \
	--provider google \
	--model gemini-2.5-pro
```

Environment variables:

- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `GOOGLE_API_KEY` or `GEMINI_API_KEY`

## Results

Each run writes a timestamped result directory under `results/` containing:

- `config_snapshot.json`: the resolved experiment configuration used for the run.
- `runs.jsonl`: one JSON object per replication, including the raw tool trace and transcript.
- `metrics.csv`: a flat per-replication table for quick analysis in pandas, R, or spreadsheets.
- `summary.json`: aggregated behavior across the five replications.

The storage format is intentionally split:

- `JSONL` is the best fit for raw traces because action logs and transcripts are nested and variable-length.
- `CSV` is useful for flat run-level metrics and downstream statistics.
- `JSON` is used for the aggregate summary because it preserves named statistics cleanly.

## Experiment specs

The two bundled specs are:

- `experiments/political_general_election.yaml`: a Lau-style scrolling election task.
- `experiments/consumer_choice_fixed_matrix.yaml`: a Payne-style fixed-matrix consumer choice task.

To adapt the framework, edit or add a new YAML file with:

- the option set,
- the attribute set,
- hidden cue values,
- normative cue scores,
- the active tool list,
- and the desired matrix mode.


# TEST
```
PYTHONPATH=src python src/tool_lab/cli.py run --config experiments/consumer_choice_fixed_matrix.yaml
```