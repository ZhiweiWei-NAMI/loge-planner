# `examples/llm` -- example / fixture inputs

Everything in this directory is a small, labelled example of the input
formats used to exercise the LLM code paths. Replace them with your own run
outputs for evaluation.

| File | What it is | Used by |
| :--- | :--- | :--- |
| `checkpoints_example.csv` | 8 checkpoints of 3 toy workflows, same schema as Stage 1 `checkpoint_init.csv` | `decompose_cli` |
| `scenario_example.json` | 3 agents / 3 workflows toy scenario | `decompose_cli` |
| `decomposition_example.json` | An example of a valid protocol-`1.0` decomposition of the table above | `decompose_cli --from-json` |
| `terminal_windows_example.csv` | 18 windows (9 classes x 2 samples, L=4) generated from hand-set prototypes + jitter | `terminal_monitor_eval --data-csv`, `prepare_terminal_sft` |
| `terminal_sft_example.jsonl` | **9** chat examples, all from `split=train` (the 9 `split=test` windows are excluded), rendered with the exact inference prompt | `train_lora` |
| `latency_replay_example.json` | Example latencies, declared `"provenance": "fixture"`, for the `--latency-replay-json` interface | `terminal_monitor_eval` |

The window fixture is produced by `src.agent_llm.terminal_data.synthesize_fixture_windows`
and is labelled `fixture_synthetic` in every report it feeds.

## 1. Cloud decomposition (real OpenAI-compatible call)

Prompt preview only (no network, no key needed):

```bash
python3 -m src.agent_llm.decompose_cli \
  --checkpoint-csv examples/llm/checkpoints_example.csv \
  --scenario-file examples/llm/scenario_example.json \
  --model gpt-4o-mini --dry-run
```

Real call (writes `checkpoint_clusters_llm.csv`, the Stage 2/3 input):

```bash
export OPENAI_API_KEY=...            # never commit the key
python3 -m src.agent_llm.decompose_cli \
  --checkpoint-csv examples/llm/checkpoints_example.csv \
  --scenario-file examples/llm/scenario_example.json \
  --model gpt-4o-mini \
  --base-url https://api.openai.com/v1 \
  --min-macro-tasks 2 --max-macro-tasks 4 \
  --output-dir /tmp/llm_decomp_example
```

Re-validate the example JSON and emit a cluster CSV without any API call:

```bash
python3 -m src.agent_llm.decompose_cli \
  --checkpoint-csv examples/llm/checkpoints_example.csv \
  --scenario-file examples/llm/scenario_example.json \
  --from-json examples/llm/decomposition_example.json \
  --output-dir /tmp/llm_decomp_example
```

On a Stage 1 run directory, add `--inplace` so the CSV lands where Stage 2/3 look
for it (`<input-root>/run_seed_<seed>/checkpoint_clusters_<method>.csv`):

```bash
python3 -m src.agent_llm.decompose_cli \
  --run-dir data/offline_init/large_mixed/run_seed_42 \
  --model gpt-4o-mini --min-macro-tasks 8 --max-macro-tasks 60 \
  --cluster-method llm --inplace
python3 -m src.offline_pipeline.stage2_plan_from_clusters --seed 42 --cluster-method llm
python3 -m src.offline_pipeline.stage3_plan_with_pddl     --seed 42 --cluster-method llm
```

## 2. Terminal-monitor SFT data -> LoRA training -> evaluation

```bash
# (a) JSONL from real windows. `--split auto` (default) exports only split=train.
#     `examples/llm/terminal_sft_example.jsonl` was generated with the same
#     command and contains only the 9 split=train windows (no test rows), so it
#     can be used directly for the example below without leaking the 9 test rows.
python3 -m src.agent_llm.prepare_terminal_sft \
  --windows-csv examples/llm/terminal_windows_example.csv --split train \
  --output-jsonl /tmp/terminal_sft.jsonl

# (b) validate the JSONL without downloading a model
python3 -m src.agent_llm.train_lora \
  --train-jsonl /tmp/terminal_sft.jsonl \
  --output-dir /tmp/qwen_terminal_lora --validate-only

# (c) real LoRA training (downloads Qwen/Qwen2.5-0.5B-Instruct once)
python3 -m src.agent_llm.train_lora \
  --train-jsonl /tmp/terminal_sft.jsonl \
  --output-dir /tmp/qwen_terminal_lora \
  --base-model Qwen/Qwen2.5-0.5B-Instruct \
  --epochs 3 --batch-size 4 --lr 2e-4 --device auto

# (d) evaluation with the real adapter
python3 -m src.validation.terminal_monitor_eval \
  --data-csv examples/llm/terminal_windows_example.csv \
  --adapter-dir /tmp/qwen_terminal_lora \
  --include-base-model \
  --output-json /tmp/terminal_eval.json
```

The example CSV has 18 windows (9 `split=train` + 9 `split=test`). Step (a)
exports only the 9 train rows, so step (d) evaluating on the same CSV scores the
9 test rows: the example is leak-free by construction. Replace it with your
own window data (long format:
`sample_id,step,accel_mps2,speed_mps,local_density,battery_pct,label[,split]`).

Leakage rules enforced by `prepare_terminal_sft`:

| Situation | Behaviour |
| :--- | :--- |
| CSV has a `split` column, `--split auto` (default) | exports only `split=train` |
| CSV has no `split` column, `--split auto` | **error**: add a `split` column, or pass `--split all` to declare the file training-only |
| `--split all` | exports everything (including any `test` rows) and prints a leakage warning; never evaluate on the same CSV |
| `--split test` | allowed for debugging, prints a warning that the evaluation is leaked |
| fixture (`--fixture-samples-per-class`) | training-only by construction: fixture seed vs `terminal_monitor_eval` fixture test seed `+101` |

Scheduler latency provenance in the evaluation report:

| `service_time_source` | Meaning |
| :--- | :--- |
| `model` | mean per-batch latency of the adapter/base-model inference in this run |
| `trace` | values read from a latency replay file (`--latency-replay-json`) |
| `fixture` | the replay file declares itself as example/fixture data |
| `default` | `--default-latency` (2.8 s nominal), when neither an LLM run nor a replay supplies a value |
