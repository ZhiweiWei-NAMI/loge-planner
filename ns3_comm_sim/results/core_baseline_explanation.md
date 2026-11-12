# Core Baseline Result Explanation

This file explains how to read the current `core_baseline` outputs under `ns3_comm_sim/results/` and what the present architecture comparison suggests.

## Where to look first

- `results/figures/core_baseline/figure_01_main_runtime_replanning.*`
  The main comparison figure. This should be the first figure to cite when discussing the architecture effect on runtime communication.

- `results/figures/core_baseline/figure_02_runtime_outcome_composition.*`
  Runtime anomaly outcome composition, separating network response, local handling, and timeout.

- `results/figures/core_baseline/figure_03_communication_qos_metrics.*`
  Control-plane communication QoS and link-support diagnostics, including packet delivery ratio, SNR, delay, and jitter.

- `results/parsed/core_baseline_summary.csv`
  The machine-readable summary used to generate the current figures.

## Main reading logic for a paper

A clean paper narrative can follow this order:

1. Show that the architecture difference matters most for runtime replanning, not for initial plan acquisition.
2. Show that `Cloud-Edge` reduces runtime delay relative to `Cloud`.
3. Show that `Cloud-Edge-Terminal` improves runtime success further by reducing the number of network-escalated anomaly events.
4. Use runtime outcome composition and control-plane QoS as supporting evidence rather than plotting low-information metrics with many zeros.

## Current headline findings

## 1. Runtime replanning is the main differentiator

From `core_baseline_summary.csv`:

- `Cloud`
  - `replan_success_rate = 0.2432`
  - `replan_mean_delay_ms = 95.86`
  - `replan_timeout_ratio = 0.7568`

- `Cloud-Edge`
  - `replan_success_rate = 0.2432`
  - `replan_mean_delay_ms = 31.89`
  - `replan_timeout_ratio = 0.7568`

- `Cloud-Edge-Terminal`
  - `replan_success_rate = 0.6351`
  - `replan_mean_delay_ms = 32.57`
  - `replan_timeout_ratio = 0.3649`
  - `replan_local_filtered_ratio = 0.4932`

Interpretation:

- Moving runtime replanning from `Cloud` to `Edge` sharply reduces runtime communication delay.
- Adding terminal-side filtering does not materially worsen the average runtime delay of completed replanning decisions.
- The main gain of `Cloud-Edge-Terminal` is reliability: many anomaly events no longer need to traverse the network path at all.

## 2. Why `local filtered ratio` should be kept only as part of a composition figure

`local filtered ratio` is zero for `Cloud` and `Cloud-Edge`, but non-zero for `Cloud-Edge-Terminal` because only that architecture has the terminal-side filtering mechanism.

It should not be plotted alone as a headline result, but it is still valuable because:

- it is the mechanism explanation for the higher runtime success of `Cloud-Edge-Terminal`
- it shows that the architecture gain is not just a cosmetic shift in average delay
- it explains why the network-required transaction count drops from `160` to `87`

Recommended use:

- keep it in a reliability/mechanism figure
- do not use it as the main result figure

## 3. Why many previous metrics were not worth plotting directly

The previous result package contained several low-value metrics for figure-making:

- `stale_ratio` was zero because transactions were typically either fast successes or hard timeouts.
- `retry_to_success` was near zero because successful cases mostly succeeded on the first viable attempt.
- the old aggregate `packet_delivery_ratio` was dominated by background contention flows and therefore not a good control-plane communication indicator.

These metrics can stay in the CSV for completeness, but they should not drive the core figure set.

## 4. How to interpret the delay figures

The visible figure labels use neutral wording such as `Average delay`.
Internally, the summary includes failure-inclusive delay accounting so that failed transactions do not disappear from the comparison.

Practical interpretation:

- use success rate as the first decision metric
- use average delay as the second decision metric
- use timeout and local filtering as supporting evidence for the communication mechanism

## Available communication metrics in the current result package

The current benchmark already exports several communication-layer indicators:

- `control_packet_delivery_ratio`
  Packet delivery ratio restricted to transaction-related control flows.

- `control_mean_flow_delay_ms`
  Mean end-to-end delay restricted to control-plane flows.

- `control_mean_flow_jitter_ms`
  Mean end-to-end jitter restricted to control-plane flows.

- `mean_snr_db`
  Mean received SNR estimated from monitor-sniffer samples.

These metrics are visible in:

- `results/parsed/core_baseline_summary.csv`
- `results/figures/core_baseline/figure_03_communication_qos_metrics.*`
- per-run raw outputs under `results/raw/core_baseline/*/flow_summary.csv`
- per-run raw outputs under `results/raw/core_baseline/*/snr_samples.csv`

## Recommended figure usage in writing

A reasonable sequence for the manuscript is:

- **Main figure**
  `figure_01_main_runtime_replanning`
  Use this figure to state the primary result: runtime replanning benefits from moving intelligence away from the cloud, and terminal-side filtering further improves feasibility.

- **Mechanism figure**
  `figure_02_runtime_outcome_composition`
  Use this to explain why `Cloud-Edge-Terminal` outperforms the others and why plotting `local filtered ratio` alone is unnecessary.

- **Diagnostic figure**
  `figure_03_communication_qos_metrics`
  Use this to support the communication interpretation with control-plane QoS evidence.

## Output checklist

The current result package is complete if the following paths exist:

- `results/parsed/core_baseline_summary.csv`
- `results/tables/core_baseline_summary.csv`
- `results/figures/core_baseline/figure_01_main_runtime_replanning.png`
- `results/figures/core_baseline/figure_02_runtime_outcome_composition.png`
- `results/figures/core_baseline/figure_03_communication_qos_metrics.png`
- `results/raw/core_baseline/core_baseline__cloud/`
- `results/raw/core_baseline/core_baseline__cloud_edge/`
- `results/raw/core_baseline/core_baseline__cloud_edge_terminal/`

## Modeling assumptions and limits of these numbers

Read every delay in this package with the following in mind.

**Processing delays are fixed constants, not measured inference latency.**

| Parameter | Value | Where it is set |
| --- | --- | --- |
| local processing | 8 ms | `ns3_comm_sim/ns3/uav_comm_benchmark.cc:1261` |
| edge processing | 20 ms | `ns3_comm_sim/ns3/uav_comm_benchmark.cc:1262` |
| cloud processing | 60 ms | `ns3_comm_sim/ns3/uav_comm_benchmark.cc:1263` |
| cloud backhaul | 12 ms | `ns3_comm_sim/configs/radio/wifi_5ghz_fixed.json` |

These constants stand in for "the decision is computed somewhere". They are **not**
timings of an LLM or of any other real planner: no language model runs inside this
benchmark, and the values are chosen assumptions, not measurements. The architecture
comparison therefore isolates *communication path* effects under a fixed compute
assumption; it cannot be read as evidence about LLM inference latency.

**The workload is synthetic.** Transactions are generated from
`configs/scenarios/core_baseline.json` (12 UAVs, 120 s, seed `20260413`) via
`scripts/run_core_baseline.py`. They are not captured from a production system.

**What is reproducible from the published files.** All delay/success/timeout metrics
can be regenerated from `results/raw/<run>/transactions.csv` +
`flow_summary.csv` with `analysis/parse_results.py`. `mean_snr_db` additionally
requires `snr_samples.csv`, which is **not** published (~9 MB per run), so the SNR
column cannot be re-derived from this repository alone. `results/figures/` is also
not published; figures must be regenerated locally.

**Provenance.** The committed outputs are the archived run of record. The runner and
all inputs (`ns3_comm_sim/inputs/`, configs, `run_manifest.json` per run) are
included, so the benchmark can be re-executed against an ns-3.40 build, but the
committed numbers were not re-executed for this release note.
