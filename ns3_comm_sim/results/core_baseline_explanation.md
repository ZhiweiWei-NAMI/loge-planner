# Core Baseline Result Notes

This note documents the three-architecture communication baseline shipped in this
repository. It states the numbers held in
`ns3_comm_sim/results/parsed/core_baseline_summary.csv`, defines every column of that
CSV, and records the configured constants on which the reported delays depend. All
figures quoted below are copied from that CSV; nothing else is used as a source of
headline values.

## 1. Run definition

| Item | Value |
| --- | --- |
| Scenario | `core_baseline` (`ns3_comm_sim/configs/scenarios/core_baseline.json`: 12 UAVs, 2.4 km × 1.6 km area, 120 s, seed `20260413`) |
| Architectures | `cloud`, `cloud_edge`, `cloud_edge_terminal` (`ns3_comm_sim/configs/architectures/`) |
| Event stream | `ns3_comm_sim/inputs/events/core_baseline.csv` — the same 227 transactions in all three runs |
| Workload composition | 48 `plan` + 148 `replan` + 31 `decouple` |
| Runner | `ns3_comm_sim/scripts/run_core_baseline.py` |
| Parser | `ns3_comm_sim/analysis/parse_results.py` |
| Radio | `ns3_comm_sim/configs/radio/wifi_5ghz_fixed.json` |
| Region | `ns3_comm_sim/configs/regions/city_canyon.json` |
| Traffic profile | `ns3_comm_sim/configs/traffic/control_messages.json` |

The three runs share one event stream and one seed; only the endpoint that serves each
task type changes:

| Architecture | `plan_target` | `replan_target` | `decouple_target` |
| --- | --- | --- | --- |
| `cloud` | cloud | cloud | cloud |
| `cloud_edge` | edge | edge | cloud |
| `cloud_edge_terminal` | edge | local | cloud |

Because the transaction set, priorities, deadlines and random seed are identical
across the three runs, differences between columns are attributable to where the
decision is served and to the resulting load on the shared radio uplink.

## 2. Headline results

### Table A — workload and aggregate outcome

| Metric (CSV column) | Cloud | Cloud-Edge | Cloud-Edge-Terminal |
| --- | --- | --- | --- |
| `total_transactions` | 227 | 227 | 227 |
| `network_required_transactions` | 227 | 227 | 79 |
| `successful_transactions` | 47 | 50 | 164 |
| `local_execution_ratio` | 0.0000 | 0.0000 | 0.6520 (65.2 %) |
| `mean_delay_ms` | 110.0000 | 40.1200 | 14.5610 |
| `penalty_mean_delay_ms` | 270.9515 | 256.3524 | 124.6432 |
| `timeout_ratio` | 0.7753 | 0.7797 | 0.2775 |

`mean_delay_ms` and the other `*_mean_delay_ms` columns average **successful**
transactions only; the `penalty_*` columns average **all** transactions, substituting
`max(delay, deadline)` for every failed transaction (see §5.3). The two families are
therefore not interchangeable.

### Table B — results by task type

Success rate:

| Task type | Cloud | Cloud-Edge | Cloud-Edge-Terminal |
| --- | --- | --- | --- |
| `plan_success_rate` | 0.2292 (22.9 %) | 0.2292 (22.9 %) | 0.2292 (22.9 %) |
| `replan_success_rate` | 0.2027 (20.3 %) | 0.2365 (23.6 %) | 1.0000 (100.0 %) |
| `decouple_success_rate` | 0.1935 (19.4 %) | 0.1290 (12.9 %) | 0.1613 (16.1 %) |

Mean delay of successful transactions / penalized mean delay / timeout ratio:

| Task type | Cloud | Cloud-Edge | Cloud-Edge-Terminal |
| --- | --- | --- | --- |
| `plan_mean_delay_ms` | 122.0909 | 36.3636 | 37.2727 |
| `plan_penalty_mean_delay_ms` | 322.8542 | 303.2083 | 303.4167 |
| `plan_timeout_ratio` | 0.7708 | 0.7708 | 0.7708 |
| `replan_mean_delay_ms` | 101.7333 | 30.0000 | 8.0000 |
| `replan_penalty_mean_delay_ms` | 228.3378 | 207.4730 | 8.0000 |
| `replan_timeout_ratio` | 0.7703 | 0.7635 | 0.0000 |
| `decouple_mean_delay_ms` | 129.1667 | 139.0000 | 158.8000 |
| `decouple_penalty_mean_delay_ms` | 394.0323 | 417.1613 | 404.7097 |
| `decouple_timeout_ratio` | 0.8065 | 0.8710 | 0.8387 |

### Table C — flow- and link-level indicators

| Metric (CSV column) | Cloud | Cloud-Edge | Cloud-Edge-Terminal |
| --- | --- | --- | --- |
| `control_packet_delivery_ratio` | 0.2946 | 0.2907 | 0.2621 |
| `control_mean_flow_delay_ms` | 15.8750 | 21.3182 | 5.5455 |
| `control_mean_flow_jitter_ms` | 1.6250 | 6.7727 | 0.8636 |
| `packet_delivery_ratio` | 0.0027 | 0.0026 | 0.0014 |
| `mean_snr_db` | 17.5183 | 17.9850 | 16.4060 |

`packet_delivery_ratio` aggregates every monitored flow, including the background
stations defined by the scenario (`background_sta_count = 6`), and is dominated by
those flows; `control_packet_delivery_ratio` restricts the same ratio to the
transaction control flows on ports 9000/9100 and is the column that corresponds to
the transaction workload.

## 3. Replanning: the two accounting scopes

The 148 `replan` transactions are the same 148 events in all three runs, with the same
priorities and the same deadlines. What differs is the scope under which they are
accounted, and the two scopes answer different questions.

### 3.1 Scope 1 — local adaptation (terminal-resident replan)

In `cloud_edge_terminal`, `replan_target = local`: the replan request is resolved at
the UAV itself, so the transaction is recorded with `local_execution = 1` and
`network_required = 0`. Its critical path is the configured local service time of
8 ms and contains no radio hop.

| Quantity (Cloud-Edge-Terminal, `replan` rows) | Value |
| --- | --- |
| `replan_local_execution_ratio` | 1.0000 (148 / 148 transactions) |
| `replan_success_rate` | 1.0000 (148 / 148) |
| `replan_mean_delay_ms` | 8.0000 |
| `replan_p95_delay_ms` / `replan_p99_delay_ms` | 8.0000 / 8.0000 |
| `replan_penalty_mean_delay_ms` | 8.0000 |
| `replan_timeout_ratio` | 0.0000 |
| `replan_stale_ratio` | 0.0000 |

The nominal control-plane replan deadline is 180 ms, scaled by priority to 144 ms
(priority 1), 180 ms (priority 2) and 216 ms (priority 3) in the event stream. An 8 ms
local critical path lies inside every one of those deadlines, which is why the success
rate in this scope is 1.0 rather than a measured probability: it is the deterministic
consequence of a local service time well below the deadline.

### 3.2 Scope 2 — wireless escalation (network-served replan)

In `cloud` and `cloud_edge`, the same 148 replan transactions must traverse the
UAV-to-access-point uplink, so each one is a 1-request / 2-response control exchange
over the impaired channel, plus the service time of the target endpoint (20 ms at the
edge; 12 ms backhaul plus 60 ms at the cloud). All 148 rows carry `network_required = 1`
and `local_execution = 0`.

| Quantity (`replan` rows) | Cloud | Cloud-Edge |
| --- | --- | --- |
| `replan_local_execution_ratio` | 0.0000 | 0.0000 |
| `replan_success_rate` | 0.2027 (30 / 148) | 0.2365 (35 / 148) |
| `replan_mean_delay_ms` | 101.7333 | 30.0000 |
| `replan_p95_delay_ms` | 209.0000 | 35.3000 |
| `replan_penalty_mean_delay_ms` | 228.3378 | 207.4730 |
| `replan_timeout_ratio` | 0.7703 (114 / 148) | 0.7635 (113 / 148) |
| `replan_stale_ratio` | 0.0270 (4 / 148) | 0.0000 |

The penalized mean is 228.3 ms in `cloud` and 207.5 ms in `cloud_edge`. In both
architectures the dominant outcome is a timeout, and a timed-out replan records the
elapsed attempt budget (240 ms, 260 ms or 288 ms) which already exceeds the 144–216 ms
deadline; `max(delay, deadline)` therefore reduces to the recorded delay, and the
penalized mean tracks those timeout values. The four `cloud` rows that are neither
successes nor timeouts are `stale_response` completions (209 ms and 219 ms against
deadlines of 144 ms and 180 ms).

### 3.3 Relationship between the two scopes

The two success rates for `replan` measure different quantities and are not
comparable as if they were the same metric:

- **20.3 % / 23.6 % (Cloud, Cloud-Edge)** is the probability that a control-plane
  replan exchange completes over the impaired uplink inside a 144–216 ms deadline. It
  is a property of the communication path, not of the replanning decision.
- **100.0 % (Cloud-Edge-Terminal)** is the outcome of serving the same replan events
  at the terminal, where the critical path is 8 ms of configured local service and no
  radio hop is involved. It is a property of local execution only.

Both are correct for their scope. Stated together, the pair shows that, for this
workload, the replan success rate is governed by whether the replan request has to
leave the UAV, and that the 20.3 % / 23.6 % figures are not a capability ceiling of
the replanning procedure. Conversely, the 100 % figure does not imply
that a terminal-resident decision would succeed if it had to be escalated: in this
configuration `replan_target` is fixed to `local`
(`ns3_comm_sim/configs/architectures/cloud_edge_terminal.json`), so the benchmark does
not model a terminal that escalates a replan request to a remote endpoint.

The overall `local_execution_ratio` of 0.6520 for `Cloud-Edge-Terminal` is the same
statement at workload level: 148 of 227 transactions (65.2 %) are served locally, and
the remaining 79 (48 `plan` + 31 `decouple`) are the `network_required_transactions`
of that run.

## 4. Plan and decouple

**Plan.** `plan_success_rate` is 0.2292 (11 of 48) in all three architectures, although
the decision endpoint moves from cloud to edge. The 48 plan requests share the same
access hop in every architecture, and the failure mode is the retry budget: with a
120 ms per-attempt timeout and `max_retries = 2`, a request whose attempts are lost
records the full three-attempt budget (360 ms, 390 ms or 432 ms in
`transactions.csv`) against a 320 ms deadline. Consistent with this, the success count
is unchanged at 11 / 48, while the delay of the successful transactions does change
with the endpoint: 122.0909 ms (cloud) versus 36.3636 ms (edge) and 37.2727 ms
(Cloud-Edge-Terminal). The three penalized means are close to each other (322.8542 /
303.2083 / 303.4167) because 37 of the 48 plan transactions time out with the same
recorded attempt budgets in every run (19 at 360 ms, 11 at 390 ms, 7 at 432 ms), so the
mean is dominated by rows that are identical across architectures; the 11 successes are
what differs.

**Decouple.** `decouple` requests are cloud-targeted in all three architectures, so
this task type does not exercise the architecture difference in decision placement.
Its success rate is the lowest of the three task types in every run (0.1935 / 0.1290 /
0.1613) and its timeout ratio the highest (0.8065 / 0.8710 / 0.8387) against deadlines
of 416 ms and 520 ms. The residual differences between runs (6 / 31, 4 / 31, 5 / 31
successes) are not attributable to the decision endpoint, which is identical; they
reflect a difference in the loss realisation on the shared uplink, which a single
seeded run per architecture does not separate from architecture effects. With one
seeded run per architecture, no run-to-run variance estimate is available for this or
any other column.

## 5. Metric dictionary (`core_baseline_summary.csv`)

The CSV has one row per run (`run_name`), 54 columns, and is produced by
`ns3_comm_sim/analysis/parse_results.py` (`SUMMARY_FIELDS`, lines 19–74;
`summarize_run`, lines 155–276). Percentiles use linear interpolation between order
statistics (`percentile` in `ns3_comm_sim/scripts/common.py:56`). A statistic computed
over an empty row set is reported as `0.0`.

### 5.1 Identity and workload

| Column | Definition |
| --- | --- |
| `run_name` | Run identifier, `<scenario>__<architecture>` |
| `scenario` | Scenario name (`core_baseline`) |
| `architecture` | `cloud` \| `cloud_edge` \| `cloud_edge_terminal` |
| `total_transactions` | Number of rows in `transactions.csv` (227) |
| `network_required_transactions` | Rows with `network_required = 1`, i.e. transactions whose critical path includes the network |
| `successful_transactions` | Rows with `success = 1` |

### 5.2 Success rates

| Column | Definition |
| --- | --- |
| `plan_success_rate` | `success = 1` rows with `event_type = plan`, divided by all `plan` rows |
| `replan_success_rate` | Same ratio for `event_type = replan` |
| `decouple_success_rate` | Same ratio for `event_type = decouple` |

A row has `success = 1` when the response arrived and its critical-path delay did not
exceed the deadline. Rows that completed after the deadline are `stale = 1` and count
as failures; rows that exhausted their retries are `timeout = 1` and also count as
failures.

### 5.3 Delay statistics — success-only and penalized families

Every delay column is a statistic over `critical_path_delay_ms` from
`transactions.csv`. Two families exist for each scope:

| Column pattern | Population | Value used per transaction |
| --- | --- | --- |
| `{scope}_mean_delay_ms`, `{scope}_p95_delay_ms`, `{scope}_p99_delay_ms` | `success = 1` rows only | actual `critical_path_delay_ms` |
| `{scope}_penalty_mean_delay_ms`, `{scope}_penalty_p95_delay_ms`, `{scope}_penalty_p99_delay_ms` | all rows | actual delay if `success = 1`, otherwise `max(delay, deadline_ms)` |

`{scope}` is empty (all transactions), `plan`, `replan` or `decouple`. Because the
success-only family excludes failures, its mean can be far below the penalized mean —
for `replan` in `cloud`, 101.7333 ms versus 228.3378 ms. The success-only family
describes the latency of the decisions that were delivered; the penalized family is a
failure-inclusive accounting that keeps failed transactions in the average instead of
dropping them.

The penalized value is a floor substitution, not an imputed latency: a timed-out
`decouple` transaction that records 240 ms of elapsed attempt budget against a 416 ms
deadline contributes 416 ms, whereas a `plan` timeout that records 360 ms against a
320 ms deadline contributes 360 ms. Which of the two dominates therefore depends on
the task type: the `decouple` deadlines (416 ms and 520 ms) exceed the recorded attempt
budgets (240 ms, 260 ms, 288 ms), so its penalized mean is pinned to the deadline,
while the `plan` and `replan` recorded budgets already exceed their deadlines, so their
penalized means track the recorded delays.

### 5.4 Outcome and execution-mode ratios

| Column | Definition |
| --- | --- |
| `stale_ratio` | Fraction of all transactions with `stale = 1` (completed after the deadline) |
| `timeout_ratio` | Fraction of all transactions with `timeout = 1` (retry budget exhausted) |
| `local_execution_ratio` | Fraction of all transactions with `local_execution = 1` (resolved at the UAV) |
| `plan_stale_ratio`, `replan_stale_ratio`, `decouple_stale_ratio` | The `stale_ratio` statistic restricted to that `event_type` |
| `plan_timeout_ratio`, `replan_timeout_ratio`, `decouple_timeout_ratio` | The `timeout_ratio` statistic restricted to that `event_type` |
| `replan_local_execution_ratio` | Fraction of `replan` rows with `local_execution = 1` |

`stale`, `timeout`, `local_execution` and `network_required` are per-row 0/1 flags
exported by the ns-3 trace writer
(`ns3_comm_sim/ns3/uav_comm_benchmark.cc:1575`); the ratios are simple counts over
`transactions.csv` rows. `local_execution` and `network_required` are complements of
each other: every transaction is resolved either at the UAV or through the network.

### 5.5 Retry columns

| Column | Definition |
| --- | --- |
| `retry_to_success` | Mean `retry_count` over rows with `success = 1` **and** `network_required = 1` |
| `plan_retry_to_success`, `replan_retry_to_success`, `decouple_retry_to_success` | The same mean restricted to that `event_type` |

Locally executed transactions never retry (all 148 locally executed `replan` rows in
`Cloud-Edge-Terminal` carry `retry_count = 0`), so `replan_retry_to_success` is 0.0000
in that run by construction. In the network-served runs the mean over successful
transactions is low (0.1000 for `cloud`, 0.0286 for `cloud_edge`) because most
transactions that lose an attempt exhaust their retries instead of succeeding on a
later one.

### 5.6 Flow- and link-level columns

Computed from `flow_summary.csv` (ns-3 FlowMonitor export) and `snr_samples.csv`
(monitor-sniffer samples) of each run.

| Column | Definition |
| --- | --- |
| `packet_delivery_ratio` | Σ `rx_packets` / Σ `tx_packets` over **all** monitored flows |
| `control_packet_delivery_ratio` | The same ratio restricted to flows whose source or destination port is 9000 or 9100 (`CONTROL_PORTS`, `parse_results.py:15`) |
| `control_mean_flow_delay_ms` | Mean of `mean_delay_ms` over control flows that received at least one packet |
| `control_mean_flow_jitter_ms` | Mean of `mean_jitter_ms` over control flows that received more than one packet |
| `mean_flow_delay_ms` | Mean of `mean_delay_ms` over all flows that received at least one packet |
| `mean_flow_jitter_ms` | Mean of `mean_jitter_ms` over all flows that received more than one packet |
| `mean_snr_db` | Mean of `snr_db` over all sniffer samples in `snr_samples.csv` |

The flow-level delay and jitter columns average per-flow means (one value per flow),
whereas the transaction-level columns average per-transaction critical paths. They
describe different populations and are not expected to agree. `mean_snr_db` depends on
`snr_samples.csv`, which is not part of the published tree (§7).

## 6. Configured endpoint service times and accounting conventions

The endpoint service times are **configured constants** of the benchmark, not measured
inference latency. No language model or other external planner executes inside the
simulation; each request is answered after a fixed processing delay that stands in for
"the decision is computed at this endpoint".

| Parameter | Value | Where it is set |
| --- | --- | --- |
| Local (terminal) processing | 8 ms | `ns3_comm_sim/ns3/uav_comm_benchmark.cc:1261`, `ns3_comm_sim/configs/traffic/control_messages.json` (`processing_ms.local`) |
| Edge processing | 20 ms | `ns3_comm_sim/ns3/uav_comm_benchmark.cc:1262`, `processing_ms.edge` |
| Cloud processing | 60 ms | `ns3_comm_sim/ns3/uav_comm_benchmark.cc:1263`, `processing_ms.cloud` |
| Cloud backhaul delay | 12 ms | `ns3_comm_sim/ns3/uav_comm_benchmark.cc:1253`, `ns3_comm_sim/configs/radio/wifi_5ghz_fixed.json` (`cloud_backhaul_delay_ms`) |
| Per-attempt timeout | 120 ms | `ns3_comm_sim/configs/traffic/control_messages.json` (`attempt_timeout_ms`) |

Task deadlines and retry limits, from
`ns3_comm_sim/configs/traffic/control_messages.json` (`task_profiles`), with the
priority scaling applied by `ns3_comm_sim/scripts/generate_event_stream.py:23-28`
(priority 1 → 0.8×, priority 2 → 1.0×, priority 3 → 1.2×):

| Task type | Base deadline | Realised deadlines | Request / response packets | Max retries |
| --- | --- | --- | --- | --- |
| `plan` | 320 ms | 320 ms | 2 / 3 | 2 |
| `replan` | 180 ms | 144 / 180 / 216 ms | 1 / 2 | 2 |
| `decouple` | 520 ms | 416 / 520 ms | 3 / 4 | 1 |

Accounting conventions that follow from these settings:

- A transaction's critical path includes the endpoint service time, the radio exchange
  on the UAV–AP hop and, where the target is the cloud, the 12 ms backhaul. A locally
  executed transaction has no radio or backhaul component, so its critical path is the
  8 ms local service time.
- The architecture comparison isolates the effect of the communication path under a
  fixed compute assumption: the service times do not vary across architectures, so
  every difference between columns traces back to the path, to the endpoint selection,
  or to the load that path carries.
- Delays are deterministic endpoint service times plus simulated radio and queueing
  delays. They are not measurements of any real planner's latency, and they should not
  be extrapolated to one.

## 7. Data availability and reproduction

**Published in this repository** (tracked under `ns3_comm_sim/results/`):

| Path | Content |
| --- | --- |
| `ns3_comm_sim/results/parsed/core_baseline_summary.csv` | The summary table quoted in §2 |
| `ns3_comm_sim/results/parsed/core_baseline__cloud.csv` | Per-run summary, one row |
| `ns3_comm_sim/results/parsed/core_baseline__cloud_edge.csv` | Per-run summary, one row |
| `ns3_comm_sim/results/parsed/core_baseline__cloud_edge_terminal.csv` | Per-run summary, one row |
| `ns3_comm_sim/results/tables/core_baseline_summary.csv` | Byte-identical copy of the parsed summary |
| `ns3_comm_sim/results/raw/core_baseline/<run>/transactions.csv` | Per-transaction trace: the source of every success, delay, timeout and stale figure |
| `ns3_comm_sim/results/raw/core_baseline/<run>/flow_summary.csv` | FlowMonitor export: the source of the flow-level columns |
| `ns3_comm_sim/results/raw/core_baseline/<run>/run_manifest.json` | Config and input paths of the run |

The transaction- and flow-level columns can be regenerated from the published files:

```
python ns3_comm_sim/analysis/parse_results.py \
  --raw-dir ns3_comm_sim/results/raw/core_baseline/core_baseline__cloud \
  --output-csv /tmp/core_baseline__cloud.csv
```

Run against the archived raw directory, the parser reproduces the published summary row
column for column; the `mean_snr_db` column is reproduced only where the local
`snr_samples.csv` is present.

**Not published:**

- `snr_samples.csv` (~4 MB per run, excluded by `.gitignore:56`). `mean_snr_db`
  depends on it and therefore cannot be recomputed from the published tree; the
  published value is the archived one.
- `ns3_comm_sim/results/figures/` (excluded by `.gitignore:46`). Figures are not part
  of the repository contents and are produced locally from
  `core_baseline_summary.csv`.

**Full re-execution** requires an ns-3.40 build:

```
python ns3_comm_sim/scripts/run_core_baseline.py --ns3-root <path-to-ns-3.40>
```

which re-runs the three architectures and rewrites
`results/parsed/core_baseline_summary.csv` and
`results/tables/core_baseline_summary.csv`. The committed outputs are the archived run
of record; they were not re-executed for this note.

**Scope of the evidence.** One seeded run per architecture, 227 transactions per run,
three architectures. No repeated runs, no confidence intervals and no sensitivity
sweep are available from this package, so differences of a few transactions between
runs (for example the `decouple` counts in §4) are not resolved as architecture
effects.
