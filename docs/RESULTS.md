# Results and Reporting

This page provides templates for recording results after running experiments. Populate the tables using artifacts produced by the runner and benchmarks. Statistical significance tests are available in [dch_pipeline/stats.py](../dch_pipeline/stats.py).

Notes:
- Artifacts directory (default): `./artifacts/<exp>*/`
  - Metrics CSV: `metrics.csv`
  - JSONL logs: `metrics.jsonl`
  - Merged config: `config.merged.json`
- Benchmarks print a single JSON line to stdout for easy capture.

---

## DVS Gesture

Enter results aggregated over seeds/runs.

| Setting | Accuracy (%) | Macro F1 | Micro F1 | Latency (ms/step) | Throughput (events/s) | Notes |
|---:|---:|---:|---:|---:|---:|---|
| DCH-only (torch-free) |  |  |  |  |  |  |
| DCH + SNN (Norse LIF) |  |  |  |  |  |  |

Ablations:
- DHG params (k, combination_order_max, δ_causal)
- Traversal (horizon, beam_size)
- Plasticity (ema_alpha, decay_lambda, prune_threshold)
- FSM (theta, lambda_decay, hold_k, min_weight)
- Abstraction enabled vs disabled

Significance (paired tests over seeds):
- Use utilities from [dch_pipeline/stats.py](../dch_pipeline/stats.py): `paired_ttest`, `wilcoxon_signed_rank`, `cohens_d_paired`, `cliffs_delta`, and `benjamini_hochberg` for multiple comparisons.
- Recommended report: mean ± std across seeds; p-values (post-correction) per comparison.

---

## N-MNIST

Enter results aggregated over seeds/runs.

| Setting | Accuracy (%) | Macro F1 | Micro F1 | Latency (ms/step) | Throughput (events/s) | Notes |
|---:|---:|---:|---:|---:|---:|---|
| DCH-only (torch-free) |  |  |  |  |  |  |
| DCH + SNN (Norse LIF) |  |  |  |  |  |  |

Ablations and significance:
- Mirror DVS Gesture ablations; prefer identical seeds for paired testing.
- Apply multiple-hypothesis correction with `benjamini_hochberg` on the p-value list.

---

## How to populate

1) Run experiments
```bash
# Torch-free example (DCH-only)
dch-run experiment=dvs_gesture snn.enabled=false

# SNN path (requires torch + norse)
dch-run experiment=nmnist snn.enabled=true model=norse_lif
```

2) Collect artifacts
- Copy `metrics.csv` and `metrics.jsonl` from each run under `./artifacts/<exp>*/`.
- Record the corresponding `config.merged.json` for provenance.

3) Aggregate and test
- Use [dch_pipeline/stats.py](../dch_pipeline/stats.py) to aggregate metrics across seeds and compute significance.
- Report mean ± std and corrected p-values.

4) Document environment
- Record Python version, OS, and optional dependency versions (torch/norse/tonic) if used.
- Include seeds and any configuration overrides (dotlist) used for the run.
