# Dynamic Causal Hypergraph DCH — Section 12 Evaluation Protocol Datasets Metrics and Ablations

Parent outline [DCH_TechSpec_Outline.md](../DCH_TechSpec_Outline.md)  
Cross references Section 1 [DCH_Section1_FormalFoundations.md](../sections/DCH_Section1_FormalFoundations.md), Section 2 [DCH_Section2_DHG_TCkNN.md](../sections/DCH_Section2_DHG_TCkNN.md), Section 4 [DCH_Section4_HyperpathEmbedding.md](../sections/DCH_Section4_HyperpathEmbedding.md), Section 5 [DCH_Section5_CreditAssignment.md](../sections/DCH_Section5_CreditAssignment.md), Section 6 [DCH_Section6_FSM.md](../sections/DCH_Section6_FSM.md), Section 7 [DCH_Section7_HierarchicalAbstraction.md](../sections/DCH_Section7_HierarchicalAbstraction.md), Section 10 [DCH_Section10_ComplexityResource.md](../sections/DCH_Section10_ComplexityResource.md), Section 11 [DCH_Section11_SoftwareBlueprint.md](../sections/DCH_Section11_SoftwareBlueprint.md)

Version v0.1

1. Objectives  
- Establish rigorous, reproducible evaluation of DCH on event vision tasks with emphasis on accuracy, sample efficiency, throughput, interpretability, and continual learning.  
- Compare against strong non DCH baselines and DCH ablations to isolate contributions of core mechanisms WL embeddings, traversal credit assignment, FSM, and HOEs.  
- Produce artifacts configs, logs, and rule libraries enabling exact replication.

2. Datasets and splits  
2.1 DVS Gesture  
- Source public DVS Gesture dataset tonic provides loader with user and gesture splits.  
- Standard split users 1 to 23 train, 24 to 29 test or k fold per literature; adopt standard to ease comparison.  
- Preprocessing keep native event stream resolution; quantize timestamps to 1 μs internal representation; no frame accumulation.  
- Labels gesture id associated with trial windows; provide supervision seeds by tagging output head vertices within ground truth windows see Section 5 credit assignment.

2.2 N MNIST  
- Source tonic N MNIST event dataset derived from MNIST with saccade patterns.  
- Train split 60k, test split 10k; use event stream directly.  
- Supervision per sample window; seed output spikes at sample end or sliding windows.

2.3 Stream construction and watermarking  
- For both datasets construct continuous event streams in sample order for streaming evaluation; optionally interleave samples for mini epochs.  
- Maintain watermark based ordering as in [DCH_Section11_SoftwareBlueprint.md](../sections/DCH_Section11_SoftwareBlueprint.md) to synchronize periodic modules.

2.4 Reproducibility controls  
- Global seed seeding PyTorch numpy and Python random.  
- Deterministic WL hashing functions; seeded traversal RNG keyed by cycle id.  
- Snapshot ids for WL SAGE and FSM rules included in logs and results.

3. Task formulations  
3.1 Classification protocol  
- Output head neurons one per class gesture or digit; decision rule integrate output spikes in a decision window W out per sample default full sample or last T tail ms and take argmax count; ties broken by earliest spike time.  
- Alternate probabilistic decoding low pass filter of spike trains with exponential kernel and argmax filtered rate.  
- Online setting compute cumulative decision at fixed cadence and measure latency to correct classification.

3.2 Continual learning protocol optional  
- Task sequence two tasks e.g., subset of gestures then full set or digits 0 to 4 then 5 to 9 with bounded memory.  
- Apply scaffolding policies from [DCH_Section8_TaskAwareScaffolding.md](../sections/DCH_Section8_TaskAwareScaffolding.md); measure forward and backward transfer and forgetting.

4. Baselines  
4.1 Surrogate gradient SNN  
- Norse based spiking network with surrogate gradient training for classification; tuned to dataset standard baselines; train with cross entropy on rate decoded outputs.  
- Report accuracy and training sample budget matched to DCH label exposure where possible.

4.2 Reservoir SNN LSM style  
- Fixed recurrent reservoir with linear readout trained by ridge regression on spike features rate or temporal basis.  
- Provides sample efficient baseline without end to end backprop.

4.3 TEH static variant  
- Static Temporal Event Hypergraph without dynamic construction and without FSM HOEs; rely on fixed precomputed edges from short windows; apply similar traversal updates.  
- Quantifies benefit of dynamic DHG and FSM abstraction.

4.4 Optional event frame CNN  
- Time surface or voxel grid CNN trained with SGD as a non spiking baseline to contextualize accuracy.

5. Metrics  
5.1 Accuracy and F1  
- Top 1 accuracy per sample; macro F1 across classes for imbalance sensitivity.  
- Online accuracy at specified latencies 50 ms, 100 ms after sample start.

5.2 Sample efficiency  
- Labeled samples required to reach accuracy thresholds e.g., 80 percent, 90 percent; area under accuracy vs labels curve.

5.3 Throughput and latency  
- End to end event lane latency and events per second sustained measured as in [DCH_Section10_ComplexityResource.md](../sections/DCH_Section10_ComplexityResource.md).  
- Traversal cycle latency per seed; FSM tick cost; percentage of event path budget consumed by WL.

5.4 Memory and footprint  
- Active edges size of E t, HOE count, embedding tables WL and SAGE memory.

5.5 Interpretability and rule quality  
- Audit success rate fraction of decisions where at least one valid hyperpath with score above τ audit explains the predicted class.  
- Rule stability fraction of promoted rules remaining active for T persist seconds.  
- Coverage fraction of seeds partially matching any active rule.  
- Path shortening factor average reduction in steps due to HOEs.

5.6 Continual learning  
- Forward transfer delta in sample efficiency for new task with reuse vs isolate policy.  
- Forgetting change in accuracy on prior tasks after learning new task.  
- Stability of reliability r e on frozen edges bounded drift.

6. Evaluation procedures  
6.1 Static task single dataset  
- Warm up calibration period to initialize buffers no labels.  
- Streaming train evaluate loop  
  - For each sample stream events to DCH; provide supervision seeds at label windows; record prediction per sample; update via traversal and plasticity cycles; FSM operates continuously.  
  - Log metrics every K samples and cadence windows.

6.2 Continual tasks  
- Train on Task A for T A seconds or samples then switch to Task B; apply scaffolding decision REUSE ISOLATE HYBRID; continue streaming.  
- Measure before after performance on A and B and compute transfer metrics.

6.3 Hyperparameter setting  
- Use defaults in [DCH_TechSpec_Outline.md](../DCH_TechSpec_Outline.md) and [DCH_Section1_FormalFoundations.md](../sections/DCH_Section1_FormalFoundations.md).  
- Limited sweep ±20 percent on key knobs alpha, τ prune, k max, Δ windows; log chosen config hash.

6.4 Latency and throughput measurement  
- Instrument event lane with histograms p50 p90 p99 per module; report averages and tail latencies.  
- Ensure measurements exclude I O by warm cache and prefetch.

6.5 Energy proxy optional  
- Report CPU utilization and estimated energy proxy proportional to CPU cycles; if available use RAPL counters; otherwise provide events per update and memory bandwidth as proxies.

7. Ablation studies  
- No FSM disable Section 6; measure accuracy, rule coverage, interpretability drop.  
- No HOE disable Section 7 path compression; measure traversal latency and path length changes.  
- No WL grouping disable Section 4 WL and LSH; rely solely on TC kNN.  
- No SAGE periodic remove refinement and measure FSM discovery stability.  
- No traversal credit assignment remove Section 5 and rely only on local watcher updates; measure learning quality.  
- k max variations 1,2,3; Δ windows narrower or wider; λ path set to 0 or 1 to isolate local vs path based learning.  
- Scaffolding off evaluate continual learning without FREEZE and regionization.

8. Statistical methodology  
- Run 5 seeds per setting; report mean ± std and 95 percent confidence intervals by bootstrap.  
- Use paired tests against closest ablation or baseline when applicable; report effect sizes Cohen s d.  
- Correct for multiple comparisons via Holm Bonferroni when testing many ablations.

9. Reporting and artifacts  
- Save per run  
  - Config YAML, commit hash, seeds, snapshot ids.  
  - Metrics CSV Parquet with timestamps and cadence.  
  - Rule descriptors active set JSON from FSM; HOE registry.  
  - Example audited paths for a random 1 percent sample of decisions.  
- Aggregate tables  
  - Accuracy and F1; sample efficiency; throughput and latency; interpretability and rule metrics; continual learning metrics.  
- Plots  
  - Accuracy over labels; latency histograms; rule promotions per minute; path length distributions; reliability trajectories.

10. Acceptance thresholds initial targets  
- DVS Gesture static task top 1 accuracy ≥ 90 percent with WL FSM HOE full DCH and ≥ 85 percent with no FSM ablation; throughput ≥ 5e5 events per second desktop target.  
- N MNIST top 1 accuracy ≥ 98 percent DCH and ≥ 96 percent without FSM; throughput ≥ 1e5 events per second.  
- Audit success rate ≥ 70 percent of correct decisions explained by a valid hyperpath with score ≥ 0.3.  
- Rule stability ≥ 60 percent of promoted rules remain active for ≥ 60 seconds.  
- Continual learning forgetting ≤ 5 percentage points under ISOLATE policy on prior task after learning new task.

11. Mermaid diagram — evaluation workflows

```mermaid
flowchart TB
DATA[Event dataset via tonic] --> STREAM[Stream construction and watermark]
STREAM --> DCH[DCH online loop]
DCH --> METRICS[Metrics logging]
DCH --> PRED[Per sample predictions]
PRED --> ACC[Accuracy and F1]
DCH --> RULES[Rules and HOEs]
RULES --> INT[Interpretability and stability]
METRICS --> THR[Throughput and latency]
INT --> REPORT[Aggregate tables and plots]
THR --> REPORT
ACC --> REPORT
```

12. Implementation hooks and harness references  
- Implement dataset adapters and run scripts in experiments per [DCH_Section11_SoftwareBlueprint.md](../sections/DCH_Section11_SoftwareBlueprint.md).  
- Provide assertions and invariants in tests for temporal validity and traversal constraints.  
- Export rule sets and HOEs for audit dashboards.

13. Risks and mitigations specific to evaluation  
- Label alignment jitter ensure supervision seeds align with measured spike times; apply tolerance band for seeds ±2 ms.  
- Class imbalance in DVS Gesture report macro F1 and per class accuracy.  
- Variance across seeds increase seeds if CI widths exceed 2 percentage points for main metrics.  
- Compute contention isolate event lane measurements by pinning threads or running offline replays.

14. Cross references  
- Complexity budgets and latency targets [DCH_Section10_ComplexityResource.md](../sections/DCH_Section10_ComplexityResource.md).  
- Module metrics and interfaces [DCH_Section9_Interfaces.md](../sections/DCH_Section9_Interfaces.md).  
- Scaffolding policies for continual learning [DCH_Section8_TaskAwareScaffolding.md](../sections/DCH_Section8_TaskAwareScaffolding.md).

End of Section 12