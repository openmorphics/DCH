# Dynamic Causal Hypergraph DCH — Section 13 Parameter Defaults and Tuning Strategy

Parent outline [DCH_TechSpec_Outline.md](../DCH_TechSpec_Outline.md)  
Cross references Section 1 [DCH_Section1_FormalFoundations.md](../sections/DCH_Section1_FormalFoundations.md), Section 2 [DCH_Section2_DHG_TCkNN.md](../sections/DCH_Section2_DHG_TCkNN.md), Section 3 [DCH_Section3_Plasticity.md](../sections/DCH_Section3_Plasticity.md), Section 4 [DCH_Section4_HyperpathEmbedding.md](../sections/DCH_Section4_HyperpathEmbedding.md), Section 5 [DCH_Section5_CreditAssignment.md](../sections/DCH_Section5_CreditAssignment.md), Section 6 [DCH_Section6_FSM.md](../sections/DCH_Section6_FSM.md), Section 8 [DCH_Section8_TaskAwareScaffolding.md](../sections/DCH_Section8_TaskAwareScaffolding.md), Section 10 [DCH_Section10_ComplexityResource.md](../sections/DCH_Section10_ComplexityResource.md), Section 11 [DCH_Section11_SoftwareBlueprint.md](../sections/DCH_Section11_SoftwareBlueprint.md)

Version v0.1

1. Purpose and scope  
- Consolidate canonical defaults for all major DCH modules.  
- Provide a practical tuning methodology with safe ranges, sensitivity guidance, and online adaptation policies.  
- Deliver dataset specific presets for DVS Gesture and N MNIST consistent with complexity and throughput targets.

2. Canonical defaults by module  
2.1 Time and windows  
- Timestamps unit μs; refractory ρ 1000 μs per neuron.  
- Delay window Δ_min 1000 μs, Δ_max 30000 μs; δ_causal 2000 μs.

2.2 DHG TC kNN (Section 2)  
- k_max 3; M_in 6 admitted unary candidates per head prior to combinations; C_cap 10 total candidates per head after scoring.  
- Candidate scoring weights w_delay 0.7, w_size 0.3; ε_init 0.05 reliability.

2.3 Plasticity (Section 3)  
- EMA step α 0.1; bounds r_min 0.02, r_max 0.98; prior r_0 0.5; discounted counts decay γ_c 0.98/s; time decay β 0.01/s.  
- Prune τ_prune 0.02; τ_age_min 2 s inactivity; τ_use_min 3; H_idle 30 s.  
- λ_path 0.5 blending path signals with local watchers.

2.4 Hyperpath embeddings (Section 4)  
- WL r_WL 2; d_WL 64; cadence Δt_WL 10 ms; frontier cap F_max 256.  
- LSH bands 8, rows_per_band 4; τ_LSH 2; cosine threshold τ_cos 0.65.  
- SAGE r_SAGE 3; d_SAGE 128; cadence Δt_SAGE 500 ms; batch 4096; S_nbr 32; S_edge 64.

2.5 Credit assignment (Section 5)  
- Seeds M 8; depth L 12; branch cap B 4; C_in 16 admissible in filter; τ_select 0.7; H_back 100 ms.  
- Feature weights w_rel 1.0, w_rec 0.5, w_rule 0.5, w_sim 0.5; λ_rec 1/100 ms; ε_norm 1.0.

2.6 FSM and rule induction (Section 6)  
- Window W 60 s; decay γ 0.98/s; s_min 50; r_min_rule 0.6; c_min 10 distinct sinks; L_max_rule 6; k_max_rule 3.  
- HH capacity K_HH 100k; CMS width 32768, depth 4; EWMA α_r 0.1; hysteresis D_stab 5 s; s_min_demote 0.5×s_min; D_demote 5 s; s_path_min 0.2.

2.7 Task aware scaffolding (Section 8)  
- Similarity thresholds θ_reuse 0.6; θ_isolate 0.3; weights w_act 0.5, w_emb 0.4, w_perf 0.1; r_act 0.7.  
- Freeze K_protect_pct 5%; K_protect_max 50k; TTL 120 s; α_review 0.02; τ_conflict 0.02.  
- Region leakage ε_leak 0.1; λ_reg 0.5; σ_max 0.6; k_alpha_up 1.5; K_head_boost 1.2; r_gate 0.8.

3. Tuning methodology and order of operations  
3.1 Fix timing and admission first  
- Validate refractory ρ and Δ windows against observed empirical delays (quantile check p10 ≥ Δ_min, p90 ≤ Δ_max).  
- Calibrate δ_causal to cluster true presyn bursts (1–3 ms typical); widen only if WL grouping is disabled.

3.2 Control combinatorics  
- Set DHG budgets C_cap and k_max; if candidate explosion or queue lag, reduce C_cap first, then k_max to 2.  
- Raise τ_prune temporarily to cull low r_e edges during burn in.

3.3 Stabilize learning signals  
- Tune α within 0.05–0.2 (higher for nonstationary streams).  
- Use λ_path sweep {0.3, 0.5, 0.7} to balance local vs. path evidence; monitor watcher confirm/miss ratio.

3.4 Traverse efficiently  
- Start with M 8, L 12, B 4; if traversal latency > budget, reduce L to 10, then C_in to 12; maintain determinism via fixed RNG seeding.  
- Increase w_rule to 0.8 when an adequate ruleset is active to improve search efficiency.

3.5 FSM sensitivity  
- Increase s_min to 75 on high churn datasets; decrease to 40 for sparse labels.  
- Raise γ to 0.995 (slower decay) when patterns should persist longer; lower to 0.95 to adapt rapidly.

3.6 Scaffolding gates  
- Calibrate θ_reuse/θ_isolate using a short pilot on Task A then Task B; target hybrid band width 0.1–0.2 to avoid oscillation.  
- Set region leakage ε_leak ≤ 0.1 to limit spurious cross links; relax when transfer is desired.

4. Safe ranges and sensitivity notes  
- Δ_min [0.5, 2] ms; Δ_max [20, 50] ms; widening increases false positives in DHG; prefer WL grouping instead.  
- α in [0.05, 0.2]; too high causes volatility; too low slows adaptation.  
- τ_prune in [0.01, 0.05]; below 0.01 increases memory; above 0.05 may prune useful edges.  
- M∈[4,12], L∈[8,16], B∈[2,6]; traversal cost scales roughly linearly in K·L·C_in (Section 10).  
- s_min∈[30,100]; lower values promote noisy rules; higher values slow rule discovery.  
- θ_reuse∈[0.5,0.7], θ_isolate∈[0.2,0.4]; too close induces policy oscillation; enforce hysteresis 0.05.

5. Dataset specific presets  
5.1 N MNIST  
- Δ_min 1 ms, Δ_max 25 ms, δ_causal 1.5 ms; k_max 2; α 0.08; τ_prune 0.02.  
- WL r 2, d 64, Δt_WL 10 ms; M 6, L 10, B 3; s_min 40; γ 0.98; θ_reuse 0.6, θ_isolate 0.3.

5.2 DVS Gesture  
- Δ_min 1 ms, Δ_max 30 ms, δ_causal 2 ms; k_max 3; α 0.1; τ_prune 0.02.  
- WL r 2, d 64, Δt_WL 10 ms; M 8, L 12, B 4; s_min 60–75; γ 0.98; θ_reuse 0.6, θ_isolate 0.3.  
- Under high churn raise s_min to 75 and increase Δt_WL to 15 ms to cap load.

6. Online adaptation policies (Meta controller)  
- Queue lag high: reduce C_cap by 20%, increase τ_prune by 25%, increase Δt_WL by 5 ms.  
- Rule churn high: raise s_min by 25%, increase γ by 0.01 absolute; lower w_rule temporarily.  
- Low traversal yield: increase w_rule to 0.8 and w_sim to 0.7; enable HOE preference; consider raising H_back to 120 ms briefly.  
- Memory pressure: increase τ_prune; lower K_head budget; demote low freshness edges using s_prune = r_e·exp(−λ_age·age).

7. Diagnostics and invariants to watch  
- DHG candidate hit rate target ≥ 30%; dedup rate ≥ 20%; rising queues → reduce C_cap.  
- Watcher confirm rate: aim 55–75% in stationary segments; sustained < 40% → shrink Δ windows or increase τ_prune.  
- Traversal valid ratio ≥ 50%; average depth 6–9; if lower, raise w_rule or adjust L.  
- FSM promotion precision proxy ≥ 70% (rules remain active ≥ 60 s).  
- HOE path shortening factor ≥ 1.5× on DVS Gesture after warm-up.

8. Auto calibration routines  
- Delay envelope fitting: per pair i→j maintain exponential histograms; update Δ_min/Δ_max to [p10, p95] with hysteresis to avoid thrash.  
- Temperature τ_select annealing: start at 0.9 for exploration; decay to 0.7 over 60 s of stable performance.  
- λ_path scheduling: increase toward 0.7 when supervised signals confirm traversal correctness; fall back to 0.5 otherwise.

9. Reproducibility and config hashing  
- Emit a config hash derived from a stable serialization of all knobs (ordered YAML) plus code revisions; attach to every metrics bundle.  
- Record snapshot ids for WL, SAGE, and FSM rulesets (see [DCH_Section9_Interfaces.md](../sections/DCH_Section9_Interfaces.md)).

10. Acceptance criteria for Section 13  
- Canonical defaults listed for all modules and consistent with earlier sections.  
- Clear order of tuning with safe ranges and sensitivity notes.  
- Dataset presets for DVS Gesture and N MNIST are provided.  
- Online adaptation rules defined for meta control and backpressure.  
- Diagnostics and invariants enable guardrail checks during runs.

11. Cross references  
- Windows and refractory semantics [DCH_Section1_FormalFoundations.md](../sections/DCH_Section1_FormalFoundations.md)  
- DHG budgets and scoring [DCH_Section2_DHG_TCkNN.md](../sections/DCH_Section2_DHG_TCkNN.md)  
- EMA and pruning [DCH_Section3_Plasticity.md](../sections/DCH_Section3_Plasticity.md)  
- Embeddings and grouping [DCH_Section4_HyperpathEmbedding.md](../sections/DCH_Section4_HyperpathEmbedding.md)  
- Traversal policy [DCH_Section5_CreditAssignment.md](../sections/DCH_Section5_CreditAssignment.md)  
- FSM thresholds [DCH_Section6_FSM.md](../sections/DCH_Section6_FSM.md)  
- Scaffolding decisions [DCH_Section8_TaskAwareScaffolding.md](../sections/DCH_Section8_TaskAwareScaffolding.md)  
- Performance budgets [DCH_Section10_ComplexityResource.md](../sections/DCH_Section10_ComplexityResource.md)

End of Section 13