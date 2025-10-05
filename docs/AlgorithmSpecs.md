# Algorithm Specifications and Complexity — Dynamic Causal Hypergraph (DCH)

Status: Draft v0.1  
Date: 2025-10-04  
Owners: DCH Maintainers

Authoritative interfaces
- Core typed contracts: [dch_core/interfaces.py](dch_core/interfaces.py)
- Planned implementations:
  - Dynamic Hypergraph Construction (TC‑kNN): [dch_core/dhg.py](dch_core/dhg.py)
  - Constrained Backward Traversal & Credit: [dch_core/traversal.py](dch_core/traversal.py)
  - Plasticity & Pruning: [dch_core/plasticity.py](dch_core/plasticity.py)
  - Streaming FSM: [dch_core/fsm.py](dch_core/fsm.py)
  - Hierarchical Abstraction (HOEs): [dch_core/abstraction.py](dch_core/abstraction.py)

Supporting documents
- Interface contracts and invariants: [docs/Interfaces.md](docs/Interfaces.md)
- Evaluation protocol (datasets, stats, CV): [docs/EVALUATION_PROTOCOL.md](docs/EVALUATION_PROTOCOL.md)
- Framework decision record: [docs/FrameworkDecision.md](docs/FrameworkDecision.md)
- Paper packaging & reproducibility: [docs/PaperPackaging.md](docs/PaperPackaging.md)

Notation
- V(t): vertices (materialized spike events) up to time t; E(t): directed hyperedges as causal hypotheses.
- Vertex v = (neuron_id, timestamp), Hyperedge e = (Tail, Head) with reliability r ∈ [0,1], temporal window [Δmin, Δmax], refractory ρ.
- B‑connectivity: an edge is traversable only if all Tail vertices are evidenced.

-------------------------------------------------------------------------------

1) Dynamic Hypergraph Construction via Temporal‑Causal kNN (TC‑kNN)

Inputs
- Hypergraph store H implementing HypergraphOps, connectivity oracle C implementing GraphConnectivity
- Head vertex v_post = (n_j, t_j)
- Window W = [t_j − Δmax, t_j − Δmin], presyn search budget k per presyn neuron
- combination_order_max ∈ {1..m}, causal_coincidence_delta (grouping proximity), budget_per_head
- init_reliability r0, refractory ρ

Output
- Candidate set of hyperedges to admit into E(t)

Pseudocode
```
procedure TCkNN_Candidates(H, C, v_post, W, k, combination_order_max, δ_causal, budget_per_head, r0, ρ):
    presyn_set ← C.presyn_sources(v_post.neuron_id)
    S ← ∅
    for n_i in presyn_set:
        # Retrieve recent spikes for presyn neuron within window W
        S_i ← top_k_recent_spikes(H, neuron=n_i, window=W, k=k)   # by time, most recent first
        S ← S ∪ S_i

    # Unary candidates
    Cand ← ∅
    for s ∈ S:
        e ← make_unary_edge(tail={s}, head=v_post, Δmin, Δmax, ρ, r0)
        if temporal_admissible(e) and refractory_ok(H, e):
            Cand.add(e)

    # Higher-order candidates: group spikes within δ_causal
    Groups ← cluster_by_time_proximity(S, δ_causal)  # small clusters of spikes close in time
    for G ∈ Groups:
        for m in 2..min(|G|, combination_order_max):
            for tail_subset in choose(G, m):
                e ← make_edge(tail=tail_subset, head=v_post, Δmin, Δmax, ρ, r0)
                if temporal_admissible(e) and refractory_ok(H, e):
                    Cand.add(e)

    Cand ← deduplicate_by_key(Cand, key=(head, sorted(tail)))
    Cand ← enforce_budget(Cand, budget_per_head, tie_break=reliability_then_recency)

    return Cand
```

Complexity
- Let d = |presyn_set|, k = per‑presyn history bound. top_k_recent_spikes per presyn is O(log N_i + k) with time index; total O(d·(log N̄ + k)).
- Grouping within δ_causal: if cluster size g ≪ S, enumerating combinations up to m yields O(Σ_g Σ_{m≤m_max} C(g, m)). Bound with m_max small (e.g., 2–3) to avoid explosion.
- Dedup and budget: O(|Cand| log |Cand|). Overall expected O(d·k + |Cand| log |Cand|), dominated by candidate enumeration for large g without budgets.

Memory
- Candidates transient O(|Cand|); persistent edge storage grows with admitted edges bounded by per‑head budgets.

Invariants and guards
- Temporal admissibility: ∀tail t_i, t_j − Δmax ≤ t_i ≤ t_j − Δmin
- Refractory: enforce spacing between heads for same postsyn neuron (ρ)
- Dedup key uniqueness on (head, sorted(tail))

-------------------------------------------------------------------------------

2) Constrained Backward Hyperpath Traversal (Randomized Beam with AND‑Frontier)

Goal
- Enumerate valid backward hyperpaths ending at a target vertex, respecting B‑connectivity and temporal logic, to support credit assignment.

Inputs
- Hypergraph H, target vertex v*, horizon T_h, beam_size B
- RNG policy π for randomized branching; refractory enforcement flag

Output
- Set of hyperpaths P with scores

Pseudocode
```
procedure BackwardTraverse(H, v*, T_h, B, π, refractory_enforce=True):
    beam ← { (head=v*, edges=[], score=1.0, t_head=timestamp(v*)) }
    results ← ∅
    while beam not empty:
        next_beam ← ∅
        for path in beam:
            incoming_edges ← admissible_incoming_edges(H, head=path.head, t_head=path.t_head,
                                                       refractory=refractory_enforce)
            if incoming_edges is empty:
                results.add(path_as_hyperpath(path))
                continue

            # B-connectivity: expand only edges whose entire tail is evidenced within horizon
            candidates ← []
            for e in incoming_edges:
                tail_vertices ← vertices(H, e.tail)
                if min(path.t_head - t(v) for v in tail_vertices) > T_h: 
                    continue  # beyond horizon
                if all_temporal_ok(e, tail_vertices, head_time=path.t_head):
                    candidates.append(e)

            # Stochastic beam expansion
            scored ← [(e, edge_score(e)) for e in candidates]
            probs ← softmax_over_scores(scored, τ)  # temperature τ
            picks ← sample_top_k_or_weighted(scored, probs, k=B, policy=π)

            for e in picks:
                new_path ← path.extend(e)
                new_path.score ← compose_scores(path.score, e.reliability, penalty=len(new_path.edges))
                new_path.head ← choose_tail_frontier(e)  # AND-frontier expansion over tail
                next_beam.add(new_path)

        beam ← top_k_by_score(next_beam, k=B)

    return finalize(results)  # dedup by canonical label, compute final scores
```

Scoring
- edge_score(e): reliability(e) (optionally penalize by tail size)
- compose_scores: product or min across path edges, with length penalty λ^L

Complexity
- Let b = average admissible in‑degree at nodes, B = beam size, D = effective depth bounded by T_h.
- Per level expansion O(B · b) with scoring; sampling/top‑k O(B·b log(B·b)). Total O(D · B · b log(B·b)).
- Memory O(D · B) for frontier.

Correctness guards
- B‑connectivity: traverse only when all tail events are present and within temporal bounds
- Refractory and horizon enforced at each step
- Canonicalization ensures isomorphic hyperpaths collapse to identical labels

-------------------------------------------------------------------------------

3) Temporal Credit Assignment and Evidence Aggregation

Goal
- Update hyperedge reliability using discrete, evidence‑based signals from valid hyperpaths spawned by supervised outcomes, errors, or rewards.

Inputs
- Hyperpaths set P (from traversal), sign ∈ {+1, −1}, now_t, PlasticityState ψ = (α, clamp, decay, freeze, prune_threshold)

Pseudocode
```
procedure UpdateReliability(H, P, sign, now_t, ψ):
    if ψ.freeze: return {}
    contrib: Map[EdgeId -> float] ← {}
    for p in P:
        w_p ← path_score(p)           # product/min with penalties, normalized
        for e in p.edges:
            contrib[e.id] ← contrib.get(e.id, 0.0) + w_p

    updates ← {}
    for (e_id, s) in contrib.items():
        e ← H.get_edge(e_id)
        if e is None: continue
        s_norm ← s / (1e-8 + sum(contrib.values()))   # normalize evidence
        r_old ← e.reliability
        r_tgt ← clip(r_old + sign * s_norm, 0.0, 1.0)
        r_new ← ψ.clamp( (1 - ψ.ema_alpha) * r_old + ψ.ema_alpha * r_tgt )
        e.reliability ← r_new
        if sign > 0: e.counts_success += 1
        else:        e.counts_miss += 1
        e.last_update_t ← now_t
        updates[e_id] = r_new

    # Optional time decay
    if ψ.decay_lambda > 0:
        for e in H.hyperedges():          # iterator over edges
            e.reliability ← ψ.clamp(e.reliability * (1 - ψ.decay_lambda))

    return updates
```

Pruning
```
procedure Prune(H, now_t, ψ):
    removed ← 0
    for e in H.hyperedges():
        if e.reliability < ψ.prune_threshold:
            H.remove_edge(e.id)
            removed += 1
    return removed
```

Complexity
- Let E_active be edges touched by P; update is O(Σ_p |p|) to aggregate + O(|E_active|) to write.
- Optional decay O(|E|). Pruning O(|E|) per sweep; amortize via periodic housekeeping.

-------------------------------------------------------------------------------

4) Adaptive Thresholding and Budget Policies

- Reliability clamp: r ∈ [r_min, r_max] prevents saturation and preserves learning headroom.
- Prune threshold: dynamic schedule possible (e.g., cosine or piecewise constant); couple to budget pressure.
- Candidate budgets: per‑head or per‑neuron caps to bound growth of E(t).
- Freeze: task‑aware scaffolding may set ψ.freeze = True for protected subgraphs.

Policy sketch
```
if memory_pressure_high() or growth_rate_exceeds(target):
    ψ.prune_threshold ← min(ψ.prune_threshold + δ, r_max - ε)
if validation_underfit():
    ψ.prune_threshold ← max(ψ.prune_threshold - δ, r_min + ε)
```

-------------------------------------------------------------------------------

5) Streaming Frequent Hyperpath Mining (FSM) and Promotion

Goal
- Detect recurring, high‑reliability hyperpaths online; promote to symbolic rules and optionally to higher‑order hyperedges.

Inputs
- Hyperpaths with canonical labels ℓ(p), weights w_p, sliding window W_fsm, thresholds θ_promote with hysteresis

Pseudocode
```
procedure FSM_Observe(stream P_t, now_t):
    for p in P_t:
        ℓ ← canonical_label(p)
        if ℓ not in counts: counts[ℓ] ← 0
        counts[ℓ] ← decay(counts[ℓ], now_t) + weight(p)
        if counts[ℓ] ≥ θ_promote and stable_above_threshold(ℓ):
            promotions.add(ℓ)
    return promotions
```

Canonical labeling
- Deterministic serialization of DAG‑like hyperpaths: order edges by topological + lexical order on (head, sorted(tail)), include timing bins if needed.
- Collision‑resistant hashing optional for compactness.

Complexity
- Per‑hyperpath O(L) to derive label (L edges), O(1) average map update; total O(|P_t|·L).
- Sliding window decay via timestamps (lazy) O(1) per touch.

-------------------------------------------------------------------------------

6) Hierarchical Abstraction: Higher‑Order Hyperedges (HOEs)

Goal
- Compress frequently validated causal chains into single high‑level hyperedges to form structural shortcuts and improve search.

Pseudocode
```
procedure PromoteToHOE(H, ℓ):
    p_template ← reconstruct_template(ℓ)
    sources ← source_vertices_of(p_template)
    sink ← sink_vertex_of(p_template)
    e_hoe ← Hyperedge(tail=sources, head=sink, Δmin', Δmax', ρ', reliability=r_init_high,
                      provenance=ℓ, budget_class="HOE")
    if guard_acyclic(H, e_hoe) and not duplicate(H, e_hoe):
        H.insert_hyperedges([e_hoe])
        return e_hoe.id
    return None
```

Guards
- Acyclicity: forbid cycles under temporal constraints
- Duplication: reject if identical (head, sorted(tail)) already exists with similar temporal params

Complexity
- Template reconstruction depends on label size; insertion O(log |E|) with indexed structures.

-------------------------------------------------------------------------------

7) Data Structures and Indices

Recommended
- Time‑indexed per‑neuron ring buffers for fast window queries (O(1) amortized append, O(log N_i + k) query).
- Packed Memory Array (PMA) or gap‑buffered arrays for dynamic edge sets with batched updates.
- Hashmaps for incoming/outgoing adjacency: VertexId → {EdgeId}
- Optional Count‑Min Sketch for FSM heavy‑hitters under memory bounds.

-------------------------------------------------------------------------------

8) Numerical Stability and Determinism

- Normalize path evidence to prevent dominance by large path counts.
- Clamp reliabilities to [r_min, r_max].
- Seed control and deterministic flags enabled globally; see seeding utilities in pipeline.

-------------------------------------------------------------------------------

9) Asymptotic Summary

Let:
- d: average presyn degree, k: recent spikes per presyn in window, g: cluster size for δ_causal
- m_max: tail combination cap, B: beam size, b: admissible branching, D: depth bound by horizon
- L: average hyperpath length, |E|: number of edges

- TC‑kNN generation: O(d·k + Σ_g Σ_{m≤m_max} C(g, m) + |Cand| log |Cand|)
- Traversal: O(D · B · b log(B·b)), memory O(D·B)
- Credit update: O(Σ_p |p| + |E_active|)
- Prune sweep: O(|E|)
- FSM observe: O(|P_t|·L)

-------------------------------------------------------------------------------

10) Traceability to Code

Primary call sites (to be implemented against interfaces):
- DHG generation and admission: [dch_core/dhg.py](dch_core/dhg.py)
- Traversal and path scoring: [dch_core/traversal.py](dch_core/traversal.py)
- Plasticity update and pruning: [dch_core/plasticity.py](dch_core/plasticity.py)
- FSM observe and promotion triggers: [dch_core/fsm.py](dch_core/fsm.py)
- HOE creation with guards: [dch_core/abstraction.py](dch_core/abstraction.py)

All modules must honor contracts in [dch_core/interfaces.py](dch_core/interfaces.py) and invariants in [docs/Interfaces.md](docs/Interfaces.md).

End of spec