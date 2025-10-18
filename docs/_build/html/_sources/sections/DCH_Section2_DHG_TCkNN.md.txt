# Dynamic Causal Hypergraph DCH — Section 2 Dynamic Hypergraph Construction with TC kNN

Parent outline [DCH_TechSpec_Outline.md](../DCH_TechSpec_Outline.md)
Cross reference Section 1 [DCH_Section1_FormalFoundations.md](../sections/DCH_Section1_FormalFoundations.md)

Version v0.1

1. Purpose and scope
- Define the online construction of E t around each postsynaptic spike event
- Specify TC kNN candidate selection higher order combination generation deduplication initialization and resource controls

2. Inputs and outputs
Inputs
- Event stream of spike vertices v equals neuron id comma timestamp
- Presynaptic adjacency Pred j for each neuron j
- Temporal parameters Δ min Δ max δ causal refractory ρ
- Budgets k max per head budget M in combinations cap C cap

Outputs
- New hyperedges added to E t with initialized attributes and provenance
- Rejected candidate statistics for meta controller feedback

3. Preliminaries and data structures
Event buffers
- For each neuron i maintain a time sorted ring buffer B i of recent spike vertices with capacity measured by time horizon T ret and item cap N ret
- Support binary search by timestamp and append amortized O one

Hyperedge stores
- incoming by head maps a head vertex id to a list of incident hyperedge ids
- by tail signature maps a canonical tail signature key to hyperedge id for deduplication
- recent heads by neuron maps neuron id to recent head vertex timestamps for enforcing refractory

Canonical tail signature
- Define sig Tail as multiset of ordered pairs neuron id comma time sorted by time then neuron id
- Key equals hash of sig Tail concatenated with head vertex id

4. TC kNN candidate generation around a post spike
Trigger
- When a spike v post equals neuron j comma time t j arrives begin candidate search

Step A presynaptic lookup
- Retrieve Pred j from adjacency
- For each i in Pred j query B i for the most recent spike u i with time t i in interval t j minus Δ max to t j minus Δ min
- If none found for i skip i else add u i to set U j

Step B unary candidates
- For each u in U j create a unary candidate Tail equals { u } and Head equals { v post }

Step C higher order candidates up to k max
- Form all combinations of U j of size m for m in 2 to k max subject to temporal coherence constraint
- Temporal coherence requires max time of tails minus min time of tails less or equal δ causal
- For each coherent combination form Tail equals that combination and Head equals { v post }

Step D candidate scoring and admission control
- Compute candidate priority score s cand using heuristics
  - Delay closeness term equals mean over u in Tail of one minus abs t j minus t u minus μ Δ divided by Δ span where μ Δ equals mid of Δ min and Δ max and Δ span equals Δ max minus Δ min
  - Fan in cost term penalizes large m for example λ m equals m divided by k max
  - Priority s cand equals w delay times delay closeness plus w size times 1 minus λ m defaults w delay equals 0.7 w size equals 0.3
- Maintain per head budget M in and global combinations cap C cap per post spike
- Select top candidates by s cand until budgets are met

Step E deduplication
- For each selected candidate build sig Tail and key
- If key exists in by tail signature skip else continue

Step F initialize hyperedge and insert
- Create edge e with attributes
  - Tail as selected set of tail vertices
  - Head equals { v post }
  - Δ min Δ max ρ inherited from defaults or per neuron tables
  - reliability score r e initialized to epsilon init default 0.05
  - created at equals t j last update time equals t j usage count equals 0 type equals event edge
- Insert e into E t and update indexes incoming by head and by tail signature and recent heads by neuron for refractory checks

5. Refractory enforcement and conflict resolution
- Before inserting any edge with Head neuron j verify that no accepted head for neuron j exists in interval t j minus ρ to t j plus ρ
- If conflict occurs apply deterministic tie break
  - Prefer the head with larger s cand
  - If equal prefer earlier timestamp to reduce pile up
- Edges referencing a rejected head are discarded

6. Provenance and audit metadata
- Record for each inserted edge the candidate score s cand and the list of presynaptic neuron ids and time lags Δ u equals t j minus t u
- Maintain a creation token equals tuple of head vertex id and tail signature to support idempotent replay

7. Parameter table defaults
- Δ min equals 1000 microseconds Δ max equals 30000 microseconds δ causal equals 2000 microseconds ρ equals 1000 microseconds
- k max equals 3 M in equals 6 C cap equals 10 epsilon init equals 0.05
- w delay equals 0.7 w size equals 0.3

8. Complexity analysis
Notation
- d j equals degree of Pred j
- b equals average buffer size per neuron within window
- c equals number of coherent tails admitted before budget

Costs per post spike
- Presyn lookup O d j
- For each i binary search in B i O log b and constant time verification per match total O d j log b
- Combination enumeration worst case sum over m from 2 to k max of C of U j choose m bounded by budgets yields O c
- Deduplication and insert O 1 amortized via hash maps

Memory
- Event buffers sum over i of size B i bounded by T ret and N ret
- Hyperedges active equals size of E t bounded by pruning policy and budgets
- Indexes proportional to number of active edges

9. Concurrency and ordering
- Process events in nondecreasing timestamp order
- Apply per neuron j critical section for refractory and insertion to avoid races
- Use lock free ring buffers for B i and atomic increments for counters
- Idempotent creation by checking creation token before insert enables safe replays

10. Failure and backpressure handling
- If budgets exceeded drop lowest s cand and increment rejected counters
- If buffer overflow occurs advance head of ring buffer and increment buffer evict counter
- If time skew detected for input events stash out of order events until watermark advances

11. Interfaces aligned to module contracts
Event ingestion to DHG
- on post spike inputs neuron id j and timestamp t j returns list of created edge ids may be empty
- get rejected stats returns struct with counts per reason and recent averages

Query
- get tails for head head vertex id returns list of tail sets and edge ids
- exists edge key returns boolean

Configuration
- set params provide Δ min Δ max δ causal ρ k max M in C cap weights and epsilon init
- get params returns current table possibly per neuron overrides

12. Optional variants and extensions
Adaptive windows
- Maintain per neuron pair i to j estimates of empirical delay distribution using exponential histograms
- Modulate Δ min and Δ max around current percentiles for that pair

Ranked K for presyn
- Instead of only the most recent u i choose top K i recent spikes in the window and allow more combinations while tightening budgets

Synaptic priors
- If the SNN exposes synaptic strength use it as a prior multiplier in s cand

13. Quality metrics for the DHG module
- Candidate hit rate equals created edges divided by evaluated candidates
- Deduplication rate equals skipped duplicates divided by evaluated candidates
- Average tails per head vertex
- Average admitted combination size
- End to end latency per post spike for DHG path

14. Mermaid diagram enhanced TC kNN flow

```mermaid
flowchart TB
POST[Post spike j at time t j] --> A1[Lookup Pred j]
A1 --> A2[Find most recent presyn spikes in window]
A2 --> A3[Form unary and multi tail candidates up to k max]
A3 --> A4[Temporal coherence filter within delta causal]
A4 --> A5[Score candidates and apply budgets]
A5 --> A6[Deduplicate by tail signature]
A6 --> A7[Refractory check and conflict resolve]
A7 --> A8[Insert edges and update indexes]
```

15. Acceptance criteria for Section 2
- TC kNN generation defined for unary and higher order tails with temporal coherence
- Budgets scoring deduplication initialization and refractory policy specified
- Complexity and memory bounds presented with symbols d j b c
- Interfaces align with contracts in outline and observability metrics defined
- Enhanced diagram included and consistent with outline

16. Cross references
- Formal symbols and validity constraints in Section 1 see [DCH_Section1_FormalFoundations.md](../sections/DCH_Section1_FormalFoundations.md)
- Plasticity updates in Section 3 see [DCH_TechSpec_Outline.md](../DCH_TechSpec_Outline.md)
- Traversal and credit assignment in Section 5 see [DCH_TechSpec_Outline.md](../DCH_TechSpec_Outline.md)

End of Section 2
## 17. Pseudocode — TC‑kNN candidate generation and admission

```pseudo
# Inputs:
#  - j: postsynaptic neuron id
#  - t_j: postsynaptic spike time
#  - Pred(j): presynaptic neuron set
#  - B_i: per-neuron ring buffers (time-sorted) for recent spikes
#  - Params: Δ_min, Δ_max, δ_causal, ρ, k_max, M_in, C_cap,
#            w_delay, w_size, ε_init
#  - Indexes: incoming_by_head, by_tail_signature, recent_heads_by_neuron

function dhg_tc_knn_on_post_spike(j, t_j, Pred, Buffers, Params, Indexes):
    U_j ← ∅
    # Step A: presynaptic lookup within temporal window
    for i in Pred(j):
        u_i ← most_recent_spike_in_window(Buffers[i], [t_j - Δ_max, t_j - Δ_min])
        if u_i ≠ ⊥:
            U_j ← U_j ∪ { u_i }

    Cands ← ∅
    # Step B: unary candidates
    for u in U_j:
        Cands.add( (Tail={u}, Head={⟨j, t_j⟩}) )

    # Step C: higher‑order candidates up to k_max with temporal coherence
    # Enumerate combinations of U_j of size m ∈ [2, k_max]
    for m in 2..k_max:
        for comb in combinations(U_j, m):
            t_min ← min(time(u) for u in comb)
            t_max ← max(time(u) for u in comb)
            if (t_max - t_min) ≤ δ_causal:
                Cands.add( (Tail=comb, Head={⟨j, t_j⟩}) )

    # Step D: candidate scoring and admission budgets
    # Delay closeness per-tail averaged to prefer the middle of [Δ_min, Δ_max]
    μ_Δ ← (Δ_min + Δ_max)/2
    Δ_span ← (Δ_max - Δ_min)
    Scored ← []
    for c in Cands:
        delays ← [ t_j - time(u) for u in c.Tail ]
        delay_closeness ← mean( 1 - |Δ - μ_Δ|/Δ_span for Δ in delays )
        λ_m ← |c.Tail| / k_max                       # fan‑in penalty
        s_cand ← w_delay * delay_closeness + w_size * (1 - λ_m)
        Scored.append( (c, s_cand) )

    # Respect per‑head and per‑post budgets
    Scored.sort_by_descending_score()
    Admitted ← take_first_k(Scored, limit=M_in, also_cap=C_cap)

    NewEdges ← []
    # Step E/F: deduplicate, refractory check, and insert
    for (c, s) in Admitted:
        key ← make_tail_signature_key(c.Tail, c.Head)   # sorted by (time, neuron)
        if by_tail_signature.contains(key):
            continue

        # Refractory: ensure no accepted head for neuron j within [t_j - ρ, t_j + ρ]
        if Indexes.recent_heads_by_neuron[j].has_collision(t_j, ρ):
            # Tie‑break if needed: prefer higher s_cand; else earlier t
            if not passes_tie_break(j, t_j, s):
                continue

        e ← Hyperedge(
                Tail=c.Tail,
                Head=c.Head,
                Δ_min=Δ_min, Δ_max=Δ_max, ρ=ρ,
                r=ε_init,
                created_at=t_j,
                last_update=t_j,
                usage_count=0,
                type="event_edge",
                provenance={ "score": s,
                             "lags": [ t_j - time(u) for u in c.Tail ],
                             "pred": list(Pred(j)) }
            )

        insert_into_E(e)
        Indexes.incoming_by_head[head_vertex_id(c.Head)].append(e.id)
        Indexes.by_tail_signature[key] ← e.id
        Indexes.recent_heads_by_neuron[j].record(t_j)
        NewEdges.append(e.id)

    return NewEdges
```

Complexity note  
- Presyn lookup: O(|Pred(j)| · log b) via buffer binary search (b = avg buffer size).  
- Combination enumeration bounded by budgets M_in and C_cap; coherence filter reduces blow‑up.  
- Dedup and insert amortized O(1) with hashing.

Figure 2 — TC‑kNN pseudocode summary  
The procedure mirrors Sections 4–7: presyn search, coherent combination, heuristic scoring, budgeted admission, deduplication, refractory safety, and indexed insertion with initialized reliability r = ε_init.