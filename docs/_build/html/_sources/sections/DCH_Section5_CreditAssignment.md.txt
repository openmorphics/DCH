# Dynamic Causal Hypergraph DCH — Section 5 Credit Assignment via Constrained Backward Hyperpath Traversal

Parent outline [DCH_TechSpec_Outline.md](../DCH_TechSpec_Outline.md)  
Cross references Section 1 [DCH_Section1_FormalFoundations.md](../sections/DCH_Section1_FormalFoundations.md), Section 2 [DCH_Section2_DHG_TCkNN.md](../sections/DCH_Section2_DHG_TCkNN.md), Section 3 [DCH_Section3_Plasticity.md](../sections/DCH_Section3_Plasticity.md), Section 4 [DCH_Section4_HyperpathEmbedding.md](../sections/DCH_Section4_HyperpathEmbedding.md)

Version v0.1

1. Purpose and scope  
- Define the discrete, evidence based credit assignment mechanism for DCH.  
- Traverse backward from target head vertices to discover valid causal hyperpaths under temporal and biological constraints.  
- Aggregate path evidence to update hyperedge reliabilities and feed online plasticity and rule induction.

2. Inputs and outputs  
Inputs  
- Seed set S seeds of head vertices tagged by supervision, error, or reward events (e.g., classifier output spikes or task signals).  
- Current DCH state E t and V t with incoming by head index from Section 2.  
- Temporal parameters Δ min, Δ max, refractory ρ; traversal policy M, L, B, temperature τ select (Section 1 defaults).  
- Optional symbolic rule priors from FSM and abstraction (Sections 6–7) as bias features.

Outputs  
- Set of discovered valid hyperpaths Π valid with per path scores s π.  
- Edge level aggregates A e positive and negative evidence contributions.  
- Optional partial paths for audit and debugging when termination budgets are hit.

3. Valid hyperpath constraints (B connectivity and temporal logic)  
- B connectivity: An expansion across a hyperedge e from its head w to its tail Tail e is admissible only if evidence exists for every tail vertex u in Tail e. Since E t encodes realized event anchored edges, tails exist, but traversal must honor bounds below.  
- Temporal window: For each u in Tail e, Δ u equals w time minus u time must satisfy Δ min less or equal Δ u less or equal Δ max (edge attributes).  
- Refractory: Along any path, no two edges with heads on the same neuron may have head timestamps closer than ρ (prevents implausible rapid refiring).  
- Horizon: Only traverse to tail vertices with timestamps greater or equal t w minus H back, where H back is a traversal horizon in time (default H back 100 ms for event vision); also obey step and branching caps.  
- Acyclicity: Because event vertices are time ordered and edges head time is strictly greater than tail times, cycles are precluded in event anchored traversal; still avoid reusing the same vertex id within a single path.

4. Traversal state and search space  
- State represented as a pair Frontier, Path where Frontier is a multiset of vertices that must be justified and Path is the ordered list of edges already included.  
- Initialization per seed w star in S seeds set Frontier equals { w star } and Path equals empty.  
- Goal condition reached when Frontier is empty; a complete hyperpath π has been discovered.  
- Expansion chooses a vertex v from Frontier, selects an admissible incoming edge e with head v, removes v from Frontier, and adds all u in Tail e into Frontier; append e to Path.  
- Sampling domain for admissible edges includes only edges meeting B connectivity and temporal checks and respecting refractory relations to edges already in Path.

5. Multi start random B walk policy  
Defaults (Section 1): M equals 8 seeds per event, L equals 12 max depth, B equals 4 branching cap per expansion, τ select equals 0.7.  
- Selection probability for candidate edge e given current v:  
  - Features  
    - f rel equals log r e (monotonic with reliability).  
    - f rec equals exp of negative lambda rec times age e where age e equals current time minus head time of e; default lambda rec equals 1 divided by 100 ms.  
    - f rule equals 1 if e aligns with a known symbolic rule motif from FSM, else 0; may be weighted by rule confidence.  
    - f sim equals average cosine similarity between WL embedding of v and WL embeddings of tails of e (Section 4).  
  - Score z e equals w rel times f rel plus w rec times f rec plus w rule times f rule plus w sim times f sim with defaults w rel 1.0, w rec 0.5, w rule 0.5, w sim 0.5.  
  - Sampling probability p e equals softmax over candidates z e divided by τ select.  
- Branching control pick up to B highest probability candidates per expansion or sample without replacement up to B; beam style control described in Section 6 below.  
- Depth control terminate any branch exceeding L expansions or reaching timestamps older than H back.

6. Search strategies and determinism  
- Randomized beam search: Maintain K beam partial states with cumulative log scores and expand the top K after each layer (K default equals M). This improves stability vs. pure random selection while preserving exploration via softmax.  
- AND frontier semantics: Because hyperedges are AND relations over their tails, a state with Frontier size greater than 1 must justify all vertices; interleave expansions by picking any vertex from Frontier that has admissible incoming edges.  
- Vertex selection heuristic: Prefer the frontier vertex with the fewest admissible incoming edges first (fail fast), ties broken by most recent timestamp.  
- Deterministic replay: Use a seeded RNG keyed by seed vertex id, current time bucket, and frontier signature. With the same seed and DCH snapshot, traversal sequences are reproducible.

7. Path scoring and aggregation  
- Path score s π equals product over e in π of r e (Section 1 default).  
- Sign of contribution:  
  - For correct or rewarded seeds, add s π to A e positive for all e in π.  
  - For erroneous seeds (wrong class spike or error signal), add s π to A e negative for all e in π.  
- Normalization per edge e:  
  - r hat path e equals A e positive divided by A e positive plus A e negative plus ε with ε small equals 1.0 for stability.  
- Emission to plasticity: Provide r hat path e and optionally per edge weights to Section 3 to blend with local watcher statistics (λ path control).

8. Temporal logic enforcement details  
- Interval algebra alignment: Treat each hyperedge as imposing interval relations where tail[u] precedes head[w] by Δ u within bounds; disallow overlaps that violate refractory for same neuron heads.  
- Head coincidence filter: If two admissible edges share the same head neuron with head times closer than ρ, only allow the earlier one in a given path branch.  
- Tail collapse: If two tails within the same expansion refer to events from the same neuron within δ causal, select the latest tail event (consistent with Section 2 construction) to avoid duplicate justifications.

9. Efficiency, pruning, and early termination  
- Candidate capping: For a vertex v, cap the number of admissible incoming edges evaluated at C in per vertex cap default 16 to bound branching.  
- Early discard rules:  
  - Upper bound pruning discard partial states whose maximum possible extension score upper bound (using max r e on remaining depth) cannot beat the current k th best; maintain per depth best beams.  
  - Time bound pruning discard branches where the next admissible steps would violate H back.  
- Memoization: Cache admissible incoming edges per head vertex within the current traversal cycle keyed by head id and snapshot id to avoid repeated filtering.

10. Data dependencies and interfaces  
- Requires incoming by head index and hyperedge attribute access from Section 2.  
- Requires WL embeddings for similarity term and optional SAGE snapshot id from Section 4.  
- Outputs r hat path e to Section 3 via plasticity update from path signals.  
- Rule priors from FSM (Section 6) provided as a lookup mapping hyperedge motifs to bias weights.

11. Parameters and defaults  
- M equals 8 seeds per event; L equals 12; B equals 4; τ select equals 0.7; H back equals 100 ms; C in equals 16.  
- Feature weights w rel equals 1.0, w rec equals 0.5, w rule equals 0.5, w sim equals 0.5; λ rec equals 1 divided by 100 ms.  
- ε equals 1.0 for r hat normalization.  
- Determinism seed composed from seed vertex id, time bucket 1 ms, and traversal cycle id.

12. Observability and metrics  
- paths per seed average and distribution, valid ratio.  
- average depth reached and branching factor.  
- positive vs negative evidence totals per cycle and per class.  
- traversal latency per seed and total budget use.  
- contribution coverage fraction of active edges touched by traversal.

13. Complexity analysis  
Notation: D equals average admissible in degree after constraints; B equals branching cap; L equals depth cap; K equals beam width (approximately M).  
- Worst case explored states O K times sum from d equals 1 to L of B to the d which is controlled by caps; practical due to tight admissibility filters and temporal horizon.  
- Per expansion work dominated by admissible edge filtering O D (bounded by C in) plus feature scoring O D and softmax.  
- Total per seed cost upper bounded by K times L times C in with small constants.

14. Failure and edge cases  
- No admissible incoming edges for a frontier vertex leads to branch dead end; continue with other vertices in Frontier; if none remain, branch fails.  
- Seeds with timestamp near stream start older than H back yield few or no paths; still emit empty aggregates to avoid bias.  
- Highly bursty inputs may create many admissible candidates; enforce C in and B, defer excess edges via priority queues, and spill stats for meta controller.

15. Mermaid diagram — backward traversal with AND frontier

```mermaid
flowchart TB
SEED[Seed head vertex w*] --> INIT[Init Frontier={w*}, Path=[]]
INIT --> EXPAND[Pick frontier vertex v with fewest admissible edges]
EXPAND --> FILTER[Filter incoming edges by temporal windows, refractory, horizon]
FILTER --> SELECT[Score edges and sample up to B via softmax]
SELECT --> BRANCH1[Choose edge e1]
SELECT --> BRANCH2[Choose edge e2]
BRANCH1 --> FRONT1[Replace v with Tail(e1) in Frontier; append e1 to Path]
BRANCH2 --> FRONT2[Replace v with Tail(e2) in Frontier; append e2 to Path]
FRONT1 --> CHECK1{Frontier empty?}
FRONT2 --> CHECK2{Frontier empty?}
CHECK1 -->|Yes| EMIT1[Emit path π1 and score s(π1)]
CHECK2 -->|Yes| EMIT2[Emit path π2 and score s(π2)]
CHECK1 -->|No| EXPAND
CHECK2 -->|No| EXPAND
```

16. Quality targets and acceptance criteria  
- Valid hyperpaths honor all temporal windows and refractory constraints; no violations in audit replays.  
- On synthetic benchmarks with planted causal chains, traversal recovers at least 90 percent of ground truth edges within H back under default caps.  
- End to end traversal latency per seed less than 1 ms on desktop prototype for K equals 8, L equals 12, C in equals 16.  
- Emitted r hat path e integrates with plasticity Section 3 and yields monotonic reliability growth on consistent tasks and depression on error tagged seeds.

17. Interfaces aligned to module contracts  
- traversal assign credit inputs seeds list, mode in {reward, error, correct}, returns edge to contribution map A e positive or A e negative and per path records.  
- traversal params set or get to adjust M, L, B, τ select, H back, C in, and feature weights.  
- traversal metrics snapshot returns counters listed in Section 12.

18. Cross references  
- Reliability aggregation and EMA update [DCH_Section3_Plasticity.md](../sections/DCH_Section3_Plasticity.md)  
- Embedding based similarity [DCH_Section4_HyperpathEmbedding.md](../sections/DCH_Section4_HyperpathEmbedding.md)  
- Formal objects and constraints [DCH_Section1_FormalFoundations.md](../sections/DCH_Section1_FormalFoundations.md)

End of Section 5
## 18. Pseudocode — randomized beam traversal with AND frontier

```pseudo
# Inputs:
#  - Seeds: list of seed head vertex ids w* with tag ∈ {reward, error, correct}
#  - E: hyperedge store with incoming_by_head index
#  - V: vertex store with timestamps
#  - Params: M, L, B, τ_select, H_back, C_in,
#            weights (w_rel, w_rec, w_rule, w_sim), λ_rec,
#            seed_time_bucket = 1 ms
#  - Priors: Active rule motifs (optional)
#  - Embeds: WL embeddings map (optional)

function assign_credit(Seeds, E, V, Params, Priors, Embeds):
    RNG.seed = make_seed(Seeds, now(), seed_time_bucket)

    Aggregates_pos ← map<edge_id, float>()
    Aggregates_neg ← map<edge_id, float>()
    Paths_emitted  ← list<PathRecord>()

    for s in Seeds[0 .. M-1]:
        beam ← init_beam_with_state( frontier={s}, path=[], score_log=0 )
        depth ← 0

        while depth < L and not beam.empty():
            next_beam ← empty structure
            for state in beam:      # randomized beam
                # Select a frontier vertex by fail-fast heuristic
                v_sel ← pick_frontier_vertex_with_fewest_admissible(state, E, V, Params, H_back)
                if v_sel == ⊥:
                    # If all frontier vertices lack admissible edges, check completion
                    if state.frontier == ∅:
                        finalize_emit(state, Aggregates_pos, Aggregates_neg, Paths_emitted, s.tag)
                    continue

                # Enumerate admissible incoming edges with B-connectivity & temporal checks
                cand_edges ← filter_incoming_edges(v_sel, state, E, V, Params, H_back, C_in)
                if cand_edges.empty():
                    # Try another frontier vertex (loop continues)
                    continue

                # Score candidates (softmax with τ_select)
                scored ← []
                for e in cand_edges:
                    z ← score_edge(e, state, Priors, Embeds, weights, λ_rec)
                    scored.append( (e, z) )
                probs ← softmax_over_scores(scored, τ_select)

                # Branch up to B edges (sample without replacement or pick top-B)
                chosen ← pick_up_to_B(probs, B, RNG)

                # Create child states
                for (e, p) in chosen:
                    child ← clone(state)
                    child.frontier.remove(v_sel)
                    for u in Tail(e):
                        child.frontier.add(u)
                    child.path.append(e)
                    child.score_log += log( max(e.r, 1e-6) )   # product of r as log-sum
                    # Refractory and temporal guards already enforced in filter
                    next_beam.add(child)

            # Keep top-K states by score (beam width K ≈ M) with some randomness
            beam ← select_topK_with_random_tiebreak(next_beam, K=M, RNG)
            depth ← depth + 1

        # Emit any completed paths in final beam layer
        for state in beam:
            if state.frontier == ∅:
                finalize_emit(state, Aggregates_pos, Aggregates_neg, Paths_emitted, s.tag)

    return Aggregates_pos, Aggregates_neg, Paths_emitted


function filter_incoming_edges(v, state, E, V, Params, H_back, C_in):
    # Incoming edges whose head == v and satisfy:
    #  - For all u∈Tail(e): Δ_u = t_head - t_u ∈ [Δ_min, Δ_max]
    #  - Refractory: no prior head on same neuron within ρ in current path
    #  - Horizon: t_u ≥ t_head - H_back
    #  - No vertex reuse in path
    cand ← []
    for e in E.incoming_by_head[v]:
        if not temporal_valid(e, V, Params.Δ_min, Params.Δ_max, H_back): continue
        if violates_refractory(e, state.path, Params.ρ): continue
        if reuses_vertex(e, state.path): continue
        cand.append(e)
        if len(cand) == C_in: break
    return cand


function score_edge(e, state, Priors, Embeds, weights, λ_rec):
    f_rel  = log( clamp(e.r, 1e-6, 1.0) )
    age    = now() - head_time(e)
    f_rec  = exp( -λ_rec * age )
    f_rule = motif_match_score(e, state, Priors)     # ∈ [0,1]
    f_sim  = wl_similarity(head_vertex(e), Tail(e), Embeds)  # ∈ [0,1]
    z = weights.w_rel*f_rel + weights.w_rec*f_rec + weights.w_rule*f_rule + weights.w_sim*f_sim
    return z


function finalize_emit(state, Agg_pos, Agg_neg, Paths_emitted, tag):
    # Path score = product of r(e) (stored as log-sum in state)
    s_pi = exp(state.score_log)
    for e in state.path:
        if tag ∈ {reward, correct}:
            Agg_pos[e] = Agg_pos.get(e, 0.0) + s_pi
        else:
            Agg_neg[e] = Agg_neg.get(e, 0.0) + s_pi
    Paths_emitted.append( make_path_record(state, s_pi) )
```

Normalization to r̂_path  
- For each edge e discovered during this cycle, compute r̂_path(e) = Agg_pos[e] / (Agg_pos[e] + Agg_neg[e] + ε) with ε ≈ 1e-6 and pass to Plasticity (Section 3).  
- Determinism: use seeded RNG based on seed id and time bucket; memoize admissible sets per head id within a cycle for repeatability.

Audit mode (optional)  
- Store compact per-branch traces (frontier signatures and chosen edge ids) behind a debug flag to enable exact replay during review without logging full paths.