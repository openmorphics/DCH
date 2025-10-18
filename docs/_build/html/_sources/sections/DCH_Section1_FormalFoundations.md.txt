# Dynamic Causal Hypergraph DCH — Section 1 Formal Foundations and Glossary

Parent outline [DCH_TechSpec_Outline.md](../DCH_TechSpec_Outline.md)

Version v0.1

1. Scope and defaults
- Time model continuous time timestamps in microseconds μs
- Path score product of edge reliabilities
- Reliability update EMA with alpha 0.1 and bounds 0.02 to 0.98
- TC kNN window 1 ms to 30 ms, k max 3, delta causal 2 ms
- Traversal seeds M 8, max depth L 12, branching cap B 4, selection temperature 0.7
- FSM window W 60 s, support s min 50, reliability threshold r min 0.6, decay gamma 0.98

2. Base sets and objects
- Neuron id space I equals {1,2,...,N}
- Time domain T equals real nonnegative measured in microseconds
- Event vertex v equals neuron id i comma timestamp t with i in I and t in T
- Vertex set V t equals { v mid v.timestamp less or equal t }
- Hypergraph DCH t equals V t comma E t

3. Hyperedge schema and attributes
- A directed hyperedge e equals Tail to Head with Head cardinality 1
- Tail e equals { u1 comma u2 comma ... comma um } subset of V t with m greater or equal 1
- Head e equals { w } with w in V t
- Attributes for e
  - delay window Δ min comma Δ max in microseconds with 0 less Δ min less Δ max
  - refractory ρ in microseconds minimum separation for heads on same neuron
  - reliability score r e in interval 0.02 to 0.98
  - created at tau c equals Head timestamp
  - last update time tau u
  - usage count c e in integers
  - type label in {event edge comma template edge}

4. Validity predicate for temporal causality
- valid e mid V requires
  - for every u in Tail e the time difference Δ u equals Head time minus u time satisfies Δ min less or equal Δ u less or equal Δ max
  - no two hyperedges e1 and e2 with the same Head neuron create head times closer than ρ
- valid e mid V implies e respects temporal logic constraints for causality

5. Event anchored and template hyperedges
- Event anchored hyperedge an e whose Head is a realized vertex w in V t
- Template hyperedge τ defines Tail schema as a set of neuron ids with relative lags and a Head neuron id with relative zero
- Instantiation of τ at time t is a mapping of its schema to concrete vertices in V t that satisfy the window constraints

6. Operations on E t
- GROW add new event anchored hyperedges produced around an observed Head vertex per TC kNN
- REFINE update reliability via predict and confirm evidence
- PRUNE remove e if r e below threshold or age large with low usage
- ABSTRACT introduce a template hyperedge that summarizes a frequent reliable hyperpath
- FREEZE gate updates for a protected subset of E t as part of task aware scaffolding

7. Hyperpaths and B connectivity
- A hyperpath π from sources to sink w is a finite collection of hyperedges in E with a partial order such that for each edge e in π every tail vertex of e is either a source vertex or the head of some edge earlier in the order
- B connectivity constraint traversal from Head to Tail is allowed only if evidence exists for all tail vertices
- Instantiated hyperpath pairs π comma g where g grounds each edge to concrete vertices in V

8. Path scoring and evidence aggregation
- Path score s π equals product over e in π of r e
- Positive evidence set P e equals valid hyperpaths that include e and originate from correct or rewarded sinks
- Negative evidence set N e equals valid hyperpaths that include e and originate from erroneous sinks
- Normalized target r hat e equals sum w in P e s w divided by sum s in P e union N e s plus epsilon where epsilon small

9. Reliability update operator
- EMA update r new e equals clip of 1 minus alpha times r old e plus alpha times r hat e bounded to 0.02 and 0.98 with alpha equals 0.1
- Update timestamp tau u set to current time and increment c e

10. Temporal windows and units
- Defaults Δ min equals 1000 microseconds and Δ max equals 30000 microseconds
- δ causal equals 2000 microseconds micro window for grouping near coincident presyn spikes
- All timestamps monotone per neuron and unique per event

11. Presynaptic adjacency and candidate sources
- Pred j equals { i in I mid synapse i to j exists } provided by the underlying SNN or connectivity map
- Candidate antecedent spikes for a post spike at neuron j and time t j are the most recent spikes from each i in Pred j within the temporal window

12. Invariants and safety constraints
- Reliability bounds 0.02 less or equal r e less or equal 0.98 for all e
- No self cycle from a head vertex back to itself
- For any neuron j and times t1 less t2 with t2 minus t1 less ρ at most one is a head of an accepted edge
- ABSTRACT does not introduce a cycle at the template level when projected to neuron graph

13. Observables and counters
- events per second lambda t
- active edges count size of E t
- average reliability bar r t equals mean over e in E t of r e
- prune rate equals removals per unit time
- traversal yield equals valid hyperpaths per seed
- rule discovery rate equals promoted templates per unit time

14. Notation summary
- i comma j neuron ids
- t comma tau time variables microseconds
- v equals i comma t event vertex
- e hyperedge Tail to Head with attributes Δ min Δ max ρ r
- π hyperpath
- s π path score
- P e positive evidence set N e negative evidence set
- r hat e normalized target probability
- alpha EMA step size default 0.1

15. Interfacing with the SNN substrate
- The DCH consumes spike events produced by an SNN or event sensor
- The DCH never writes to neuron state directly but may export symbolic rules and policy hints to guide search or initialization

16. Minimal worked example
- Suppose neuron A fires at time 10000 microseconds and neuron B at 11700 microseconds and neuron C at 21000 microseconds with Δ min equals 1000 and Δ max equals 30000
- GROW proposes unary edges {A at 10000} to {C at 21000} and {B at 11700} to {C at 21000} and a binary edge {A at 10000 comma B at 11700} to {C at 21000} since B minus A equals 1700 within δ causal equals 2000
- Later the pattern {A then B} repeats and C fires within window generating positive evidence that increases r for the binary edge by the EMA rule

17. Cross references
- Detailed construction algorithms appear in Section 2 see [DCH_TechSpec_Outline.md](../DCH_TechSpec_Outline.md)
- Traversal and credit assignment appear in Section 5 see [DCH_TechSpec_Outline.md](../DCH_TechSpec_Outline.md)
- Online rule induction appears in Section 6 see [DCH_TechSpec_Outline.md](../DCH_TechSpec_Outline.md)

18. Acceptance criteria for Section 1
- Formal definitions for V t E t vertices hyperedges attributes and validity are present
- Reliability update operator EMA with bounds and target aggregation is specified
- Hyperpath and B connectivity are defined with default path scoring
- Invariants and units are defined with defaults for Δ and δ causal
- Cross references align with the overall outline

File index
- This section file [DCH_Section1_FormalFoundations.md](../sections/DCH_Section1_FormalFoundations.md)
- Parent outline file [DCH_TechSpec_Outline.md](../DCH_TechSpec_Outline.md)

End of Section 1