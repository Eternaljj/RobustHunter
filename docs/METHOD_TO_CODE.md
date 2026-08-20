# Method-to-code map

This map follows the equation numbering in the 2026-08-20 manuscript, **RobustHunter: Limiting Error Propagation in Threat Hunting Under Uncertain Query Graphs**.

| Paper definition | Equations | Implementation |
| --- | --- | --- |
| Initial node features and shared relation-aware encoding | (6)-(9) | `compile_graph_features()`, `SharedRGATEncoder`, and `RGATLayer` in `src/robusthunter/rewritten/rgat.py` |
| Relation attention details | (46)-(48) | `RGATLayer.forward()` in `src/robusthunter/rewritten/rgat.py` |
| Cross-graph alignment matrix, similarity, and joint candidate/NULL normalization | (10)-(14) | `soft_align()` in `src/robusthunter/rewritten/alignment.py` |
| Type, attribute, and embedding similarity | (11), (49)-(51) | `type_compatibility()` and `soft_align()` in `src/robusthunter/rewritten/alignment.py` |
| Asymmetric match/NULL decision | (15) | `soft_align()` and `RewrittenMatcher.forward()` |
| Instance/stage structural-support weights | (16) | `structural_support_weights()` in `src/robusthunter/rewritten/matcher.py` |
| Weighted unsupported-query rate | (17) | `soft_align()` in `src/robusthunter/rewritten/alignment.py` |
| Attack-instance behavior consistency and weak stage order | (18)-(20), (52)-(55) | `behavior_consistency()` in `src/robusthunter/rewritten/behavior_consistency.py` |
| Entity score, behavior score, and final ranking score | (21)-(23) | `soft_align()`, `behavior_consistency()`, and `RewrittenMatcher.forward()` |
| Positive/negative training triple | (24) | `PairwiseTrainingExample` in `src/robusthunter/training.py` |
| Margin ranking loss | (25)-(26) | `matcher_losses()` in `src/robusthunter/rewritten/matcher.py` |
| Weak node-alignment loss | (27) | `one_to_many_alignment_loss()` and `matcher_losses()` |
| Behavior margin and total objective | (28)-(30) | `matcher_losses()` and `train_pairwise()` |

## Stage-3 data flow

```text
layered query + retained candidate paths
                |
                v
shared relation-aware encoder
                |
                v
joint candidate/NULL soft alignment
                |
                +----> weighted entity support and NULL penalty
                |
                +----> behavior-path consistency and stage penalty
                |
                v
final candidate score and ranked explanation
```

## Behavior-path implementation detail

Stage 2 supplies a bounded set of retained paths. `reconstruct_retained_paths()` verifies edge references, directed continuity, canonical relations, and timestamps. For every behavior edge in an attack instance, the current experimental implementation scores all retained valid paths, chooses the strongest path evidence, and averages the selected edge-level scores within the instance. The resulting instance scores are averaged and multiplied by the weak stage-order penalty.

This edge-decomposed calculation preserves the four consistency components in (18) and the max-supported-path mechanism in (19), while permitting different member edges of one attack instance to select different retained paths. It is documented here so the repository describes the exact implementation used by the experiments.

## Inference isolation

The matcher recursively rejects evaluator-only fields such as `ground_truth`, `ground_truth_nodes`, `ground_truth_edges`, `gt_nodes`, `gt_edges`, `anchor_precision`, and `anchor_relevance`. Labels are not matcher inputs.
