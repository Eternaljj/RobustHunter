# Reproducibility notes

## Frozen public defaults

The bundled configuration uses a 64-dimensional deterministic attribute hash,
64 hidden dimensions, two shared relation-aware layers, Top-3 alignment
evidence, score weights `(0.4, 0.5, 0.1)`, structural weights
`mu_instance=1`, `mu_stage=1`, and stage penalty `beta_stage=0.2`.

The paper configuration uses AdamW with learning rate `1e-3`, weight decay
`1e-5`, ranking/behavior margin `0.2`, gradient clipping at `5.0`, and
validation early-stopping patience 10. The small public `train_pairwise()`
helper exposes the same optimization and loss terms but deliberately leaves
dataset splitting, validation-only early stopping, and private label handling
to the experiment protocol.

## Required experiment provenance

For a reported result, archive and report:

1. release commit/tag and `SHA256SUMS`;
2. layered-query and candidate-manifest hashes;
3. train/validation/test split protocol and seed;
4. exact model configuration and checkpoint hash;
5. validation-only epoch and pair-threshold selection;
6. evaluator version and the point at which labels were opened.

Candidate generation and evaluation labels are outside this minimal Stage-3
artifact. The matcher must never receive candidate labels, ground-truth nodes or
edges, anchor precision, or anchor relevance.

## Determinism

The public builder seeds Python and PyTorch, uses deterministic algorithms, and
sets one PyTorch CPU thread. Hardware/library differences can still affect
floating-point behavior; report Python, PyTorch, operating system, and device.

## Version compatibility

The current model uses a shared learnable NULL baseline normalized jointly with
compatible candidate nodes. Checkpoints from the historical
`threshold_without_virtual_node` implementation are not numerically compatible
and are rejected when their saved configuration identifies that mode.
