# RobustHunter - Stage-3 Graph Matching

Official Stage-3 code artifact for **RobustHunter: Limiting Error Propagation in Threat Hunting Under Uncertain Query Graphs**.

Repository: <https://github.com/Eternaljj/RobustHunter>

This release implements the paper's graph-matching stage: shared relation-aware encoding, joint candidate/NULL soft alignment, structurally weighted asymmetric partial matching, attack-behavior consistency, weak attack-stage constraints, and candidate ranking. Stage 1 query construction and Stage 2 dual-anchor sampling are intentionally outside this repository; their layered query and retained candidate paths are the inputs to Stage 3.

## Paper-to-code overview

| Paper component | Equations | Main code |
| --- | --- | --- |
| Relation-aware graph encoding | (6)-(9), (46)-(48) | `src/robusthunter/rewritten/rgat.py` |
| Similarity and joint candidate/NULL alignment | (10)-(14), (49)-(51) | `src/robusthunter/rewritten/alignment.py` |
| Structural weights and asymmetric partial matching | (15)-(17) | `src/robusthunter/rewritten/matcher.py` |
| Behavior consistency and stage-order constraint | (18)-(20), (52)-(55) | `src/robusthunter/rewritten/behavior_consistency.py` |
| Entity/behavior/NULL aggregation | (21)-(23) | `src/robusthunter/rewritten/matcher.py` |
| Ranking, alignment, and behavior losses | (24)-(30) | `src/robusthunter/rewritten/matcher.py`, `src/robusthunter/training.py` |

The complete mapping is in [`docs/METHOD_TO_CODE.md`](docs/METHOD_TO_CODE.md).

## Repository layout

```text
config/                         Stage-3 model configuration
docs/                           Method map, I/O contract, reproducibility notes
examples/                       Small non-scientific smoke inputs
src/robusthunter/rewritten/     Core Stage-3 numerical implementation
src/robusthunter/               Public API, CLI, and training/checkpoint helpers
tests/                          Deterministic focused tests
```

## Install

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e .
```

PyTorch is the only runtime dependency. For CUDA-specific builds, install the appropriate PyTorch wheel before `pip install -e .`.

## Smoke test

The bundled example uses deterministic random initialization to verify the software path only. It is not a scientific result.

```bash
robusthunter-match \
  --query examples/query.json \
  --candidate examples/candidate.json \
  --config config/graph_matching_v3.json \
  --allow-untrained \
  --output outputs/smoke.json

python tests/run_tests.py
python examples/train_smoke.py
```

## Scientific inference

A trained, configuration-compatible checkpoint is mandatory for reporting performance:

```bash
robusthunter-match \
  --query path/to/layered_query.json \
  --candidate path/to/candidate_graph.json \
  --config config/graph_matching_v3.json \
  --checkpoint path/to/checkpoint.pt \
  --output outputs/match.json
```

```python
import json
from robusthunter import build_matcher, load_checkpoint, match_graphs

model = build_matcher("config/graph_matching_v3.json", device="cpu")
load_checkpoint(model, "checkpoint.pt")

with open("query.json", encoding="utf-8") as query_file:
    query = json.load(query_file)
with open("candidate.json", encoding="utf-8") as candidate_file:
    candidate = json.load(candidate_file)

result = match_graphs(model, query, candidate)
print(result["score"], result["explanation"])
```

Report the repository commit, configuration, checkpoint hash, candidate-pool hash, split protocol, seed, and evaluator version with every experiment. See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

## Implementation boundary

- Candidate paths are bounded inputs retained by Stage 2; Stage 3 does not perform an unbounded search over a full provenance graph.
- Ground-truth and evaluator-only fields are recursively rejected at inference.
- The released experimental implementation scores retained paths for each behavior edge and averages the selected evidence within an attack instance. This is the edge-decomposed implementation of the behavior-consistency term used by the current experiments.
- Historical threshold-based checkpoints are incompatible with the joint candidate/NULL model and are rejected when their saved configuration identifies the legacy mode.
- The public pairwise trainer exposes the paper's loss terms but does not bundle private dataset splits, evaluator labels, or trained checkpoints.

## Integrity

Verify the release files with:

```bash
sha256sum -c SHA256SUMS
```
