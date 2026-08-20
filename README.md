# RobustHunter

This repository contains the main implementation for the paper **RobustHunter: Limiting Error Propagation in Threat Hunting Under Uncertain Query Graphs**.

## Paper-to-code overview

| Paper component | Equations | Main code |
| --- | --- | --- |
| Relation-aware graph encoding | (6)-(9), (46)-(48) | `src/robusthunter/rewritten/rgat.py` |
| Similarity and soft alignment | (10)-(14), (49)-(51) | `src/robusthunter/rewritten/alignment.py` |
| Structural weights and asymmetric partial matching | (15)-(17) | `src/robusthunter/rewritten/matcher.py` |
| Behavior consistency and stage-order constraint | (18)-(20), (52)-(55) | `src/robusthunter/rewritten/behavior_consistency.py` |
| Entity/behavior/NULL aggregation | (21)-(23) | `src/robusthunter/rewritten/matcher.py` |
| Ranking, alignment and behavior losses | (24)-(30) | `src/robusthunter/rewritten/matcher.py`, `src/robusthunter/training.py` |


## Install

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e .
```

PyTorch is the only runtime dependency. For CUDA-specific builds, install the appropriate PyTorch wheel before `pip install -e .`.

## test

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

## Integrity

Verify the release files with:

```bash
sha256sum -c SHA256SUMS
```
