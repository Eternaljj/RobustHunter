# RobustHunter

This repository contains the main implementation for the paper **RobustHunter: Limiting Error Propagation in Threat Hunting Under Uncertain Query Graphs**.

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
