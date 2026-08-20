from __future__ import annotations

import json
from pathlib import Path

from robusthunter import (
    PairwiseTrainingExample,
    build_matcher,
    save_checkpoint,
    train_pairwise,
)


ROOT = Path(__file__).resolve().parents[1]
query = json.loads((ROOT / "examples/query.json").read_text(encoding="utf-8"))
positive = json.loads((ROOT / "examples/candidate.json").read_text(encoding="utf-8"))
negative = json.loads(json.dumps(positive))
negative["candidate_id"] = "paper_smoke_negative"
negative["nodes"][0]["name"] = "unrelated-benign-process"

model = build_matcher(ROOT / "config/graph_matching_v3.json", seed=20260714)
history = train_pairwise(
    model,
    [PairwiseTrainingExample(query, positive, negative)],
    epochs=1,
    seed=20260714,
)
output = ROOT / "outputs/smoke_checkpoint.pt"
save_checkpoint(model, output, metadata={"purpose": "software-smoke-only"})
print(json.dumps({"checkpoint": str(output), "history": history}, indent=2))
