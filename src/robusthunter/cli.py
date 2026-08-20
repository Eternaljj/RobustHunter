from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .graph_matching import build_matcher, load_checkpoint, match_graphs


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score one layered RobustHunter query against one candidate provenance subgraph."
    )
    parser.add_argument("--query", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--allow-untrained", action="store_true")
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.checkpoint is None and not args.allow_untrained:
        parser.error(
            "--checkpoint is required for scientific inference; use --allow-untrained only for the smoke example"
        )

    model = build_matcher(args.config, seed=args.seed, device=args.device)
    checkpoint = (
        load_checkpoint(model, args.checkpoint, device=args.device)
        if args.checkpoint is not None
        else None
    )
    result = {
        "format": "robusthunter-graph-match-v1",
        "scientific_inference": checkpoint is not None,
        "checkpoint": checkpoint,
        "query_path": str(args.query),
        "candidate_path": str(args.candidate),
        **match_graphs(model, load(args.query), load(args.candidate)),
    }
    text = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is None:
        sys.stdout.write(text)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
