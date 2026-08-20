from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch

from .rewritten.matcher import RewrittenMatcher


@dataclass(frozen=True)
class GraphMatcherConfig:
    """Public, serializable configuration of the current Stage-3 matcher."""

    attribute_dim: int = 64
    hidden_dim: int = 64
    layers: int = 2
    top_k: int = 3
    threshold: float = 0.5
    lambda_entity: float = 0.4
    lambda_behavior: float = 0.5
    lambda_null: float = 0.1
    mu_instance: float = 1.0
    mu_stage: float = 1.0
    null_baseline_init: float = 0.0
    beta_stage: float = 0.2
    variant: str = "Full"

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "GraphMatcherConfig":
        payload = value.get("model", value)
        if not isinstance(payload, Mapping):
            raise TypeError("matcher configuration must be an object")
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise ValueError(f"unknown matcher configuration fields: {unknown}")
        return cls(**{key: payload[key] for key in payload})

    @classmethod
    def from_json(cls, path: str | Path) -> "GraphMatcherConfig":
        return cls.from_mapping(json.loads(Path(path).read_text(encoding="utf-8")))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def set_determinism(seed: int, *, threads: int = 1) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)


def build_matcher(
    config: GraphMatcherConfig | Mapping[str, Any] | str | Path | None = None,
    *,
    seed: int = 20260714,
    device: str | torch.device = "cpu",
) -> RewrittenMatcher:
    if config is None:
        resolved = GraphMatcherConfig()
    elif isinstance(config, GraphMatcherConfig):
        resolved = config
    elif isinstance(config, (str, Path)):
        resolved = GraphMatcherConfig.from_json(config)
    else:
        resolved = GraphMatcherConfig.from_mapping(config)
    set_determinism(seed)
    model = RewrittenMatcher(**resolved.to_dict())
    return model.to(device)


def _safe_torch_load(path: str | Path, map_location: str | torch.device) -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:  # PyTorch versions before weights_only was introduced.
        return torch.load(path, map_location=map_location)


def load_checkpoint(
    model: RewrittenMatcher,
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    checkpoint = _safe_torch_load(path, device)
    if not isinstance(checkpoint, dict):
        raise TypeError("checkpoint must contain a mapping")
    state = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state, dict):
        raise TypeError("checkpoint state_dict must be a mapping")
    saved_config = checkpoint.get("model_config")
    if isinstance(saved_config, dict):
        runtime_config = model.configuration()
        compared = (
            "attribute_dim",
            "hidden_dim",
            "layers",
            "top_k",
            "threshold",
            "lambda",
            "structural_support",
            "behavior_stage_penalty",
            "null_mode",
            "variant",
        )
        mismatches = {
            key: {"checkpoint": saved_config.get(key), "runtime": runtime_config.get(key)}
            for key in compared
            if saved_config.get(key) != runtime_config.get(key)
        }
        if mismatches:
            raise ValueError(f"checkpoint/model configuration mismatch: {mismatches}")
    incompatible = model.load_state_dict(state, strict=strict)
    return {
        "path": str(Path(path)),
        "format": checkpoint.get("format"),
        "fold": checkpoint.get("fold"),
        "seed": checkpoint.get("seed"),
        "model_config": saved_config,
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
    }


def _json_copy(value: Any) -> Any:
    """Reject non-JSON input early and detach the caller's mutable objects."""
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


def match_graphs(
    model: RewrittenMatcher,
    query: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    model.eval()
    with torch.no_grad():
        output = model(_json_copy(query), _json_copy(candidate))
    score = float(output.score)
    if not math.isfinite(score):
        raise ValueError("matcher produced a non-finite score")
    return {
        "score": score,
        "entity_score": float(output.entity_score),
        "behavior_score": float(output.behavior_score),
        "null_rate": float(output.null_rate),
        "explanation": output.explanation,
    }


def score_candidates(
    model: RewrittenMatcher,
    query: Mapping[str, Any],
    candidates: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for index, candidate in enumerate(candidates):
        result = match_graphs(model, query, candidate)
        rows.append(
            {
                "candidate_index": index,
                "candidate_id": str(candidate.get("candidate_id") or f"candidate_{index}"),
                **result,
            }
        )
    return sorted(rows, key=lambda row: (-float(row["score"]), str(row["candidate_id"])))
