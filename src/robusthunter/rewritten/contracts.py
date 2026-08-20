from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CONTRACT_ID = "rewritten-pipeline-v2.0"
NODE_TYPES = {"Process", "File", "Socket", "Registry"}
STAGE_VOCABULARY = (
    "Initial Compromise",
    "Establish Foothold",
    "Escalate Privilege",
    "Internal Reconnaissance",
    "Move Laterally",
    "Maintain Persistence",
    "Complete Mission",
)

_FORBIDDEN_MATCHER_FIELDS = {
    "anchor_precision",
    "anchor_relevance",
    "ground_truth",
    "ground_truth_nodes",
    "ground_truth_edges",
    "gt_nodes",
    "gt_edges",
    "label_source",
}


@dataclass(frozen=True)
class ContractError:
    code: str
    path: str
    message: str


@dataclass(frozen=True)
class ValidationReport:
    accepted: bool
    errors: tuple[ContractError, ...]


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_rewritten_config(config_root: str | Path | None = None) -> dict[str, Any]:
    root = (
        Path(config_root)
        if config_root is not None
        else Path(__file__).resolve().parents[2] / "config"
    )
    path = root / "rewritten_pipeline_v2.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("contract_id") != CONTRACT_ID:
        raise ValueError(f"wrong rewritten contract id in {path}")
    if set(config.get("node_types", [])) != NODE_TYPES:
        raise ValueError(f"rewritten node vocabulary mismatch in {path}")
    if tuple(config.get("stage_vocabulary", [])) != STAGE_VOCABULARY:
        raise ValueError(f"rewritten stage vocabulary mismatch in {path}")
    return {
        "config": config,
        "config_sha256": canonical_sha256(config),
        **config,
    }


def _is_sha256(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text)


def _contract_errors(
    artifact: dict[str, Any],
    *,
    expected_stage: str,
) -> list[ContractError]:
    contract = artifact.get("contract")
    if not isinstance(contract, dict):
        return [
            ContractError(
                "missing_contract",
                "contract",
                "artifact contract metadata is required",
            )
        ]

    errors: list[ContractError] = []
    if contract.get("contract_id") != CONTRACT_ID:
        errors.append(
            ContractError(
                "wrong_contract_id",
                "contract.contract_id",
                f"expected {CONTRACT_ID}",
            )
        )
    if contract.get("stage") != expected_stage:
        errors.append(
            ContractError(
                "wrong_stage",
                "contract.stage",
                f"expected {expected_stage}",
            )
        )
    for field in ("input_sha256", "config_sha256"):
        if not _is_sha256(contract.get(field)):
            errors.append(
                ContractError(
                    f"invalid_{field}",
                    f"contract.{field}",
                    f"{field} must be a lowercase SHA-256 value",
                )
            )
    return errors


def validate_layered_artifact(artifact: dict[str, Any]) -> ValidationReport:
    errors = _contract_errors(artifact, expected_stage="layered_query")
    for index, row in enumerate(artifact.get("attack_stages", [])):
        stage = row.get("stage") if isinstance(row, dict) else None
        if stage not in STAGE_VOCABULARY:
            errors.append(
                ContractError(
                    "unknown_stage",
                    f"attack_stages[{index}].stage",
                    f"unknown seven-stage value: {stage!r}",
                )
            )
    return ValidationReport(not errors, tuple(errors))


def validate_candidate_artifact(
    artifact: dict[str, Any],
    *,
    expected_layered_sha256: str,
) -> ValidationReport:
    errors = _contract_errors(artifact, expected_stage="candidate_sampling")
    contract = artifact.get("contract", {})
    actual = contract.get("parent_artifact_sha256")
    if actual != expected_layered_sha256:
        errors.append(
            ContractError(
                "parent_hash_mismatch",
                "contract.parent_artifact_sha256",
                "candidate artifact does not reference the validated layered artifact",
            )
        )
    return ValidationReport(not errors, tuple(errors))


def _walk_fields(value: Any, path: str = ""):
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            yield str(key), child_path
            yield from _walk_fields(child, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk_fields(child, f"{path}[{index}]")


def validate_matcher_input(payload: dict[str, Any]) -> ValidationReport:
    errors = []
    for field, path in _walk_fields(payload):
        if field.lower() in _FORBIDDEN_MATCHER_FIELDS:
            errors.append(
                ContractError(
                    "forbidden_matcher_field",
                    path,
                    "ground-truth and anchor-quality fields cannot enter matcher inference",
                )
            )
    return ValidationReport(not errors, tuple(errors))


def prepare_immutable_run_directory(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=False)
    return target
