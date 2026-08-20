from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


CONTRACT_ID = "method-contract-v1.0"
QUERY_SCHEMA_VERSION = "query-layer-v1"
ANCHOR_SCHEMA_VERSION = "anchor-library-v2"
ATTACK_VERSION = "enterprise-attack-15.1"

NODE_TYPES = frozenset({"Process", "File", "Socket", "Registry"})
RELATIONS = frozenset(
    {"read", "write", "send", "recv", "fork", "exec", "exit", "mmap", "unlink", "connect"}
)
STAGE_ORDER = (
    "Initial Compromise",
    "Establish Foothold",
    "Escalate Privilege",
    "Internal Reconnaissance",
    "Move Laterally",
    "Maintain Persistence",
    "Complete Mission",
)
STAGES = frozenset(STAGE_ORDER)

ENDPOINT_RULES: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    "read": (frozenset({"File", "Socket", "Registry"}), frozenset({"Process"})),
    "write": (frozenset({"Process"}), frozenset({"File", "Registry"})),
    "send": (frozenset({"Process"}), frozenset({"Socket"})),
    "recv": (frozenset({"Socket"}), frozenset({"Process"})),
    "fork": (frozenset({"Process"}), frozenset({"Process"})),
    "exec": (frozenset({"File", "Process"}), frozenset({"Process"})),
    "exit": (frozenset({"Process"}), frozenset({"Process"})),
    "mmap": (frozenset({"Process"}), frozenset({"File"})),
    "unlink": (frozenset({"Process"}), frozenset({"File", "Registry"})),
    "connect": (frozenset({"Process"}), frozenset({"Socket"})),
}

TECHNIQUE_ID_RE = re.compile(r"^T\d{4}(?:\.\d{3})?$")


@dataclass(frozen=True)
class ContractError:
    code: str
    message: str
    path: str = ""


@dataclass
class ValidationReport:
    accepted: bool
    errors: list[ContractError] = field(default_factory=list)
    warnings: list[ContractError] = field(default_factory=list)


def _value(obj: dict[str, Any], *keys: str, default: Any = "") -> Any:
    for key in keys:
        if key in obj and obj[key] not in (None, ""):
            return obj[key]
    return default


def stable_json_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _node_id(node: dict[str, Any]) -> str:
    return str(_value(node, "id", "graph_node_id", "node_id"))


def _edge_id(edge: dict[str, Any], index: int) -> str:
    return str(_value(edge, "id", "edge_id", default=f"edge_{index}"))


def _node_type(node: dict[str, Any]) -> str:
    return str(_value(node, "type", "node_type"))


def _edge_relation(edge: dict[str, Any]) -> str:
    return str(_value(edge, "relation", "type", "label")).lower()


def _edge_endpoint(edge: dict[str, Any], source: bool) -> str:
    return str(
        _value(
            edge,
            *("source", "source_graph_node_id", "src", "from")
            if source
            else ("target", "target_graph_node_id", "dst", "to"),
        )
    )


def validate_behavior_graph(graph: dict[str, Any]) -> ValidationReport:
    errors: list[ContractError] = []
    node_ids: set[str] = set()
    nodes = graph.get("nodes")
    edges = graph.get("edges")
    if not isinstance(nodes, list):
        errors.append(ContractError("nodes_not_list", "nodes must be a list", "nodes"))
        nodes = []
    if not isinstance(edges, list):
        errors.append(ContractError("edges_not_list", "edges must be a list", "edges"))
        edges = []
    for index, node in enumerate(nodes):
        if not isinstance(node, dict):
            errors.append(ContractError("node_not_object", "node must be an object", f"nodes[{index}]"))
            continue
        identifier = _node_id(node)
        node_type = _node_type(node)
        if not identifier:
            errors.append(ContractError("empty_node_id", "node id must be non-empty", f"nodes[{index}].id"))
        elif identifier in node_ids:
            errors.append(ContractError("duplicate_node_id", f"duplicate node id: {identifier}", f"nodes[{index}].id"))
        node_ids.add(identifier)
        if node_type not in NODE_TYPES:
            errors.append(ContractError("unknown_node_type", f"unknown node type: {node_type}", f"nodes[{index}].type"))
    edge_ids: set[str] = set()
    for index, edge in enumerate(edges):
        if not isinstance(edge, dict):
            errors.append(ContractError("edge_not_object", "edge must be an object", f"edges[{index}"))
            continue
        identifier = _edge_id(edge, index)
        relation = _edge_relation(edge)
        source = _edge_endpoint(edge, True)
        target = _edge_endpoint(edge, False)
        if identifier in edge_ids:
            errors.append(ContractError("duplicate_edge_id", f"duplicate edge id: {identifier}", f"edges[{index}].id"))
        edge_ids.add(identifier)
        if relation not in RELATIONS:
            errors.append(ContractError("unknown_relation", f"unknown relation: {relation}", f"edges[{index}].relation"))
            continue
        if source not in node_ids or target not in node_ids:
            errors.append(ContractError("missing_edge_endpoint", "edge endpoint is not a known node", f"edges[{index}"))
            continue
        source_type = next(_node_type(node) for node in nodes if isinstance(node, dict) and _node_id(node) == source)
        target_type = next(_node_type(node) for node in nodes if isinstance(node, dict) and _node_id(node) == target)
        allowed_source, allowed_target = ENDPOINT_RULES[relation]
        if source_type not in allowed_source or target_type not in allowed_target:
            errors.append(
                ContractError(
                    "invalid_endpoint_direction",
                    f"{relation} does not allow {source_type}->{target_type}",
                    f"edges[{index}]",
                )
            )
    return ValidationReport(accepted=not errors, errors=errors)


def validate_layered_query(layered: dict[str, Any]) -> ValidationReport:
    errors: list[ContractError] = []
    contract = layered.get("contract") or {}
    if contract.get("contract_id") != CONTRACT_ID:
        errors.append(ContractError("contract_id_mismatch", "layered query contract id mismatch", "contract.contract_id"))
    for key in ("behavior_nodes", "behavior_edges", "procedure_instances", "stage_intentions"):
        if not isinstance(layered.get(key), list):
            errors.append(ContractError("missing_layer", f"{key} must be a list", key))
    stage_ids: set[str] = set()
    for index, stage in enumerate(layered.get("stage_intentions") or []):
        if not isinstance(stage, dict):
            errors.append(ContractError("stage_not_object", "stage must be an object", f"stage_intentions[{index}"))
            continue
        label = str(stage.get("stage", ""))
        if label not in STAGES:
            errors.append(ContractError("unknown_stage", f"unknown stage: {label}", f"stage_intentions[{index}].stage"))
        identifier = str(stage.get("id", ""))
        if identifier in stage_ids:
            errors.append(ContractError("duplicate_stage_id", f"duplicate stage id: {identifier}", f"stage_intentions[{index}].id"))
        stage_ids.add(identifier)
    for index, instance in enumerate(layered.get("procedure_instances") or []):
        if not isinstance(instance, dict):
            errors.append(ContractError("procedure_not_object", "procedure must be an object", f"procedure_instances[{index}"))
            continue
        technique = str(instance.get("technique_id", instance.get("technique", "")))
        if technique != "UNASSIGNED" and not TECHNIQUE_ID_RE.fullmatch(technique):
            errors.append(ContractError("invalid_technique_id", f"invalid technique id: {technique}", f"procedure_instances[{index}].technique_id"))
        if not instance.get("behavior_edges"):
            errors.append(ContractError("empty_procedure_members", "procedure must contain behavior edges", f"procedure_instances[{index}]"))
    return ValidationReport(accepted=not errors, errors=errors)


def load_frozen_config(config_root: Path | None = None) -> dict[str, Any]:
    root = config_root or Path(__file__).resolve().parents[1] / "config"
    manifest_path = root / "method_contract_v1.json"
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("contract_id") != CONTRACT_ID or manifest.get("attack_version") != ATTACK_VERSION:
        raise ValueError("frozen method contract metadata mismatch")
    result = {"manifest": manifest}
    for filename, expected_hash in (manifest.get("source_files") or {}).items():
        path = root / str(filename)
        if not path.exists() or sha256_file(path).lower() != str(expected_hash).lower():
            raise ValueError(f"frozen source hash mismatch: {filename}")
    for filename in manifest.get("files", []):
        path = root / str(filename)
        expected_hash = str((manifest.get("file_hashes") or {}).get(str(filename), "")).lower()
        actual_hash = sha256_file(path).lower()
        if not expected_hash or actual_hash != expected_hash:
            raise ValueError(f"frozen config hash mismatch: {filename}")
        with path.open("r", encoding="utf-8") as handle:
            result[path.stem] = json.load(handle)
    return result
