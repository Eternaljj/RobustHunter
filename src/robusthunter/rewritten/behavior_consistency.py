from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .contracts import STAGE_VOCABULARY
from .rgat import SYSCALL_TO_RELATION


@dataclass(frozen=True)
class PathRecord:
    index: int
    edge_ids: tuple[str, ...]
    node_indices: tuple[int, ...]
    relations: tuple[str, ...]
    times: tuple[float, ...]


@dataclass(frozen=True)
class BehaviorResult:
    score: torch.Tensor
    instance_evidence: tuple[dict[str, Any], ...]
    weak_order_conflicts: tuple[dict[str, Any], ...]
    stage_violation_ratio: float
    stage_penalty: float


def _edge_id(edge: dict[str, Any], fallback: int) -> str:
    return str(edge.get("hash_id") or edge.get("id") or edge.get("edge_id") or edge.get("Unnamed: 0") or fallback).replace(".0", "")


def _endpoint(edge: dict[str, Any], source: bool) -> str:
    keys = ("source", "src", "source_graph_node_id") if source else ("target", "dst", "target_graph_node_id")
    for key in keys:
        if edge.get(key) not in (None, ""):
            return str(edge[key]).replace(".0", "")
    return ""


def reconstruct_retained_paths(candidate: dict[str, Any]) -> tuple[PathRecord, ...]:
    nodes = list(candidate.get("nodes") or [])
    aliases: dict[str, int] = {}
    for index, node in enumerate(nodes):
        # Candidate exporters use ``row_id`` for edge endpoints while retaining
        # ``n_id`` as a stable node identifier; both must resolve to one node.
        for key in ("id", "graph_node_id", "node_id", "Unnamed: 0", "n_id", "row_id"):
            if node.get(key) not in (None, ""):
                aliases[str(node[key]).replace(".0", "")] = index
    edges = list(candidate.get("edges") or [])
    by_id = {_edge_id(edge, index): edge for index, edge in enumerate(edges)}
    output: list[PathRecord] = []
    for path_index, raw_path in enumerate(candidate.get("paths") or []):
        edge_ids = tuple(str(value).replace(".0", "") for value in raw_path.get("edge_ids") or [])
        selected = [by_id.get(edge_id) for edge_id in edge_ids]
        if not selected or any(edge is None for edge in selected):
            continue
        node_indices: list[int] = []
        valid = True
        for edge_index, edge in enumerate(selected):
            assert edge is not None
            source = aliases.get(_endpoint(edge, True))
            target = aliases.get(_endpoint(edge, False))
            if source is None or target is None:
                valid = False
                break
            if edge_index == 0:
                node_indices.extend((source, target))
            elif node_indices[-1] == source:
                node_indices.append(target)
            else:
                valid = False
                break
        if not valid:
            continue
        relations = tuple(
            SYSCALL_TO_RELATION.get(
                str(edge.get("syscall") or edge.get("relation") or "").replace(".0", "").lower(),
                str(edge.get("relation") or "").lower(),
            )
            for edge in selected
            if edge is not None
        )
        times = tuple(float(edge.get("time") or edge.get("timestamp") or 0.0) for edge in selected if edge is not None)
        output.append(PathRecord(path_index, edge_ids, tuple(node_indices), relations, times))
    return tuple(output)


def _query_node_id(node: dict[str, Any], fallback: int) -> str:
    return str(node.get("id") or node.get("graph_node_id") or node.get("node_id") or fallback)


def _query_edge_id(edge: dict[str, Any], fallback: int) -> str:
    return str(edge.get("id") or edge.get("edge_id") or edge.get("behavior_id") or f"edge_{fallback}")


def _query_endpoint(edge: dict[str, Any], source: bool) -> str:
    keys = ("source", "src", "source_graph_node_id") if source else ("target", "dst", "target_graph_node_id")
    for key in keys:
        if edge.get(key) not in (None, ""):
            return str(edge[key])
    return ""


def _instance_members(instance: dict[str, Any]) -> set[str]:
    raw = (
        instance.get("behavior_edge_ids")
        or instance.get("behavior_edges")
        or instance.get("behaviors")
        or instance.get("members")
        or instance.get("member_edge_ids")
        or []
    )
    return {str(value) for value in raw}


def behavior_consistency(
    layered_query: dict[str, Any],
    candidate: dict[str, Any],
    node_similarity: torch.Tensor,
    *,
    beta_start: float = 0.25,
    beta_relation: float = 0.25,
    beta_direction: float = 0.25,
    beta_causal: float = 0.25,
    beta_stage: float = 0.2,
    use_stage_constraint: bool = True,
) -> BehaviorResult:
    if abs(beta_start + beta_relation + beta_direction + beta_causal - 1.0) > 1e-7:
        raise ValueError("behavior weights must sum to one")
    if not 0.0 <= beta_stage <= 1.0:
        raise ValueError("stage penalty weight must be in [0, 1]")
    graph = (
        layered_query.get("behavior_graph")
        or layered_query.get("behavior_evidence")
        or {
            "nodes": layered_query.get("behavior_nodes") or [],
            "edges": layered_query.get("behavior_edges") or [],
        }
    )
    nodes = list(graph.get("nodes") or [])
    edges = list(graph.get("edges") or [])
    node_index = {_query_node_id(node, index): index for index, node in enumerate(nodes)}
    retained_paths = reconstruct_retained_paths(candidate)
    instances = list(layered_query.get("attack_instances") or layered_query.get("instances") or [])
    if not instances:
        instances = [{"id": "implicit_all", "behavior_edge_ids": [_query_edge_id(edge, index) for index, edge in enumerate(edges)]}]
    stages_by_instance: dict[str, list[str]] = {}
    for row in layered_query.get("attack_stages") or layered_query.get("stages") or []:
        instance_ids = row.get("instance_ids") or [
            row.get("instance_id") or row.get("instance") or ""
        ]
        stage = str(row.get("stage") or "")
        if stage and stage not in STAGE_VOCABULARY:
            raise ValueError(f"unknown seven-stage value: {stage!r}")
        for raw_instance_id in instance_ids:
            instance_id = str(raw_instance_id)
            if instance_id and stage:
                stages_by_instance.setdefault(instance_id, []).append(stage)
    for raw_instance_id, raw_stages in (layered_query.get("instance_to_stage") or {}).items():
        for raw_stage in raw_stages if isinstance(raw_stages, list) else [raw_stages]:
            stage = str(raw_stage)
            if stage not in STAGE_VOCABULARY:
                raise ValueError(f"unknown seven-stage value: {stage!r}")
            stages_by_instance.setdefault(str(raw_instance_id), []).append(stage)
    stages_by_instance = {
        instance_id: list(dict.fromkeys(stages))
        for instance_id, stages in stages_by_instance.items()
    }

    instance_scores: list[torch.Tensor] = []
    evidence: list[dict[str, Any]] = []
    observed_stage_times: list[tuple[int, float, str]] = []
    for instance_index, instance in enumerate(instances):
        instance_id = str(instance.get("id") or instance.get("instance_id") or f"instance_{instance_index}")
        members = _instance_members(instance)
        member_edges = [
            (edge_index, edge)
            for edge_index, edge in enumerate(edges)
            if not members or _query_edge_id(edge, edge_index) in members
        ]
        edge_scores: list[torch.Tensor] = []
        chosen_rows: list[dict[str, Any]] = []
        chosen_times: list[float] = []
        for edge_index, edge in member_edges:
            source_index = node_index.get(_query_endpoint(edge, True))
            target_index = node_index.get(_query_endpoint(edge, False))
            relation = str(edge.get("relation") or edge.get("type") or "").lower()
            if source_index is None or target_index is None:
                continue
            best_score = node_similarity.new_tensor(0.0)
            best_row: dict[str, Any] | None = None
            for path in retained_paths:
                if not path.node_indices:
                    continue
                start_index = path.node_indices[0]
                end_index = path.node_indices[-1]
                start = (node_similarity[source_index, start_index] + node_similarity[target_index, end_index]) / 2.0
                relation_score = node_similarity.new_tensor(float(relation in path.relations))
                monotonic = all(left <= right for left, right in zip(path.times, path.times[1:]))
                direction = node_similarity.new_tensor(float(monotonic))
                causal = node_similarity.new_tensor(
                    float(
                        len(path.node_indices) == len(path.edge_ids) + 1
                        and len(path.edge_ids) > 0
                        and len(path.edge_ids) <= 3
                    )
                )
                score = (
                    beta_start * start
                    + beta_relation * relation_score
                    + beta_direction * direction
                    + beta_causal * causal
                )
                if best_row is None or float(score.detach()) > float(best_score.detach()):
                    best_score = score
                    best_row = {
                        "query_edge_id": _query_edge_id(edge, edge_index),
                        "path_index": path.index,
                        "edge_ids": list(path.edge_ids),
                        "start_support": float(start.detach()),
                        "relation_support": float(relation_score.detach()),
                        "direction_support": float(direction.detach()),
                        "causal_support": float(causal.detach()),
                        "score": float(score.detach()),
                    }
                    if path.times:
                        best_row["time"] = sum(path.times) / len(path.times)
            edge_scores.append(best_score)
            if best_row is not None:
                chosen_rows.append(best_row)
                if "time" in best_row:
                    chosen_times.append(float(best_row["time"]))
        instance_score = torch.stack(edge_scores).mean() if edge_scores else node_similarity.sum() * 0.0
        instance_scores.append(instance_score)
        stage_values = stages_by_instance.get(instance_id, [])
        mean_time = sum(chosen_times) / len(chosen_times) if chosen_times else None
        for stage in stage_values:
            if use_stage_constraint and mean_time is not None:
                observed_stage_times.append((STAGE_VOCABULARY.index(stage), mean_time, instance_id))
        evidence.append(
            {
                "instance_id": instance_id,
                "member_edge_count": len(member_edges),
                "matched_edge_count": sum(row["score"] > 0 for row in chosen_rows),
                "score": float(instance_score.detach()),
                "stages": stage_values,
                "paths": chosen_rows,
            }
        )
    conflicts: list[dict[str, Any]] = []
    ordered = sorted(observed_stage_times, key=lambda row: row[0])
    comparable = 0
    for left_index, left in enumerate(ordered):
        for right in ordered[left_index + 1 :]:
            if left[0] >= right[0]:
                continue
            comparable += 1
            if left[1] > right[1]:
                conflicts.append(
                    {
                        "earlier_instance": left[2],
                        "later_instance": right[2],
                        "earlier_stage_rank": left[0],
                        "later_stage_rank": right[0],
                    }
                )
    base = torch.stack(instance_scores).mean() if instance_scores else node_similarity.sum() * 0.0
    violation_ratio = len(conflicts) / max(1, comparable)
    penalty = 1.0 - beta_stage * violation_ratio if use_stage_constraint else 1.0
    return BehaviorResult(
        base * penalty,
        tuple(evidence),
        tuple(conflicts),
        violation_ratio,
        penalty,
    )
