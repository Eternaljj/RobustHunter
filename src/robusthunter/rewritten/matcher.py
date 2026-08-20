from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from .alignment import (
    AlignmentResult,
    hard_one_to_one_align,
    one_to_many_alignment_loss,
    soft_align,
)
from .behavior_consistency import BehaviorResult, behavior_consistency
from .contracts import canonical_sha256, validate_matcher_input
from .rgat import SharedGCNEncoder, SharedRGATEncoder, compile_graph_features


MATCHER_VARIANTS = {
    "Full",
    "FlatQuery",
    "NoRelationEncoder",
    "NoPartialMatching",
    "NoBehaviorConsistency",
    "NoStageConstraint",
}


def structural_support_weights(
    layered_query: dict[str, Any],
    query_graph: dict[str, Any],
    *,
    device: torch.device,
    dtype: torch.dtype,
    mu_instance: float = 1.0,
    mu_stage: float = 1.0,
) -> torch.Tensor:
    """Implement Eq. (16) from behavior/instance/stage memberships."""
    if min(mu_instance, mu_stage) < 0:
        raise ValueError("structural support coefficients must be non-negative")
    nodes = list(query_graph.get("nodes") or [])
    edges = list(query_graph.get("edges") or [])
    node_ids = [
        str(node.get("id") or node.get("graph_node_id") or node.get("node_id") or index)
        for index, node in enumerate(nodes)
    ]
    edge_nodes: dict[str, set[str]] = {}
    for index, edge in enumerate(edges):
        edge_id = str(edge.get("id") or edge.get("edge_id") or edge.get("behavior_id") or f"edge_{index}")
        edge_nodes[edge_id] = {
            str(edge.get("source") or edge.get("src") or edge.get("source_graph_node_id") or ""),
            str(edge.get("target") or edge.get("dst") or edge.get("target_graph_node_id") or ""),
        } - {""}

    instances = list(layered_query.get("attack_instances") or layered_query.get("instances") or [])
    instance_nodes: dict[str, set[str]] = {}
    for index, instance in enumerate(instances):
        instance_id = str(instance.get("id") or instance.get("instance_id") or f"instance_{index}")
        members = (
            instance.get("behavior_edge_ids")
            or instance.get("behavior_edges")
            or instance.get("behaviors")
            or instance.get("members")
            or []
        )
        supported: set[str] = set()
        for member in members:
            member_id = str(member)
            supported.update(edge_nodes.get(member_id, set()))
            if member_id in node_ids:
                supported.add(member_id)
        instance_nodes[instance_id] = supported

    for member, raw_instances in (layered_query.get("behavior_to_instance") or {}).items():
        instance_ids = raw_instances if isinstance(raw_instances, list) else [raw_instances]
        supported = edge_nodes.get(str(member), {str(member)} if str(member) in node_ids else set())
        for instance_id in instance_ids:
            instance_nodes.setdefault(str(instance_id), set()).update(supported)

    staged_instances: set[str] = set(map(str, (layered_query.get("instance_to_stage") or {}).keys()))
    for row in layered_query.get("attack_stages") or layered_query.get("stages") or []:
        staged_instances.update(
            str(value)
            for value in (
                row.get("instance_ids")
                or [row.get("instance_id") or row.get("instance") or ""]
            )
            if str(value)
        )
    instance_supported = set().union(*instance_nodes.values()) if instance_nodes else set()
    stage_supported = set().union(
        *(instance_nodes.get(instance_id, set()) for instance_id in staged_instances)
    ) if staged_instances else set()
    return torch.tensor(
        [
            1.0
            + mu_instance * float(node_id in instance_supported)
            + mu_stage * float(node_id in stage_supported)
            for node_id in node_ids
        ],
        dtype=dtype,
        device=device,
    )


@dataclass(frozen=True)
class MatcherOutput:
    score: torch.Tensor
    entity_score: torch.Tensor
    behavior_score: torch.Tensor
    null_rate: torch.Tensor
    alignment: AlignmentResult
    behavior: BehaviorResult
    explanation: dict[str, Any]


class RewrittenMatcher(nn.Module):
    def __init__(
        self,
        *,
        attribute_dim: int = 64,
        hidden_dim: int = 64,
        layers: int = 2,
        top_k: int = 3,
        threshold: float = 0.3,
        lambda_entity: float = 0.4,
        lambda_behavior: float = 0.5,
        lambda_null: float = 0.1,
        mu_instance: float = 1.0,
        mu_stage: float = 1.0,
        null_baseline_init: float = 0.0,
        beta_stage: float = 0.2,
        variant: str = "Full",
    ):
        super().__init__()
        if variant not in MATCHER_VARIANTS:
            raise ValueError(f"unknown matcher variant: {variant}")
        if min(lambda_entity, lambda_behavior, lambda_null) < 0:
            raise ValueError("matcher weights must be non-negative")
        encoder_cls = SharedGCNEncoder if variant == "NoRelationEncoder" else SharedRGATEncoder
        self.encoder = encoder_cls(
            attribute_dim=attribute_dim,
            hidden_dim=hidden_dim,
            layers=layers,
        )
        self.top_k = top_k
        self.threshold = threshold
        self.lambda_entity = lambda_entity
        self.lambda_behavior = lambda_behavior
        self.lambda_null = lambda_null
        self.mu_instance = mu_instance
        self.mu_stage = mu_stage
        self.beta_stage = beta_stage
        self.null_baseline = nn.Parameter(torch.tensor(float(null_baseline_init)))
        self.variant = variant

    def configuration(self) -> dict[str, Any]:
        return {
            "attribute_dim": self.encoder.attribute_dim,
            "hidden_dim": self.encoder.input_projection.out_features,
            "layers": len(self.encoder.layers),
            "top_k": self.top_k,
            "threshold": self.threshold,
            "alpha": [0.2, 0.3, 0.5],
            "lambda": [self.lambda_entity, self.lambda_behavior, self.lambda_null],
            "structural_support": {"mu_instance": self.mu_instance, "mu_stage": self.mu_stage},
            "behavior_stage_penalty": self.beta_stage,
            "null_mode": "joint_candidate_null_softmax",
            "variant": self.variant,
        }

    def forward(
        self,
        layered_query: dict[str, Any],
        candidate: dict[str, Any],
    ) -> MatcherOutput:
        report = validate_matcher_input({"query": layered_query, "candidate": candidate})
        if not report.accepted:
            raise ValueError("; ".join(f"{error.path}: {error.message}" for error in report.errors))
        query_graph = (
            layered_query.get("behavior_graph")
            or layered_query.get("behavior_evidence")
            or (
                {
                    "nodes": layered_query.get("behavior_nodes") or [],
                    "edges": layered_query.get("behavior_edges") or [],
                }
                if "behavior_nodes" in layered_query or "behavior_edges" in layered_query
                else layered_query
            )
        )
        query_features = compile_graph_features(
            query_graph,
            attribute_dim=self.encoder.attribute_dim,
        )
        candidate_features = compile_graph_features(
            candidate,
            attribute_dim=self.encoder.attribute_dim,
        )
        query_embedding = self.encoder(query_features)
        candidate_embedding = self.encoder(candidate_features)
        node_weights = structural_support_weights(
            layered_query,
            query_graph,
            device=query_embedding.device,
            dtype=query_embedding.dtype,
            mu_instance=self.mu_instance,
            mu_stage=self.mu_stage,
        )
        aligner = (
            hard_one_to_one_align
            if self.variant == "NoPartialMatching"
            else soft_align
        )
        alignment = aligner(
            query_features.node_types,
            candidate_features.node_types,
            query_features.attributes,
            candidate_features.attributes,
            query_embedding,
            candidate_embedding,
            **(
                {}
                if self.variant == "NoPartialMatching"
                else {
                    "top_k": self.top_k,
                    "threshold": self.threshold,
                    "null_logit": self.null_baseline,
                    "structural_weights": node_weights,
                }
            ),
            **(
                {"structural_weights": node_weights}
                if self.variant == "NoPartialMatching"
                else {}
            ),
        )
        behavior_query = layered_query
        if self.variant == "FlatQuery":
            behavior_query = {"behavior_graph": query_graph}
        behavior = behavior_consistency(
            behavior_query,
            candidate,
            alignment.similarity,
            beta_stage=self.beta_stage,
            use_stage_constraint=self.variant != "NoStageConstraint",
        )
        behavior_score = (
            behavior.score
            if self.variant != "NoBehaviorConsistency"
            else behavior.score * 0.0
        )
        score = (
            self.lambda_entity * alignment.entity_score
            + self.lambda_behavior * behavior_score
            - self.lambda_null * alignment.null_rate
        )
        decisions = []
        for query_index, query_id in enumerate(query_features.node_ids):
            candidates = []
            for rank in range(alignment.topk_indices.shape[1]):
                candidate_index = int(alignment.topk_indices[query_index, rank])
                if not bool(alignment.compatibility[query_index, candidate_index]):
                    continue
                candidates.append(
                    {
                        "candidate_node_id": candidate_features.node_ids[candidate_index],
                        "probability": float(alignment.topk_scores[query_index, rank].detach()),
                        "similarity": float(alignment.similarity[query_index, candidate_index].detach()),
                    }
                )
            decisions.append(
                {
                    "query_node_id": query_id,
                    "matched": bool(alignment.matched[query_index]),
                    "structural_weight": float(alignment.structural_weights[query_index].detach()),
                    "null_probability": float(alignment.null_probability[query_index].detach()),
                    "top_k": candidates,
                }
            )
        config = self.configuration()
        explanation = {
            "score": float(score.detach()),
            "S_E": float(alignment.entity_score.detach()),
            "S_B": float(behavior_score.detach()),
            "P_null": float(alignment.null_rate.detach()),
            "node_decisions": decisions,
            "instance_path_evidence": list(behavior.instance_evidence),
            "weak_order_conflicts": list(behavior.weak_order_conflicts),
            "stage_violation_ratio": behavior.stage_violation_ratio,
            "stage_penalty": behavior.stage_penalty,
            "query_feature_sha256": query_features.feature_sha256,
            "candidate_feature_sha256": candidate_features.feature_sha256,
            "config": config,
            "config_sha256": canonical_sha256(config),
        }
        return MatcherOutput(
            score,
            alignment.entity_score,
            behavior_score,
            alignment.null_rate,
            alignment,
            behavior,
            explanation,
        )


def matcher_losses(
    positive: MatcherOutput,
    negative: MatcherOutput,
    *,
    margin: float = 0.2,
    acceptable_node_matches: list[list[int] | None] | None = None,
    alignment_mask: list[bool] | None = None,
    behavior_supervision_available: bool = True,
    gamma_alignment: float = 1.0,
    gamma_behavior: float = 1.0,
) -> dict[str, torch.Tensor | int]:
    rank = F.relu(positive.score.new_tensor(margin) - positive.score + negative.score)
    if acceptable_node_matches is None:
        align = positive.score * 0.0
        alignment_count = 0
    else:
        alignment = positive.alignment
        revised_probabilities = hasattr(alignment, "transport")
        align, alignment_count = one_to_many_alignment_loss(
            alignment.transport if revised_probabilities else alignment.similarity,
            acceptable_node_matches,
            mask=alignment_mask,
            probabilities=revised_probabilities,
            null_probability=(
                alignment.null_probability if revised_probabilities else None
            ),
            structural_weights=(
                alignment.structural_weights if revised_probabilities else None
            ),
        )
    behavior = (
        F.relu(
            positive.score.new_tensor(margin)
            - positive.behavior_score
            + negative.behavior_score
        )
        if behavior_supervision_available
        else positive.score * 0.0
    )
    total = rank + gamma_alignment * align + gamma_behavior * behavior
    return {
        "loss": total,
        "rank_loss": rank,
        "alignment_loss": align,
        "behavior_loss": behavior,
        "alignment_supervised_count": alignment_count,
        "behavior_supervised_count": int(behavior_supervision_available),
    }
