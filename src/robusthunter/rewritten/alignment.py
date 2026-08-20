from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class AlignmentResult:
    similarity: torch.Tensor
    transport: torch.Tensor
    null_probability: torch.Tensor
    compatibility: torch.Tensor
    topk_indices: torch.Tensor
    topk_scores: torch.Tensor
    matched: torch.Tensor
    entity_score: torch.Tensor
    null_rate: torch.Tensor
    structural_weights: torch.Tensor


def type_compatibility(
    query_types: tuple[str, ...] | list[str],
    candidate_types: tuple[str, ...] | list[str],
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    return torch.tensor(
        [[query == candidate for candidate in candidate_types] for query in query_types],
        dtype=torch.bool,
        device=device,
    )


def soft_align(
    query_types: tuple[str, ...] | list[str],
    candidate_types: tuple[str, ...] | list[str],
    query_attributes: torch.Tensor,
    candidate_attributes: torch.Tensor,
    query_embeddings: torch.Tensor,
    candidate_embeddings: torch.Tensor,
    *,
    alpha_type: float = 0.2,
    alpha_attribute: float = 0.3,
    alpha_embedding: float = 0.5,
    top_k: int = 3,
    threshold: float = 0.3,
    null_logit: torch.Tensor | float = 0.0,
    structural_weights: torch.Tensor | None = None,
) -> AlignmentResult:
    if abs(alpha_type + alpha_attribute + alpha_embedding - 1.0) > 1e-7:
        raise ValueError("alignment weights must sum to one")
    if top_k < 1:
        raise ValueError("top_k must be positive")
    query_count = len(query_types)
    candidate_count = len(candidate_types)
    device = query_embeddings.device
    compatibility = type_compatibility(query_types, candidate_types, device=device)
    weights = (
        query_embeddings.new_ones((query_count,))
        if structural_weights is None
        else structural_weights.to(device=device, dtype=query_embeddings.dtype).reshape(-1)
    )
    if weights.shape != (query_count,) or bool((weights <= 0).any()):
        raise ValueError("structural weights must be a positive value per query node")
    if query_count == 0:
        empty = query_embeddings.new_empty((0, candidate_count))
        return AlignmentResult(
            empty,
            empty,
            query_embeddings.new_empty((0,)),
            compatibility,
            torch.empty((0, 0), dtype=torch.long, device=device),
            query_embeddings.new_empty((0, 0)),
            torch.empty((0,), dtype=torch.bool, device=device),
            query_embeddings.new_tensor(0.0),
            query_embeddings.new_tensor(0.0),
            weights,
        )
    if candidate_count == 0:
        return AlignmentResult(
            query_embeddings.new_zeros((query_count, 0)),
            query_embeddings.new_zeros((query_count, 0)),
            query_embeddings.new_ones((query_count,)),
            compatibility,
            torch.empty((query_count, 0), dtype=torch.long, device=device),
            query_embeddings.new_empty((query_count, 0)),
            torch.zeros(query_count, dtype=torch.bool, device=device),
            query_embeddings.new_tensor(0.0),
            query_embeddings.new_tensor(1.0),
            weights,
        )
    attribute_similarity = (
        F.normalize(query_attributes.to(device), dim=1)
        @ F.normalize(candidate_attributes.to(device), dim=1).T
    ).clamp(-1.0, 1.0)
    attribute_similarity = (attribute_similarity + 1.0) / 2.0
    embedding_similarity = (
        F.normalize(query_embeddings, dim=1)
        @ F.normalize(candidate_embeddings, dim=1).T
    ).clamp(-1.0, 1.0)
    embedding_similarity = (embedding_similarity + 1.0) / 2.0
    similarity = (
        alpha_type * compatibility.float()
        + alpha_attribute * attribute_similarity
        + alpha_embedding * embedding_similarity
    )
    similarity = similarity.masked_fill(~compatibility, 0.0)
    candidate_logits = similarity.masked_fill(~compatibility, float("-inf"))
    null = torch.as_tensor(null_logit, dtype=similarity.dtype, device=device)
    if null.ndim == 0:
        null = null.expand(query_count)
    else:
        null = null.reshape(-1)
    if null.shape != (query_count,):
        raise ValueError("NULL logit must be shared or have one value per query node")
    joint = torch.cat([candidate_logits, null[:, None]], dim=1)
    joint_probability = torch.softmax(joint, dim=1)
    transport = joint_probability[:, :candidate_count]
    null_probability = joint_probability[:, candidate_count]
    k = min(top_k, candidate_count)
    masked_for_topk = transport.masked_fill(~compatibility, float("-inf"))
    topk_scores, topk_indices = torch.topk(masked_for_topk, k=k, dim=1)
    topk_scores = torch.where(torch.isfinite(topk_scores), topk_scores, torch.zeros_like(topk_scores))
    best_similarity = similarity.masked_fill(~compatibility, float("-inf")).max(dim=1).values
    matched = compatibility.any(dim=1) & (best_similarity >= threshold)
    expected_support = (transport * similarity).sum(dim=1)
    denominator = weights.sum().clamp_min(torch.finfo(weights.dtype).eps)
    entity_score = (weights * expected_support).sum() / denominator
    null_rate = (weights * null_probability).sum() / denominator
    return AlignmentResult(
        similarity,
        transport,
        null_probability,
        compatibility,
        topk_indices,
        topk_scores,
        matched,
        entity_score,
        null_rate,
        weights,
    )


def hard_one_to_one_align(
    query_types: tuple[str, ...] | list[str],
    candidate_types: tuple[str, ...] | list[str],
    query_attributes: torch.Tensor,
    candidate_attributes: torch.Tensor,
    query_embeddings: torch.Tensor,
    candidate_embeddings: torch.Tensor,
    *,
    alpha_type: float = 0.2,
    alpha_attribute: float = 0.3,
    alpha_embedding: float = 0.5,
    structural_weights: torch.Tensor | None = None,
) -> AlignmentResult:
    """Greedy type-constrained one-to-one alignment without NULL rejection.

    This deliberately restrictive matcher is the manuscript's
    ``w/o Partial Matching`` ablation. Selection indices are discrete, while
    the selected similarities remain differentiable.
    """

    base = soft_align(
        query_types,
        candidate_types,
        query_attributes,
        candidate_attributes,
        query_embeddings,
        candidate_embeddings,
        alpha_type=alpha_type,
        alpha_attribute=alpha_attribute,
        alpha_embedding=alpha_embedding,
        top_k=1,
        threshold=0.0,
        structural_weights=structural_weights,
    )
    query_count, candidate_count = base.similarity.shape
    device = query_embeddings.device
    if query_count == 0 or candidate_count == 0:
        return AlignmentResult(
            base.similarity,
            base.transport,
            base.null_probability * 0.0,
            base.compatibility,
            base.topk_indices,
            base.topk_scores,
            torch.zeros(query_count, dtype=torch.bool, device=device),
            query_embeddings.new_tensor(0.0),
            query_embeddings.new_tensor(0.0),
            base.structural_weights,
        )

    available_queries = set(range(query_count))
    available_candidates = set(range(candidate_count))
    assignments: dict[int, int] = {}
    detached = base.similarity.detach()
    while available_queries and available_candidates:
        choices = [
            (float(detached[q, c]), q, c)
            for q in available_queries
            for c in available_candidates
            if bool(base.compatibility[q, c])
        ]
        if not choices:
            break
        _, query_index, candidate_index = max(
            choices,
            key=lambda row: (row[0], -row[1], -row[2]),
        )
        assignments[query_index] = candidate_index
        available_queries.remove(query_index)
        available_candidates.remove(candidate_index)

    indices = torch.zeros((query_count, 1), dtype=torch.long, device=device)
    scores = query_embeddings.new_zeros((query_count, 1))
    matched = torch.zeros(query_count, dtype=torch.bool, device=device)
    for query_index, candidate_index in assignments.items():
        indices[query_index, 0] = candidate_index
        scores[query_index, 0] = base.similarity[query_index, candidate_index]
        matched[query_index] = True
    transport = query_embeddings.new_zeros((query_count, candidate_count))
    for query_index, candidate_index in assignments.items():
        transport[query_index, candidate_index] = 1.0
    denominator = base.structural_weights.sum().clamp_min(
        torch.finfo(base.structural_weights.dtype).eps
    )
    entity_score = (
        base.structural_weights * scores[:, 0]
    ).sum() / denominator
    return AlignmentResult(
        base.similarity,
        transport,
        query_embeddings.new_zeros((query_count,)),
        base.compatibility,
        indices,
        scores,
        matched,
        entity_score,
        query_embeddings.new_tensor(0.0),
        base.structural_weights,
    )


def one_to_many_alignment_loss(
    similarity: torch.Tensor,
    acceptable: list[list[int] | tuple[int, ...] | set[int] | None],
    *,
    mask: list[bool] | None = None,
    epsilon: float = 1e-8,
    probabilities: bool = False,
    null_probability: torch.Tensor | None = None,
    structural_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, int]:
    losses: list[torch.Tensor] = []
    weights: list[torch.Tensor] = []
    for query_index, indices in enumerate(acceptable):
        if mask is not None and not mask[query_index]:
            continue
        if indices is None:
            continue
        valid = sorted({int(index) for index in (indices or []) if 0 <= int(index) < similarity.shape[1]})
        probability = similarity[query_index] if probabilities else torch.softmax(similarity[query_index], dim=0)
        if valid:
            support = probability[valid].sum()
        elif null_probability is not None:
            support = null_probability[query_index]
        else:
            continue
        losses.append(-torch.log(support.clamp_min(epsilon)))
        weights.append(
            similarity.new_tensor(1.0)
            if structural_weights is None
            else structural_weights[query_index].to(similarity)
        )
    zero = similarity.sum() * 0.0
    if not losses:
        return zero, 0
    loss_tensor = torch.stack(losses)
    weight_tensor = torch.stack(weights)
    return (
        (loss_tensor * weight_tensor).sum() / weight_tensor.sum().clamp_min(epsilon),
        len(losses),
    )
