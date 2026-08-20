from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch

from .rewritten.matcher import RewrittenMatcher, matcher_losses


@dataclass(frozen=True)
class PairwiseTrainingExample:
    """One weakly supervised (query, positive, negative) training triple."""

    query: Mapping[str, Any]
    positive: Mapping[str, Any]
    negative: Mapping[str, Any]
    behavior_supervision_available: bool = True


def train_pairwise(
    model: RewrittenMatcher,
    examples: Iterable[PairwiseTrainingExample],
    *,
    epochs: int = 7,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    margin: float = 0.2,
    gradient_clip: float = 5.0,
    gamma_alignment: float = 1.0,
    gamma_behavior: float = 1.0,
    seed: int = 20260714,
) -> list[dict[str, float | int]]:
    """Train with the current ranking + alignment + behavior objective.

    Node-alignment supervision is intentionally absent from this minimal weakly
    supervised API. Call ``matcher_losses`` directly when an audited
    ``acceptable_node_matches`` construction is available.
    """

    rows = list(examples)
    if not rows:
        raise ValueError("at least one pairwise training example is required")
    if epochs <= 0 or learning_rate <= 0 or weight_decay < 0 or gradient_clip <= 0:
        raise ValueError("invalid training hyperparameters")
    random.seed(seed)
    torch.manual_seed(seed)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    history: list[dict[str, float | int]] = []
    for epoch in range(epochs):
        model.train()
        shuffled = list(rows)
        random.shuffle(shuffled)
        totals = {"loss": 0.0, "rank_loss": 0.0, "alignment_loss": 0.0, "behavior_loss": 0.0}
        for example in shuffled:
            optimizer.zero_grad(set_to_none=True)
            positive = model(dict(example.query), dict(example.positive))
            negative = model(dict(example.query), dict(example.negative))
            terms = matcher_losses(
                positive,
                negative,
                margin=margin,
                behavior_supervision_available=example.behavior_supervision_available,
                gamma_alignment=gamma_alignment,
                gamma_behavior=gamma_behavior,
            )
            terms["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
            for key in totals:
                totals[key] += float(terms[key].detach())
        history.append(
            {
                "epoch": epoch + 1,
                "example_count": len(shuffled),
                **{f"mean_{key}": value / len(shuffled) for key, value in totals.items()},
            }
        )
    return history


def save_checkpoint(
    model: RewrittenMatcher,
    path: str | Path,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "robusthunter-public-graph-matcher-checkpoint-v1",
        "model_config": model.configuration(),
        "state_dict": model.state_dict(),
        "metadata": dict(metadata or {}),
    }
    torch.save(payload, target)
