"""Public package for the RobustHunter graph-matching paper artifact."""

from .graph_matching import (
    GraphMatcherConfig,
    build_matcher,
    load_checkpoint,
    match_graphs,
    score_candidates,
)
from .training import PairwiseTrainingExample, save_checkpoint, train_pairwise

__all__ = [
    "GraphMatcherConfig",
    "build_matcher",
    "load_checkpoint",
    "match_graphs",
    "score_candidates",
    "PairwiseTrainingExample",
    "save_checkpoint",
    "train_pairwise",
]

__version__ = "0.1.0"
