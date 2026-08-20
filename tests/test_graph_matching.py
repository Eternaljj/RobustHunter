from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from robusthunter import (
    GraphMatcherConfig,
    PairwiseTrainingExample,
    build_matcher,
    load_checkpoint,
    match_graphs,
    save_checkpoint,
    train_pairwise,
)


ROOT = Path(__file__).resolve().parents[1]


class GraphMatchingSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.query = json.loads((ROOT / "examples/query.json").read_text(encoding="utf-8"))
        cls.candidate = json.loads((ROOT / "examples/candidate.json").read_text(encoding="utf-8"))

    def test_smoke_result_is_deterministic_and_explained(self) -> None:
        first = match_graphs(build_matcher(GraphMatcherConfig(), seed=17), self.query, self.candidate)
        second = match_graphs(build_matcher(GraphMatcherConfig(), seed=17), self.query, self.candidate)
        self.assertTrue(math.isfinite(first["score"]))
        self.assertEqual(first, second)
        self.assertEqual(first["explanation"]["config"]["null_mode"], "joint_candidate_null_softmax")
        self.assertEqual(len(first["explanation"]["instance_path_evidence"]), 1)
        self.assertTrue(first["explanation"]["instance_path_evidence"][0]["paths"])

    def test_ground_truth_fields_are_rejected(self) -> None:
        query = dict(self.query)
        query["ground_truth_nodes"] = ["10"]
        with self.assertRaisesRegex(ValueError, "ground-truth"):
            match_graphs(build_matcher(seed=17), query, self.candidate)

    def test_unknown_config_field_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown matcher configuration"):
            GraphMatcherConfig.from_mapping({"hidden_dim": 64, "not_a_parameter": 1})

    def test_one_pairwise_training_step_and_checkpoint_round_trip(self) -> None:
        negative = json.loads(json.dumps(self.candidate))
        negative["candidate_id"] = "negative"
        negative["nodes"][0]["name"] = "unrelated"
        model = build_matcher(seed=19)
        history = train_pairwise(
            model,
            [PairwiseTrainingExample(self.query, self.candidate, negative)],
            epochs=1,
            seed=19,
        )
        self.assertEqual(history[0]["example_count"], 1)
        self.assertTrue(math.isfinite(float(history[0]["mean_loss"])))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            save_checkpoint(model, path, metadata={"purpose": "unit-test"})
            restored = build_matcher(seed=20)
            metadata = load_checkpoint(restored, path)
            self.assertEqual(metadata["format"], "robusthunter-public-graph-matcher-checkpoint-v1")
            self.assertEqual(match_graphs(model, self.query, self.candidate), match_graphs(restored, self.query, self.candidate))

    def test_legacy_null_mode_checkpoint_is_rejected(self) -> None:
        model = build_matcher(seed=21)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.pt"
            payload = {
                "state_dict": model.state_dict(),
                "model_config": {**model.configuration(), "null_mode": "threshold_without_virtual_node"},
            }
            import torch

            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, "configuration mismatch"):
                load_checkpoint(build_matcher(seed=21), path)


if __name__ == "__main__":
    unittest.main()
