"""Tests for per-run model parameter range overrides."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from app.core import pipeline
from app.core.data import DataConfig
from app.core.models import MODELS, get
from app.core.optimize import OptConfig
from app.core.space import Param, clone_space, validate_space


class ModelSpaceOverrideTests(unittest.TestCase):
    def test_all_registered_spaces_are_valid(self) -> None:
        for spec in MODELS.values():
            with self.subTest(model=spec.key):
                validate_space(spec.space)

    def test_model_override_is_isolated_from_registry_and_input(self) -> None:
        original = get("xgboost")
        custom = clone_space(original.space)
        custom[0].low = 0.02

        overridden = original.with_space(custom)
        custom[0].low = 0.03

        self.assertEqual(original.space[0].low, 0.01)
        self.assertEqual(overridden.space[0].low, 0.02)

    def test_invalid_ranges_are_rejected(self) -> None:
        invalid = [Param("depth", "int", low=10, high=3)]
        with self.assertRaisesRegex(ValueError, "下限必须小于上限"):
            validate_space(invalid)

        invalid_log = [Param("rate", "float", low=0, high=1, log=True)]
        with self.assertRaisesRegex(ValueError, "下限必须大于 0"):
            validate_space(invalid_log)

        invalid_choices = [Param("kernel", "cat", choices=[])]
        with self.assertRaisesRegex(ValueError, "至少需要一个候选值"):
            validate_space(invalid_choices)

    def test_pipeline_uses_override_for_this_run(self) -> None:
        df = pd.DataFrame({
            "x": list(range(20)),
            "y": [value * 2 + 1 for value in range(20)],
        })
        data_cfg = DataConfig(
            target="y",
            features=["x"],
            categorical=[],
            non_standardize=[],
            test_size=0.2,
            random_state=42,
        )
        override = [
            Param("max_depth", "int", low=2, high=4),
            Param("min_samples_split", "int", low=2, high=5),
            Param("min_samples_leaf", "int", low=1, high=3),
        ]
        captured: dict[str, object] = {}

        def fake_optimize(spec, *_args, **_kwargs):
            captured["space"] = clone_space(spec.space)
            return dict(spec.default_params)

        with patch("app.core.pipeline.opt_mod.optimize", side_effect=fake_optimize):
            result = pipeline.train_one(
                df,
                data_cfg,
                "decision_tree",
                OptConfig(),
                model_space=override,
            )

        self.assertEqual(captured["space"], override)
        self.assertEqual(result.spec.space, override)
        self.assertEqual(get("decision_tree").space[0].low, 10)


if __name__ == "__main__":
    unittest.main()
