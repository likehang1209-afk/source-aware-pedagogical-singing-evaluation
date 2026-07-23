from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ReleaseLogicTests(unittest.TestCase):
    def test_weighted_confusion(self):
        module = load_script("reproduce_metrics.py")
        matrix = module.weighted_confusion(
            np.array([0, 0, 1]),
            np.array([0, 1, 1]),
            np.array([1.0, 1.0, 2.0]),
            2,
        )
        np.testing.assert_allclose(matrix, [[1.0, 1.0], [0.0, 2.0]])

    def test_source_weights_sum_to_source_count(self):
        module = load_script("reproduce_metrics.py")
        frame = pd.DataFrame({"source_uid": ["a", "a", "b"]})
        weight = module.observation_weights(frame, "source_uid")
        self.assertAlmostEqual(float(weight.sum()), 2.0)

    def test_inner_support_summary_has_expected_columns(self):
        module = load_script("summarize_inner_support.py")
        frame = pd.DataFrame(
            {
                "role": ["inner_dev", "inner_dev"],
                "target": ["vibrato", "vibrato"],
                "class": [2, 2],
                "count": [5, 7],
            }
        )
        summary = module.summarize(frame)
        self.assertEqual(int(summary.iloc[0]["minimum"]), 5)
        self.assertEqual(int(summary.iloc[0]["zero_support_cells"]), 0)


if __name__ == "__main__":
    unittest.main()
