import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from lib.models.layers.attn import HMoE
from lib.models.layers.bilift import BiLift
from tools.profile_lifttrack import (
    build_profile_result,
    count_target_modules,
    percentile,
    measure_training_steps,
    validate_profile_result,
    write_profile_json,
)


class LiftTrackProfilerTests(unittest.TestCase):
    def test_profiler_can_run_as_a_direct_script(self):
        repo_root = Path(__file__).resolve().parents[1]
        completed = subprocess.run(
            [sys.executable, str(repo_root / "tools" / "profile_lifttrack.py"), "--help"],
            cwd=repo_root,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--config", completed.stdout)

    def test_percentile_uses_linear_interpolation(self):
        values = [4.0, 1.0, 3.0, 2.0]

        self.assertEqual(percentile(values, 0), 1.0)
        self.assertEqual(percentile(values, 50), 2.5)
        self.assertAlmostEqual(percentile(values, 90), 3.7)
        self.assertEqual(percentile(values, 100), 4.0)
        with self.assertRaisesRegex(ValueError, "non-empty"):
            percentile([], 50)
        with self.assertRaisesRegex(ValueError, "between 0 and 100"):
            percentile(values, 101)

    def test_module_counting_is_recursive_and_type_specific(self):
        model = nn.Sequential(
            BiLift(dim=16, rank=4),
            nn.Sequential(
                HMoE(dim=16, experts=2, slots=2, hid_dim=2),
                BiLift(dim=16, rank=2),
            ),
        )

        self.assertEqual(count_target_modules(model), {"hmoe": 1, "bilift": 2})

    def test_profile_result_has_stable_json_schema(self):
        result = build_profile_result(
            config="rgbt_lifttrack",
            checkpoint=None,
            device="cuda:0",
            warmup=2,
            iterations=4,
            parameter_counts={"total": 100, "trainable": 20},
            module_counts={"hmoe": 0, "bilift": 2},
            latencies_ms=[4.0, 1.0, 3.0, 2.0],
            memory_bytes={"peak_allocated": 1000, "peak_reserved": 2000},
            output_shapes={"pred_boxes": [1, 1, 4], "score_map": [1, 1, 16, 16]},
            finite_output=True,
            input_metadata={"source": "lasher", "sequence": "smoke_sequence_000"},
        )

        validate_profile_result(result)
        self.assertEqual(result["schema_version"], 1)
        self.assertEqual(result["latency_ms"]["mean"], 2.5)
        self.assertEqual(result["latency_ms"]["median"], 2.5)
        self.assertAlmostEqual(result["latency_ms"]["p90"], 3.7)
        self.assertEqual(result["fps"], 400.0)
        self.assertTrue(result["outputs"]["finite"])

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "profile.json"
            write_profile_json(output, result)
            loaded = json.loads(output.read_text())
        self.assertEqual(loaded, result)

        invalid = dict(result)
        invalid.pop("memory_bytes")
        with self.assertRaisesRegex(ValueError, "memory_bytes"):
            validate_profile_result(invalid)

    def test_training_measurement_runs_forward_backward_and_step(self):
        class TinyTracker(nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = nn.Parameter(torch.tensor(1.0))

            def forward(self, template, search):
                value = self.scale * (template.mean() + search.mean())
                return {
                    "pred_boxes": value.expand(1, 1, 4),
                    "score_map": value.expand(1, 1, 2, 2),
                }

        model = TinyTracker()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        template = torch.ones(1, 6, 4, 4)
        search = torch.ones(1, 6, 8, 8)
        before = model.scale.detach().clone()

        output, latencies, memory, loss = measure_training_steps(
            model,
            optimizer,
            template,
            search,
            torch.device("cpu"),
            warmup=1,
            iterations=2,
        )

        self.assertEqual(len(latencies), 2)
        self.assertTrue(all(value > 0 for value in latencies))
        self.assertEqual(memory, {"peak_allocated": 0, "peak_reserved": 0})
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(output["pred_boxes"]).all())
        self.assertFalse(torch.equal(model.scale.detach(), before))


if __name__ == "__main__":
    unittest.main()
