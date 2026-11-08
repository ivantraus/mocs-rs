import importlib.util
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


smoke = load_script("smoke")
fetch_zoo = load_script("fetch_zoo")


class WorkflowScriptTests(unittest.TestCase):
    def test_smoke_config_matrix_records_sbx_as_numeric_only(self):
        config = smoke.toml_config(problem="sf2", crossover="simulated_binary", mutation="adaptive")
        self.assertIn('kind = "sf2"', config)
        self.assertIn("population = 8", config)
        self.assertIn("offspring = 8", config)
        self.assertIn('crossover = "simulated_binary"', config)
        self.assertIn('mutation = "adaptive"', config)
        self.assertNotIn("[training]", config)
        model_config = smoke.toml_config(model="rule", crossover="variable_length")
        self.assertIn('crossover = "variable_length"', model_config)

    def test_smoke_rejects_nonfinite_evidence_values(self):
        with self.assertRaisesRegex(ValueError, "not finite"):
            smoke.finite_tree({"front": [[float("nan")]]})

    def test_front_validation_rejects_dominated_points_but_allows_duplicates(self):
        smoke.validate_front([[1.0, 2.0], [1.0, 2.0], [2.0, 1.0]])
        with self.assertRaisesRegex(ValueError, "dominated point"):
            smoke.validate_front([[1.0, 2.0], [2.0, 3.0]])

    def test_compact_summary_preserves_case_counts_and_diagnostics(self):
        report = {key: 0 for key in (
            "loss", "conditional_loss", "coverage", "generality", "size", "rule_count",
            "unused_rule_fraction", "abstentions", "fallbacks", "out_of_range", "rows",
        )}
        cases = [
            {"case": "numeric:nsga2:sf2", "algorithm": "nsga2", "problem": "sf2",
             "evaluations": 16, "front_size": 1},
            {"case": "operators:k_point:uniform:random", "crossover": "k_point",
             "mutation": "uniform", "mating": "random", "evaluations": 16, "front_size": 1},
            {"case": "model:nsga2:rule:loss_size:best_accuracy", "algorithm": "nsga2",
             "model": "rule", "fitness": "loss_size", "decision": "best_accuracy",
             "evaluations": 16, "front_size": 1, "training_report": report, "heldout_report": report},
            {"case": "historical:bupa:rule:fold0", "training_report": report,
             "heldout_report": report, "train_sha256": "a", "test_sha256": "b"},
        ]
        compact = smoke.compact_evidence({
            "suite": "runtime_smoke_v1", "scope": "tiny", "executable_sha256": "a",
            "script_sha256": "b", "fixtures_sha256": {}, "runs": cases,
        })
        self.assertEqual(compact["counts"], {
            "numeric": 1, "operator_combinations": 1, "model_combinations": 1,
            "historical_diagnostics": 1,
        })
        self.assertEqual(compact["historical"][0]["test_sha256"], "b")

    def test_zoo_conversion_removes_identifier_and_keeps_rows(self):
        archive_path = pathlib.Path("/tmp/zoo.zip")
        if not archive_path.is_file():
            self.skipTest("official Zoo archive is not cached locally")
        archive_bytes = archive_path.read_bytes()
        import hashlib
        self.assertEqual(hashlib.sha256(archive_bytes).hexdigest(), fetch_zoo.SHA256)
        arff, row_count = fetch_zoo.convert_zoo(archive_bytes)
        self.assertEqual(row_count, 101)
        self.assertIn("@attribute type {1,2,3,4,5,6,7}", arff)
        self.assertEqual(arff.count("\n"), 121)
        self.assertNotIn("aardvark", arff)
        self.assertTrue(arff.endswith("\n"))


if __name__ == "__main__":
    unittest.main()
