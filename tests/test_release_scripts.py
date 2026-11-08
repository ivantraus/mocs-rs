import importlib.util
import json
import math
import pathlib
import tempfile
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


quality = load_script("quality")
benchmark = load_script("benchmark")
perf_study = load_script("perf_study")
audit_performance = load_script("audit_performance")
audit_quality = load_script("audit_quality")


class ReleaseScriptTests(unittest.TestCase):
    def test_quality_manifest_persists_and_rejects_incompatible_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            base = pathlib.Path(folder)
            manifest_path, raw_path = base / "manifest.json", base / "raw.jsonl"
            manifest = {"gate_version": "v1", "binary": "abc"}
            first_id = quality.prepare_manifest(manifest_path, raw_path, manifest)
            self.assertEqual(first_id, quality.prepare_manifest(manifest_path, raw_path, manifest))
            with self.assertRaisesRegex(RuntimeError, "manifest differs"):
                quality.prepare_manifest(manifest_path, raw_path, {**manifest, "binary": "changed"})

    def test_quality_manifest_binds_zoo_experiment_thread_count(self):
        with tempfile.TemporaryDirectory() as folder:
            base = pathlib.Path(folder)
            manifest_path, raw_path = base / "manifest.json", base / "raw.jsonl"
            manifest = {"gate_version": "v1", "zoo_experiment_threads": 1}
            quality.prepare_manifest(manifest_path, raw_path, manifest)
            with self.assertRaisesRegex(RuntimeError, "manifest differs"):
                quality.prepare_manifest(manifest_path, raw_path, {**manifest, "zoo_experiment_threads": 4})

    def test_quality_refuses_unmanifested_raw_results(self):
        with tempfile.TemporaryDirectory() as folder:
            base = pathlib.Path(folder)
            raw_path = base / "raw.jsonl"
            raw_path.write_text("{}\n")
            with self.assertRaisesRegex(RuntimeError, "without a manifest"):
                quality.prepare_manifest(base / "manifest.json", raw_path, {"gate_version": "v1"})

    def test_closed_form_true_front_references_match_dense_hypervolume_sweeps(self):
        self.assertEqual(quality.sf2_true_hv(), 67.0 / 3.0)
        sf2 = [[(2 * i / 50000) ** 2, (2 * i / 50000 - 2) ** 2] for i in range(50001)]
        self.assertLessEqual(quality.hv2(sf2, [5.0, 5.0]), quality.sf2_true_hv())
        self.assertLess(quality.sf2_true_hv() - quality.hv2(sf2, [5.0, 5.0]), 0.001)
        zdt6 = []
        for i in range(50001):
            x = i / 50000
            f1 = 1 - math.exp(-4 * x) * math.sin(6 * math.pi * x) ** 6
            zdt6.append([f1, 1 - f1 * f1])
        self.assertAlmostEqual(quality.zdt6_true_hv(), 0.5078775146545909, places=14)
        self.assertLessEqual(quality.hv2(zdt6, [1.1, 1.1]), quality.zdt6_true_hv())
        self.assertLess(quality.zdt6_true_hv() - quality.hv2(zdt6, [1.1, 1.1]), 0.001)

    def test_dense_regression_fixture_mean_baseline_is_preregistered_value(self):
        train = ROOT / "examples/quality/regression-dense-train.arff"
        test = ROOT / "examples/quality/regression-dense-test.arff"
        self.assertEqual(len(quality.read_numeric_targets(train)), 51)
        self.assertEqual(len(quality.read_numeric_targets(test)), 50)
        baseline = quality.mean_baseline_mse(train, test)
        self.assertLessEqual(abs(baseline - 0.3332), 1e-12)
        self.assertLessEqual(abs(0.5 * baseline - 0.1666), 1e-12)
        fixtures = quality.fixture_names("regression_dense")
        self.assertEqual(set(fixtures), {"regression-dense-train.arff", "regression-dense-test.arff"})
        self.assertTrue(all(quality.digest(ROOT / "examples/quality" / name) for name in fixtures))

    def test_benchmark_scientific_identity_catches_changed_front(self):
        a = {"config": {"problem": {"kind": "sf2"}, "algorithm": {"kind": "nsga2"}, "threads": 1},
             "seed": 0, "input_sha256": ["data"], "evaluations": 100, "front": [[1.0, 2.0]]}
        b = {**a, "config": {**a["config"], "threads": 8}}
        self.assertEqual(benchmark.scientific_record(a), benchmark.scientific_record(b))
        b["front"] = [[1.1, 2.0]]
        self.assertNotEqual(benchmark.scientific_record(a), benchmark.scientific_record(b))

    def test_scientific_fingerprint_distinguishes_signed_zero_and_numeric_types(self):
        record = {
            "config": {"problem": {"kind": "sf2"}},
            "seed": 0,
            "input_sha256": [],
            "evaluations": 1,
            "front": [[0.0, 1]],
        }
        negative_zero = {**record, "front": [[-0.0, 1]]}
        float_integer = {**record, "front": [[0.0, 1.0]]}
        self.assertEqual(benchmark.scientific_record(record), benchmark.scientific_record(negative_zero))
        self.assertNotEqual(
            benchmark.scientific_fingerprint(record),
            benchmark.scientific_fingerprint(negative_zero),
        )
        self.assertNotEqual(
            benchmark.scientific_fingerprint(record),
            benchmark.scientific_fingerprint(float_integer),
        )
        self.assertNotEqual(
            perf_study.scientific_fingerprint(record),
            perf_study.scientific_fingerprint(negative_zero),
        )

    def test_performance_audit_catches_signed_zero_pair_mismatch(self):
        base = {
            "config": {"algorithm": {"kind": "nsga2"}, "threads": 1},
            "seed": 0,
            "input_sha256": [],
            "evaluations": 1,
            "front": [[0.0]],
        }
        candidate = {**base, "front": [[-0.0]]}
        manifest = {
            "case_identity": [{"name": "fixture"}],
            "seeds": [0],
            "paired": {
                "fixture\t0": {
                    "baseline": {"scientific_record": base, "run_record": base},
                    "candidate": {"scientific_record": candidate, "run_record": candidate},
                }
            },
            "comparison": {"all_pass": True},
        }
        with tempfile.TemporaryDirectory() as folder:
            path = pathlib.Path(folder) / "manifest.json"
            path.write_text(json.dumps(manifest))
            result = audit_performance.audit_manifest(path, expected_pairs_per_case=1)
        self.assertEqual(result["pairs_checked"], 1)
        self.assertEqual(result["mismatch_count"], 2)
        self.assertFalse(result["audit_passed"])

    def test_performance_resume_binds_script_host_and_gate_settings(self):
        with tempfile.TemporaryDirectory() as folder:
            binary = pathlib.Path(folder) / "mocs"
            binary.write_bytes(b"frozen binary")
            cases = []
            host = {"node": "host-a", "platform": "test"}
            settings = {"primary": "sf2-nsga2", "bootstrap_samples": 20, "bootstrap_seed": 17}
            manifest = {
                "baseline": {"sha256": perf_study.sha256_file(binary)},
                "candidate": None,
                "case_identity": [],
                "seeds": [0, 1],
                "warmups": 1,
                "script_sha256": "script-a",
                "host": host,
                "settings": settings,
            }
            kwargs = {"script_sha256": "script-a", "host": host, "settings": settings}
            perf_study.check_identity(manifest, binary, None, cases, [0, 1], 1, **kwargs)
            for changed in (
                {**kwargs, "script_sha256": "script-b"},
                {**kwargs, "host": {"node": "host-b", "platform": "test"}},
                {**kwargs, "settings": {**settings, "primary": "sf2-spea2"}},
            ):
                with self.assertRaisesRegex(ValueError, "resumable manifest"):
                    perf_study.check_identity(manifest, binary, None, cases, [0, 1], 1, **changed)

    def test_all_synthetic_model_benchmarks_use_variable_length_crossover(self):
        cases = perf_study.load_cases(
            ROOT / "examples/bench", ROOT / "examples/bench/synthetic-200x10.arff"
        )
        models = [case for case in cases if case["command"] == "train"]
        self.assertEqual(len(models), 3)
        for case in models:
            import tomllib

            with case["config_path"].open("rb") as stream:
                config = tomllib.load(stream)
            self.assertEqual(config["operators"]["crossover"], "variable_length")

    def test_performance_parent_error_includes_worker_stderr(self):
        failure = mock.Mock(returncode=1, stderr="SBX requires equal genome lengths", stdout="")
        with mock.patch.object(perf_study.subprocess, "run", return_value=failure):
            with self.assertRaisesRegex(RuntimeError, "SBX requires equal genome lengths"):
                perf_study.run_isolated(
                    pathlib.Path("/bin/false"),
                    {
                        "name": "synthetic-instance",
                        "config_path": pathlib.Path("instance.toml"),
                        "data_path": None,
                    },
                    0,
                )

    def test_quality_audit_front_and_budget_invariants(self):
        self.assertTrue(audit_quality.valid_front([[1.0, 2.0], [1.0, 2.0], [2.0, 1.0]]))
        self.assertFalse(audit_quality.valid_front([[1.0, 2.0], [2.0, 3.0]]))
        self.assertFalse(audit_quality.valid_front([[1.0, float("inf")]]))
        self.assertFalse(audit_quality.valid_front([[1.0], [1.0, 2.0]]))
        self.assertEqual(audit_quality.budget_errors(
            {"requested_evaluations": 100, "evaluations": 150, "budget_overshoot": 50},
            100, 25, "fixture"), [])
        self.assertTrue(audit_quality.budget_errors(
            {"requested_evaluations": 100, "evaluations": 151, "budget_overshoot": 51},
            100, 25, "fixture"))
        self.assertTrue(audit_quality.budget_errors(
            {"requested_evaluations": 100, "evaluations": 99, "budget_overshoot": 0},
            100, 25, "fixture"))
        report = {"loss": 0.0, "coverage": 1.0, "conditional_loss": 0.0, "generality": 0.5,
                  "size": 1.0, "unused_rule_fraction": 0.0, "rows": 10, "rule_count": 2,
                  "abstentions": 0, "fallbacks": 0, "out_of_range": 0, "objectives": [0.0, 1.0]}
        self.assertTrue(audit_quality.valid_report(report))
        self.assertFalse(audit_quality.valid_report({**report, "coverage": None}))
        self.assertFalse(audit_quality.valid_report({**report, "loss": float("inf")}))

    def test_quality_audit_expected_suite_has_30_seeds_per_case(self):
        keys = audit_quality.expected_keys(range(30))
        self.assertEqual(len(keys), 420)
        self.assertEqual(sum(key[0] == "zoo_ten_fold" for key in keys), 90)

    def test_quality_audit_refuses_incomplete_study_without_writing_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            base = pathlib.Path(folder)
            (base / "manifest.json").write_text("{}")
            (base / "raw.jsonl").write_text("")
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                audit_quality.audit(base)
            self.assertFalse((base / "audit.json").exists())


if __name__ == "__main__":
    unittest.main()
