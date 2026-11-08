import importlib.util
import json
import pathlib
import tempfile
import unittest


SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts/regression_compare.py"
SPEC = importlib.util.spec_from_file_location("regression_compare", SCRIPT)
rc = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(rc)


class RegressionCompareTests(unittest.TestCase):
    def test_centered_ols_matches_line_and_uses_normalized_predictor(self):
        fitted = rc.fit_ols([10.0, 15.0, 20.0], [1.0, 2.0, 3.0])
        self.assertFalse(fitted["fallback"])
        self.assertAlmostEqual(fitted["slope"], 2.0)
        self.assertEqual(rc.predict_ols(fitted, [12.5, 17.5]), [1.5, 2.5])
        self.assertEqual(rc.predict_ols(fitted, [25.0]), [4.0])

    def test_scaled_mean_stays_finite_for_large_representable_mean(self):
        self.assertEqual(rc.finite_mean([1e308, 1e308]), 1e308)
        with self.assertRaises(ValueError):
            rc.finite_mean([float("nan")])

    def test_ols_missing_predictor_uses_full_target_mean_and_constant_fallback(self):
        missing = rc.fit_ols([None, 0.0, 1.0], [0.0, 2.0, 4.0])
        self.assertEqual(rc.predict_ols(missing, [None]), [2.0])
        constant = rc.fit_ols([3.0, 3.0], [1.0, 5.0])
        self.assertEqual(constant["reason"], "constant_predictor")
        self.assertEqual(rc.predict_ols(constant, [3.0, None]), [3.0, 3.0])

    def test_ols_fit_has_no_heldout_input_and_does_not_leak(self):
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            train = root / "train.arff"
            test = root / "test.arff"
            train.write_text("@relation train\n@attribute x numeric\n@attribute y numeric\n@data\n10,1\n15,2\n20,3\n")
            test.write_text("@relation test\n@attribute x numeric\n@attribute y numeric\n@data\n12.5,1.5\n")
            first = rc.baseline_rows(root, root / "first", "m", "case",
                                     "train.arff", "test.arff")
            first_fit = next(row["fit"] for row in first if row["method"] == "ols")
            test.write_text("@relation test\n@attribute x numeric\n@attribute y numeric\n@data\n12.5,9999\n")
            second = rc.baseline_rows(root, root / "second", "m", "case",
                                      "train.arff", "test.arff")
            second_fit = next(row["fit"] for row in second if row["method"] == "ols")
            self.assertEqual(first_fit, second_fit)
            self.assertNotEqual(first[1]["heldout_mse"], second[1]["heldout_mse"])

    def test_mse_recomputes_prediction_score_and_rejects_bad_shapes(self):
        self.assertEqual(rc.mse([1.0, 2.0], [1.0, 4.0]), 2.0)
        with self.assertRaises(ValueError):
            rc.mse([1.0], [1.0, 2.0])

    def test_resume_manifest_rejects_changed_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            manifest_path, raw_path = root / "manifest.json", root / "raw.jsonl"
            manifest = {"binary": "a", "script": "b", "fixture": "c", "protocol": 1}
            first_id = rc.prepare_manifest(manifest_path, raw_path, manifest)
            self.assertEqual(first_id, rc.canonical_id(manifest))
            self.assertEqual(rc.prepare_manifest(manifest_path, raw_path, manifest), first_id)
            changed = {**manifest, "binary": "changed"}
            with self.assertRaisesRegex(RuntimeError, "manifest differs"):
                rc.prepare_manifest(manifest_path, raw_path, changed)

    def test_resume_rejects_duplicate_unknown_and_mixed_records(self):
        with tempfile.TemporaryDirectory() as temp:
            path = pathlib.Path(temp) / "raw.jsonl"
            row = {"manifest_id": "m", "key": "sparse:ols"}
            path.write_text(json.dumps(row) + "\n")
            loaded = rc.load_completed(path, "m", {"sparse:ols", "dense:ols"})
            self.assertEqual(set(loaded), {"sparse:ols"})
            self.assertEqual({"sparse:ols", "dense:ols"} - loaded.keys(), {"dense:ols"})
            path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
            with self.assertRaisesRegex(RuntimeError, "duplicate"):
                rc.load_completed(path, "m", {"sparse:ols"})
            path.write_text(json.dumps({**row, "key": "other"}) + "\n")
            with self.assertRaisesRegex(RuntimeError, "unknown key"):
                rc.load_completed(path, "m", {"sparse:ols"})
            path.write_text(json.dumps({**row, "manifest_id": "other"}) + "\n")
            with self.assertRaisesRegex(RuntimeError, "another manifest"):
                rc.load_completed(path, "m", {"sparse:ols"})

    def test_train_retry_clears_native_record_left_by_uncheckpointed_attempt(self):
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            fake = root / "fake-mocs"
            fake.write_text("""#!/usr/bin/env python3
import csv, hashlib, json, pathlib, sys
args = sys.argv[1:]
def option(name): return pathlib.Path(args[args.index(name) + 1])
if args[0] == 'train':
    option('--model-out').write_text('{}')
    budget = int(args[args.index('--max-evaluations') + 1])
    seed = int(args[args.index('--seed') + 1])
    config = {'seed': seed, 'training': {'model': 'regression_rule'}}
    run = {'seed': seed, 'config': config,
           'input_sha256': [hashlib.sha256(option('--data').read_bytes()).hexdigest(),
                            hashlib.sha256(option('--test').read_bytes()).hexdigest()],
           'requested_evaluations': budget, 'evaluations': budget, 'budget_overshoot': 0,
           'training_selection': {'loss': 0.0, 'size': 1.0, 'rule_count': 1},
           'heldout': {'loss': 0.0, 'fallbacks': 0, 'abstentions': 0}}
    with option('--output').open('a') as stream: stream.write(json.dumps(run) + '\\n')
elif args[0] == 'predict':
    data = option('--data').read_text().split('@data', 1)[1]
    targets = [float(next(csv.reader([line]))[-1]) for line in data.splitlines() if line.strip()]
    option('--output').write_text(json.dumps([{'value': {'kind': 'numeric', 'value': target}}
                                               for target in targets]))
""")
            fake.chmod(fake.stat().st_mode | 0o111)
            quality = SCRIPT.parent.parent / "examples/quality"
            native = root / "models/sparse/regression_rule/seed-00.run.jsonl"
            native.parent.mkdir(parents=True)
            native.write_text('{"stale":true}\n')
            row = rc.train_one(fake, quality, root, "manifest", "sparse",
                               "regression-train.arff", "regression-test.arff",
                               100_000, "regression_rule", 0)
            lines = native.read_text().splitlines()
            self.assertEqual(len(lines), 1)
            self.assertEqual(json.loads(lines[0])["evaluations"], 100_000)
            self.assertEqual(row["evaluations"], 100_000)
            self.assertEqual(row["training_mse"], 0.0)
            self.assertEqual(row["heldout_mse"], 0.0)
            for artifact, digest in row["artifact_sha256"].items():
                self.assertEqual(rc.sha256(root / row["artifact_paths"][artifact]), digest)
            rc.validate_completed_artifacts(root, {row["key"]: row})
            (root / row["model_path"]).write_text("corrupted")
            with self.assertRaisesRegex(RuntimeError, "changed model artifact"):
                rc.validate_completed_artifacts(root, {row["key"]: row})

    def test_summary_pairs_each_evolution_seed_with_singleton_baselines(self):
        rows = [
            {"case": "sparse", "method": "regression_linear_rule", "seed": 0,
             "heldout_mse": 0.1, "training_mse": 0.0},
            {"case": "sparse", "method": "regression_linear_rule", "seed": 1,
             "heldout_mse": 0.3, "training_mse": 0.0},
            {"case": "sparse", "method": "ols", "seed": None,
             "heldout_mse": 0.2, "training_mse": 0.0},
            {"case": "sparse", "method": "training_mean", "seed": None,
             "heldout_mse": 0.4, "training_mse": 0.0},
        ]
        case = rc.summarize(rows, "sparse")["cases"]["sparse"]
        ols = case["paired"]["regression_linear_rule_minus_ols"]
        mean = case["paired"]["regression_linear_rule_minus_training_mean"]
        self.assertEqual(ols["seeds"], [0, 1])
        self.assertEqual((ols["left_wins"], ols["ties"], ols["right_wins"]), (1, 0, 1))
        self.assertAlmostEqual(ols["median_difference"], 0.0)
        self.assertEqual((mean["left_wins"], mean["ties"], mean["right_wins"]), (2, 0, 0))
        target = case["new_variant_target"]
        self.assertEqual(target["threshold"], 0.125)
        self.assertFalse(target["pass"])
        self.assertFalse(target["original_release_gate_changed"])

    def test_fixture_protocol_counts_and_linear_config_delta(self):
        quality = SCRIPT.parent.parent / "examples/quality"
        sparse_x, sparse_y = rc.read_xy(quality / "regression-train.arff")
        quadratic_x, quadratic_y = rc.read_xy(quality / "regression-quadratic-train.arff")
        v_x, v_y = rc.read_xy(quality / "regression-absolute-train.arff")
        self.assertEqual((len(sparse_x), len(sparse_y)), (3, 3))
        self.assertEqual((len(quadratic_x), len(quadratic_y)), (51, 51))
        self.assertEqual((len(v_x), len(v_y)), (51, 51))
        self.assertEqual(quadratic_y[25], 0.25)
        self.assertEqual(v_y[25], 0.0)
        with (quality / "regression.toml").open("rb") as source:
            original = __import__("tomllib").load(source)
        with (quality / "regression-linear.toml").open("rb") as source:
            linear = __import__("tomllib").load(source)
        self.assertEqual({**original["training"], "model": "regression_linear_rule"}, linear["training"])
        original.pop("training")
        linear.pop("training")
        self.assertEqual(original, linear)

    def test_manifest_binds_json_protocol_definition_and_new_identity(self):
        protocol = SCRIPT.parent / "protocols/regression-linear-v2.json"
        definition = json.loads(protocol.read_text())
        self.assertEqual(definition["protocol"], rc.PROTOCOL)
        self.assertEqual(rc.PROTOCOL, "regression-linear-protocol-v2")
        self.assertEqual(protocol.relative_to(SCRIPT.parent).as_posix(),
                         "protocols/regression-linear-v2.json")


if __name__ == "__main__":
    unittest.main()
