import importlib.util
import json
import pathlib
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("audit_regression_compare", ROOT / "scripts/audit_regression_compare.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


class RegressionCompareAuditTests(unittest.TestCase):
    def test_old_protocol_identity_is_rejected_by_manifest_auditor(self):
        quality = ROOT / "examples/quality"
        cases = audit.expected_plan("sparse")
        configs = ("regression.toml", "regression-linear.toml")
        fixtures = sorted({path for _, train, test, _, _ in cases for path in (train, test)})
        with tempfile.TemporaryDirectory() as temp:
            binary = pathlib.Path(temp) / "fake-mocs"
            binary.write_bytes(b"binary fixture")
            manifest = {
                "protocol": audit.compare.PROTOCOL,
                "suite": "sparse",
                "cases": [list(case) for case in cases],
                "models": list(audit.compare.MODELS),
                "binary_sha256": audit.compare.sha256(binary),
                "script_sha256": audit.compare.sha256(ROOT / "scripts/regression_compare.py"),
                "protocol_definition_sha256": audit.compare.sha256(ROOT / "scripts/protocols/regression-linear-v2.json"),
                "configs": {name: audit.compare.sha256(quality / name) for name in configs},
                "fixtures": {path: audit.compare.sha256(quality / path) for path in fixtures},
                "sources": {str(path.relative_to(ROOT)): audit.compare.sha256(path)
                            for path in sorted((ROOT / "src").glob("*.rs"))},
            }
            audit.verify_manifest(ROOT, binary, manifest)
            manifest["protocol"] = "regression-linear-protocol-v1"
            with self.assertRaisesRegex(ValueError, "manifest does not match"):
                audit.verify_manifest(ROOT, binary, manifest)

    def test_full_protocol_has_168_expected_records(self):
        cases = audit.expected_plan("all")
        keys = audit.expected_keys(cases)
        self.assertEqual(len(keys), 168)
        self.assertIn("sparse:regression_linear_rule:29", keys)
        self.assertIn("vshape:ols", keys)
        self.assertIn("quadratic:training_mean", keys)

    def test_front_audit_rejects_nonfinite_and_dominated_points(self):
        self.assertTrue(audit.nondominated([[0.1, 0.8], [0.2, 0.4]]))
        self.assertFalse(audit.nondominated([[0.1, 0.8], [0.2, 0.9]]))
        self.assertFalse(audit.nondominated([[float("nan"), 0.8]]))
        self.assertFalse(audit.nondominated([]))

    def test_saved_prediction_reader_checks_shape_type_and_finiteness(self):
        with tempfile.TemporaryDirectory() as temp:
            path = pathlib.Path(temp) / "predictions.json"
            path.write_text('[{"row":0,"value":{"kind":"numeric","value":2.5}}]')
            self.assertEqual(audit.prediction_values(path, 1), [2.5])
            path.write_text('[{"row":1,"value":{"kind":"numeric","value":2.5}}]')
            with self.assertRaisesRegex(ValueError, "row order"):
                audit.prediction_values(path, 1)
            path.write_text('[{"row":0,"value":{"kind":"nominal","value":"x"}}]')
            with self.assertRaisesRegex(ValueError, "nonnumeric"):
                audit.prediction_values(path, 1)
            path.write_text('[{"row":0,"value":{"kind":"numeric","value":NaN}}]')
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                audit.prediction_values(path, 1)

    def test_baseline_audit_recomputes_ols_from_training_data_and_checks_artifact(self):
        quality = ROOT / "examples/quality"
        with tempfile.TemporaryDirectory() as temp:
            out = pathlib.Path(temp)
            row = audit.compare.baseline_rows(quality, out, "manifest", "sparse",
                                               "regression-train.arff", "regression-test.arff")[1]
            audit.audit_baseline(out, row, "sparse", "regression-train.arff",
                                 "regression-test.arff", "ols")
            row["fit"]["slope"] += 1.0
            with self.assertRaisesRegex(ValueError, "baseline fit cannot be reproduced"):
                audit.audit_baseline(out, row, "sparse", "regression-train.arff",
                                     "regression-test.arff", "ols")
            valid = audit.compare.baseline_rows(quality, out, "manifest", "sparse",
                                                "regression-train.arff", "regression-test.arff")[1]
            valid["case"] = "dense"
            with self.assertRaisesRegex(ValueError, "baseline identity"):
                audit.audit_baseline(out, valid, "sparse", "regression-train.arff",
                                     "regression-test.arff", "ols")
            for field, bad_value in (("method", "training_mean"),
                                     ("requested_evaluations", 1), ("budget_overshoot", 1)):
                valid = audit.compare.baseline_rows(quality, out, "manifest", "sparse",
                                                    "regression-train.arff", "regression-test.arff")[1]
                valid[field] = bad_value
                with self.subTest(field=field), self.assertRaisesRegex(ValueError, "baseline identity"):
                    audit.audit_baseline(out, valid, "sparse", "regression-train.arff",
                                         "regression-test.arff", "ols")

    def test_evolution_raw_duplicate_fields_must_match_native_report(self):
        native = {"budget_overshoot": 2,
                  "training_selection": {"rule_count": 3},
                  "heldout": {"fallbacks": 1, "abstentions": 0}}
        row = {"key": "sparse:regression_rule:0", "budget_overshoot": 2,
               "rule_count": 3, "fallback_count": 1, "abstention_count": 0}
        audit.check_duplicate_fields(row, native)
        for field, bad_value in (("budget_overshoot", 1), ("rule_count", 4),
                                 ("fallback_count", 0), ("abstention_count", 2)):
            changed = {**row, field: bad_value}
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
                audit.check_duplicate_fields(changed, native)

    def test_native_audit_checks_config_budget_inputs_and_pareto_front(self):
        quality = ROOT / "examples/quality"
        train, test = quality / "regression-train.arff", quality / "regression-test.arff"
        native = {
            "seed": 2,
            "config": {"seed": 2, "population": 100, "offspring": 100,
                       "generations": 100_000, "max_evaluations": 100_000, "threads": 1,
                       "algorithm": {"kind": "nsga2"},
                       "training": {"model": "regression_linear_rule", "fitness": "loss_size",
                                    "decision": "best_accuracy", "initial_rules": 15},
                       "operators": {"mating": "tournament", "crossover": "variable_length",
                                     "mutation": "uniform", "crossover_probability": 0.8,
                                     "mutation_probability": 1.0}},
            "input_sha256": [audit.compare.sha256(train), audit.compare.sha256(test)],
            "requested_evaluations": 100_000, "evaluations": 100_000,
            "budget_overshoot": 0, "front": [[0.1, 0.8], [0.2, 0.4]],
        }
        audit.check_native({"key": "sparse:regression_linear_rule:2"}, "sparse", "regression_linear_rule", 2, 100_000,
                           train, test, native)
        native["front"] = [[0.1, 0.8], [0.2, 0.9]]
        with self.assertRaisesRegex(ValueError, "dominated"):
            audit.check_native({"key": "sparse:regression_linear_rule:2"}, "sparse", "regression_linear_rule", 2, 100_000,
                               train, test, native)

    def test_optional_previous_quality_check_uses_scientific_fingerprints(self):
        with tempfile.TemporaryDirectory() as temp:
            previous = pathlib.Path(temp)
            manifest = {"suite": "all"}
            (previous / "manifest.json").write_text(json.dumps(manifest))
            manifest_id = audit.hashlib.sha256(json.dumps(
                manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            rows, current = [], {}
            for gate, case in (("regression", "sparse"), ("regression_dense", "dense")):
                for seed in range(30):
                    run = {"config": {"threads": 1, "training": {"model": "regression_rule"}},
                           "seed": seed, "input_sha256": [gate], "evaluations": 10,
                           "generations": 2, "front": [[0.1, 0.2]],
                           "training_selection": {"loss": 0.01}, "heldout": {"loss": 0.02}}
                    rows.append({"manifest_id": manifest_id, "gate_id": gate, "seed": seed, "run": run})
                    current[f"{case}:regression_rule:{seed}"] = {"native_record": run}
            (previous / "raw.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            check = audit.audit_previous_quality(previous, current)
            self.assertEqual(check["status"], "pass")
            self.assertEqual(check["records"], {"sparse": 30, "dense": 30})
            current["sparse:regression_rule:0"]["native_record"]["evaluations"] += 1
            with self.assertRaisesRegex(ValueError, "scientific record changed"):
                audit.audit_previous_quality(previous, current)


if __name__ == "__main__":
    unittest.main()
