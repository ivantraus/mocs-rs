#!/usr/bin/env python3
"""Independent integrity and score audit for regression_compare.py output."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location("frozen_regression_compare", ROOT / "scripts/regression_compare.py")
compare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compare)
BENCHMARK_SPEC = importlib.util.spec_from_file_location("regression_scientific_fingerprint", ROOT / "scripts/benchmark.py")
benchmark = importlib.util.module_from_spec(BENCHMARK_SPEC)
BENCHMARK_SPEC.loader.exec_module(benchmark)


def close(a: object, b: object, tol: float = 1e-12) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isfinite(float(a)) and math.isfinite(float(b)) and math.isclose(
            float(a), float(b), rel_tol=tol, abs_tol=tol)
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(close(a[k], b[k], tol) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(close(x, y, tol) for x, y in zip(a, b))
    return a == b


def finite_tree(value: object) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_tree(item) for item in value)
    return True


def nondominated(front: object) -> bool:
    if not isinstance(front, list) or not front:
        return False
    if any(not isinstance(point, list) or len(point) != 2 or
           not all(isinstance(x, (int, float)) and math.isfinite(x) for x in point)
           for point in front):
        return False
    for i, point in enumerate(front):
        for j, other in enumerate(front):
            if i != j and all(a <= b for a, b in zip(other, point)) and any(
                    a < b for a, b in zip(other, point)):
                return False
    return True


def prediction_values(path: pathlib.Path, expected_count: int) -> list[float]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, list) or len(payload) != expected_count:
        raise ValueError(f"wrong prediction row count in {path}")
    values = []
    for index, row in enumerate(payload):
        if row.get("row") != index:
            raise ValueError(f"prediction row order mismatch in {path} at index {index}")
        value = row.get("value")
        if not isinstance(value, dict) or value.get("kind") not in ("numeric", "integer"):
            raise ValueError(f"nonnumeric prediction in {path}")
        number = float(value["value"])
        if not math.isfinite(number):
            raise ValueError(f"nonfinite prediction in {path}")
        values.append(number)
    return values


def expected_plan(suite: str) -> tuple[tuple, ...]:
    if suite not in ("all", *compare.SUITES):
        raise ValueError(f"unknown suite {suite!r}")
    return tuple(spec for name in compare.SUITES if suite == "all" or suite == name
                 for spec in compare.SUITES[name])


def expected_keys(cases: tuple[tuple, ...]) -> set[str]:
    keys = set()
    for case, _train, _test, seeds, _budget in cases:
        keys.update(f"{case}:{model}:{seed}" for model in compare.MODELS for seed in range(seeds))
        keys.update((f"{case}:ols", f"{case}:training_mean"))
    return keys


def verify_manifest(out: pathlib.Path, binary: pathlib.Path, manifest: dict) -> tuple[tuple, ...]:
    cases = expected_plan(manifest.get("suite"))
    quality = ROOT / "examples/quality"
    config_names = ("regression.toml", "regression-linear.toml")
    fixtures = sorted({p for _, train, test, _, _ in cases for p in (train, test)})
    source_files = sorted((ROOT / "src").glob("*.rs"))
    protocol = ROOT / "scripts/protocols/regression-linear-v2.json"
    expected = {
        "protocol": compare.PROTOCOL,
        "suite": manifest.get("suite"),
        "cases": [list(case) for case in cases],
        "models": list(compare.MODELS),
        "binary_sha256": compare.sha256(binary),
        "script_sha256": compare.sha256(ROOT / "scripts/regression_compare.py"),
        "protocol_definition_sha256": compare.sha256(protocol),
        "configs": {name: compare.sha256(quality / name) for name in config_names},
        "fixtures": {name: compare.sha256(quality / name) for name in fixtures},
        "sources": {str(path.relative_to(ROOT)): compare.sha256(path) for path in source_files},
    }
    if manifest != expected:
        raise ValueError("manifest does not match frozen binary, script, protocol, configs, fixtures, and source hashes")
    return cases


def check_native(row: dict, case: str, model: str, seed: int, budget: int,
                 train: pathlib.Path, test: pathlib.Path, native: dict) -> None:
    key = row["key"]
    config = native.get("config", {})
    training = config.get("training", {})
    operators = config.get("operators", {})
    algorithm = config.get("algorithm", {})
    expected_config = {
        "seed": seed, "population": 100, "offspring": 100, "generations": 100_000,
        "max_evaluations": budget, "threads": 1,
    }
    for field, expected in expected_config.items():
        if config.get(field) != expected:
            raise ValueError(f"{key}: native config {field} mismatch")
    if algorithm.get("kind") != "nsga2" or training.get("model") != model:
        raise ValueError(f"{key}: native algorithm/model mismatch")
    if training.get("fitness") != "loss_size" or training.get("decision") != "best_accuracy":
        raise ValueError(f"{key}: native training selection config mismatch")
    if training.get("initial_rules") != 15:
        raise ValueError(f"{key}: native initial rule count mismatch")
    if (operators.get("mating") != "tournament" or operators.get("crossover") != "variable_length" or
            operators.get("mutation") != "uniform" or operators.get("crossover_probability") != 0.8 or
            operators.get("mutation_probability") != 1.0):
        raise ValueError(f"{key}: native operator config mismatch")
    if native.get("seed") != seed or native.get("input_sha256") != [compare.sha256(train), compare.sha256(test)]:
        raise ValueError(f"{key}: native seed or ordered input hashes mismatch")
    evaluations = native.get("evaluations")
    overshoot = native.get("budget_overshoot")
    if native.get("requested_evaluations") != budget or not isinstance(evaluations, int) or evaluations < budget:
        raise ValueError(f"{key}: native evaluation budget mismatch")
    if overshoot != evaluations - budget or not 0 <= overshoot <= 200:
        raise ValueError(f"{key}: native evaluation overshoot mismatch")
    if not finite_tree(native.get("front")) or not nondominated(native.get("front")):
        raise ValueError(f"{key}: native front has nonfinite, malformed, or dominated points")


def check_duplicate_fields(row: dict, native: dict) -> None:
    key = row.get("key", "unknown")
    training = native.get("training_selection") or {}
    heldout = native.get("heldout") or {}
    for raw_name, native_value in (
            ("budget_overshoot", native.get("budget_overshoot")),
            ("rule_count", training.get("rule_count")),
            ("fallback_count", heldout.get("fallbacks")),
            ("abstention_count", heldout.get("abstentions"))):
        if row.get(raw_name) != native_value:
            raise ValueError(f"{key}: raw {raw_name} disagrees with native report")


def audit_evolution(out: pathlib.Path, row: dict, case: str, train_name: str, test_name: str,
                    budget: int, model: str, seed: int) -> None:
    key = f"{case}:{model}:{seed}"
    if row.get("key") != key or (row.get("case"), row.get("method"), row.get("seed")) != (case, model, seed):
        raise ValueError(f"{key}: raw record identity mismatch")
    quality = ROOT / "examples/quality"
    train, test = quality / train_name, quality / test_name
    for field, path in (("train_sha256", train), ("test_sha256", test)):
        if row.get(field) != compare.sha256(path):
            raise ValueError(f"{key}: raw {field} mismatch")
    paths, hashes = row.get("artifact_paths", {}), row.get("artifact_sha256", {})
    if set(paths) != {"model", "native_record", "train_predictions", "heldout_predictions"}:
        raise ValueError(f"{key}: incomplete artifact path set")
    artifacts = {name: out / relative for name, relative in paths.items()}
    for name, path in artifacts.items():
        if not path.is_file() or compare.sha256(path) != hashes.get(name):
            raise ValueError(f"{key}: missing or changed {name} artifact")
    native_lines = [line for line in artifacts["native_record"].read_text().splitlines() if line.strip()]
    if len(native_lines) != 1:
        raise ValueError(f"{key}: expected one saved native run record")
    native = json.loads(native_lines[0])
    if native != row.get("native_record"):
        raise ValueError(f"{key}: raw embedded native record differs from saved record")
    check_native(row, case, model, seed, budget, train, test, native)
    check_duplicate_fields(row, native)
    saved_model = json.loads(artifacts["model"].read_text())
    expected_semantics = 2 if model == "regression_linear_rule" else 1
    if (saved_model.get("config", {}).get("model") != model or
            saved_model.get("semantics_version") != expected_semantics or not finite_tree(saved_model)):
        raise ValueError(f"{key}: saved model tag, semantics, or finite-value audit failed")
    train_x, train_y = compare.read_xy(train)
    test_x, test_y = compare.read_xy(test)
    train_pred = prediction_values(artifacts["train_predictions"], len(train_y))
    test_pred = prediction_values(artifacts["heldout_predictions"], len(test_y))
    independent_train = compare.mse(train_y, train_pred)
    independent_test = compare.mse(test_y, test_pred)
    if not close(train_pred, row.get("train_predictions")) or not close(test_pred, row.get("heldout_predictions")):
        raise ValueError(f"{key}: raw prediction arrays differ from prediction artifacts")
    for label, independent, raw_value, native_report in (
            ("training", independent_train, row.get("training_mse"), native.get("training_selection")),
            ("held-out", independent_test, row.get("heldout_mse"), native.get("heldout"))):
        if not close(independent, raw_value) or not native_report or not close(independent, native_report.get("loss")):
            raise ValueError(f"{key}: {label} MSE disagrees across predictions, raw record, and native report")
    if row.get("requested_evaluations") != budget or row.get("evaluations") != native.get("evaluations"):
        raise ValueError(f"{key}: raw and native evaluation records disagree")


def audit_baseline(out: pathlib.Path, row: dict, case: str, train_name: str,
                   test_name: str, method: str) -> None:
    key = f"{case}:{method}"
    if (row.get("key") != key or row.get("case") != case or row.get("method") != method or
            row.get("seed") is not None or row.get("requested_evaluations") != 0 or
            row.get("evaluations") != 0 or row.get("budget_overshoot") != 0):
        raise ValueError(f"{key}: baseline identity/evaluation mismatch")
    quality = ROOT / "examples/quality"
    train, test = quality / train_name, quality / test_name
    if row.get("train_sha256") != compare.sha256(train) or row.get("test_sha256") != compare.sha256(test):
        raise ValueError(f"{key}: baseline fixture hash mismatch")
    artifact = out / row.get("artifact_path", "")
    if not artifact.is_file():
        raise ValueError(f"{key}: baseline artifact is missing")
    saved = json.loads(artifact.read_text())
    tx, ty = compare.read_xy(train)
    vx, vy = compare.read_xy(test)
    fit = compare.fit_ols(tx, ty) if method == "ols" else {
        "target_mean": compare.finite_mean(ty), "fallback": True}
    if not close(fit, row.get("fit")) or not close(fit, saved.get("fit")):
        raise ValueError(f"{key}: baseline fit cannot be reproduced from training data")
    train_pred = compare.predict_ols(fit, tx) if method == "ols" else [fit["target_mean"]] * len(ty)
    test_pred = compare.predict_ols(fit, vx) if method == "ols" else [fit["target_mean"]] * len(vy)
    if not close(train_pred, row.get("train_predictions")) or not close(test_pred, row.get("heldout_predictions")):
        raise ValueError(f"{key}: baseline prediction arrays mismatch")
    if not close(train_pred, saved.get("train_predictions")) or not close(test_pred, saved.get("heldout_predictions")):
        raise ValueError(f"{key}: baseline artifact prediction arrays mismatch")
    if not close(compare.mse(ty, train_pred), row.get("training_mse")) or not close(
            compare.mse(vy, test_pred), row.get("heldout_mse")):
        raise ValueError(f"{key}: baseline MSE mismatch")


def audit_previous_quality(previous: pathlib.Path, records: dict[str, dict]) -> dict:
    manifest = json.loads((previous / "manifest.json").read_text())
    manifest_id = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    old: dict[tuple[str, int], dict] = {}
    for number, line in enumerate((previous / "raw.jsonl").read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        gate = row.get("gate_id")
        if gate not in ("regression", "regression_dense"):
            continue
        if row.get("manifest_id") != manifest_id:
            raise ValueError(f"previous quality line {number}: manifest ID mismatch")
        seed = row.get("seed")
        key = (gate, seed)
        if not isinstance(seed, int) or key in old:
            raise ValueError(f"previous quality line {number}: invalid or duplicate regression seed")
        old[key] = row["run"]
    checked = {}
    for gate, case in (("regression", "sparse"), ("regression_dense", "dense")):
        seeds = sorted(seed for found_gate, seed in old if found_gate == gate)
        if seeds != list(range(30)):
            raise ValueError(f"previous quality gate {gate} does not contain seeds 0..29")
        for seed in seeds:
            current = records[f"{case}:regression_rule:{seed}"]["native_record"]
            if benchmark.scientific_fingerprint(old[(gate, seed)]) != benchmark.scientific_fingerprint(current):
                raise ValueError(f"original {gate} scientific record changed at seed {seed}")
        checked[case] = len(seeds)
    return {"status": "pass", "records": checked,
            "quality_manifest_id": manifest_id,
            "fingerprint_helper_sha256": compare.sha256(ROOT / "scripts/benchmark.py")}


def audit_directory(out: pathlib.Path, binary: pathlib.Path,
                    previous_quality: pathlib.Path | None = None) -> dict:
    errors: list[str] = []
    checked_rows = checked_artifacts = 0
    manifest_path, raw_path = out / "manifest.json", out / "raw.jsonl"
    try:
        manifest = json.loads(manifest_path.read_text())
        cases = verify_manifest(out, binary, manifest)
        manifest_id = compare.canonical_id(manifest)
        expected = expected_keys(cases)
        records: dict[str, dict] = {}
        for number, line in enumerate(raw_path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = row.get("key")
            if row.get("manifest_id") != manifest_id:
                errors.append(f"raw record line {number}: manifest id mismatch")
            if key not in expected:
                errors.append(f"raw record line {number}: unknown key {key!r}")
            elif key in records:
                errors.append(f"duplicate raw key {key}")
            else:
                records[key] = row
        missing = expected - records.keys()
        if missing:
            errors.append(f"missing {len(missing)} expected records: {sorted(missing)[:8]}")
        extras = records.keys() - expected
        if extras:
            errors.append(f"unexpected keys: {sorted(extras)[:8]}")
        for case, train_name, test_name, nseeds, budget in cases:
            keys = [f"{case}:{method}:{seed}" for method in compare.MODELS for seed in range(nseeds)]
            keys += [f"{case}:ols", f"{case}:training_mean"]
            for key in keys:
                if key not in records:
                    continue
                row = records[key]
                try:
                    if key.endswith(":ols") or key.endswith(":training_mean"):
                        method = key.rsplit(":", 1)[-1]
                        audit_baseline(out, row, case, train_name, test_name, method)
                    else:
                        _, model, seed = key.split(":")
                        audit_evolution(out, row, case, train_name, test_name, budget, model, int(seed))
                        checked_artifacts += 4
                    checked_rows += 1
                except Exception as exc:
                    errors.append(f"{key}: {exc}")
        summary = json.loads((out / "summary.json").read_text())
        expected_summary = compare.summarize(list(records.values()), manifest["suite"])
        if summary != expected_summary:
            errors.append("summary.json does not match raw records")
        previous_check = None
        if previous_quality is not None:
            try:
                previous_check = audit_previous_quality(previous_quality, records)
            except Exception as exc:
                errors.append(f"previous quality comparison: {exc}")
                previous_check = {"status": "fail", "error": str(exc),
                                  "fingerprint_helper_sha256": compare.sha256(ROOT / "scripts/benchmark.py")}
        return {"status": "fail" if errors else "pass", "protocol": compare.PROTOCOL,
                "suite": manifest.get("suite"), "manifest_id": manifest_id,
                "expected_records": len(expected), "checked_records": checked_rows,
                "checked_evolution_artifacts": checked_artifacts,
                "previous_quality_exact_check": previous_check, "errors": errors}
    except Exception as exc:
        errors.append(str(exc))
        return {"status": "fail", "protocol": compare.PROTOCOL,
                "expected_records": None, "checked_records": checked_rows,
                "checked_evolution_artifacts": checked_artifacts, "errors": errors}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/regression-compare")
    parser.add_argument("--mocs", default="target/release/mocs")
    parser.add_argument("--json-out", help="audit JSON destination; defaults to OUT/audit.json")
    parser.add_argument("--previous-quality", help="also compare original regression records in this quality-study directory")
    parser.add_argument("--compact-json", action="store_true", help="write audit JSON on one line")
    opt = parser.parse_args()
    out, binary = pathlib.Path(opt.out).resolve(), pathlib.Path(opt.mocs).resolve()
    previous_quality = pathlib.Path(opt.previous_quality).resolve() if opt.previous_quality else None
    report = audit_directory(out, binary, previous_quality)
    destination = pathlib.Path(opt.json_out).resolve() if opt.json_out else out / "audit.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if opt.compact_json:
        destination.write_text(json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n")
    else:
        destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
