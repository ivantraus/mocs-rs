#!/usr/bin/env python3
"""Prospective, resumable regression-rule comparison (protocol v2)."""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import pathlib
import statistics
import subprocess

PROTOCOL = "regression-linear-protocol-v2"
MODELS = ("regression_rule", "regression_linear_rule")
SUITES = {
    "sparse": (("sparse", "regression-train.arff", "regression-test.arff", 30, 100_000),),
    "dense": (("dense", "regression-dense-train.arff", "regression-dense-test.arff", 30, 100_000),),
    "quadratic": (("quadratic", "regression-quadratic-train.arff", "regression-quadratic-test.arff", 10, 10_000),),
    "vshape": (("vshape", "regression-absolute-train.arff", "regression-absolute-test.arff", 10, 10_000),),
}


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_id(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read_xy(path: pathlib.Path) -> tuple[list[float | None], list[float]]:
    names: list[str] = []
    rows: list[tuple[float | None, float]] = []
    in_data = False
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("%"):
            continue
        low = line.lower()
        if low.startswith("@attribute"):
            names.append(line.split()[1].strip("'\"").lower())
        elif low == "@data":
            in_data = True
        elif in_data:
            fields = next(csv.reader([line], skipinitialspace=True))
            try:
                xi, yi = names.index("x"), names.index("y")
            except ValueError as exc:
                raise ValueError(f"{path}: expected x and y attributes") from exc
            xv, yv = fields[xi].strip().strip("'\""), fields[yi].strip().strip("'\"")
            x = None if xv == "?" else float(xv)
            y = float(yv)
            if (x is not None and not math.isfinite(x)) or not math.isfinite(y):
                raise ValueError(f"{path}: nonfinite observation")
            rows.append((x, y))
    if not rows:
        raise ValueError(f"{path}: no observations")
    return [row[0] for row in rows], [row[1] for row in rows]


def mse(actual: list[float], predicted: list[float]) -> float:
    if len(actual) != len(predicted) or not actual:
        raise ValueError("MSE requires equally sized, nonempty arrays")
    value = math.fsum((a - p) ** 2 for a, p in zip(actual, predicted)) / len(actual)
    if not math.isfinite(value):
        raise ValueError("MSE overflowed")
    return value


def finite_mean(values: list[float]) -> float:
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("mean requires nonempty finite values")
    scale = max(abs(value) for value in values)
    if scale == 0.0:
        return 0.0
    value = (math.fsum(item / scale for item in values) / len(values)) * scale
    if not math.isfinite(value):
        raise ValueError("mean overflowed")
    return value


def fit_ols(train_x: list[float | None], train_y: list[float]) -> dict:
    if len(train_x) != len(train_y) or not train_y:
        raise ValueError("invalid OLS training data")
    ybar = finite_mean(train_y)
    observed = [(x, y) for x, y in zip(train_x, train_y) if x is not None]
    if not observed:
        return {"fallback": True, "reason": "all_predictors_missing", "target_mean": ybar,
                "x_min": None, "x_max": None, "x_center": None, "y_center": ybar, "slope": 0.0}
    xs = [x for x, _ in observed]
    xmin, xmax = min(xs), max(xs)
    if xmin == xmax:
        return {"fallback": True, "reason": "constant_predictor", "target_mean": ybar,
                "x_min": xmin, "x_max": xmax, "x_center": None, "y_center": ybar, "slope": 0.0}
    scale = xmax - xmin
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("OLS predictor range overflowed")
    zs = [(x - xmin) / scale for x in xs]
    zbar = math.fsum(zs) / len(zs)
    yobs = [y for _, y in observed]
    yobsbar = finite_mean(yobs)
    numerator = math.fsum((z - zbar) * (y - yobsbar) for z, y in zip(zs, yobs))
    denominator = math.fsum((z - zbar) ** 2 for z in zs)
    slope = numerator / denominator if denominator else 0.0
    if not all(math.isfinite(v) for v in (zbar, yobsbar, slope)):
        raise ValueError("OLS fit overflowed")
    return {"fallback": False, "reason": None, "target_mean": ybar, "x_min": xmin,
            "x_max": xmax, "x_center": zbar, "y_center": yobsbar, "slope": slope}


def predict_ols(model: dict, xs: list[float | None]) -> list[float]:
    if model["fallback"]:
        return [model["target_mean"]] * len(xs)
    span = model["x_max"] - model["x_min"]
    predictions = [model["target_mean"] if x is None else
                   model["y_center"] + model["slope"] * (((x - model["x_min"]) / span) - model["x_center"])
                   for x in xs]
    if not all(math.isfinite(v) for v in predictions):
        raise ValueError("OLS prediction overflowed")
    return predictions


def load_completed(path: pathlib.Path, manifest_id: str, expected: set[str]) -> dict[str, dict]:
    completed: dict[str, dict] = {}
    if not path.exists():
        return completed
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        key = row.get("key")
        if row.get("manifest_id") != manifest_id:
            raise RuntimeError(f"raw record {number} belongs to another manifest")
        if key not in expected:
            raise RuntimeError(f"raw record {number} has an unknown key: {key}")
        if key in completed:
            raise RuntimeError(f"duplicate completed key: {key}")
        completed[key] = row
    return completed


def validate_completed_artifacts(out: pathlib.Path, completed: dict[str, dict]) -> None:
    for row in completed.values():
        if row.get("method") not in MODELS:
            continue
        paths = row.get("artifact_paths", {})
        hashes = row.get("artifact_sha256", {})
        if set(paths) != {"model", "native_record", "train_predictions", "heldout_predictions"}:
            raise RuntimeError(f"completed record {row['key']} has incomplete artifact metadata")
        for name, relative in paths.items():
            artifact = out / relative
            if not artifact.is_file():
                raise RuntimeError(f"completed record {row['key']} is missing {name} artifact {artifact}")
            if sha256(artifact) != hashes.get(name):
                raise RuntimeError(f"completed record {row['key']} has changed {name} artifact {artifact}")


def prepare_manifest(path: pathlib.Path, raw: pathlib.Path, manifest: dict) -> str:
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise RuntimeError("resume manifest differs from executable, config, source, data, script, or protocol")
    elif raw.exists() and raw.stat().st_size:
        raise RuntimeError("raw records exist without a manifest")
    else:
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return canonical_id(manifest)


def parse_prediction_json(path: pathlib.Path) -> list[float]:
    rows = json.loads(path.read_text())
    result = []
    for row in rows:
        value = row.get("value")
        if value is None or value.get("kind") not in ("numeric", "integer"):
            raise RuntimeError("model produced a missing or nonnumeric prediction")
        result.append(float(value["value"]))
    if not all(math.isfinite(v) for v in result):
        raise RuntimeError("model produced a nonfinite prediction")
    return result


def run_command(binary: pathlib.Path, args: list[object]) -> None:
    proc = subprocess.run([str(binary), *map(str, args)], text=True, capture_output=True)
    if proc.returncode:
        raise RuntimeError(f"command failed: {proc.stderr.strip()}")


def reset_native_output(path: pathlib.Path) -> None:
    """Remove an uncheckpointed prior attempt before the CLI appends its run record."""
    path.unlink(missing_ok=True)


def train_one(binary: pathlib.Path, quality: pathlib.Path, out: pathlib.Path, manifest_id: str,
              case: str, train_name: str, test_name: str, budget: int, model: str, seed: int) -> dict:
    key = f"{case}:{model}:{seed}"
    train, test = quality / train_name, quality / test_name
    config_name = "regression.toml" if model == "regression_rule" else "regression-linear.toml"
    artifact = out / "models" / case / model / f"seed-{seed:02}.json"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    native = out / "models" / case / model / f"seed-{seed:02}.run.jsonl"
    pred_train = artifact.with_suffix(".train-predictions.json")
    pred_test = artifact.with_suffix(".test-predictions.json")
    reset_native_output(native)
    run_command(binary, ["train", "--config", quality / config_name, "--seed", seed,
                         "--max-evaluations", budget, "--threads", 1, "--data", train,
                         "--test", test, "--model-out", artifact, "--output", native])
    run_command(binary, ["predict", "--model", artifact, "--data", train, "--output", pred_train])
    run_command(binary, ["predict", "--model", artifact, "--data", test, "--output", pred_test])
    runs = [json.loads(line) for line in native.read_text().splitlines() if line.strip()]
    if len(runs) != 1:
        raise RuntimeError(f"expected exactly one native run record for {key}")
    run = runs[0]
    train_x, train_y = read_xy(train)
    test_x, test_y = read_xy(test)
    train_predictions = parse_prediction_json(pred_train)
    test_predictions = parse_prediction_json(pred_test)
    if len(train_predictions) != len(train_y) or len(test_predictions) != len(test_y):
        raise RuntimeError(f"prediction row count mismatch for {key}")
    train_mse, heldout_mse = mse(train_y, train_predictions), mse(test_y, test_predictions)
    expected_inputs = [sha256(train), sha256(test)]
    config = run.get("config", {})
    training_config = config.get("training", {})
    evaluations = run.get("evaluations")
    requested = run.get("requested_evaluations")
    overshoot = run.get("budget_overshoot")
    if run.get("seed") != seed or config.get("seed") != seed:
        raise RuntimeError(f"native seed mismatch for {key}")
    if training_config.get("model") != model:
        raise RuntimeError(f"native model mismatch for {key}")
    if run.get("input_sha256") != expected_inputs:
        raise RuntimeError(f"native train/test input hashes mismatch for {key}")
    if requested != budget or evaluations is None or evaluations < budget:
        raise RuntimeError(f"native evaluation budget mismatch for {key}")
    if overshoot != evaluations - budget or overshoot > 200:
        raise RuntimeError(f"native evaluation overshoot is invalid for {key}")
    actual_training = run.get("training_selection")
    actual_heldout = run.get("heldout")
    for label, report, independent in (("training", actual_training, train_mse),
                                       ("held-out", actual_heldout, heldout_mse)):
        native_loss = report.get("loss") if report else None
        if native_loss is None or not math.isfinite(native_loss) or not math.isclose(
                native_loss, independent, rel_tol=1e-12, abs_tol=1e-12):
            raise RuntimeError(f"native {label} MSE disagrees with independent predictions for {key}")
    relative_artifacts = {
        "model": str(artifact.relative_to(out)),
        "native_record": str(native.relative_to(out)),
        "train_predictions": str(pred_train.relative_to(out)),
        "heldout_predictions": str(pred_test.relative_to(out)),
    }
    return {"manifest_id": manifest_id, "key": key, "case": case, "method": model, "seed": seed,
            "requested_evaluations": budget, "evaluations": run["evaluations"],
            "budget_overshoot": overshoot,
            "training_mse": train_mse,
            "heldout_mse": heldout_mse,
            "actual_training_selection": actual_training,
            "actual_heldout_report": actual_heldout,
            "rule_count": (actual_training or {}).get("rule_count"),
            "fallback_count": (actual_heldout or {}).get("fallbacks"),
            "abstention_count": (actual_heldout or {}).get("abstentions"),
            "train_predictions": train_predictions, "heldout_predictions": test_predictions,
            "model_path": relative_artifacts["model"], "native_record": run,
            "artifact_paths": relative_artifacts,
            "artifact_sha256": {name: sha256(out / path) for name, path in relative_artifacts.items()},
            "train_sha256": sha256(train), "test_sha256": sha256(test)}


def baseline_rows(quality: pathlib.Path, out: pathlib.Path, manifest_id: str,
                  case: str, train_name: str, test_name: str) -> list[dict]:
    train, test = quality / train_name, quality / test_name
    tx, ty = read_xy(train)
    vx, vy = read_xy(test)
    ols = fit_ols(tx, ty)
    methods = (("training_mean", {"predictions": [finite_mean(ty)] * len(vy)}),
               ("ols", {"model": ols, "predictions": predict_ols(ols, vx)}))
    rows = []
    for name, data in methods:
        pred_train = [data["model"]["target_mean"]] * len(ty) if name == "ols" and data["model"]["fallback"] else (
            predict_ols(data["model"], tx) if name == "ols" else [finite_mean(ty)] * len(ty))
        if name == "training_mean":
            pred_test = data["predictions"]
            model_info = {"target_mean": finite_mean(ty), "fallback": True}
        else:
            pred_test = data["predictions"]
            model_info = data["model"]
        artifact = out / "baselines" / case / f"{name}.json"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text(json.dumps({"fit": model_info, "train_predictions": pred_train,
                                       "heldout_predictions": pred_test}, indent=2, allow_nan=False) + "\n")
        rows.append({"manifest_id": manifest_id, "key": f"{case}:{name}", "case": case,
                     "method": name, "seed": None, "requested_evaluations": 0,
                     "evaluations": 0, "budget_overshoot": 0,
                     "training_mse": mse(ty, pred_train), "heldout_mse": mse(vy, pred_test),
                     "fit": model_info, "train_predictions": pred_train,
                     "heldout_predictions": pred_test, "artifact_path": str(artifact.relative_to(out)),
                     "train_sha256": sha256(train), "test_sha256": sha256(test)})
    return rows


def summarize(rows: list[dict], suite: str) -> dict:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row["case"], []).append(row)
    result = {"protocol": PROTOCOL, "suite": suite,
              "interpretation": "descriptive seeded optimizer variation; no inference over datasets",
              "original_release_gate_unchanged": True,
              "cases": {}}
    for case, records in groups.items():
        methods = sorted({r["method"] for r in records})
        case_result = {"methods": {}, "paired": {}}
        for method in methods:
            method_rows = sorted((r for r in records if r["method"] == method),
                                 key=lambda r: -1 if r["seed"] is None else r["seed"])
            sample = [r["heldout_mse"] for r in method_rows]
            training = [r["training_mse"] for r in method_rows]
            case_result["methods"][method] = {"runs": len(sample), "median_heldout_mse": statistics.median(sample),
                "min_heldout_mse": min(sample), "max_heldout_mse": max(sample),
                "median_training_mse": statistics.median(training),
                "per_seed_heldout_mse": [{"seed": r["seed"], "mse": r["heldout_mse"]}
                                          for r in method_rows if r["seed"] is not None]}
        for left, right in (("regression_linear_rule", "regression_rule"),
                            ("regression_linear_rule", "ols"),
                            ("regression_linear_rule", "training_mean")):
            lmap = {r["seed"]: r["heldout_mse"] for r in records if r["method"] == left and r["seed"] is not None}
            rmap = {r["seed"]: r["heldout_mse"] for r in records if r["method"] == right and r["seed"] is not None}
            baseline = next((r["heldout_mse"] for r in records
                             if r["method"] == right and r["seed"] is None), None)
            common = sorted(lmap.keys() if baseline is not None else lmap.keys() & rmap.keys())
            if common:
                diffs = [lmap[s] - (baseline if baseline is not None else rmap[s]) for s in common]
                case_result["paired"][f"{left}_minus_{right}"] = {"seeds": common,
                    "median_difference": statistics.median(diffs), "left_wins": sum(d < 0 for d in diffs),
                    "ties": sum(d == 0 for d in diffs), "right_wins": sum(d > 0 for d in diffs)}
        targets = {"sparse": 0.125, "dense": 0.1666}
        if case in targets:
            new_rows = [r for r in records if r["method"] == "regression_linear_rule"]
            median = statistics.median(r["heldout_mse"] for r in new_rows) if new_rows else None
            case_result["new_variant_target"] = {"threshold": targets[case],
                "median_heldout_mse": median,
                "pass": median <= targets[case] if median is not None else None,
                "original_release_gate_changed": False}
        else:
            case_result["interpretation"] = "descriptive control; no acceptance threshold"
        result["cases"][case] = case_result
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mocs", default="target/release/mocs")
    parser.add_argument("--out", default="results/regression-compare")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--suite", choices=("all", *SUITES), default="all")
    opt = parser.parse_args()
    if opt.workers < 1:
        parser.error("--workers must be positive")
    binary = pathlib.Path(opt.mocs).resolve()
    out = pathlib.Path(opt.out).resolve()
    quality = pathlib.Path(__file__).resolve().parent.parent / "examples/quality"
    if not binary.is_file():
        raise FileNotFoundError(binary)
    cases = tuple(spec for name in SUITES if opt.suite == "all" or opt.suite == name for spec in SUITES[name])
    configs = ("regression.toml", "regression-linear.toml")
    fixtures = sorted({p for _, train, test, _, _ in cases for p in (train, test)})
    source_files = sorted((pathlib.Path(__file__).resolve().parent.parent / "src").glob("*.rs"))
    protocol_path = pathlib.Path(__file__).resolve().parent / "protocols/regression-linear-v2.json"
    manifest = {"protocol": PROTOCOL, "suite": opt.suite, "cases": [list(c) for c in cases],
        "models": list(MODELS), "binary_sha256": sha256(binary), "script_sha256": sha256(pathlib.Path(__file__)),
        "protocol_definition_sha256": sha256(protocol_path),
        "configs": {p: sha256(quality / p) for p in configs},
        "fixtures": {p: sha256(quality / p) for p in fixtures},
        "sources": {str(p.relative_to(quality.parent.parent)): sha256(p) for p in source_files}}
    out.mkdir(parents=True, exist_ok=True)
    manifest_path, raw_path = out / "manifest.json", out / "raw.jsonl"
    manifest_id = prepare_manifest(manifest_path, raw_path, manifest)
    jobs = [(case, train, test, budget, model, seed) for case, train, test, nseeds, budget in cases
            for model in MODELS for seed in range(nseeds)]
    expected = {f"{case}:{model}:{seed}" for case, _, _, _, model, seed in jobs}
    expected |= {f"{case}:{baseline}" for case, *_ in cases for baseline in ("ols", "training_mean")}
    completed = load_completed(raw_path, manifest_id, expected)
    for case, train, test, _, _ in cases:
        for row in baseline_rows(quality, out, manifest_id, case, train, test):
            if row["key"] not in completed:
                with raw_path.open("a") as stream:
                    stream.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")
                completed[row["key"]] = row
    validate_completed_artifacts(out, completed)
    pending_jobs = [job for job in jobs if f"{job[0]}:{job[4]}:{job[5]}" not in completed]
    keys = [f"{case}:{model}:{seed}" for case, _, _, _, model, seed in jobs]
    next_write = next((i for i, key in enumerate(keys) if key not in completed), len(keys))
    if pending_jobs:
        with concurrent.futures.ThreadPoolExecutor(max_workers=opt.workers) as pool:
            future_to_key = {pool.submit(train_one, binary, quality, out, manifest_id, *job):
                             f"{job[0]}:{job[4]}:{job[5]}" for job in pending_jobs}
            ready: dict[str, dict] = {}
            for future in concurrent.futures.as_completed(future_to_key):
                key = future_to_key[future]
                ready[key] = future.result()
                completed[key] = ready[key]
                while next_write < len(keys) and keys[next_write] in completed:
                    append = completed[keys[next_write]]
                    if keys[next_write] in ready:
                        with raw_path.open("a") as stream:
                            stream.write(json.dumps(append, allow_nan=False, separators=(",", ":")) + "\n")
                    next_write += 1
    rows_by_key = load_completed(raw_path, manifest_id, expected)
    missing = expected - rows_by_key.keys()
    if missing:
        raise RuntimeError(f"study ended with missing records: {sorted(missing)}")
    rows = list(rows_by_key.values())
    summary = summarize(rows, opt.suite)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(f"saved {len(rows)} records to {out}")


if __name__ == "__main__":
    main()
