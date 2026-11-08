#!/usr/bin/env python3
"""Evaluate fixed quality thresholds with resumable raw records.

Thresholds are diagnostic acceptance targets, not a claim that every model passes.
The sparse constant-consequence regression fixture is a known limitation.
See README.md for scope and scripts/protocols/ for the separate regression comparison."""
import argparse
import csv
import hashlib
import json
import math
import pathlib
import statistics
import subprocess

ALGORITHMS = ("moga", "moes", "ngga", "nsga2", "spea2")
MODELS = ("rule", "instance", "mixed")
SEEDS = tuple(range(30))
GATE_VERSION = "mocs-release-gates-v3"


def hv2(points, ref):
    candidates = sorted((p[0], p[1]) for p in points
                        if len(p) == 2 and all(math.isfinite(x) for x in p)
                        and p[0] <= ref[0] and p[1] <= ref[1])
    area, best_y = 0.0, ref[1]
    for x, y in candidates:
        if y < best_y:
            area += (ref[0] - x) * (best_y - y)
            best_y = y
    return area


def sf2_true_hv():
    """Exact SF2 minimization-front hypervolume for reference (5,5)."""
    return 67.0 / 3.0


def zdt6_true_hv(ref=(1.1, 1.1)):
    """Exact ZDT6 front hypervolume for the standard continuous front."""
    x_star = math.atan(9.0 * math.pi) / (6.0 * math.pi)
    a = 1.0 - math.exp(-4.0 * x_star) * math.sin(6.0 * math.pi * x_star) ** 6
    rx, ry = ref
    if not all(math.isfinite(v) for v in ref) or rx < 1.0 or ry < 1.0:
        raise ValueError("ZDT6 reference must be finite and at least (1,1)")
    return (ry - 1.0) * (1.0 - a) + (1.0 - a**3) / 3.0 + (rx - 1.0) * ry


def digest(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def call(binary, args):
    proc = subprocess.run([str(binary), *map(str, args)], text=True, capture_output=True)
    if proc.returncode:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(map(str,args))}\n{proc.stderr.strip()}")
    return proc


def append_result(path, row):
    with path.open("a") as stream:
        stream.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")


def load_completed(path, manifest_id):
    completed = {}
    if path.exists():
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("manifest_id") != manifest_id:
                raise RuntimeError(f"raw result line {number} belongs to another gate manifest")
            key = (row["gate"], row.get("algorithm"), row.get("model"), row.get("seed"))
            if key in completed:
                raise RuntimeError(f"duplicate completed result {key}")
            completed[key] = row
    return completed


def prepare_manifest(manifest_path, raw_path, manifest):
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise RuntimeError("quality resume manifest differs from executable, config, fixture, data, suite, or gate version")
    elif raw_path.exists() and raw_path.stat().st_size:
        raise RuntimeError("raw results exist without a manifest; refusing an unverified resume")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read_labels(path):
    rows, in_data = [], False
    for line in pathlib.Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("%"):
            continue
        if line.lower() == "@data":
            in_data = True
        elif in_data:
            rows.append(next(csv.reader([line], skipinitialspace=True))[-1].strip().strip("'\""))
    if not rows:
        raise RuntimeError(f"no target rows in {path}")
    return rows


def read_numeric_targets(path):
    targets, in_data = [], False
    for line in pathlib.Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("%"):
            continue
        if line.lower() == "@data":
            in_data = True
        elif in_data:
            fields = next(csv.reader([line], skipinitialspace=True))
            value = float(fields[-1].strip().strip("'\""))
            if not math.isfinite(value):
                raise RuntimeError(f"nonfinite numeric target in {path}")
            targets.append(value)
    if not targets:
        raise RuntimeError(f"no numeric target rows in {path}")
    return targets


def mean_baseline_mse(train_path, test_path):
    train_targets = read_numeric_targets(train_path)
    test_targets = read_numeric_targets(test_path)
    mean = statistics.mean(train_targets)
    return statistics.mean((target - mean) ** 2 for target in test_targets)


def fixture_names(suite):
    fixtures = []
    if suite in ("all", "binary"):
        fixtures.append("binary.arff")
    if suite in ("all", "regression"):
        fixtures.extend(("regression-train.arff", "regression-test.arff"))
    if suite in ("all", "regression_dense"):
        fixtures.extend(("regression-dense-train.arff", "regression-dense-test.arff"))
    return fixtures


def result_value(completed, key, field):
    if key not in completed:
        raise RuntimeError(f"missing resumable gate record {key}")
    value = completed[key][field]
    if not math.isfinite(value):
        raise RuntimeError(f"nonfinite {field} in {key}")
    return value


def run_one(binary, out, manifest_id, completed, key, field, command, extra):
    if key in completed:
        return result_value(completed, key, field), completed[key]
    record_file = out / "pending-run.jsonl"
    record_file.write_text("")
    call(binary, [*command, "--output", record_file])
    lines = [line for line in record_file.read_text().splitlines() if line.strip()]
    if len(lines) != 1:
        raise RuntimeError(f"expected one run record from {' '.join(map(str, command))}, got {len(lines)}")
    run = json.loads(lines[0])
    value = field
    row = {"manifest_id": manifest_id, "gate_id": extra.get("gate"), **extra,
           "evaluations": run["evaluations"], "run": run}
    if field == "hypervolume":
        value = hv2(run["front"], extra["reference"])
    else:
        value = run["heldout"]["loss"]
    row[field] = value
    append_result(out / "raw.jsonl", row)
    completed[key] = row
    return value, row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mocs", default="target/release/mocs")
    parser.add_argument("--out", default="results/quality")
    parser.add_argument("--suite", choices=("all", "sf2", "zdt6", "binary", "regression", "regression_dense", "zoo"), default="all")
    parser.add_argument("--experiment-threads", type=int, default=1,
                        help="outer worker count for Zoo fold experiments only")
    parser.add_argument("--legacy-data-root", type=pathlib.Path)
    parser.add_argument("--zoo-train-pattern", default="{root}/train{fold}.arff")
    parser.add_argument("--zoo-test-pattern", default="{root}/test{fold}.arff")
    opt = parser.parse_args()
    if opt.experiment_threads < 1:
        raise ValueError("--experiment-threads must be positive")
    binary = pathlib.Path(opt.mocs).resolve()
    out = pathlib.Path(opt.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    quality = pathlib.Path(__file__).resolve().parent.parent / "examples/quality"

    configs = ([f"sf2-{name}.toml" for name in ALGORITHMS] if opt.suite in ("all", "sf2") else [])
    if opt.suite in ("all", "zdt6"):
        configs.append("zdt6-nsga2.toml")
    if opt.suite in ("all", "binary"):
        configs.extend(f"binary-{name}.toml" for name in MODELS)
    if opt.suite in ("all", "regression", "regression_dense"):
        configs.append("regression.toml")
    if opt.suite in ("all", "zoo"):
        configs.extend(f"zoo-{name}.toml" for name in MODELS)
    fixtures = fixture_names(opt.suite)
    zoo_files = []
    if opt.legacy_data_root:
        root = opt.legacy_data_root.resolve()
        for fold in range(10):
            for pattern in (opt.zoo_train_pattern, opt.zoo_test_pattern):
                path = pathlib.Path(pattern.format(root=root, fold=fold))
                if not path.is_file():
                    raise FileNotFoundError(f"Zoo fold file not found: {path}")
                zoo_files.append(path)
    if opt.suite == "zoo" and not opt.legacy_data_root:
        raise ValueError("--suite zoo requires --legacy-data-root")
    manifest = {
        "gate_version": GATE_VERSION, "suite": opt.suite, "seeds": list(SEEDS),
        "zoo_experiment_threads": opt.experiment_threads,
        "binary_sha256": digest(binary), "script_sha256": digest(__file__),
        "configs": {name: digest(quality / name) for name in configs},
        "fixtures": {name: digest(quality / name) for name in fixtures},
        "legacy_data_root": str(opt.legacy_data_root.resolve()) if opt.legacy_data_root else None,
        "zoo_patterns": [opt.zoo_train_pattern, opt.zoo_test_pattern],
        "zoo_inputs": {str(path): digest(path) for path in zoo_files},
    }
    manifest_path = out / "manifest.json"
    raw_path = out / "raw.jsonl"
    manifest_id = prepare_manifest(manifest_path, raw_path, manifest)
    completed = load_completed(raw_path, manifest_id)
    gates = []

    if opt.suite in ("all", "sf2"):
        reference = [5.0, 5.0]
        true_hv = sf2_true_hv()
        for algorithm in ALGORITHMS:
            values = []
            for seed in SEEDS:
                key = ("sf2_all_algorithms", algorithm, None, seed)
                value, _ = run_one(binary, out, manifest_id, completed, key, "hypervolume",
                    ["optimize", "--config", quality / f"sf2-{algorithm}.toml", "--seed", seed],
                    {"gate": key[0], "algorithm": algorithm, "seed": seed, "reference": reference})
                values.append(value)
            median = statistics.median(values)
            gates.append({"gate": f"sf2_{algorithm}", "threshold": .90 * true_hv, "median": median, "pass": median >= .90 * true_hv})
            if algorithm == "nsga2":
                gates.append({"gate": "sf2_nsga2_canonical", "threshold": .95 * true_hv, "median": median, "pass": median >= .95 * true_hv})

    if opt.suite in ("all", "zdt6"):
        reference = [1.1, 1.1]
        true_hv = zdt6_true_hv(reference)
        values = []
        for seed in SEEDS:
            key = ("zdt6_nsga2", None, None, seed)
            value, _ = run_one(binary, out, manifest_id, completed, key, "hypervolume",
                ["optimize", "--config", quality / "zdt6-nsga2.toml", "--seed", seed],
                {"gate": key[0], "seed": seed, "reference": reference})
            values.append(value)
        median = statistics.median(values)
        gates.append({"gate": "zdt6_nsga2", "threshold": .80 * true_hv, "median": median, "pass": median >= .80 * true_hv})

    if opt.suite in ("all", "binary"):
        data = quality / "binary.arff"
        for model in MODELS:
            values = []
            for seed in SEEDS:
                key = ("binary_zero_loss", None, model, seed)
                value, _ = run_one(binary, out, manifest_id, completed, key, "test_error",
                    ["train", "--config", quality / f"binary-{model}.toml", "--data", data, "--test", data,
                     "--model-out", out / "pending-model.json", "--seed", seed],
                    {"gate": key[0], "model": model, "seed": seed})
                values.append(value)
            median = statistics.median(values)
            gates.append({"gate": f"binary_{model}", "threshold": 0.0, "median": median, "pass": median == 0.0})

    if opt.suite in ("all", "regression"):
        train = quality / "regression-train.arff"
        test = quality / "regression-test.arff"
        training_x = [0.0, .5, 1.0]
        test_y = [2 * .25 + 1, 2 * .75 + 1]
        baseline = statistics.mean((y - statistics.mean([2*x+1 for x in training_x])) ** 2 for y in test_y)
        values = []
        for seed in SEEDS:
            key = ("regression", None, None, seed)
            value, row = run_one(binary, out, manifest_id, completed, key, "mse",
                ["train", "--config", quality / "regression.toml", "--data", train, "--test", test,
                 "--model-out", out / "pending-model.json", "--seed", seed],
                {"gate": key[0], "seed": seed, "baseline_mse": baseline})
            values.append(value)
        median = statistics.median(values)
        gates.append({"gate": "regression", "threshold": .5 * baseline, "baseline_mse": baseline, "median": median, "pass": median <= .5 * baseline})

    if opt.suite in ("all", "regression_dense"):
        train = quality / "regression-dense-train.arff"
        test = quality / "regression-dense-test.arff"
        baseline = mean_baseline_mse(train, test)
        if not math.isclose(baseline, 0.3332, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError(f"dense regression fixture baseline changed: expected 0.3332, got {baseline:.17g}")
        values = []
        for seed in SEEDS:
            key = ("regression_dense", None, None, seed)
            value, _ = run_one(binary, out, manifest_id, completed, key, "mse",
                ["train", "--config", quality / "regression.toml", "--data", train, "--test", test,
                 "--model-out", out / "pending-model.json", "--seed", seed],
                {"gate": key[0], "seed": seed, "baseline_mse": baseline})
            values.append(value)
        median = statistics.median(values)
        gates.append({"gate": "regression_dense", "threshold": .5 * baseline,
                      "baseline_mse": baseline, "median": median, "pass": median <= .5 * baseline,
                      "note": "prospective dense-data study; does not replace the preregistered sparse regression gate"})

    if opt.suite in ("all", "zoo"):
        if not opt.legacy_data_root:
            gates.append({"gate": "zoo_ten_fold", "pass": None, "skipped": True,
                          "reason": "pass --legacy-data-root to include the historical folds"})
        else:
            root = opt.legacy_data_root.resolve()
            train_pattern = opt.zoo_train_pattern.format(root=root, fold="{fold}")
            test_pattern = opt.zoo_test_pattern.format(root=root, fold="{fold}")
            train_paths = [pathlib.Path(train_pattern.format(fold=i)) for i in range(10)]
            test_paths = [pathlib.Path(test_pattern.format(fold=i)) for i in range(10)]
            baseline_errors, rows_total = 0, 0
            for tr, te in zip(train_paths, test_paths):
                labels = read_labels(tr)
                counts = {label: labels.count(label) for label in set(labels)}
                majority = max(counts, key=counts.get)
                heldout = read_labels(te)
                baseline_errors += sum(label != majority for label in heldout)
                rows_total += len(heldout)
            baseline = baseline_errors / rows_total
            for model in MODELS:
                values = []
                for seed in SEEDS:
                    key = ("zoo_ten_fold", None, model, seed)
                    if key in completed:
                        value = result_value(completed, key, "test_error")
                    else:
                        raw = out / "pending-experiment.jsonl"
                        raw.write_text("")
                        call(binary, ["experiment", "--config", quality / f"zoo-{model}.toml",
                            "--train-pattern", train_pattern, "--test-pattern", test_pattern,
                            "--folds", "10", "--fold-start", "0", "--seeds", seed,
                            "--threads", opt.experiment_threads, "--output", raw])
                        folds = [json.loads(line) for line in raw.read_text().splitlines() if line.strip()]
                        if len(folds) != 10:
                            raise RuntimeError(f"Zoo experiment returned {len(folds)} records, expected 10")
                        error_rows = sum(run["heldout"]["rows"] for run in folds)
                        value = sum(run["heldout"]["loss"] * run["heldout"]["rows"] for run in folds) / error_rows
                        row = {"manifest_id": manifest_id, "gate": key[0], "model": model, "seed": seed,
                               "gate_id": key[0],
                               "test_error": value, "baseline_error": baseline, "folds": folds}
                        append_result(raw_path, row)
                        completed[key] = row
                    values.append(value)
                median = statistics.median(values)
                gates.append({"gate": f"zoo_{model}", "threshold": baseline - .05,
                              "baseline_error": baseline, "median": median, "pass": median <= baseline - .05})

    evaluated = [gate for gate in gates if not gate.get("skipped")]
    skipped = any(gate.get("skipped") for gate in gates)
    result = {"suite": opt.suite, "gates": gates,
              "all_evaluated_pass": bool(evaluated) and all(gate["pass"] for gate in evaluated),
              "all_pass": bool(evaluated) and not skipped and all(gate["pass"] for gate in evaluated),
              "failed_gates": [gate["gate"] for gate in evaluated if not gate["pass"]],
              "skipped_gates": [gate["gate"] for gate in gates if gate.get("skipped")]}
    (out / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["all_pass"] else 1)


if __name__ == "__main__":
    main()
