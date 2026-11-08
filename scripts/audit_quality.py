#!/usr/bin/env python3
"""Audit a completed quality study without changing its preregistered results."""
import argparse, hashlib, json, math, pathlib, statistics, sys, tomllib

ALGORITHMS = ("moga", "moes", "ngga", "nsga2", "spea2")
MODELS = ("rule", "instance", "mixed")

def digest(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()

def manifest_id(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

def dominates(a, b):
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))

def valid_front(points):
    if not isinstance(points, list) or not points or any(not isinstance(p, list) or not p for p in points):
        return False
    dim = len(points[0])
    if any(len(p) != dim or any(not isinstance(x, (int, float)) or not math.isfinite(x) for x in p)
           for p in points):
        return False
    return not any(dominates(a, b) for i, a in enumerate(points) for j, b in enumerate(points) if i != j)

def expected_keys(seeds):
    return ({("sf2_all_algorithms", a, None, s) for s in seeds for a in ALGORITHMS} | {(g, None, None, s) for s in seeds for g in ("zdt6_nsga2", "regression", "regression_dense")} | {(g, None, m, s) for s in seeds for g in ("binary_zero_loss", "zoo_ten_fold") for m in MODELS})

def budget_errors(run, cap, offspring, context):
    actual, requested, over = (run.get(k) for k in ("evaluations", "requested_evaluations", "budget_overshoot"))
    errors = []
    if requested != cap:
        errors.append(f"{context}: requested cap {requested!r}, expected {cap}")
    if not isinstance(actual, int) or actual < cap:
        errors.append(f"{context}: invalid or below-cap evaluations {actual!r}")
    else:
        expected = max(0, actual - cap)
        if over != expected:
            errors.append(f"{context}: overshoot {over!r}, expected {expected}")
        if actual > cap + 2 * offspring:
            errors.append(f"{context}: exceeded cap by more than two offspring")
    return errors

def finite_tree(x):
    if isinstance(x, float):
        return math.isfinite(x)
    if isinstance(x, dict):
        return all(finite_tree(v) for v in x.values())
    if isinstance(x, (list, tuple)):
        return all(finite_tree(v) for v in x)
    return True

def valid_report(r):
    numeric = ("loss", "coverage", "conditional_loss", "generality", "size", "unused_rule_fraction", "objectives")
    counts = ("rows", "rule_count", "abstentions", "fallbacks", "out_of_range")
    return (isinstance(r, dict) and finite_tree(r)
            and all(isinstance(r.get(k), (int, float)) and math.isfinite(r[k]) for k in numeric[:-1])
            and isinstance(r.get("objectives"), list) and bool(r["objectives"])
            and all(isinstance(x, (int, float)) and math.isfinite(x) for x in r["objectives"])
            and all(isinstance(r.get(k), int) and r[k] >= 0 for k in counts))

def audit(directory):
    directory = pathlib.Path(directory)
    paths = {k: directory / f for k, f in (("manifest", "manifest.json"), ("raw", "raw.jsonl"), ("summary", "summary.json"))}
    if not all(p.is_file() for p in paths.values()):
        raise RuntimeError("quality study is incomplete: manifest.json, raw.jsonl, or summary.json is missing")
    manifest, summary = json.loads(paths["manifest"].read_text()), json.loads(paths["summary"].read_text())
    seeds = manifest.get("seeds", [])
    if manifest.get("suite") != "all" or len(seeds) != 30 or len(set(seeds)) != 30:
        raise RuntimeError("expected suite=all with 30 unique seeds")
    expected, mid, rows, found = expected_keys(seeds), manifest_id(manifest), [], {}
    for n, line in enumerate(paths["raw"].read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"invalid raw JSON line {n}: {exc}") from exc
        key = row["gate"], row.get("algorithm"), row.get("model"), row.get("seed")
        if key in found or row.get("manifest_id") != mid:
            raise RuntimeError(f"duplicate record or manifest ID mismatch: {key}")
        found[key] = row
        rows.append(row)
    missing, extra = expected - found.keys(), found.keys() - expected
    if missing or extra:
        raise RuntimeError(f"incomplete or unexpected records: {len(missing)} missing, {len(extra)} extra")
    root = pathlib.Path(__file__).resolve().parent.parent
    qdir, configs, errors = root / "examples/quality", {}, []
    for name, expected_hash in manifest.get("configs", {}).items():
        path = qdir / name
        if not path.is_file() or digest(path) != expected_hash:
            errors.append(f"config digest mismatch: {name}")
        else:
            configs[name] = tomllib.loads(path.read_text())
    inputs = {**manifest.get("fixtures", {}), **manifest.get("zoo_inputs", {})}
    for name, expected_hash in inputs.items():
        path = pathlib.Path(name) if name in manifest.get("zoo_inputs", {}) else qdir / name
        if not path.is_file() or digest(path) != expected_hash:
            errors.append(f"input digest mismatch: {name}")
    def config_for(row):
        gate = row["gate"]
        name = (f"sf2-{row['algorithm']}.toml" if gate == "sf2_all_algorithms" else
                "zdt6-nsga2.toml" if gate == "zdt6_nsga2" else
                f"binary-{row['model']}.toml" if gate == "binary_zero_loss" else
                f"zoo-{row['model']}.toml" if gate == "zoo_ten_fold" else "regression.toml")
        return configs.get(name, {})

    def expected_inputs(row, fold):
        gate = row["gate"]
        if gate == "binary_zero_loss":
            return [manifest["fixtures"]["binary.arff"]] * 2
        if gate in ("regression", "regression_dense"):
            prefix = "regression-dense" if gate == "regression_dense" else "regression"
            return [manifest["fixtures"][f"{prefix}-{part}.arff"] for part in ("train", "test")]
        if gate == "zoo_ten_fold":
            root = manifest["legacy_data_root"]
            paths = [pathlib.Path(pattern.format(root=root, fold=fold)) for pattern in manifest["zoo_patterns"]]
            return [manifest["zoo_inputs"][str(path)] for path in paths]
        return []

    for row in rows:
        cfg = config_for(row)
        cap, offspring = cfg.get("max_evaluations"), cfg.get("offspring")
        if not isinstance(cap, int) or not isinstance(offspring, int):
            errors.append(f"missing config budget: {row['gate']}")
            continue
        zoo = row["gate"] == "zoo_ten_fold"
        runs = row.get("folds", []) if zoo else [row.get("run")]
        if zoo:
            folds = [str(r.get("fold_id")) for r in runs if isinstance(r, dict)]
            if len(runs) != 10 or set(folds) != {str(i) for i in range(10)}:
                errors.append(f"{row['model']} seed {row['seed']}: expected folds 0..9 exactly once")
        for i, run in enumerate(runs):
            fold = run.get("fold_id") if isinstance(run, dict) else i
            ctx = f"{row['gate']} seed {row['seed']} fold {fold}"
            if not isinstance(run, dict):
                errors.append(f"{ctx}: missing run record")
                continue
            if run.get("seed") != row.get("seed"):
                errors.append(f"{ctx}: run seed mismatch")
            errors.extend(budget_errors(run, cap, offspring, ctx))
            if not valid_front(run.get("front")):
                errors.append(f"{ctx}: missing, nonfinite, inconsistent, or dominated front")
            if run.get("heldout") is not None and not valid_report(run["heldout"]):
                errors.append(f"{ctx}: missing or nonfinite heldout metrics")
            hashes = run.get("input_sha256")
            if hashes != expected_inputs(row, fold):
                errors.append(f"{ctx}: ordered input hashes do not match manifested case/fold")
        if not zoo and row.get("evaluations") != (row.get("run") or {}).get("evaluations"):
            errors.append(f"{row['gate']} seed {row['seed']}: row/run evaluation mismatch")

    zoo = {}
    for model in MODELS:
        reps = []
        for seed in seeds:
            folds = found[("zoo_ten_fold", None, model, seed)]["folds"]
            nrows = sum(f["heldout"]["rows"] for f in folds)
            weighted = lambda field: sum(f["heldout"][field] * f["heldout"]["rows"] for f in folds) / nrows
            reps.append({"error": weighted("loss"), "coverage": weighted("coverage"),
                         **{k: sum(f["heldout"][k] for f in folds)
                            for k in ("abstentions", "fallbacks", "out_of_range")},
                         "evaluations": sum(f["evaluations"] for f in folds)})
        zoo[model] = {f"median_{k}": statistics.median(r[k] for r in reps)
                      for k in ("error", "coverage", "abstentions", "fallbacks", "out_of_range", "evaluations")}
        zoo[model].update(repetitions=len(reps), folds_per_repetition=10)

    runs = [r for row in rows for r in (row.get("folds", []) if row["gate"] == "zoo_ten_fold" else [row.get("run")]) if isinstance(r, dict)]
    revisions = sorted({r.get("revision") for r in runs})
    if len(revisions) != 1 or any(r.get("source_dirty") is not False for r in runs):
        errors.append("runs do not share one clean source revision")
    return {"audit_passed": not errors, "audit_errors": errors, "study": str(directory),
            "suite": manifest["suite"], "manifest_id": mid,
            "hashes": {"manifest_sha256": digest(paths["manifest"]), "raw_sha256": digest(paths["raw"]),
                       "binary_sha256": manifest.get("binary_sha256"),
                       "quality_script_sha256": manifest.get("script_sha256"),
                       "audit_script_sha256": digest(__file__)},
            "binary_revision": revisions[0] if len(revisions) == 1 else revisions,
            "records": {"expected": len(expected), "observed": len(rows), "seeds": len(seeds)},
            "zoo_cv": {"inferential_unit": "one seeded 10-fold repetition; folds pooled within seed/model",
                       "models": zoo},
            "release_gates": {k: summary.get(k) for k in
                               ("all_pass", "all_evaluated_pass", "failed_gates", "skipped_gates", "gates")}}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=pathlib.Path, help="completed quality study directory")
    parser.add_argument("--output", type=pathlib.Path, help="audit JSON path (default: <study>/audit.json)")
    args = parser.parse_args()
    try:
        result = audit(args.study)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    output = args.output or args.study / "audit.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"audit_passed": result["audit_passed"], "output": str(output),
                      "records": result["records"], "audit_errors": result["audit_errors"]}, indent=2))
    return 0 if result["audit_passed"] else 1

if __name__ == "__main__":
    raise SystemExit(main())
