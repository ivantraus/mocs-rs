#!/usr/bin/env python3
"""Run resumable, paired performance studies for fixed mocs-rs workloads.

First freeze a release binary and collect its pilot rows:

  python3 scripts/perf_study.py --baseline target/release/mocs --manifest results/perf-study.json

After a candidate binary exists, pass `--candidate`; the script warms both
binaries once per case, then alternates baseline-first and candidate-first
pairs. It writes a checkpoint after every child process so interrupted studies
can resume without losing completed pairs.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import random
import resource
import statistics
import subprocess
import sys
import tempfile
import time
import tomllib


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 2
_BENCHMARK_SPEC = importlib.util.spec_from_file_location(
    "_mocs_benchmark_analysis", Path(__file__).with_name("benchmark.py")
)
benchmark_analysis = importlib.util.module_from_spec(_BENCHMARK_SPEC)
assert _BENCHMARK_SPEC.loader is not None
_BENCHMARK_SPEC.loader.exec_module(benchmark_analysis)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def current_host_identity() -> dict:
    return {
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "node": platform.node(),
        "python": sys.version.split()[0],
    }


def parse_seeds(value: str, count: int) -> list[int]:
    if value == "default":
        return list(range(count))
    if ":" in value:
        start, stop = (int(x) for x in value.split(":", 1))
        seeds = list(range(start, stop))
    else:
        seeds = [int(x) for x in value.split(",") if x.strip()]
    if not seeds or any(seed < 0 or seed >= 2**64 for seed in seeds):
        raise ValueError("seeds must be a nonempty list of unsigned 64-bit integers")
    if len(seeds) != len(set(seeds)):
        raise ValueError("seed list contains duplicates")
    return seeds


def scientific_record(record: dict) -> dict:
    """Return exact resolved workload and result fields, omitting build/timing metadata."""
    keys = (
        "schema_version",
        "semantic_version",
        "rng_version",
        "config",
        "input_sha256",
        "dataset_id",
        "fold_id",
        "split_id",
        "split_seed",
        "train_indices_sha256",
        "test_indices_sha256",
        "seed",
        "evaluations",
        "generations",
        "requested_evaluations",
        "front",
        "training_selection",
        "heldout",
    )
    return {key: record.get(key) for key in keys}


def scientific_fingerprint(record: dict) -> str:
    return benchmark_analysis.canonical_fingerprint(scientific_record(record))


def _peak_rss_mb(value: float) -> float:
    # getrusage reports bytes on macOS and KiB on Linux and the BSDs.
    bytes_used = value if sys.platform == "darwin" else value * 1024.0
    return bytes_used / (1024.0 * 1024.0)


def run_worker(binary: Path, case: dict, seed: int, workdir: Path) -> dict:
    output_path = workdir / "run.jsonl"
    if case["command"] == "optimize":
        command = [
            str(binary), "optimize", "--config", str(case["config_path"]),
            "--seed", str(seed), "--output", str(output_path),
        ]
    else:
        model_out = workdir / "model.json"
        command = [
            str(binary), "train", "--config", str(case["config_path"]),
            "--data", str(case["data_path"]), "--seed", str(seed),
            "--model-out", str(model_out), "--output", str(output_path),
        ]
    started = time.perf_counter()
    completed = subprocess.run(command, capture_output=True, text=True)
    process_wall = time.perf_counter() - started
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    if completed.returncode:
        raise RuntimeError(
            f"{case['name']} seed={seed} exited {completed.returncode}: {completed.stderr.strip()}"
        )
    records = [line for line in output_path.read_text().splitlines() if line.strip()]
    if len(records) != 1:
        raise RuntimeError(f"{case['name']} seed={seed} wrote {len(records)} run records")
    record = json.loads(records[0])
    if record.get("schema_version") != 1:
        raise RuntimeError(f"{case['name']} seed={seed} emitted an unsupported record schema")
    elapsed = record.get("elapsed_seconds")
    if not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed <= 0:
        raise RuntimeError(f"{case['name']} seed={seed} emitted invalid internal elapsed time")
    return {
        "benchmark_case": case["name"],
        "seed": seed,
        "internal_elapsed_seconds": elapsed,
        "process_wall_seconds": process_wall,
        "child_cpu_seconds": usage.ru_utime + usage.ru_stime,
        "child_user_seconds": usage.ru_utime,
        "child_system_seconds": usage.ru_stime,
        "child_peak_rss_mb": _peak_rss_mb(usage.ru_maxrss),
        "config_signature": hashlib.sha256(canonical(record.get("config")).encode()).hexdigest(),
        "scientific_record": scientific_record(record),
        "run_record": record,
    }


def _worker_main(args: argparse.Namespace) -> None:
    binary = Path(args.worker_binary).resolve()
    config_path = Path(args.worker_config).resolve()
    case = {
        "name": args.worker_case,
        "command": "train" if args.worker_data else "optimize",
        "config_path": config_path,
        "data_path": Path(args.worker_data).resolve() if args.worker_data else None,
    }
    with tempfile.TemporaryDirectory(prefix="mocs-perf-child-") as tmp:
        result = run_worker(binary, case, args.worker_seed, Path(tmp))
    print(json.dumps(result, allow_nan=False, separators=(",", ":")))


def run_isolated(binary: Path, case: dict, seed: int) -> dict:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--worker-binary", str(binary),
        "--worker-case", case["name"],
        "--worker-config", str(case["config_path"]),
        "--worker-seed", str(seed),
    ]
    if case["data_path"] is not None:
        command.extend(["--worker-data", str(case["data_path"])])
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip() or "(worker produced no diagnostic output)"
        detail = detail[-4000:]
        raise RuntimeError(
            f"performance worker failed for {case['name']} seed={seed} "
            f"(exit {result.returncode}):\n{detail}"
        )
    return json.loads(result.stdout)


def load_cases(directory: Path, data: Path) -> list[dict]:
    cases = []
    for config_path in sorted(directory.glob("*.toml")):
        with config_path.open("rb") as stream:
            config = tomllib.load(stream)
        is_model = "model" in config.get("training", {})
        cases.append({
            "name": config_path.stem,
            "command": "train" if is_model else "optimize",
            "config_path": config_path.resolve(),
            "config_sha256": sha256_file(config_path),
            "config_signature": hashlib.sha256(canonical(config).encode()).hexdigest(),
            "data_path": data.resolve() if is_model else None,
            "data_sha256": sha256_file(data) if is_model else None,
        })
    if not cases:
        raise ValueError(f"no TOML cases found in {directory}")
    return cases


def save_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


def check_identity(
    manifest: dict,
    baseline: Path,
    candidate: Path | None,
    cases: list[dict],
    seeds: list[int],
    warmups: int,
    *,
    script_sha256: str,
    host: dict,
    settings: dict,
) -> None:
    if manifest["baseline"]["sha256"] != sha256_file(baseline):
        raise ValueError("baseline binary hash differs from the resumable manifest")
    prior_candidate = manifest.get("candidate")
    if candidate is not None and prior_candidate and prior_candidate["sha256"] != sha256_file(candidate):
        raise ValueError("candidate binary hash differs from the resumable manifest")
    if manifest["case_identity"] != [
        {key: case[key] for key in ("name", "config_sha256", "config_signature", "data_sha256")}
        for case in cases
    ]:
        raise ValueError("case configs or input data differ from the resumable manifest")
    if manifest["seeds"] != seeds or manifest["warmups"] != warmups:
        raise ValueError("seed or warmup settings differ from the resumable manifest")
    if manifest.get("script_sha256") != script_sha256:
        raise ValueError("performance script hash differs from the resumable manifest")
    if manifest.get("host") != host:
        raise ValueError("host identity differs from the resumable manifest")
    if manifest.get("settings") != settings:
        raise ValueError("primary benchmark or performance gate settings differ from the resumable manifest")


def key(case_name: str, seed: int) -> str:
    return f"{case_name}\t{seed}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--cases", type=Path, default=ROOT / "examples/bench")
    parser.add_argument("--data", type=Path, default=ROOT / "examples/bench/synthetic-200x10.arff")
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--seeds", default="default", help="comma-separated values or start:stop; default is 0..repetitions-1")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=17)
    parser.add_argument("--primary", default="sf2-spea2-p100-g25")
    parser.add_argument("--minimum-improvement", type=float, default=0.05)
    parser.add_argument("--maximum-regression", type=float, default=0.05)
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="write and validate the run identity manifest without warming or timing binaries",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-binary")
    parser.add_argument("--worker-case")
    parser.add_argument("--worker-config")
    parser.add_argument("--worker-data")
    parser.add_argument("--worker-seed", type=int)
    args = parser.parse_args()
    if args.worker:
        _worker_main(args)
        return
    if args.baseline is None or args.manifest is None:
        parser.error("--baseline and --manifest are required outside worker mode")
    if args.repetitions < 1 or args.warmups < 0 or args.bootstrap_samples < 1:
        raise ValueError("repetitions/bootstrap samples must be positive and warmups nonnegative")
    baseline = args.baseline.resolve()
    candidate = args.candidate.resolve() if args.candidate else None
    if not baseline.is_file() or not os.access(baseline, os.X_OK):
        raise ValueError(f"baseline binary must be executable: {baseline}")
    if candidate and (not candidate.is_file() or not os.access(candidate, os.X_OK)):
        raise ValueError(f"candidate binary must be executable: {candidate}")
    cases = load_cases(args.cases.resolve(), args.data.resolve())
    seeds = parse_seeds(args.seeds, args.repetitions)
    script_sha256 = sha256_file(Path(__file__).resolve())
    host = current_host_identity()
    settings = {
        "primary": args.primary,
        "minimum_improvement": args.minimum_improvement,
        "maximum_regression": args.maximum_regression,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
        "comparison_gate": "paired-bootstrap-median-v1",
    }
    manifest_path = args.manifest.resolve()
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported performance manifest schema")
        check_identity(manifest, baseline, candidate, cases, seeds, args.warmups,
                       script_sha256=script_sha256, host=host, settings=settings)
    else:
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "baseline": {"path": str(baseline), "sha256": sha256_file(baseline)},
            "candidate": None,
            "host": host,
            "script_sha256": script_sha256,
            "settings": settings,
            "seeds": seeds,
            "warmups": args.warmups,
            "case_identity": [
                {key: case[key] for key in ("name", "config_sha256", "config_signature", "data_sha256")}
                for case in cases
            ],
            "cases": {
                case["name"]: {
                    "name": case["name"],
                    "command": case["command"],
                    "config_path": str(case["config_path"]),
                    "config_sha256": case["config_sha256"],
                    "config_signature": case["config_signature"],
                    "data_path": str(case["data_path"]) if case["data_path"] else None,
                    "data_sha256": case["data_sha256"],
                }
                for case in cases
            },
            "pilot": {},
            "warmup_results": {},
            "paired": {},
        }
        save_manifest(manifest_path, manifest)
    if candidate:
        if manifest.get("candidate") is None:
            manifest["candidate"] = {"path": str(candidate), "sha256": sha256_file(candidate)}
        save_manifest(manifest_path, manifest)

    if args.prepare_only:
        print(f"performance identity preregistered: {manifest_path}")
        return

    binaries = {"baseline": baseline}
    if candidate:
        binaries["candidate"] = candidate
    for case in cases:
        for role, binary in binaries.items():
            warmup_key = f"{role}\t{case['name']}"
            for warmup in range(args.warmups):
                item = f"{warmup_key}\t{warmup}"
                if item not in manifest["warmup_results"]:
                    print(f"warmup {role} {case['name']} {warmup + 1}/{args.warmups}", flush=True)
                    manifest["warmup_results"][item] = run_isolated(binary, case, 0)
                    save_manifest(manifest_path, manifest)

    if not candidate:
        for case in cases:
            for seed in seeds:
                item = key(case["name"], seed)
                if item in manifest["pilot"]:
                    continue
                print(f"baseline pilot {case['name']} seed={seed}", flush=True)
                result = run_isolated(baseline, case, seed)
                result["binary_sha256"] = manifest["baseline"]["sha256"]
                manifest["pilot"][item] = result
                save_manifest(manifest_path, manifest)
        export_case_jsonl(manifest, manifest_path)
        print(f"baseline pilot complete: {manifest_path}")
        return

    for seed_index, seed in enumerate(seeds):
        for case in cases:
            item = key(case["name"], seed)
            pair = manifest["paired"].setdefault(item, {})
            order = ("baseline", "candidate") if seed_index % 2 == 0 else ("candidate", "baseline")
            for role in order:
                if role in pair:
                    continue
                print(f"paired run {role} {case['name']} seed={seed}", flush=True)
                result = run_isolated(binaries[role], case, seed)
                result["binary_sha256"] = manifest[role]["sha256"]
                pair[role] = result
                save_manifest(manifest_path, manifest)
            if scientific_fingerprint(pair["baseline"]["scientific_record"]) != scientific_fingerprint(pair["candidate"]["scientific_record"]):
                raise ValueError(f"scientific records differ for {case['name']} seed={seed}")

    comparison = compare(manifest, args)
    manifest["comparison"] = comparison
    save_manifest(manifest_path, manifest)
    export_case_jsonl(manifest, manifest_path)
    print(json.dumps(comparison, indent=2, allow_nan=False))
    if not comparison["all_pass"]:
        raise SystemExit(1)


def compare(manifest: dict, args: argparse.Namespace) -> dict:
    groups = {}
    for item, pair in manifest["paired"].items():
        if "baseline" not in pair or "candidate" not in pair:
            raise ValueError(f"incomplete pair {item}")
        case, seed_str = item.split("\t", 1)
        base = pair["baseline"]
        cand = pair["candidate"]
        improvement = 1.0 - cand["process_wall_seconds"] / base["process_wall_seconds"]
        groups.setdefault(case, []).append({
            "seed": int(seed_str),
            "process_wall_improvement": improvement,
            "baseline": {k: base[k] for k in ("internal_elapsed_seconds", "process_wall_seconds", "child_cpu_seconds", "child_peak_rss_mb")},
            "candidate": {k: cand[k] for k in ("internal_elapsed_seconds", "process_wall_seconds", "child_cpu_seconds", "child_peak_rss_mb")},
        })
    rng = random.Random(args.bootstrap_seed)
    results = []
    for case_name, rows in sorted(groups.items()):
        values = [row["process_wall_improvement"] for row in rows]
        bootstrap = sorted(statistics.median(rng.choices(values, k=len(values))) for _ in range(args.bootstrap_samples))
        low = bootstrap[int(0.025 * (len(bootstrap) - 1))]
        high = bootstrap[int(0.975 * (len(bootstrap) - 1))]
        median = statistics.median(values)
        primary = case_name == args.primary
        passed = median >= args.minimum_improvement and low > 0 if primary else median >= -args.maximum_regression
        results.append({
            "benchmark_case": case_name,
            "primary": primary,
            "pairs": len(rows),
            "median_process_wall_improvement": median,
            "process_wall_ci95": [low, high],
            "median_baseline_child_cpu_seconds": statistics.median(row["baseline"]["child_cpu_seconds"] for row in rows),
            "median_candidate_child_cpu_seconds": statistics.median(row["candidate"]["child_cpu_seconds"] for row in rows),
            "median_baseline_peak_rss_mb": statistics.median(row["baseline"]["child_peak_rss_mb"] for row in rows),
            "median_candidate_peak_rss_mb": statistics.median(row["candidate"]["child_peak_rss_mb"] for row in rows),
            "pass": passed,
            "pairs_by_seed": rows,
        })
    if not any(row["primary"] for row in results):
        raise ValueError(f"primary benchmark {args.primary!r} is absent")
    return {
        "metric": "process_wall_seconds",
        "criteria": {
            "primary": args.primary,
            "minimum_primary_improvement": args.minimum_improvement,
            "maximum_other_regression": args.maximum_regression,
            "primary_ci_lower_must_exceed": 0,
        },
        "workload_identity": "case, resolved config, seed, data hashes, evaluation counts, and scientific results must match exactly",
        "cases": results,
        "all_pass": bool(results) and all(row["pass"] for row in results),
    }


def export_case_jsonl(manifest: dict, manifest_path: Path) -> None:
    """Emit standard run records in one file per case for benchmark.py compatibility."""
    root = manifest_path.with_suffix("").with_name(manifest_path.stem + "-runs")
    for role in ("baseline", "candidate"):
        source = manifest.get("pilot", {}) if role == "baseline" and manifest.get("candidate") is None else manifest.get("paired", {})
        rows_by_case = {}
        for item, value in source.items():
            if role == "baseline" and "scientific_record" in value:
                measurement = value
            else:
                measurement = value.get(role)
            if measurement is None:
                continue
            case_name, _ = item.split("\t", 1)
            record = dict(measurement["run_record"])
            record["benchmark_case"] = case_name
            record["benchmark_config_signature"] = measurement["config_signature"]
            record["benchmark_binary_sha256"] = measurement["binary_sha256"]
            record["process_wall_seconds"] = measurement["process_wall_seconds"]
            record["child_cpu_seconds"] = measurement["child_cpu_seconds"]
            record["child_peak_rss_mb"] = measurement["child_peak_rss_mb"]
            rows_by_case.setdefault(case_name, []).append(record)
        for case_name, records in rows_by_case.items():
            destination = root / role / f"{case_name}.jsonl"
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text("".join(json.dumps(record, allow_nan=False) + "\n" for record in records))
    print(f"per-case JSONL exports: {root}")


if __name__ == "__main__":
    main()
