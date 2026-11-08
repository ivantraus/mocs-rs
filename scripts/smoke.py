#!/usr/bin/env python3
"""Small reproducible runtime matrix for CLI and workflow coverage.

This is a compatibility smoke, not a quality study. It uses tiny seeded runs and
writes one compact JSON evidence file under results/smoke by default.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import subprocess
import tempfile
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parent.parent
ALGORITHMS = ("moga", "moes", "ngga", "nsga2", "spea2")
NUMERIC_PROBLEMS = ("sf2", "kur", "okk", "zdt6")
MODELS = {
    "rule": "binary.arff",
    "instance": "binary.arff",
    "mixed": "binary.arff",
    "regression_rule": "regression-train.arff",
}
FITNESS = ("penalized_loss", "loss_size", "loss_generality_size", "coverage_size")
DECISIONS = ("best_accuracy", "compromise")
CROSSOVERS = ("k_point", "variable_length", "simulated_binary")
MUTATIONS = ("uniform", "normal", "polynomial", "adaptive")
MATINGS = ("random", "tournament", "nearest_neighbour", "fitness_proportionate")


def toml_config(*, algorithm: str = "nsga2", problem: str = "sf2",
                model: str | None = None, fitness: str = "loss_size",
                decision: str = "best_accuracy", crossover: str = "simulated_binary",
                mutation: str = "polynomial", mating: str = "tournament") -> str:
    lines = [
        "seed = 0", "population = 8", "offspring = 8", "generations = 1", "threads = 1",
        "", "[algorithm]", f'kind = "{algorithm}"',
        "", "[problem]", f'kind = "{problem}"',
    ]
    if problem == "zdt6":
        lines.append("dimensions = 4")
    elif problem == "kur":
        lines.append("dimensions = 3")
    if model:
        lines += [
            "", "[training]", f'model = "{model}"', f'fitness = "{fitness}"',
            f'decision = "{decision}"', "initial_rules = 3", "prune_probability = 0.0",
        ]
    lines += [
        "", "[operators]", f'mating = "{mating}"', f'crossover = "{crossover}"',
        f'mutation = "{mutation}"', "crossover_probability = 0.8",
        "mutation_probability = 1.0", "per_locus_probability = 1.0",
        "crossover_points = 1", "crossover_alignment = 1", "sbx_eta = 15.0",
        "mutation_eta = 15.0", "normal_epsilon = 0.000001",
    ]
    return "\n".join(lines) + "\n"


def finite_tree(value: Any, path: str = "record") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{path} is not finite")
    if isinstance(value, list):
        for i, item in enumerate(value):
            finite_tree(item, f"{path}[{i}]")
    elif isinstance(value, dict):
        for key, item in value.items():
            finite_tree(item, f"{path}.{key}")


def sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_front(front: list[list[float]]) -> None:
    if not front:
        raise ValueError("final front is empty")
    width = len(front[0])
    if width == 0 or any(len(point) != width for point in front):
        raise ValueError("final front has inconsistent objective dimensions")
    for i, a in enumerate(front):
        for j, b in enumerate(front):
            if i == j or a == b:
                continue  # exact duplicates are allowed in a reported front
            if all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b)):
                raise ValueError(f"final front contains dominated point {j} (dominated by point {i})")


def report_summary(report: dict[str, Any]) -> dict[str, Any]:
    fields = ("loss", "conditional_loss", "coverage", "generality", "size", "rule_count",
              "unused_rule_fraction", "abstentions", "fallbacks", "out_of_range", "rows")
    return {field: report[field] for field in fields}


def compact_evidence(evidence: dict[str, Any]) -> dict[str, Any]:
    """Keep every matrix cell in a reviewable, low-overhead JSON summary."""
    numeric, operators, models, historical = [], [], [], []
    for run in evidence["runs"]:
        if run["case"].startswith("numeric:"):
            numeric.append([run[k] for k in ("algorithm", "problem", "evaluations", "front_size")])
        elif run["case"].startswith("operators:"):
            operators.append([run[k] for k in ("crossover", "mutation", "mating", "evaluations", "front_size")])
        elif run["case"].startswith("model:"):
            models.append([run[k] for k in (
                "algorithm", "model", "fitness", "decision", "evaluations", "front_size",
                "training_report", "heldout_report",
            )])
        else:
            historical.append({k: v for k, v in run.items() if k != "case"} | {"case": run["case"]})
    return {
        "suite": evidence["suite"],
        "scope": evidence["scope"],
        "binary_sha256": evidence["executable_sha256"],
        "script_sha256": evidence["script_sha256"],
        "fixtures_sha256": evidence["fixtures_sha256"],
        "settings": {"seed": 0, "population": 8, "offspring": 8, "generations": 1, "threads": 1},
        "counts": {"numeric": len(numeric), "operator_combinations": len(operators),
                   "model_combinations": len(models), "historical_diagnostics": len(historical)},
        "numeric": numeric,
        "operators": operators,
        "models": models,
        "historical": historical,
    }


def run_record(binary: pathlib.Path, config: pathlib.Path, *, data: pathlib.Path | None,
               test_data: pathlib.Path | None = None, model_out: pathlib.Path | None,
               output: pathlib.Path) -> dict[str, Any]:
    command = [str(binary), "optimize" if data is None else "train", "--config", str(config),
               "--output", str(output)]
    if data is not None:
        if model_out is None:
            raise ValueError("training smoke needs a model output path")
        command += ["--data", str(data), "--test", str(test_data or data), "--model-out", str(model_out)]
    proc = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    if proc.returncode:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(command)}\n{proc.stderr.strip()}")
    rows = [line for line in output.read_text().splitlines() if line.strip()]
    if len(rows) != 1:
        raise ValueError(f"expected one JSONL record, got {len(rows)} for {command}")
    record = json.loads(rows[0])
    finite_tree(record)
    required = ("schema_version", "config", "evaluations", "generations", "front", "training_selection", "heldout")
    missing = [name for name in required if name not in record]
    if missing:
        raise ValueError(f"record missing fields: {missing}")
    if record["schema_version"] != 1 or record["evaluations"] < 1 or record["generations"] < 1:
        raise ValueError(f"invalid record schema or counts: {record}")
    if not isinstance(record["front"], list):
        raise ValueError("front is not a list")
    validate_front(record["front"])
    if data is None and (record["training_selection"] is not None or record["heldout"] is not None):
        raise ValueError("numeric optimization record unexpectedly contains learning reports")
    if data is not None:
        if not isinstance(record["training_selection"], dict) or not isinstance(record["heldout"], dict):
            raise ValueError("training record is missing selection or held-out diagnostics")
        for report_name in ("training_selection", "heldout"):
            report = record[report_name]
            for field in ("loss", "conditional_loss", "coverage", "abstentions", "fallbacks",
                          "out_of_range", "rows", "rule_count"):
                if field not in report:
                    raise ValueError(f"{report_name} is missing {field}")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mocs", type=pathlib.Path, default=ROOT / "target/debug/mocs")
    parser.add_argument("--out", type=pathlib.Path, default=ROOT / "results/smoke")
    parser.add_argument("--diagnostics", action="store_true",
                        help="also run Bupa and Sonar historical-data compatibility diagnostics")
    parser.add_argument("--legacy-data-root", type=pathlib.Path,
                        help="directory holding externally obtained bupa/ and sonar/ fold files")
    parser.add_argument("--baseline-summary", type=pathlib.Path,
                        help="also write the compact review summary to this path")
    args = parser.parse_args()
    if args.diagnostics and args.legacy_data_root is None:
        parser.error("--diagnostics requires --legacy-data-root")
    binary = args.mocs.resolve()
    if not binary.is_file():
        raise FileNotFoundError(f"mocs executable not found: {binary}")
    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    quality = ROOT / "examples/quality"
    evidence: dict[str, Any] = {
        "suite": "runtime_smoke_v1",
        "executable": str(binary),
        "executable_sha256": sha256_file(binary),
        "script_sha256": sha256_file(pathlib.Path(__file__).resolve()),
        "fixtures_sha256": {
            fixture: sha256_file(quality / fixture)
            for fixture in sorted(set(MODELS.values()))
        },
        "runs": [],
    }
    raw_records: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="mocs-smoke-") as temp:
        temp_dir = pathlib.Path(temp)

        def run(case: str, *, problem: str = "sf2", algorithm: str = "nsga2",
                data: pathlib.Path | None = None, test_data: pathlib.Path | None = None,
                model: str | None = None,
                fitness: str = "loss_size", decision: str = "best_accuracy",
                crossover: str = "simulated_binary", mutation: str = "polynomial",
                mating: str = "tournament", index: int) -> None:
            cfg = temp_dir / f"case-{index}.toml"
            cfg.write_text(toml_config(algorithm=algorithm, problem=problem, model=model,
                                       fitness=fitness, decision=decision, crossover=crossover,
                                       mutation=mutation, mating=mating))
            record_path = temp_dir / f"case-{index}.jsonl"
            model_path = temp_dir / f"case-{index}.model.json" if data is not None else None
            record = run_record(binary, cfg, data=data, test_data=test_data,
                                model_out=model_path, output=record_path)
            raw_records.append({"case": case, "record": record})
            evidence["runs"].append({
                "case": case,
                "algorithm": record["config"]["algorithm"]["kind"],
                "problem": record["config"]["problem"]["kind"],
                "model": record["config"]["training"]["model"] if data is not None else None,
                "fitness": record["config"]["training"]["fitness"] if data is not None else None,
                "decision": record["config"]["training"]["decision"] if data is not None else None,
                "crossover": record["config"]["operators"]["crossover"],
                "mutation": record["config"]["operators"]["mutation"],
                "mating": record["config"]["operators"]["mating"],
                "evaluations": record["evaluations"],
                "generations": record["generations"],
                "front_size": len(record["front"]),
                "input_sha256": record["input_sha256"],
                "dataset_id": record["dataset_id"],
                "train_sha256": sha256_file(data) if data is not None else None,
                "test_sha256": sha256_file(test_data or data) if data is not None else None,
                "training_report": report_summary(record["training_selection"]) if data is not None else None,
                "heldout_report": report_summary(record["heldout"]) if data is not None else None,
            })

        index = 0
        for algorithm in ALGORITHMS:
            for problem in NUMERIC_PROBLEMS:
                run(f"numeric:{algorithm}:{problem}", algorithm=algorithm, problem=problem, index=index)
                index += 1

        # SBX is exercised only on fixed-length real numeric genomes. K-point and
        # variable-length crossover are also crossed with every mutation family.
        for crossover in CROSSOVERS:
            for mutation in MUTATIONS:
                for mating in MATINGS:
                    run(f"operators:{crossover}:{mutation}:{mating}", problem="kur",
                        crossover=crossover, mutation=mutation, mating=mating, index=index)
                    index += 1

        for algorithm in ALGORITHMS:
            for model, fixture in MODELS.items():
                data = quality / fixture
                for fitness in FITNESS:
                    for decision in DECISIONS:
                        run(f"model:{algorithm}:{model}:{fitness}:{decision}", algorithm=algorithm,
                            model=model, fitness=fitness, decision=decision, data=data,
                            crossover="variable_length", mutation="uniform", index=index)
                        index += 1

        if args.diagnostics:
            for dataset in ("bupa", "sonar"):
                for model in ("rule", "instance", "mixed"):
                    data_root = args.legacy_data_root / dataset
                    train = data_root / "train0.arff"
                    test = data_root / "test0.arff"
                    if not train.is_file() or not test.is_file():
                        raise FileNotFoundError(f"historical fold-0 inputs not found under {data_root}")
                    cfg = temp_dir / f"case-{index}.toml"
                    cfg.write_text(toml_config(model=model, crossover="variable_length", mutation="uniform"))
                    record_path = temp_dir / f"case-{index}.jsonl"
                    model_path = temp_dir / f"case-{index}.model.json"
                    record = run_record(binary, cfg, data=train, test_data=test,
                                        model_out=model_path, output=record_path)
                    raw_records.append({"case": f"historical:{dataset}:{model}:fold0", "record": record})
                    evidence["runs"].append({
                        "case": f"historical:{dataset}:{model}:fold0",
                        "algorithm": record["config"]["algorithm"]["kind"],
                        "problem": record["config"]["problem"]["kind"],
                        "model": model,
                        "fitness": record["config"]["training"]["fitness"],
                        "decision": record["config"]["training"]["decision"],
                        "crossover": record["config"]["operators"]["crossover"],
                        "mutation": record["config"]["operators"]["mutation"],
                        "mating": record["config"]["operators"]["mating"],
                        "evaluations": record["evaluations"],
                        "generations": record["generations"],
                        "front_size": len(record["front"]),
                        "input_sha256": record["input_sha256"],
                        "dataset_id": record["dataset_id"],
                        "train_sha256": sha256_file(train),
                        "test_sha256": sha256_file(test),
                        "training_report": report_summary(record["training_selection"]),
                        "heldout_report": report_summary(record["heldout"]),
                        "interpretation": (
                            "compatibility diagnostic only; Bupa target is the historical selector field, "
                            "not a disease-presence label" if dataset == "bupa" else
                            "compatibility diagnostic on historical Sonar classification labels"
                        ),
                    })
                    index += 1

    evidence["run_count"] = len(evidence["runs"])
    evidence["scope"] = "tiny seeded compatibility checks; not a quality or performance study"
    evidence_path = out_dir / "evidence.json"
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False) + "\n")
    raw_path = out_dir / "raw.jsonl"
    raw_path.write_text("".join(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
                              for row in raw_records))
    if args.baseline_summary:
        summary_path = args.baseline_summary.resolve()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary = compact_evidence(evidence)
        summary["front_audit"] = {
            "result": "pass",
            "records": len(raw_records),
            "check": "finite objectives; consistent objective dimensions; no point strictly Pareto-dominates another; exact duplicates allowed",
            "raw_jsonl_sha256": sha256_file(raw_path),
            "audit_script_sha256": sha256_file(pathlib.Path(__file__).resolve()),
            "baseline_script_sha256": evidence["script_sha256"],
        }
        summary_path.write_text(json.dumps(summary, sort_keys=True,
                                           separators=(",", ":"), allow_nan=False) + "\n")
    print(f"wrote {evidence_path} ({evidence['run_count']} runs)")


if __name__ == "__main__":
    main()
