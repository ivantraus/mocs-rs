#!/usr/bin/env python3
"""Audit exact paired scientific identities in a saved performance manifest."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


_BENCHMARK_SPEC = importlib.util.spec_from_file_location(
    "_mocs_benchmark_audit", Path(__file__).with_name("benchmark.py")
)
benchmark = importlib.util.module_from_spec(_BENCHMARK_SPEC)
assert _BENCHMARK_SPEC.loader is not None
_BENCHMARK_SPEC.loader.exec_module(benchmark)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def audit_manifest(manifest_path: Path, expected_pairs_per_case: int = 30) -> dict:
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    cases = [item["name"] for item in manifest.get("case_identity", [])]
    if not cases or len(cases) != len(set(cases)):
        raise ValueError("manifest must contain unique benchmark case identities")
    if expected_pairs_per_case < 1:
        raise ValueError("expected pairs per case must be positive")

    expected_seeds = set(manifest.get("seeds", []))
    if len(expected_seeds) != expected_pairs_per_case:
        raise ValueError(
            f"manifest has {len(expected_seeds)} unique seeds; expected {expected_pairs_per_case}"
        )
    paired = manifest.get("paired", {})
    counts = {case: 0 for case in cases}
    seen = {case: set() for case in cases}
    mismatches = []
    incomplete = []
    for pair_id, pair in sorted(paired.items()):
        try:
            case, seed_text = pair_id.split("\t", 1)
            seed = int(seed_text)
        except (ValueError, TypeError) as error:
            incomplete.append({"pair": pair_id, "reason": f"invalid pair key: {error}"})
            continue
        if case not in counts:
            incomplete.append({"pair": pair_id, "reason": "unknown benchmark case"})
            continue
        if seed in seen[case]:
            incomplete.append({"pair": pair_id, "reason": "duplicate seed"})
        seen[case].add(seed)
        counts[case] += 1
        if seed not in expected_seeds:
            incomplete.append({"pair": pair_id, "reason": "seed is not preregistered"})
        if not {"baseline", "candidate"}.issubset(pair):
            incomplete.append({"pair": pair_id, "reason": "missing baseline or candidate measurement"})
            continue
        baseline = pair["baseline"]
        candidate = pair["candidate"]
        base_science = baseline.get("scientific_record")
        candidate_science = candidate.get("scientific_record")
        base_run = baseline.get("run_record")
        candidate_run = candidate.get("run_record")
        if any(value is None for value in (base_science, candidate_science, base_run, candidate_run)):
            incomplete.append({"pair": pair_id, "reason": "missing scientific or source run record"})
            continue
        if benchmark.canonical_fingerprint(base_science) != benchmark.canonical_fingerprint(candidate_science):
            mismatches.append({"pair": pair_id, "source": "scientific_record"})
        if benchmark.scientific_fingerprint(base_run) != benchmark.scientific_fingerprint(candidate_run):
            mismatches.append({"pair": pair_id, "source": "run_record"})

    for case in cases:
        if counts[case] != expected_pairs_per_case:
            incomplete.append({
                "case": case,
                "reason": f"has {counts[case]} pairs; expected {expected_pairs_per_case}",
            })
        missing = sorted(expected_seeds - seen[case])
        if missing:
            incomplete.append({"case": case, "reason": f"missing seeds: {missing}"})

    comparison = manifest.get("comparison", {})
    all_pairs_exact = not mismatches and not incomplete
    return {
        "schema_version": 1,
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256(raw),
        "manifest_script_sha256_at_collection": manifest.get("script_sha256"),
        "audit_script_sha256": sha256(Path(__file__).read_bytes()),
        "fingerprint_script_sha256": sha256(Path(__file__).with_name("benchmark.py").read_bytes()),
        "fingerprint": "canonical sorted compact JSON, rejecting nonfinite values and preserving signed zero and JSON number types",
        "pairs_checked": len(paired),
        "expected_pairs_per_case": expected_pairs_per_case,
        "cases": counts,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "incomplete_count": len(incomplete),
        "incomplete": incomplete,
        "performance_gate_passed": comparison.get("all_pass") is True,
        "all_pairs_exact": all_pairs_exact,
        "audit_passed": all_pairs_exact and len(paired) == len(cases) * expected_pairs_per_case,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--pairs-per-case", type=int, default=30)
    args = parser.parse_args()
    result = audit_manifest(args.manifest, args.pairs_per_case)
    output = args.output or args.manifest.with_name("scientific-audit.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key not in ("mismatches", "incomplete")}, indent=2))
    if not result["audit_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
