#!/usr/bin/env python3
"""Compare paired release-run JSONL timings with bootstrap confidence intervals."""
import argparse, hashlib, json, math, pathlib, random, statistics


def canonical_fingerprint(value):
    """Canonical JSON identity that distinguishes int/float and signed zero."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)

def read(path, metric):
    rows=[]
    for line_no,line in enumerate(pathlib.Path(path).read_text().splitlines(),1):
        if not line.strip(): continue
        r=json.loads(line)
        if r.get("schema_version")!=1: raise ValueError(f"{path}:{line_no}: unsupported record schema")
        timing_fields={"internal": "elapsed_seconds", "process_wall": "process_wall_seconds", "child_cpu": "child_cpu_seconds"}
        field=timing_fields[metric]
        if not math.isfinite(r.get(field,float("nan"))) or r[field]<=0: raise ValueError(f"{path}:{line_no}: invalid {field}")
        c=r["config"]
        signature=r.get("benchmark_config_signature")
        if signature is None:
            normalized=json.dumps(c,sort_keys=True,separators=(",",":"),allow_nan=False)
            signature=hashlib.sha256(normalized.encode()).hexdigest()
        key=(r.get("benchmark_case"),signature,c.get("problem",{}).get("kind"),c.get("algorithm",{}).get("kind"),r.get("seed"),tuple(r.get("input_sha256",[])))
        rows.append((key,r))
    return rows

def scientific_record(r):
    """Identity and outputs that must match before comparing elapsed time."""
    config = json.loads(json.dumps(r.get("config")))
    if isinstance(config, dict):
        config.pop("threads", None)
    return {
        "config": config,
        "seed": r.get("seed"),
        "input_sha256": r.get("input_sha256"),
        "dataset_id": r.get("dataset_id"),
        "fold_id": r.get("fold_id"),
        "split_id": r.get("split_id"),
        "split_seed": r.get("split_seed"),
        "train_indices_sha256": r.get("train_indices_sha256"),
        "test_indices_sha256": r.get("test_indices_sha256"),
        "evaluations": r.get("evaluations"),
        "generations": r.get("generations"),
        "front": r.get("front"),
        "training_selection": r.get("training_selection"),
        "heldout": r.get("heldout"),
    }


def scientific_fingerprint(record):
    return canonical_fingerprint(scientific_record(record))

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--baseline",required=True);p.add_argument("--candidate",required=True);p.add_argument("--output",required=True)
    p.add_argument("--bootstrap-samples",type=int,default=20000);p.add_argument("--seed",type=int,default=17);p.add_argument("--primary",required=True,help="primary benchmark_case, or problem:algorithm for legacy records")
    p.add_argument("--metric",choices=["internal","process_wall","child_cpu"],default="internal")
    p.add_argument("--minimum-improvement",type=float,default=.05);p.add_argument("--maximum-regression",type=float,default=.05)
    a=p.parse_args()
    if a.bootstrap_samples < 1: raise ValueError("--bootstrap-samples must be positive")
    base={};candidate={}
    metric_field={"internal": "elapsed_seconds", "process_wall": "process_wall_seconds", "child_cpu": "child_cpu_seconds"}[a.metric]
    for key,row in read(a.baseline,a.metric):
        if key in base: raise ValueError(f"duplicate baseline pair {key}")
        base[key]=row
    for key,row in read(a.candidate,a.metric):
        if key in candidate: raise ValueError(f"duplicate candidate pair {key}")
        candidate[key]=row
    if base.keys()!=candidate.keys(): raise ValueError("baseline and candidate run pairs differ")
    for key in sorted(base):
        if scientific_fingerprint(base[key]) != scientific_fingerprint(candidate[key]):
            raise ValueError(f"scientific workload or output differs for pair {key}; timing comparison is invalid")
    groups={}
    for key in sorted(base):
        identity=(key[0],key[1],key[2],key[3]);speedup=1.0-candidate[key][metric_field]/base[key][metric_field]
        groups.setdefault(identity,[]).append(speedup)
    rng=random.Random(a.seed);cases=[]
    for case,values in sorted(groups.items()):
        boot=[]
        for _ in range(a.bootstrap_samples):boot.append(statistics.median(rng.choices(values,k=len(values))))
        boot.sort();lo=boot[int(.025*(len(boot)-1))];hi=boot[int(.975*(len(boot)-1))];med=statistics.median(values)
        case_name,signature,problem,algorithm=case
        primary=(case_name==a.primary) if case_name is not None else (problem+":"+algorithm)==a.primary
        passed=(med>=a.minimum_improvement and lo>0) if primary else med>=-a.maximum_regression
        cases.append({"benchmark_case":case_name,"config_signature":signature,"problem":problem,"algorithm":algorithm,"primary":primary,"pairs":len(values),"median_improvement":med,"ci95":[lo,hi],"pass":passed})
    if not any(x["primary"] for x in cases): raise ValueError(f"primary benchmark {a.primary!r} is absent")
    result={"criteria":{"primary":a.primary,"metric":a.metric,"minimum_primary_improvement":a.minimum_improvement,"maximum_other_regression":a.maximum_regression,"primary_ci_lower_must_exceed":0},"workload_identity":"benchmark case, normalized full config signature, and scientific records must match exactly","cases":cases,"all_pass":bool(cases) and all(x["pass"] for x in cases)}
    pathlib.Path(a.output).write_text(json.dumps(result,indent=2,allow_nan=False)+"\n");print(json.dumps(result,indent=2))
    raise SystemExit(0 if result["all_pass"] else 1)
if __name__=="__main__":main()
