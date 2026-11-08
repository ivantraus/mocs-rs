use std::{
    fs,
    path::PathBuf,
    process::Command,
    sync::atomic::{AtomicU64, Ordering},
};

static NEXT: AtomicU64 = AtomicU64::new(0);
fn scratch() -> PathBuf {
    let p = std::env::temp_dir().join(format!(
        "mocs-cli-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir_all(&p).unwrap();
    p
}
fn mocs() -> Command {
    Command::new(env!("CARGO_BIN_EXE_mocs"))
}

fn ordinary_run_record(algorithm: &str, input_hashes: &[&str]) -> serde_json::Value {
    serde_json::json!({
        "schema_version": 1,
        "semantic_version": "mocs-rs-1",
        "rng_version": "chacha8-sampler-1",
        "config": {"algorithm": {"kind": algorithm}},
        "input_sha256": input_hashes,
        "dataset_id": "shared-dataset",
        "fold_id": null,
        "split_id": null,
        "split_seed": null,
        "train_indices_sha256": null,
        "test_indices_sha256": null,
        "seed": 42,
        "revision": null,
        "source_tree_id": null,
        "source_dirty": null,
        "rustc": "test",
        "target": "test",
        "build_profile": "test",
        "threads": 1,
        "outer_experiment_threads": null,
        "elapsed_seconds": 0.1,
        "evaluations": 1,
        "generations": 1,
        "requested_evaluations": null,
        "budget_overshoot": 0,
        "front": [[1.0, 1.0]],
        "training_selection": null,
        "heldout": null
    })
}

fn write_records(path: &std::path::Path, records: &[serde_json::Value]) {
    let content = records
        .iter()
        .map(serde_json::Value::to_string)
        .collect::<Vec<_>>()
        .join("\n");
    fs::write(path, content + "\n").unwrap();
}

#[test]
fn numeric_run_metrics_and_analysis_cli_work_end_to_end() {
    let d = scratch();
    let runs = d.join("runs.jsonl");
    let summary = d.join("analysis");
    let status = mocs()
        .args([
            "optimize",
            "--seed",
            "0",
            "--population",
            "4",
            "--offspring",
            "2",
            "--generations",
            "2",
            "--output",
        ])
        .arg(&runs)
        .status()
        .unwrap();
    assert!(status.success());
    let line = fs::read_to_string(&runs).unwrap();
    let record: serde_json::Value = serde_json::from_str(line.trim()).unwrap();
    assert_eq!(record["schema_version"], 1);
    assert_eq!(record["semantic_version"], "mocs-rs-1");
    assert_eq!(record["seed"], 0);
    assert!(record["evaluations"].as_u64().unwrap() >= 4);
    assert!(
        mocs()
            .args(["metrics", "--input"])
            .arg(&runs)
            .args(["--reference", "1,1"])
            .status()
            .unwrap()
            .success()
    );
    assert!(
        mocs()
            .args(["analyze", "--input"])
            .arg(&runs)
            .args(["--output-dir"])
            .arg(&summary)
            .args(["--reference", "1,1"])
            .status()
            .unwrap()
            .success()
    );
    assert!(summary.join("summary.csv").exists());
    assert!(summary.join("fronts.csv").exists());
    fs::remove_dir_all(d).unwrap();
}

#[test]
fn cv_repetition_analysis_reports_missing_fold_ids_without_panicking() {
    let d = scratch();
    let input = d.join("ordinary-runs.jsonl");
    write_records(
        &input,
        &[
            ordinary_run_record("nsga2", &["training-hash", "heldout-hash"]),
            ordinary_run_record("moga", &["training-hash", "heldout-hash"]),
        ],
    );
    let result = mocs()
        .args(["analyze", "--input"])
        .arg(&input)
        .args(["--output-dir"])
        .arg(d.join("analysis"))
        .args([
            "--compare",
            "nsga2:moga",
            "--unit",
            "cv_repetition",
            "--folds",
            "1",
            "--metric",
            "front_size",
        ])
        .output()
        .unwrap();
    let diagnostic = String::from_utf8_lossy(&result.stderr);
    assert!(!result.status.success());
    assert!(
        diagnostic.contains("requires fold IDs on both sides"),
        "{diagnostic}"
    );
    assert!(!diagnostic.contains("panicked"), "{diagnostic}");
    fs::remove_dir_all(d).unwrap();
}

#[test]
fn seeded_run_analysis_rejects_different_input_hashes() {
    let d = scratch();
    let input = d.join("mismatched-input-runs.jsonl");
    write_records(
        &input,
        &[
            ordinary_run_record("nsga2", &["training-hash", "heldout-hash-a"]),
            ordinary_run_record("moga", &["training-hash", "heldout-hash-b"]),
        ],
    );
    let result = mocs()
        .args(["analyze", "--input"])
        .arg(&input)
        .args(["--output-dir"])
        .arg(d.join("analysis"))
        .args([
            "--compare",
            "nsga2:moga",
            "--unit",
            "seeded_run",
            "--metric",
            "front_size",
        ])
        .output()
        .unwrap();
    let diagnostic = String::from_utf8_lossy(&result.stderr);
    assert!(!result.status.success());
    assert!(
        diagnostic.contains("input hashes do not match"),
        "{diagnostic}"
    );
    fs::remove_dir_all(d).unwrap();
}

#[test]
fn train_saves_a_model_and_predict_accepts_targetless_arff() {
    let d = scratch();
    let train = d.join("train.arff");
    let predict = d.join("predict.arff");
    let model = d.join("model.json");
    let output = d.join("prediction.json");
    let run_output = d.join("training.jsonl");
    fs::write(&train,"@relation tiny\n@attribute x numeric\n@attribute class {a,b}\n@data\n0,a\n0.2,a\n0.8,b\n1,b\n").unwrap();
    fs::write(
        &predict,
        "@relation targetless\n@attribute x numeric\n@data\n0.1\n0.9\n",
    )
    .unwrap();
    let status = mocs()
        .args(["train", "--data"])
        .arg(&train)
        .args(["--model-out"])
        .arg(&model)
        .args(["--output"])
        .arg(&run_output)
        .args([
            "--seed",
            "0",
            "--population",
            "6",
            "--offspring",
            "3",
            "--generations",
            "2",
            "--config",
        ])
        .arg(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/classification.toml"))
        .status()
        .unwrap();
    assert!(status.success());
    assert!(model.exists());
    let run: serde_json::Value =
        serde_json::from_str(fs::read_to_string(&run_output).unwrap().trim()).unwrap();
    assert_eq!(run["semantic_version"], "mocs-rs-3");
    let saved: serde_json::Value = serde_json::from_slice(&fs::read(&model).unwrap()).unwrap();
    assert_eq!(saved["semantics_version"], 1);
    let status = mocs()
        .args(["predict", "--model"])
        .arg(&model)
        .args(["--data"])
        .arg(&predict)
        .args(["--output"])
        .arg(&output)
        .status()
        .unwrap();
    assert!(status.success());
    let rows: serde_json::Value = serde_json::from_slice(&fs::read(&output).unwrap()).unwrap();
    assert_eq!(rows.as_array().unwrap().len(), 2);
    fs::remove_dir_all(d).unwrap();
}

#[test]
fn analyze_writes_paired_test_results_for_declared_seed_units() {
    let d = scratch();
    let nsga_cfg = d.join("nsga.toml");
    let moga_cfg = d.join("moga.toml");
    let template =
        fs::read_to_string(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/sf2.toml"))
            .unwrap();
    fs::write(&nsga_cfg, &template).unwrap();
    fs::write(
        &moga_cfg,
        template.replace("kind = \"nsga2\"", "kind = \"moga\""),
    )
    .unwrap();
    let combined = d.join("runs.jsonl");
    for (cfg, _) in [(&nsga_cfg, "nsga2"), (&moga_cfg, "moga")] {
        for seed in 0..3 {
            assert!(
                mocs()
                    .args(["optimize", "--config"])
                    .arg(cfg)
                    .args(["--seed"])
                    .arg(seed.to_string())
                    .args([
                        "--population",
                        "6",
                        "--offspring",
                        "6",
                        "--generations",
                        "2",
                        "--output"
                    ])
                    .arg(&combined)
                    .status()
                    .unwrap()
                    .success()
            );
        }
    }
    let dest = d.join("analysis");
    assert!(
        mocs()
            .args(["analyze", "--input"])
            .arg(&combined)
            .args(["--output-dir"])
            .arg(&dest)
            .args([
                "--compare",
                "nsga2:moga",
                "--unit",
                "seeded_run",
                "--metric",
                "front_size"
            ])
            .status()
            .unwrap()
            .success()
    );
    let csv = fs::read_to_string(dest.join("comparisons.csv")).unwrap();
    assert!(csv.contains("nsga2:moga"));
    assert!(csv.contains("t_holm"));
    assert!(csv.lines().count() == 2);
    fs::remove_dir_all(d).unwrap();
}

#[test]
fn cv_analysis_aggregates_complete_folds_and_rejects_missing_fold_pairs() {
    let d = scratch();
    let data = d.join("cv.arff");
    fs::write(
        &data,
        "@relation cv\n@attribute x numeric\n@attribute class {a,b}\n@data\n0,a\n0.1,a\n0.2,a\n0.3,a\n0.7,b\n0.8,b\n0.9,b\n1.0,b\n",
    )
    .unwrap();
    let template = fs::read_to_string(
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/classification.toml"),
    )
    .unwrap();
    let nsga = d.join("nsga.toml");
    let moga = d.join("moga.toml");
    fs::write(&nsga, &template).unwrap();
    fs::write(
        &moga,
        template.replace("kind = \"nsga2\"", "kind = \"moga\""),
    )
    .unwrap();
    let all = d.join("all.jsonl");
    for cfg in [&nsga, &moga] {
        assert!(
            mocs()
                .args(["experiment", "--config"])
                .arg(cfg)
                .args(["--data"])
                .arg(&data)
                .args([
                    "--folds",
                    "2",
                    "--split-seed",
                    "71",
                    "--seeds",
                    "0,1",
                    "--output"
                ])
                .arg(&all)
                .status()
                .unwrap()
                .success()
        );
    }
    let analysis = d.join("analysis");
    let status = mocs()
        .args(["analyze", "--input"])
        .arg(&all)
        .args(["--output-dir"])
        .arg(&analysis)
        .args([
            "--compare",
            "nsga2:moga",
            "--unit",
            "cv_repetition",
            "--folds",
            "2",
        ])
        .status()
        .unwrap();
    assert!(status.success());
    let csv = fs::read_to_string(analysis.join("comparisons.csv")).unwrap();
    assert!(csv.contains("heldout_loss"));
    assert!(csv.lines().count() == 2);

    let dataset_analysis = d.join("dataset-analysis");
    let status = mocs()
        .args(["analyze", "--input"])
        .arg(&all)
        .args(["--output-dir"])
        .arg(&dataset_analysis)
        .args([
            "--compare",
            "nsga2:moga",
            "--unit",
            "dataset",
            "--folds",
            "2",
        ])
        .status()
        .unwrap();
    assert!(status.success());
    let csv = fs::read_to_string(dataset_analysis.join("comparisons.csv")).unwrap();
    assert!(csv.lines().count() == 2);

    let lines = fs::read_to_string(&all)
        .unwrap()
        .lines()
        .map(str::to_owned)
        .collect::<Vec<_>>();
    let missing_fold = d.join("missing.jsonl");
    let mut removed_moga_fold = false;
    let mut selected = Vec::new();
    for line in lines {
        let row: serde_json::Value = serde_json::from_str(&line).unwrap();
        let alg = row["config"]["algorithm"]["kind"].as_str().unwrap();
        if alg == "moga" && !removed_moga_fold {
            removed_moga_fold = true;
            continue;
        }
        selected.push(line);
    }
    fs::write(&missing_fold, selected.join("\n") + "\n").unwrap();
    let status = mocs()
        .args(["analyze", "--input"])
        .arg(&missing_fold)
        .args(["--output-dir"])
        .arg(d.join("incomplete"))
        .args([
            "--compare",
            "nsga2:moga",
            "--unit",
            "cv_repetition",
            "--folds",
            "2",
        ])
        .status()
        .unwrap();
    assert!(!status.success());

    let missing_both = d.join("missing-both.jsonl");
    let mut selected = Vec::new();
    for line in fs::read_to_string(&all).unwrap().lines() {
        let row: serde_json::Value = serde_json::from_str(line).unwrap();
        if row["fold_id"] == "1" {
            continue;
        }
        selected.push(line.to_owned());
    }
    fs::write(&missing_both, selected.join("\n") + "\n").unwrap();
    let status = mocs()
        .args(["analyze", "--input"])
        .arg(&missing_both)
        .args(["--output-dir"])
        .arg(d.join("incomplete-both"))
        .args([
            "--compare",
            "nsga2:moga",
            "--unit",
            "dataset",
            "--folds",
            "2",
        ])
        .status()
        .unwrap();
    assert!(!status.success());
    fs::remove_dir_all(d).unwrap();
}

#[test]
fn numeric_experiment_runs_seeded_repetitions_and_records_budget_overshoot() {
    let d = scratch();
    let runs = d.join("numeric-repetitions.jsonl");
    let status = mocs()
        .args(["experiment", "--config"])
        .arg(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/sf2.toml"))
        .args([
            "--seeds",
            "0,1",
            "--population",
            "4",
            "--offspring",
            "4",
            "--generations",
            "2",
            "--max-evaluations",
            "8",
            "--output",
        ])
        .arg(&runs)
        .status()
        .unwrap();
    assert!(status.success());
    let rows = fs::read_to_string(&runs)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str::<serde_json::Value>(line).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0]["seed"], 0);
    assert_eq!(rows[1]["seed"], 1);
    for row in &rows {
        assert_eq!(row["semantic_version"], "mocs-rs-1");
        assert!(row["heldout"].is_null());
        assert_eq!(row["requested_evaluations"], 8);
        let expected = row["evaluations"].as_u64().unwrap().saturating_sub(8);
        assert_eq!(row["budget_overshoot"], expected);
    }
    let invalid = mocs()
        .args(["experiment", "--folds", "2"])
        .output()
        .unwrap();
    assert!(!invalid.status.success());
    fs::remove_dir_all(d).unwrap();
}

#[cfg(feature = "parallel")]
#[test]
fn numeric_experiment_parallel_runs_match_serial_science_and_order() {
    let d = scratch();
    let config = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/sf2.toml");
    let mut outputs = Vec::new();
    for threads in [1, 2] {
        let path = d.join(format!("threads-{threads}.jsonl"));
        let status = mocs()
            .args(["experiment", "--config"])
            .arg(&config)
            .args([
                "--seeds",
                "3,0,2",
                "--population",
                "4",
                "--offspring",
                "4",
                "--generations",
                "2",
                "--max-evaluations",
                "12",
                "--threads",
            ])
            .arg(threads.to_string())
            .args(["--output"])
            .arg(&path)
            .status()
            .unwrap();
        assert!(status.success());
        let rows = fs::read_to_string(path)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str::<serde_json::Value>(line).unwrap())
            .collect::<Vec<_>>();
        outputs.push(rows);
    }
    assert_eq!(outputs[0].len(), 3);
    assert_eq!(outputs[1].len(), 3);
    for (serial, parallel) in outputs[0].iter().zip(&outputs[1]) {
        for field in [
            "seed",
            "front",
            "evaluations",
            "generations",
            "budget_overshoot",
        ] {
            assert_eq!(serial[field], parallel[field], "scientific field {field}");
        }
    }
    assert_eq!(outputs[1][0]["seed"], 3);
    assert_eq!(outputs[1][1]["seed"], 0);
    assert_eq!(outputs[1][2]["seed"], 2);
    assert_eq!(outputs[1][0]["outer_experiment_threads"], 2);
    fs::remove_dir_all(d).unwrap();
}
