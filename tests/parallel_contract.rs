#![cfg(feature = "parallel")]

use serde_json::Value;
use std::{fs, path::Path, process::Command};

fn run(output: &Path, data: &Path, threads: usize) {
    let result = Command::new(env!("CARGO_BIN_EXE_mocs"))
        .args([
            "experiment",
            "--data",
            data.to_str().unwrap(),
            "--folds",
            "2",
            "--seeds",
            "3,7",
            "--split-seed",
            "91",
            "--population",
            "8",
            "--offspring",
            "4",
            "--generations",
            "3",
            "--threads",
            &threads.to_string(),
            "--output",
            output.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
}

fn records(path: &Path) -> Vec<Value> {
    fs::read_to_string(path)
        .unwrap()
        .lines()
        .map(|line| {
            let mut value: Value = serde_json::from_str(line).unwrap();
            let object = value.as_object_mut().unwrap();
            object.remove("elapsed_seconds");
            object.remove("threads");
            object.remove("outer_experiment_threads");
            if let Some(config) = object.get_mut("config").and_then(Value::as_object_mut) {
                config.remove("threads");
            }
            value
        })
        .collect()
}

#[test]
fn serial_and_parallel_experiment_jobs_have_identical_scientific_records() {
    let temp = std::env::temp_dir().join(format!("mocs-parallel-{}", std::process::id()));
    fs::create_dir_all(&temp).unwrap();
    let data = temp.join("tiny.arff");
    fs::write(
        &data,
        "@relation tiny\n@attribute x numeric\n@attribute class {a,b}\n@data\n0,a\n1,a\n2,a\n3,a\n4,b\n5,b\n6,b\n7,b\n",
    )
    .unwrap();
    let serial = temp.join("serial.jsonl");
    let parallel = temp.join("parallel.jsonl");
    run(&serial, &data, 1);
    run(&parallel, &data, 2);
    assert_eq!(records(&serial), records(&parallel));
    fs::remove_dir_all(temp).unwrap();
}
