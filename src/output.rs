//! Versioned run records and compact analysis exports.

use crate::evaluation::EvaluationReport;
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File, OpenOptions},
    io::{BufWriter, Write},
    path::Path,
};

/// Schema version of [`RunRecord`]; [`append_jsonl`] rejects other versions.
pub const RUN_RECORD_VERSION: u32 = 1;
/// Default semantics label written to [`RunRecord::semantic_version`].
pub const SEMANTIC_VERSION: &str = "mocs-rs-1";

/// Provenance and results of one run, written as one JSON line by [`append_jsonl`].
///
/// [`RunRecord::new`] fills the build metadata; callers set the optional
/// experiment fields (dataset, fold, split, budget and evaluation reports).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunRecord {
    /// Record schema version ([`RUN_RECORD_VERSION`]).
    pub schema_version: u32,
    /// Label of the algorithm/model semantics that produced the result.
    pub semantic_version: String,
    /// Random-stream contract label.
    pub rng_version: String,
    /// The resolved run configuration as JSON.
    pub config: serde_json::Value,
    /// Hex SHA-256 of each input file, in the order given.
    pub input_sha256: Vec<String>,
    /// Identifier of the dataset (e.g. its SHA-256), when known.
    #[serde(default)]
    pub dataset_id: Option<String>,
    /// Cross-validation fold label, for experiment runs.
    pub fold_id: Option<String>,
    /// Identifier of the train/test split, for experiment runs.
    #[serde(default)]
    pub split_id: Option<String>,
    /// Seed used to build generated folds.
    #[serde(default)]
    pub split_seed: Option<u64>,
    /// SHA-256 of the JSON-encoded training row indices of a generated fold.
    #[serde(default)]
    pub train_indices_sha256: Option<String>,
    /// SHA-256 of the JSON-encoded test row indices of a generated fold.
    #[serde(default)]
    pub test_indices_sha256: Option<String>,
    /// Run seed.
    pub seed: u64,
    /// Short git revision of the build, when available.
    pub revision: Option<String>,
    /// Hash of the packaged source files at build time, when available.
    #[serde(default)]
    pub source_tree_id: Option<String>,
    /// Whether the git working tree had changes at build time, when known.
    #[serde(default)]
    pub source_dirty: Option<bool>,
    /// `rustc --version --verbose` of the build, or `unknown`.
    pub rustc: String,
    /// Build target triple (or `arch-os` fallback).
    pub target: String,
    /// Cargo build profile, or `unknown`.
    pub build_profile: String,
    /// Evaluation threads used by the run.
    pub threads: usize,
    /// Threads used to run experiment folds concurrently, if any.
    #[serde(default)]
    pub outer_experiment_threads: Option<usize>,
    /// Wall-clock duration of the run in seconds.
    pub elapsed_seconds: f64,
    /// Problem evaluations performed.
    pub evaluations: usize,
    /// Generations completed.
    pub generations: usize,
    /// Configured evaluation budget (`max_evaluations`), if any.
    pub requested_evaluations: Option<usize>,
    /// Evaluations performed beyond `requested_evaluations` (the last generation is not truncated).
    #[serde(default)]
    pub budget_overshoot: usize,
    /// Objective vectors of the final front.
    pub front: Vec<Vec<f64>>,
    /// Training evaluation of the model chosen from the front (learning runs).
    pub training_selection: Option<EvaluationReport>,
    /// Held-out evaluation of the chosen model, when test data was scored.
    pub heldout: Option<EvaluationReport>,
}

impl RunRecord {
    /// Create a record, hashing `input_paths` and capturing build metadata. Optional
    /// experiment fields start empty. Errors if an input cannot be read or
    /// `elapsed_seconds` is negative or nonfinite.
    #[allow(clippy::too_many_arguments)] // Record creation has a fixed, explicit run metadata boundary.
    pub fn new(
        config: serde_json::Value,
        input_paths: &[impl AsRef<Path>],
        seed: u64,
        threads: usize,
        elapsed_seconds: f64,
        evaluations: usize,
        generations: usize,
        front: Vec<Vec<f64>>,
    ) -> Result<Self> {
        ensure!(
            elapsed_seconds.is_finite() && elapsed_seconds >= 0.0,
            "invalid elapsed time"
        );
        let hashes = input_paths
            .iter()
            .map(sha256_file)
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            schema_version: RUN_RECORD_VERSION,
            semantic_version: SEMANTIC_VERSION.into(),
            rng_version: "chacha8-sampler-1".into(),
            config,
            input_sha256: hashes,
            dataset_id: None,
            fold_id: None,
            split_id: None,
            split_seed: None,
            train_indices_sha256: None,
            test_indices_sha256: None,
            seed,
            revision: option_env!("MOCS_GIT_REVISION").map(str::to_owned),
            source_tree_id: option_env!("MOCS_SOURCE_TREE_ID").map(str::to_owned),
            source_dirty: option_env!("MOCS_SOURCE_DIRTY").map(|x| x == "true"),
            rustc: option_env!("MOCS_RUSTC_VERSION")
                .unwrap_or("unknown")
                .into(),
            target: option_env!("MOCS_BUILD_TARGET")
                .map(str::to_owned)
                .unwrap_or_else(|| format!("{}-{}", std::env::consts::ARCH, std::env::consts::OS)),
            build_profile: option_env!("MOCS_BUILD_PROFILE")
                .unwrap_or("unknown")
                .into(),
            threads,
            outer_experiment_threads: None,
            elapsed_seconds,
            evaluations,
            generations,
            requested_evaluations: None,
            budget_overshoot: 0,
            front,
            training_selection: None,
            heldout: None,
        })
    }
}

/// Lowercase hex SHA-256 of a file's contents. Errors if it cannot be read.
pub fn sha256_file(path: impl AsRef<Path>) -> Result<String> {
    let bytes = fs::read(path.as_ref())
        .with_context(|| format!("read input {}", path.as_ref().display()))?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

/// Lowercase hex SHA-256 of `bytes`.
pub fn sha256_bytes(bytes: impl AsRef<[u8]>) -> String {
    format!("{:x}", Sha256::digest(bytes.as_ref()))
}

/// Append one JSON object per line. A single record is serialized before opening the file,
/// so a serialization error never leaves a partial record behind.
pub fn append_jsonl(path: impl AsRef<Path>, record: &RunRecord) -> Result<()> {
    ensure!(
        record.schema_version == RUN_RECORD_VERSION,
        "unsupported run-record schema version {}",
        record.schema_version
    );
    let mut line = serde_json::to_vec(record)?;
    line.push(b'\n');
    let file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path.as_ref())
        .with_context(|| format!("open JSONL output {}", path.as_ref().display()))?;
    let mut writer = BufWriter::new(file);
    writer.write_all(&line)?;
    writer.flush()?;
    Ok(())
}

/// Summary statistics for one metric of one group of runs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SummaryRow {
    /// Group label (e.g. configuration name); empty from [`summarize`].
    pub group: String,
    /// Metric name; empty from [`summarize`].
    pub metric: String,
    /// Number of values summarized.
    pub runs: usize,
    /// Arithmetic mean.
    pub mean: f64,
    /// Sample standard deviation (`n - 1` denominator); 0 for a single value.
    pub stddev: f64,
    /// Minimum value.
    pub min: f64,
    /// Maximum value.
    pub max: f64,
}

/// Summarize a sample, leaving `group` and `metric` empty. Errors if the sample is
/// empty, contains nonfinite values or the statistics overflow.
pub fn summarize(values: &[f64]) -> Result<SummaryRow> {
    ensure!(!values.is_empty(), "cannot summarize an empty sample");
    ensure!(
        values.iter().all(|v| v.is_finite()),
        "summary has nonfinite values"
    );
    let scale = values.iter().map(|v| v.abs()).fold(0.0, f64::max);
    let mean = if scale == 0.0 {
        0.0
    } else {
        (values.iter().map(|v| v / scale).sum::<f64>() / values.len() as f64) * scale
    };
    let stddev = if values.len() < 2 {
        0.0
    } else {
        if scale == 0.0 {
            0.0
        } else {
            scale
                * (values
                    .iter()
                    .map(|v| (v / scale - mean / scale).powi(2))
                    .sum::<f64>()
                    / (values.len() - 1) as f64)
                    .sqrt()
        }
    };
    ensure!(
        mean.is_finite() && stddev.is_finite(),
        "summary statistics overflowed"
    );
    Ok(SummaryRow {
        group: String::new(),
        metric: String::new(),
        runs: values.len(),
        mean,
        stddev,
        min: values.iter().copied().fold(f64::INFINITY, f64::min),
        max: values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
    })
}

/// Write summary rows as CSV (`group,metric,runs,mean,stddev,min,max`), quoting
/// the labels and printing floats with 17 decimals. Overwrites `path`.
pub fn write_summary_csv(path: impl AsRef<Path>, rows: &[SummaryRow]) -> Result<()> {
    let file = File::create(path.as_ref())
        .with_context(|| format!("create CSV {}", path.as_ref().display()))?;
    let mut w = BufWriter::new(file);
    writeln!(w, "group,metric,runs,mean,stddev,min,max")?;
    for r in rows {
        writeln!(
            w,
            "\"{}\",\"{}\",{},{:.17},{:.17},{:.17},{:.17}",
            r.group.replace('"', "\"\""),
            r.metric.replace('"', "\"\""),
            r.runs,
            r.mean,
            r.stddev,
            r.min,
            r.max
        )?;
    }
    w.flush()?;
    Ok(())
}

/// Write fronts as CSV with columns `run,index,objective_1..`, one row per point
/// (`run` and `index` are zero-based). Errors if points have inconsistent or
/// empty dimensions or nonfinite values. Overwrites `path`.
pub fn write_front_csv(path: impl AsRef<Path>, fronts: &[Vec<Vec<f64>>]) -> Result<()> {
    let dim = fronts.iter().flatten().next().map(Vec::len).unwrap_or(0);
    ensure!(
        dim > 0 || fronts.iter().all(Vec::is_empty),
        "empty objective vector"
    );
    let file = File::create(path.as_ref())
        .with_context(|| format!("create front CSV {}", path.as_ref().display()))?;
    let mut w = BufWriter::new(file);
    write!(w, "run,index")?;
    for i in 0..dim {
        write!(w, ",objective_{}", i + 1)?;
    }
    writeln!(w)?;
    for (ri, front) in fronts.iter().enumerate() {
        for (pi, p) in front.iter().enumerate() {
            ensure!(
                p.len() == dim && p.iter().all(|x| x.is_finite()),
                "front dimensions or values are invalid"
            );
            write!(w, "{},{}", ri, pi)?;
            for x in p {
                write!(w, ",{x:.17}")?;
            }
            writeln!(w)?;
        }
    }
    w.flush()?;
    Ok(())
}
