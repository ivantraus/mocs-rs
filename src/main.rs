//! The `mocs` command-line program: optimize, train, predict, run experiments and
//! analyze results using the `mocs_rs` library. See `README.md` for usage.

#![forbid(unsafe_code)]

use anyhow::{Context, Result, ensure};
use clap::{Parser, Subcommand};
use mocs_rs::{
    algorithms,
    config::RunConfig,
    data::Dataset,
    execution::ExecutionContext,
    experiment, metrics,
    models::{LearningProblem, SavedModel, learning_run_semantics_version},
    output::{self, RunRecord, SummaryRow},
    problems::NumericProblem,
    stats::{self, ComparisonUnit, Observation},
};
use std::{path::PathBuf, time::Instant};

#[derive(Parser)]
#[command(
    name = "mocs",
    version,
    about = "Multi-objective evolutionary optimization and learning"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Optimize a built-in numeric problem.
    Optimize(RunArgs),
    /// Train a learning model and save the training-selected nondominated solution.
    Train(TrainArgs),
    /// Predict rows with a saved model.
    Predict(PredictArgs),
    /// Run explicit seeded repetitions over existing folds or reproducible generated folds.
    Experiment(ExperimentArgs),
    /// Compute front metrics from versioned JSONL run records.
    Metrics(MetricsArgs),
    /// Export grouped CSV summaries and plot-ready front data from JSONL records.
    Analyze(AnalyzeArgs),
}

#[derive(clap::Args, Clone)]
struct ConfigArgs {
    /// Native TOML config files, merged in order.
    #[arg(long = "config", value_name = "FILE")]
    config: Vec<PathBuf>,
    #[arg(long)]
    seed: Option<u64>,
    #[arg(long)]
    population: Option<usize>,
    #[arg(long)]
    offspring: Option<usize>,
    #[arg(long)]
    generations: Option<usize>,
    #[arg(long = "max-evaluations")]
    max_evaluations: Option<usize>,
    #[arg(long)]
    threads: Option<usize>,
}
#[derive(clap::Args)]
struct RunArgs {
    #[command(flatten)]
    cfg: ConfigArgs,
    #[arg(long, default_value = "runs.jsonl")]
    output: PathBuf,
    #[arg(long)]
    front_csv: Option<PathBuf>,
}
#[derive(clap::Args)]
struct TrainArgs {
    #[command(flatten)]
    cfg: ConfigArgs,
    #[arg(long)]
    data: PathBuf,
    #[arg(long)]
    test: Option<PathBuf>,
    #[arg(long)]
    model_out: PathBuf,
    #[arg(long, default_value = "runs.jsonl")]
    output: PathBuf,
}
#[derive(clap::Args)]
struct PredictArgs {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    data: PathBuf,
    #[arg(long)]
    output: Option<PathBuf>,
}
#[derive(clap::Args)]
struct ExperimentArgs {
    #[command(flatten)]
    cfg: ConfigArgs,
    #[arg(long)]
    data: Option<PathBuf>,
    /// Number of generated folds; only valid with --data.
    #[arg(long)]
    folds: Option<usize>,
    /// Existing fold path with `{fold}` replaced by the fold number (0 through 9 by default).
    #[arg(long)]
    train_pattern: Option<String>,
    #[arg(long)]
    test_pattern: Option<String>,
    /// First fold number substituted into existing-fold path patterns.
    #[arg(long)]
    fold_start: Option<usize>,
    /// Comma-separated explicit seeds, including zero if requested.
    #[arg(long, default_value = "0")]
    seeds: String,
    /// Seed used to generate row splits; independent of the optimization seeds.
    #[arg(long)]
    split_seed: Option<u64>,
    #[arg(long, default_value = "runs.jsonl")]
    output: PathBuf,
}
#[derive(clap::Args)]
struct MetricsArgs {
    #[arg(long)]
    input: PathBuf,
    #[arg(long, required = true)]
    reference: String,
    #[arg(long)]
    output: Option<PathBuf>,
}
#[derive(clap::Args)]
struct AnalyzeArgs {
    #[arg(long)]
    input: PathBuf,
    #[arg(long)]
    output_dir: PathBuf,
    #[arg(long)]
    reference: Option<String>,
    /// Summary metric; defaults to hypervolume with --reference, or heldout_loss when available.
    #[arg(long)]
    metric: Option<String>,
    /// Repeatable algorithm contrasts, for example `nsga2:moga`.
    #[arg(long = "compare")]
    comparisons: Vec<String>,
    /// Pairing unit: seeded_run, cv_repetition, or dataset.
    #[arg(long, default_value = "seeded_run")]
    unit: String,
    /// Required complete fold count when comparing CV repetitions.
    #[arg(long)]
    folds: Option<usize>,
}

fn config(a: &ConfigArgs) -> Result<RunConfig> {
    let mut c = RunConfig::load_unvalidated(&a.config)?;
    if let Some(v) = a.seed {
        c.seed = v
    }
    if let Some(v) = a.population {
        c.population = v
    }
    if let Some(v) = a.offspring {
        c.offspring = v
    }
    if let Some(v) = a.generations {
        c.generations = v
    }
    if let Some(v) = a.max_evaluations {
        c.max_evaluations = Some(v)
    }
    if let Some(v) = a.threads {
        c.threads = v
    }
    c.validate()?;
    Ok(c)
}
fn run_record(
    cfg: &RunConfig,
    paths: &[PathBuf],
    started: Instant,
    result: &algorithms::RunResult,
) -> Result<RunRecord> {
    let value = serde_json::to_value(cfg.resolved())?;
    let mut record = RunRecord::new(
        value,
        paths,
        cfg.seed,
        cfg.threads,
        started.elapsed().as_secs_f64(),
        result.evaluations,
        result.generations,
        result
            .front
            .iter()
            .map(|x| {
                x.evaluation()
                    .expect("run front is evaluated")
                    .objectives
                    .clone()
            })
            .collect(),
    )?;
    record.dataset_id = record.input_sha256.first().cloned().or_else(|| {
        serde_json::to_vec(&cfg.problem)
            .ok()
            .map(output::sha256_bytes)
    });
    record.requested_evaluations = cfg.max_evaluations;
    record.budget_overshoot = cfg
        .max_evaluations
        .map_or(0, |requested| result.evaluations.saturating_sub(requested));
    Ok(record)
}
fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Optimize(a) => {
            let c = config(&a.cfg)?;
            let problem = NumericProblem::new(&c.problem)?;
            problem.validate_operators(&c.operators)?;
            let start = Instant::now();
            let result = algorithms::run(&problem, &c)?;
            let record = run_record(&c, &[], start, &result)?;
            output::append_jsonl(&a.output, &record)?;
            if let Some(path) = a.front_csv {
                let fronts = vec![record.front.clone()];
                output::write_front_csv(path, &fronts)?;
            }
            eprintln!(
                "completed {} generations, {} evaluations, {} nondominated solutions",
                result.generations,
                result.evaluations,
                result.front.len()
            );
        }
        Command::Train(a) => {
            let c = config(&a.cfg)?;
            c.operators.validate_for_training(&c.training)?;
            let data = Dataset::load(&a.data, c.training.target.as_deref())?;
            let problem = LearningProblem::new(data, c.training.clone(), c.operators.clone())?;
            let start = Instant::now();
            let result = algorithms::run(&problem, &c)?;
            let selected = problem.choose(&result.front)?;
            problem.save_model(selected.genome(), &a.model_out)?;
            let heldout = if let Some(test) = &a.test {
                Some(problem.score_file(selected.genome(), test)?)
            } else {
                None
            };
            let paths = std::iter::once(a.data.clone())
                .chain(a.test.clone())
                .collect::<Vec<_>>();
            let mut record = run_record(&c, &paths, start, &result)?;
            record.semantic_version = format!(
                "mocs-rs-{}",
                learning_run_semantics_version(c.training.model)
            );
            record.training_selection = selected.evaluation().cloned();
            record.heldout = heldout;
            output::append_jsonl(&a.output, &record)?;
            eprintln!(
                "saved selected model to {} (training loss {:.6})",
                a.model_out.display(),
                selected.evaluation().unwrap().loss
            );
        }
        Command::Predict(a) => {
            let model = SavedModel::load(&a.model)?;
            let rows = model.predict_file(&a.data)?;
            if let Some(p) = a.output {
                std::fs::write(&p, serde_json::to_vec_pretty(&rows)?)
                    .with_context(|| format!("write predictions {}", p.display()))?;
            } else {
                println!("{}", serde_json::to_string_pretty(&rows)?);
            }
        }
        Command::Experiment(a) => run_experiment(a)?,
        Command::Metrics(a) => run_metrics(a)?,
        Command::Analyze(a) => run_analyze(a)?,
    }
    Ok(())
}

fn run_experiment(a: ExperimentArgs) -> Result<()> {
    let base = config(&a.cfg)?;
    let seeds = parse_seeds(&a.seeds)?;
    if a.data.is_some() || a.train_pattern.is_some() {
        base.operators.validate_for_training(&base.training)?;
    }
    let mut jobs = Vec::<ExperimentJob>::new();
    match (&a.data, &a.train_pattern, &a.test_pattern) {
        (Some(path), None, None) => {
            ensure!(
                a.fold_start.is_none(),
                "--fold-start is only valid with existing-fold patterns"
            );
            let folds = a.folds.context("--folds is required with --data")?;
            let full = Dataset::load(path, base.training.target.as_deref())?;
            let split_seed = a.split_seed.unwrap_or(base.seed);
            let splits = if full.is_classification() {
                experiment::stratified_folds(&full, folds, split_seed)?
            } else {
                experiment::shuffled_folds(&full, folds, split_seed)?
            };
            let dataset_id = output::sha256_file(path)?;
            for fold in splits {
                for &seed in &seeds {
                    let train = full.subset_rows(&fold.train_indices)?;
                    let heldout = full.subset_rows_for_prediction(&fold.test_indices)?;
                    jobs.push(ExperimentJob {
                        train: Some(train),
                        heldout: Some(heldout),
                        paths: vec![path.clone()],
                        seed,
                        dataset_id: dataset_id.clone(),
                        fold_id: Some((fold.id + 1).to_string()),
                        split_id: Some(format!("cv:{split_seed}:{}", fold.id)),
                        split_seed: Some(split_seed),
                        train_indices_sha256: Some(output::sha256_bytes(serde_json::to_vec(
                            &fold.train_indices,
                        )?)),
                        test_indices_sha256: Some(output::sha256_bytes(serde_json::to_vec(
                            &fold.test_indices,
                        )?)),
                    });
                }
            }
        }
        (None, Some(train_pattern), Some(test_pattern)) => {
            ensure!(
                a.split_seed.is_none(),
                "--split-seed is only valid with generated folds from --data"
            );
            ensure!(
                a.folds.is_some(),
                "--folds is required with existing-fold patterns"
            );
            let n = a.folds.unwrap();
            ensure!(n >= 2, "fold count must be at least 2");
            let fold_start = a.fold_start.unwrap_or(0);
            let manifest = serde_json::json!({"train_pattern":train_pattern,"test_pattern":test_pattern,"fold_start":fold_start,"folds":n});
            let dataset_id = output::sha256_bytes(serde_json::to_vec(&manifest)?);
            for offset in 0..n {
                let fold = fold_start + offset;
                let train_path = PathBuf::from(train_pattern.replace("{fold}", &fold.to_string()));
                let test_path = PathBuf::from(test_pattern.replace("{fold}", &fold.to_string()));
                let train = Dataset::load(&train_path, base.training.target.as_deref())?;
                let test_input = std::fs::read_to_string(&test_path)
                    .with_context(|| format!("read test fold {}", test_path.display()))?;
                let heldout = Dataset::parse_str_for_prediction(&test_input, train.schema())?;
                for &seed in &seeds {
                    jobs.push(ExperimentJob {
                        train: Some(train.clone()),
                        heldout: Some(heldout.clone()),
                        paths: vec![train_path.clone(), test_path.clone()],
                        seed,
                        dataset_id: dataset_id.clone(),
                        fold_id: Some(fold.to_string()),
                        split_id: Some(format!("existing:{fold}")),
                        split_seed: None,
                        train_indices_sha256: None,
                        test_indices_sha256: None,
                    });
                }
            }
        }
        (None, None, None)
            if a.folds.is_none() && a.fold_start.is_none() && a.split_seed.is_none() =>
        {
            let dataset_id = output::sha256_bytes(serde_json::to_vec(&base.problem)?);
            for &seed in &seeds {
                jobs.push(ExperimentJob {
                    train: None,
                    heldout: None,
                    paths: Vec::new(),
                    seed,
                    dataset_id: dataset_id.clone(),
                    fold_id: None,
                    split_id: None,
                    split_seed: None,
                    train_indices_sha256: None,
                    test_indices_sha256: None,
                });
            }
        }
        _ => anyhow::bail!(
            "provide numeric repetitions without fold flags, --data with --folds, or both existing fold patterns with --folds"
        ),
    }
    validate_job_operators(&base, &jobs)?;
    let records = execute_experiment_jobs(&base, &jobs)?;
    for record in &records {
        output::append_jsonl(&a.output, record)?;
    }
    Ok(())
}

/// Check crossover compatibility with every distinct training split before any
/// job starts evolving; fold subsets can drop constant predictors and so change
/// the genome layout.
fn validate_job_operators(cfg: &RunConfig, jobs: &[ExperimentJob]) -> Result<()> {
    let mut seen = std::collections::HashSet::new();
    for job in jobs {
        if !seen.insert(job.split_id.as_deref()) {
            continue;
        }
        if let Some(train) = &job.train {
            LearningProblem::new(train.clone(), cfg.training.clone(), cfg.operators.clone())
                .with_context(|| format!("fold {}", job.fold_id.as_deref().unwrap_or("unknown")))?;
        } else {
            NumericProblem::new(&cfg.problem)?.validate_operators(&cfg.operators)?;
        }
    }
    Ok(())
}

#[derive(Clone)]
struct ExperimentJob {
    train: Option<Dataset>,
    heldout: Option<Dataset>,
    paths: Vec<PathBuf>,
    seed: u64,
    dataset_id: String,
    fold_id: Option<String>,
    split_id: Option<String>,
    split_seed: Option<u64>,
    train_indices_sha256: Option<String>,
    test_indices_sha256: Option<String>,
}

fn execute_experiment_job(
    base: &RunConfig,
    job: &ExperimentJob,
    context: &ExecutionContext,
    outer_threads: Option<usize>,
) -> Result<RunRecord> {
    let mut cfg = base.clone();
    cfg.seed = job.seed;
    cfg.validate()?;
    let started = Instant::now();
    let (result, selected, score) = if let (Some(train), Some(heldout)) = (&job.train, &job.heldout)
    {
        let problem =
            LearningProblem::new(train.clone(), cfg.training.clone(), cfg.operators.clone())?;
        let result = algorithms::run_with_context(&problem, &cfg, context)?;
        let chosen = problem.choose(&result.front)?;
        let score = problem.score_dataset(chosen.genome(), heldout)?;
        let selected = chosen.evaluation().cloned();
        (result, selected, Some(score))
    } else {
        ensure!(
            job.train.is_none() && job.heldout.is_none(),
            "experiment job has incomplete dataset state"
        );
        let problem = NumericProblem::new(&cfg.problem)?;
        (
            algorithms::run_with_context(&problem, &cfg, context)?,
            None,
            None,
        )
    };
    let mut record = run_record(&cfg, &job.paths, started, &result)?;
    if job.train.is_some() {
        record.semantic_version = format!(
            "mocs-rs-{}",
            learning_run_semantics_version(cfg.training.model)
        );
    }
    record.dataset_id = Some(job.dataset_id.clone());
    record.fold_id = job.fold_id.clone();
    record.split_id = job.split_id.clone();
    record.split_seed = job.split_seed;
    record.train_indices_sha256 = job.train_indices_sha256.clone();
    record.test_indices_sha256 = job.test_indices_sha256.clone();
    record.outer_experiment_threads = outer_threads;
    record.training_selection = selected;
    record.heldout = score;
    Ok(record)
}

fn execute_experiment_jobs(base: &RunConfig, jobs: &[ExperimentJob]) -> Result<Vec<RunRecord>> {
    if jobs.len() <= 1 {
        let context = ExecutionContext::new(base.threads)?;
        return jobs
            .iter()
            .map(|job| execute_experiment_job(base, job, &context, None))
            .collect();
    }
    if base.threads == 1 {
        let serial = ExecutionContext::new(1)?;
        return jobs
            .iter()
            .map(|job| execute_experiment_job(base, job, &serial, None))
            .collect();
    }
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        let context = ExecutionContext::new(base.threads)?;
        let serial = ExecutionContext::new(1)?;
        let results = context.install(|| {
            jobs.par_iter()
                .map(|job| {
                    execute_experiment_job(
                        &RunConfig {
                            threads: 1,
                            ..base.clone()
                        },
                        job,
                        &serial,
                        (base.threads > 1).then_some(base.threads),
                    )
                })
                .collect::<Vec<_>>()
        })?;
        results.into_iter().collect()
    }
    #[cfg(not(feature = "parallel"))]
    {
        ensure!(
            base.threads == 1,
            "experiment threads > 1 require rebuilding with --features parallel"
        );
        let serial = ExecutionContext::new(1)?;
        jobs.iter()
            .map(|job| execute_experiment_job(base, job, &serial, None))
            .collect()
    }
}
fn parse_seeds(s: &str) -> Result<Vec<u64>> {
    let out = s
        .split(',')
        .map(|v| v.trim().parse::<u64>().map_err(Into::into))
        .collect::<Result<Vec<_>>>()?;
    ensure!(!out.is_empty(), "at least one seed is required");
    ensure!(
        out.iter()
            .copied()
            .collect::<std::collections::BTreeSet<_>>()
            .len()
            == out.len(),
        "experiment seeds must be unique"
    );
    Ok(out)
}

fn read_records(path: &PathBuf) -> Result<Vec<RunRecord>> {
    let input =
        std::fs::read_to_string(path).with_context(|| format!("read JSONL {}", path.display()))?;
    let mut out = Vec::new();
    for (i, line) in input.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let r: RunRecord = serde_json::from_str(line)
            .with_context(|| format!("decode JSONL record on line {}", i + 1))?;
        ensure!(
            r.schema_version == output::RUN_RECORD_VERSION,
            "unsupported run-record version {} on line {}",
            r.schema_version,
            i + 1
        );
        out.push(r);
    }
    Ok(out)
}
fn run_metrics(a: MetricsArgs) -> Result<()> {
    let records = read_records(&a.input)?;
    ensure!(!records.is_empty(), "input contains no run records");
    let reference = parse_reference(&a.reference)?;
    let mut rows = String::from("run,hypervolume,outside_points,historical_s,historical_spacing\n");
    for (i, r) in records.iter().enumerate() {
        let hv = metrics::hypervolume(&r.front, &reference)?;
        let s = if reference.len() == 2 {
            metrics::historical_s(&r.front, &reference)?
        } else {
            f64::NAN
        };
        let sp = if reference.len() == 2 {
            metrics::historical_spacing(&r.front)?
        } else {
            f64::NAN
        };
        rows.push_str(&format!(
            "{i},{:.17},{},{:.17},{:.17}\n",
            hv.value, hv.outside_points, s, sp
        ));
    }
    if let Some(path) = a.output {
        std::fs::write(&path, rows).with_context(|| format!("write metrics {}", path.display()))?;
    } else {
        print!("{rows}");
    }
    Ok(())
}
fn run_analyze(a: AnalyzeArgs) -> Result<()> {
    let records = read_records(&a.input)?;
    ensure!(!records.is_empty(), "input contains no run records");
    std::fs::create_dir_all(&a.output_dir)?;
    let reference = a.reference.as_deref().map(parse_reference).transpose()?;
    let metric = a.metric.clone().unwrap_or_else(|| {
        if reference.is_some() {
            "hypervolume".into()
        } else {
            "heldout_loss".into()
        }
    });
    let mut groups = std::collections::BTreeMap::<String, Vec<f64>>::new();
    let mut fronts = Vec::new();
    for r in &records {
        let name = r
            .config
            .pointer("/algorithm/kind")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string();
        let value = record_metric(r, &metric, reference.as_deref())?;
        groups.entry(name).or_default().push(value);
        fronts.push(r.front.clone());
    }
    let mut summaries = Vec::<SummaryRow>::new();
    for (group, values) in groups {
        let mut row = output::summarize(&values)?;
        row.group = group;
        row.metric = metric.clone();
        summaries.push(row);
    }
    output::write_summary_csv(a.output_dir.join("summary.csv"), &summaries)?;
    output::write_front_csv(a.output_dir.join("fronts.csv"), &fronts)?;
    if !a.comparisons.is_empty() {
        let unit = match a.unit.as_str() {
            "seeded_run" => ComparisonUnit::SeededRun,
            "cv_repetition" => ComparisonUnit::CvRepetition,
            "dataset" => ComparisonUnit::Dataset,
            _ => anyhow::bail!("unknown comparison unit {:?}", a.unit),
        };
        let compare_metric = a.metric.clone().unwrap_or_else(|| {
            if records.iter().all(|r| r.heldout.is_some()) {
                "heldout_loss".into()
            } else {
                "hypervolume".into()
            }
        });
        let mut t_results = Vec::new();
        let mut w_results = Vec::new();
        let mut labels = Vec::new();
        for contrast in &a.comparisons {
            let (left, right) = contrast
                .split_once(':')
                .context("--compare must use ALGORITHM:ALGORITHM")?;
            ensure!(
                !left.is_empty() && !right.is_empty() && left != right,
                "invalid algorithm contrast {contrast:?}"
            );
            let (x, y) = paired_values(
                &records,
                left,
                right,
                unit,
                &compare_metric,
                reference.as_deref(),
                a.folds,
            )?;
            t_results.push(stats::paired_t_test(unit, &x, &y)?);
            w_results.push(stats::wilcoxon_signed_rank(unit, &x, &y)?);
            labels.push(contrast.clone());
        }
        let t_adj = stats::holm_adjust(&mut t_results);
        let w_adj = stats::holm_adjust(&mut w_results);
        let mut csv = String::from(
            "contrast,unit,metric,n,mean_difference,t_statistic,t_p,t_holm,w_statistic,w_p,w_holm,status\n",
        );
        for i in 0..labels.len() {
            csv.push_str(&format!(
                "{},{},{},{},{},{},{},{},{},{},{},{:?}\n",
                labels[i],
                a.unit,
                compare_metric,
                t_results[i].n,
                fmt_opt(t_results[i].mean_difference),
                fmt_opt(t_results[i].statistic),
                fmt_opt(t_results[i].p_value),
                fmt_opt(t_adj[i]),
                fmt_opt(w_results[i].statistic),
                fmt_opt(w_results[i].p_value),
                fmt_opt(w_adj[i]),
                t_results[i].status
            ));
        }
        std::fs::write(a.output_dir.join("comparisons.csv"), csv)?;
    }
    Ok(())
}

fn parse_reference(s: &str) -> Result<Vec<f64>> {
    let values = s
        .split(',')
        .map(|x| x.trim().parse::<f64>().map_err(Into::into))
        .collect::<Result<Vec<_>>>()?;
    ensure!(
        (values.len() == 2 || values.len() == 3) && values.iter().all(|x| x.is_finite()),
        "reference must contain two or three finite comma-separated values"
    );
    Ok(values)
}

fn fmt_opt(x: Option<f64>) -> String {
    x.map(|v| format!("{v:.17}")).unwrap_or_default()
}
fn record_metric(r: &RunRecord, metric: &str, reference: Option<&[f64]>) -> Result<f64> {
    let v = match metric {
        "heldout_loss" => {
            r.heldout
                .as_ref()
                .context("paired heldout_loss needs scored run records")?
                .loss
        }
        "hypervolume" => {
            metrics::hypervolume(
                &r.front,
                reference.context("paired hypervolume needs --reference")?,
            )?
            .value
        }
        "front_size" => r.front.len() as f64,
        _ => anyhow::bail!("unsupported paired metric {metric:?}"),
    };
    ensure!(v.is_finite(), "paired metric is nonfinite");
    Ok(v)
}
fn paired_values(
    records: &[RunRecord],
    left: &str,
    right: &str,
    unit: ComparisonUnit,
    metric: &str,
    reference: Option<&[f64]>,
    expected_folds: Option<usize>,
) -> Result<(Vec<Observation>, Vec<Observation>)> {
    #[derive(Clone)]
    struct Sample {
        fold: Option<String>,
        split: Option<String>,
        seed: u64,
        split_seed: Option<u64>,
        input_hashes: Vec<String>,
        value: f64,
        rows: usize,
    }
    let mut grouped =
        std::collections::BTreeMap::<String, std::collections::BTreeMap<String, Vec<Sample>>>::new(
        );
    for r in records {
        let alg = r
            .config
            .pointer("/algorithm/kind")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");
        if alg != left && alg != right {
            continue;
        }
        let dataset = r.dataset_id.clone().context(
            "paired analysis requires dataset_id metadata; regenerate records with current CLI",
        )?;
        ensure!(
            !dataset.is_empty(),
            "run record has no dataset_id for paired analysis"
        );
        let pair = match unit {
            ComparisonUnit::SeededRun => {
                ensure!(
                    r.fold_id.is_none(),
                    "seeded_run comparison requires run-level records without fold IDs"
                );
                ensure!(
                    r.split_id.is_none(),
                    "seeded_run comparison cannot contain split metadata"
                );
                format!("{dataset}|seed={}", r.seed)
            }
            ComparisonUnit::CvRepetition => format!(
                "{dataset}|seed={}|split={}",
                r.seed,
                r.split_seed
                    .map(|v| format!("generated:{v}"))
                    .unwrap_or_else(|| "existing".into())
            ),
            ComparisonUnit::Dataset => dataset,
        };
        let values = grouped
            .entry(pair)
            .or_default()
            .entry(alg.to_string())
            .or_default();
        ensure!(
            !values
                .iter()
                .any(|s| s.fold == r.fold_id && s.seed == r.seed && s.split_seed == r.split_seed),
            "ambiguous duplicate record for algorithm {alg}, fold {:?}, seed {}",
            r.fold_id,
            r.seed
        );
        let rows = r.heldout.as_ref().map(|report| report.rows).unwrap_or(1);
        ensure!(rows > 0, "paired record has zero scored rows");
        values.push(Sample {
            fold: r.fold_id.clone(),
            split: r.split_id.clone(),
            seed: r.seed,
            split_seed: r.split_seed,
            input_hashes: r.input_sha256.clone(),
            value: record_metric(r, metric, reference)?,
            rows,
        });
    }
    ensure!(
        !grouped.is_empty(),
        "no records found for comparison {left}:{right}"
    );
    let (mut a, mut b) = (Vec::new(), Vec::new());
    for (pair, algos) in grouped {
        let l = algos
            .get(left)
            .context(format!("missing {left} pair {pair}"))?;
        let r = algos
            .get(right)
            .context(format!("missing {right} pair {pair}"))?;
        if unit == ComparisonUnit::SeededRun {
            ensure!(
                l.len() == 1 && r.len() == 1,
                "seeded_run comparison requires one record per algorithm for pair {pair}"
            );
            ensure!(
                l[0].input_hashes.as_slice() == r[0].input_hashes.as_slice(),
                "input hashes do not match for pair {pair}"
            );
        }
        let reduce = |v: &Vec<Sample>| -> Result<f64> {
            if unit == ComparisonUnit::CvRepetition {
                ensure!(
                    v.iter().all(|sample| sample.fold.is_some()),
                    "CV repetition comparison requires fold IDs"
                );
                let n =
                    expected_folds.context("--folds is required for cv_repetition comparisons")?;
                ensure!(
                    v.len() == n,
                    "pair {pair} has {} folds; expected {n}",
                    v.len()
                );
                ensure!(!v.is_empty(), "CV repetition {pair} has no folds");
                ensure!(
                    metric == "heldout_loss",
                    "cv_repetition comparisons require --metric heldout_loss"
                );
                Ok(v.iter().map(|s| s.value * s.rows as f64).sum::<f64>()
                    / v.iter().map(|s| s.rows).sum::<usize>() as f64)
            } else if unit == ComparisonUnit::Dataset {
                ensure!(!v.is_empty(), "dataset pair {pair} has no runs");
                if metric == "heldout_loss" {
                    Ok(v.iter().map(|s| s.value * s.rows as f64).sum::<f64>()
                        / v.iter().map(|s| s.rows).sum::<usize>() as f64)
                } else {
                    Ok(v.iter().map(|s| s.value).sum::<f64>() / v.len() as f64)
                }
            } else {
                ensure!(v.len() == 1, "pair {pair} has repeated seeded run records");
                Ok(v[0].value)
            }
        };
        if unit == ComparisonUnit::CvRepetition {
            ensure!(
                l.iter().all(|sample| sample.fold.is_some())
                    && r.iter().all(|sample| sample.fold.is_some()),
                "CV repetition comparison for pair {pair} requires fold IDs on both sides"
            );
            let li = l
                .iter()
                .filter_map(|s| s.fold.as_ref())
                .collect::<std::collections::BTreeSet<_>>();
            let ri = r
                .iter()
                .filter_map(|s| s.fold.as_ref())
                .collect::<std::collections::BTreeSet<_>>();
            ensure!(li == ri, "fold IDs do not match in pair {pair}");
            ensure!(
                l.iter()
                    .map(|s| (s.fold.as_ref(), s.split.as_ref()))
                    .collect::<std::collections::BTreeSet<_>>()
                    == r.iter()
                        .map(|s| (s.fold.as_ref(), s.split.as_ref()))
                        .collect::<std::collections::BTreeSet<_>>(),
                "split IDs do not match in pair {pair}"
            );
            ensure!(
                l.iter()
                    .map(|s| (s.fold.as_ref(), &s.input_hashes))
                    .collect::<std::collections::BTreeSet<_>>()
                    == r.iter()
                        .map(|s| (s.fold.as_ref(), &s.input_hashes))
                        .collect::<std::collections::BTreeSet<_>>(),
                "input hashes do not match by fold in pair {pair}"
            );
        }
        if unit == ComparisonUnit::Dataset {
            let signature = |items: &Vec<Sample>| {
                items
                    .iter()
                    .map(|s| (s.seed, s.split_seed, s.fold.clone()))
                    .collect::<std::collections::BTreeSet<_>>()
            };
            ensure!(
                signature(l) == signature(r),
                "seed/fold sets do not match for dataset {pair}"
            );
            let data_signature = |items: &Vec<Sample>| {
                items
                    .iter()
                    .map(|s| {
                        (
                            s.seed,
                            s.split_seed,
                            s.fold.clone(),
                            s.split.clone(),
                            s.input_hashes.clone(),
                        )
                    })
                    .collect::<std::collections::BTreeSet<_>>()
            };
            ensure!(
                data_signature(l) == data_signature(r),
                "split IDs or input hashes differ for dataset {pair}"
            );
            if l.iter().any(|sample| sample.fold.is_some()) {
                let n = expected_folds
                    .context("--folds is required for dataset comparisons over fold records")?;
                let mut repetitions =
                    std::collections::BTreeMap::<(u64, Option<u64>), usize>::new();
                for sample in l {
                    ensure!(
                        sample.fold.is_some(),
                        "cannot mix folded and non-folded records in dataset {pair}"
                    );
                    *repetitions
                        .entry((sample.seed, sample.split_seed))
                        .or_default() += 1;
                }
                ensure!(
                    repetitions.values().all(|count| *count == n),
                    "dataset {pair} contains an incomplete fold repetition; expected {n} folds each"
                );
            }
        }
        a.push(Observation {
            pair_id: pair.clone(),
            value: reduce(l)?,
        });
        b.push(Observation {
            pair_id: pair.clone(),
            value: reduce(r)?,
        });
    }
    Ok((a, b))
}

#[cfg(all(test, feature = "parallel"))]
mod experiment_execution_tests {
    use super::*;

    fn job(seed: u64) -> ExperimentJob {
        ExperimentJob {
            train: None,
            heldout: None,
            paths: Vec::new(),
            seed,
            dataset_id: format!("dataset-{seed}"),
            fold_id: None,
            split_id: None,
            split_seed: None,
            train_indices_sha256: None,
            test_indices_sha256: None,
        }
    }

    fn base_config(threads: usize) -> RunConfig {
        RunConfig {
            threads,
            population: 4,
            offspring: 4,
            generations: 1,
            ..RunConfig::default()
        }
    }

    #[test]
    fn singleton_experiment_uses_inner_budget_and_multiple_jobs_use_outer_budget() {
        let base = base_config(2);
        let one = execute_experiment_jobs(&base, &[job(1)]).unwrap();
        assert_eq!(one[0].threads, 2);
        assert_eq!(one[0].outer_experiment_threads, None);

        let mut ngga = base_config(2);
        ngga.algorithm.kind = mocs_rs::config::AlgorithmKind::Ngga;
        let ngga_one = execute_experiment_jobs(&ngga, &[job(3)]).unwrap();
        assert_eq!(ngga_one[0].threads, 2);
        assert_eq!(ngga_one[0].outer_experiment_threads, None);

        let many = execute_experiment_jobs(&base, &[job(1), job(2)]).unwrap();
        assert_eq!(many.iter().map(|r| r.seed).collect::<Vec<_>>(), vec![1, 2]);
        assert!(many.iter().all(|r| r.threads == 1));
        assert!(many.iter().all(|r| r.outer_experiment_threads == Some(2)));

        let serial_many = execute_experiment_jobs(&base_config(1), &[job(4), job(5)]).unwrap();
        assert!(serial_many.iter().all(|r| r.threads == 1));
        assert!(
            serial_many
                .iter()
                .all(|r| r.outer_experiment_threads.is_none())
        );
    }
}
