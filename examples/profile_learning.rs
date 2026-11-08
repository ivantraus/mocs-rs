use anyhow::{Context, Result, ensure};
use clap::Parser;
use mocs_rs::{
    algorithms,
    config::RunConfig,
    data::Dataset,
    evaluation::{EvaluationReport, Problem},
    genome::{Genome, Individual},
    models::LearningProblem,
    operators::{BuiltInOperators, EvolutionOperators},
    output,
    rng::Random,
};
use serde_json::json;
use std::{
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, AtomicUsize, Ordering},
    time::Instant,
};

#[derive(Debug, Parser)]
#[command(about = "Diagnostic phase profiler for a deterministic learning run")]
struct Args {
    #[arg(long)]
    config: PathBuf,
    #[arg(long)]
    data: PathBuf,
    #[arg(long)]
    seed: Option<u64>,
    #[arg(long)]
    generations: Option<usize>,
    #[arg(long)]
    verify_reference: bool,
}

#[derive(Default)]
struct Timings {
    initialize_ns: AtomicU64,
    initialize_calls: AtomicUsize,
    initialized_loci: AtomicUsize,
    evaluate_ns: AtomicU64,
    evaluate_calls: AtomicUsize,
    evaluated_rules: AtomicUsize,
    max_evaluated_rules: AtomicUsize,
    evaluated_loci: AtomicUsize,
    max_evaluated_loci: AtomicUsize,
    prune_ns: AtomicU64,
    prune_calls: AtomicUsize,
    prune_changes: AtomicUsize,
    mate_ns: AtomicU64,
    mate_calls: AtomicUsize,
    crossover_ns: AtomicU64,
    crossover_calls: AtomicUsize,
    crossover_parent_loci: AtomicUsize,
    crossover_child_loci: AtomicUsize,
    mutation_ns: AtomicU64,
    mutation_calls: AtomicUsize,
    mutation_loci: AtomicUsize,
    mutation_changes: AtomicUsize,
}

fn elapsed_ns(start: Instant) -> u64 {
    start.elapsed().as_nanos().min(u64::MAX as u128) as u64
}

fn add_ns(counter: &AtomicU64, elapsed: u64) {
    counter.fetch_add(elapsed, Ordering::Relaxed);
}

struct TimedProblem<'a> {
    inner: &'a LearningProblem,
    timings: &'a Timings,
}

impl Problem for TimedProblem<'_> {
    fn initialize(&self, rng: &mut Random) -> Result<Genome> {
        let started = Instant::now();
        let result = self.inner.initialize(rng);
        add_ns(&self.timings.initialize_ns, elapsed_ns(started));
        self.timings
            .initialize_calls
            .fetch_add(1, Ordering::Relaxed);
        if let Ok(genome) = &result {
            self.timings
                .initialized_loci
                .fetch_add(genome.values.len(), Ordering::Relaxed);
        }
        result
    }

    fn evaluate(&self, genome: &Genome) -> Result<EvaluationReport> {
        let started = Instant::now();
        let result = self.inner.evaluate(genome);
        add_ns(&self.timings.evaluate_ns, elapsed_ns(started));
        self.timings.evaluate_calls.fetch_add(1, Ordering::Relaxed);
        self.timings
            .evaluated_loci
            .fetch_add(genome.values.len(), Ordering::Relaxed);
        self.timings
            .max_evaluated_loci
            .fetch_max(genome.values.len(), Ordering::Relaxed);
        if let Ok(report) = &result {
            self.timings
                .evaluated_rules
                .fetch_add(report.rule_count, Ordering::Relaxed);
            self.timings
                .max_evaluated_rules
                .fetch_max(report.rule_count, Ordering::Relaxed);
        }
        result
    }

    fn prune(&self, genome: &mut Genome, report: &EvaluationReport) -> Result<bool> {
        let started = Instant::now();
        let result = self.inner.prune(genome, report);
        add_ns(&self.timings.prune_ns, elapsed_ns(started));
        self.timings.prune_calls.fetch_add(1, Ordering::Relaxed);
        if result.as_ref().is_ok_and(|changed| *changed) {
            self.timings.prune_changes.fetch_add(1, Ordering::Relaxed);
        }
        result
    }
}

struct TimedOperators<'a> {
    inner: BuiltInOperators,
    timings: &'a Timings,
}

impl EvolutionOperators for TimedOperators<'_> {
    fn mate(
        &self,
        population: &[Individual],
        scores: &[f64],
        config: &mocs_rs::config::OperatorsConfig,
        rng: &mut Random,
    ) -> Result<(usize, usize)> {
        let started = Instant::now();
        let result = self.inner.mate(population, scores, config, rng);
        add_ns(&self.timings.mate_ns, elapsed_ns(started));
        self.timings.mate_calls.fetch_add(1, Ordering::Relaxed);
        result
    }

    fn crossover(
        &self,
        first: &Individual,
        second: &Individual,
        config: &mocs_rs::config::OperatorsConfig,
        rng: &mut Random,
        child_id: u64,
    ) -> Result<(Individual, Individual)> {
        let started = Instant::now();
        let parent_loci = first.genome().values.len() + second.genome().values.len();
        let result = self.inner.crossover(first, second, config, rng, child_id);
        add_ns(&self.timings.crossover_ns, elapsed_ns(started));
        self.timings.crossover_calls.fetch_add(1, Ordering::Relaxed);
        self.timings
            .crossover_parent_loci
            .fetch_add(parent_loci, Ordering::Relaxed);
        if let Ok((first_child, second_child)) = &result {
            self.timings.crossover_child_loci.fetch_add(
                first_child.genome().values.len() + second_child.genome().values.len(),
                Ordering::Relaxed,
            );
        }
        result
    }

    fn mutate(
        &self,
        individual: &mut Individual,
        config: &mocs_rs::config::OperatorsConfig,
        rng: &mut Random,
    ) -> Result<bool> {
        let started = Instant::now();
        let loci = individual.genome().values.len();
        let result = self.inner.mutate(individual, config, rng);
        add_ns(&self.timings.mutation_ns, elapsed_ns(started));
        self.timings.mutation_calls.fetch_add(1, Ordering::Relaxed);
        self.timings
            .mutation_loci
            .fetch_add(loci, Ordering::Relaxed);
        if result.as_ref().is_ok_and(|changed| *changed) {
            self.timings
                .mutation_changes
                .fetch_add(1, Ordering::Relaxed);
        }
        result
    }
}

fn atomic_value(value: &AtomicUsize) -> usize {
    value.load(Ordering::Relaxed)
}

fn run() -> Result<()> {
    let args = Args::parse();
    let mut config = RunConfig::load(std::slice::from_ref(&args.config))?;
    if let Some(seed) = args.seed {
        config.seed = seed;
    }
    if let Some(generations) = args.generations {
        config.generations = generations;
    }
    config.threads = 1;
    config.validate()?;

    let dataset = Dataset::load(&args.data, config.training.target.as_deref())?;
    let problem = LearningProblem::new(dataset, config.training.clone(), config.operators.clone())?;
    let timings = Timings::default();
    let timed_problem = TimedProblem {
        inner: &problem,
        timings: &timings,
    };
    let timed_operators = TimedOperators {
        inner: BuiltInOperators,
        timings: &timings,
    };

    let total_started = Instant::now();
    let result = algorithms::run_with_operators(&timed_problem, &config, &timed_operators)?;
    let selected = problem.choose(&result.front)?;
    let total_ns = elapsed_ns(total_started);
    let measured = timings.initialize_ns.load(Ordering::Relaxed)
        + timings.evaluate_ns.load(Ordering::Relaxed)
        + timings.prune_ns.load(Ordering::Relaxed)
        + timings.mate_ns.load(Ordering::Relaxed)
        + timings.crossover_ns.load(Ordering::Relaxed)
        + timings.mutation_ns.load(Ordering::Relaxed);
    let phase_ns = json!({
        "initialize": timings.initialize_ns.load(Ordering::Relaxed),
        "evaluate": timings.evaluate_ns.load(Ordering::Relaxed),
        "prune": timings.prune_ns.load(Ordering::Relaxed),
        "mate": timings.mate_ns.load(Ordering::Relaxed),
        "crossover": timings.crossover_ns.load(Ordering::Relaxed),
        "mutation": timings.mutation_ns.load(Ordering::Relaxed),
        "unaccounted_selection_ranking_cloning_and_rest": total_ns.saturating_sub(measured),
    });
    if args.verify_reference {
        let reference = algorithms::run(&problem, &config)?;
        let timed_json = serde_json::to_vec(&result)?;
        let reference_json = serde_json::to_vec(&reference)?;
        ensure!(
            timed_json == reference_json,
            "instrumented run differs from ordinary algorithms::run"
        );
    }

    let front_sizes: Vec<usize> = result
        .history
        .iter()
        .map(|generation| generation.front.len())
        .collect();
    let front_fractions: Vec<f64> = front_sizes
        .iter()
        .map(|size| *size as f64 / config.population as f64)
        .collect();
    let evaluations = atomic_value(&timings.evaluate_calls);
    let measured_rules = atomic_value(&timings.evaluated_rules);
    let source = Path::new(file!());
    let source_hashes = source_hashes(source)?;
    let output = json!({
        "instrumented": true,
        "verification_requested": args.verify_reference,
        "verified_equal_to_uninstrumented": args.verify_reference,
        "crate_version": env!("CARGO_PKG_VERSION"),
        "config": &config,
        "config_sha256": output::sha256_file(&args.config)?,
        "input_path": args.data.display().to_string(),
        "input_sha256": output::sha256_file(&args.data)?,
        "source_hashes": source_hashes,
        "total_ns": total_ns,
        "phase_ns": phase_ns,
        "counts": {
            "evaluations": evaluations,
            "initialize_calls": atomic_value(&timings.initialize_calls),
            "prune_calls": atomic_value(&timings.prune_calls),
            "prune_changes": atomic_value(&timings.prune_changes),
            "mate_calls": atomic_value(&timings.mate_calls),
            "crossover_calls": atomic_value(&timings.crossover_calls),
            "mutation_calls": atomic_value(&timings.mutation_calls),
            "mutation_changes": atomic_value(&timings.mutation_changes),
        },
        "loci": {
            "initialized_total": atomic_value(&timings.initialized_loci),
            "evaluated_total": atomic_value(&timings.evaluated_loci),
            "evaluated_max": atomic_value(&timings.max_evaluated_loci),
            "crossover_parent_total": atomic_value(&timings.crossover_parent_loci),
            "crossover_child_total": atomic_value(&timings.crossover_child_loci),
            "mutation_total": atomic_value(&timings.mutation_loci),
        },
        "evaluated_rules": {
            "mean": (measured_rules as f64 / evaluations as f64),
            "max": atomic_value(&timings.max_evaluated_rules),
        },
        "run": {
            "evaluations": result.evaluations,
            "generations": result.generations,
            "final_front_size": result.front.len(),
            "final_front_fraction_of_population": result.front.len() as f64 / config.population as f64,
            "generation_front_sizes": front_sizes,
            "generation_front_fractions": front_fractions,
            "selected_report": selected.evaluation(),
        },
    });
    println!("{}", serde_json::to_string_pretty(&output)?);
    Ok(())
}

fn source_hashes(example: &Path) -> Result<serde_json::Value> {
    let example_path = if example.is_absolute() {
        example.to_path_buf()
    } else {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(example)
    };
    let mut files = vec![example_path];
    let source_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src");
    for entry in std::fs::read_dir(&source_root)
        .with_context(|| format!("read source directory {}", source_root.display()))?
    {
        let path = entry?.path();
        if path.extension().is_some_and(|extension| extension == "rs") {
            files.push(path);
        }
    }
    files.push(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml"));
    files.push(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("Cargo.lock"));
    files.sort();
    let mut hashes = serde_json::Map::new();
    for path in files {
        let label = path
            .strip_prefix(env!("CARGO_MANIFEST_DIR"))
            .unwrap_or(&path)
            .to_string_lossy()
            .to_string();
        hashes.insert(label, json!(output::sha256_file(path)?));
    }
    Ok(serde_json::Value::Object(hashes))
}

fn main() -> Result<()> {
    run()
}
