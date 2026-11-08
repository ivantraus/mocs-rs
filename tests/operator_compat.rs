//! Crossover/mutation compatibility with every numeric problem and learning model.
//! Validation must reject exactly the combinations that cannot run, before any
//! evolution starts, and every accepted combination must run.

use std::{
    fs,
    path::PathBuf,
    process::Command,
    sync::atomic::{AtomicUsize, Ordering},
};

use mocs_rs::{
    algorithms,
    config::{
        AlgorithmKind, CrossoverKind, MatingKind, ModelKind, MutationKind, OperatorsConfig,
        ProblemKind, RunConfig, TrainingConfig,
    },
    data::Dataset,
    genome::Individual,
    models::LearningProblem,
    operators::{BuiltInOperators, EvolutionOperators},
    problems::NumericProblem,
    rng::Random,
};

const CROSSOVERS: [CrossoverKind; 3] = [
    CrossoverKind::KPoint,
    CrossoverKind::VariableLength,
    CrossoverKind::SimulatedBinary,
];
const MUTATIONS: [MutationKind; 4] = [
    MutationKind::Uniform,
    MutationKind::Normal,
    MutationKind::Polynomial,
    MutationKind::Adaptive,
];
const ALGORITHMS: [AlgorithmKind; 5] = [
    AlgorithmKind::Nsga2,
    AlgorithmKind::Spea2,
    AlgorithmKind::Moga,
    AlgorithmKind::Moes,
    AlgorithmKind::Ngga,
];

const REAL_GENES: &str = "requires real-valued genes";
const EQUAL_LENGTHS: &str = "requires equal-length genomes";
const ALIGNMENT: &str = "operators.crossover_alignment =";

/// A regression fixture with a nominal predictor, whose genome has integer loci.
const NOMINAL_REGRESSION: &str = "@relation nominal_regression\n\
@attribute x numeric\n@attribute colour {red,blue}\n@attribute y numeric\n@data\n\
0,red,0\n5,blue,4\n10,red,10\n2,blue,3\n";

/// Enough regression rows that pruning leaves genomes with different rule counts.
const WIDER_REGRESSION: &str = "@relation wider_regression\n\
@attribute x numeric\n@attribute y numeric\n@data\n\
0,0\n1,2\n2,1\n3,5\n4,3\n5,8\n6,4\n7,9\n8,7\n9,12\n10,10\n11,6\n";

fn root_file(path: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(path)
}

fn classification() -> Dataset {
    Dataset::load(root_file("examples/data/classification.arff"), None).unwrap()
}

fn regression() -> Dataset {
    Dataset::load(root_file("tests/fixtures/regression.arff"), None).unwrap()
}

fn run_config(case: usize, crossover: CrossoverKind, mutation: MutationKind) -> RunConfig {
    let mut cfg = RunConfig {
        seed: case as u64,
        population: 6,
        offspring: 6,
        generations: 3,
        threads: 1,
        ..RunConfig::default()
    };
    cfg.algorithm.kind = ALGORITHMS[case % ALGORITHMS.len()];
    cfg.operators.crossover = crossover;
    cfg.operators.mutation = mutation;
    cfg.operators.crossover_points = 1 + case % 3;
    cfg.operators.mating = MatingKind::Random;
    cfg
}

/// The rejection the validation contract promises, derived independently of
/// the operator code: SBX blends real values of equal-length genomes only, and
/// an explicit variable-length alignment must equal the layout block width.
fn expected_rejection(
    crossover: CrossoverKind,
    alignment: usize,
    width: usize,
    classification_model: bool,
    integer_loci: bool,
    pruning: bool,
) -> Option<&'static str> {
    match crossover {
        CrossoverKind::SimulatedBinary if classification_model => Some(REAL_GENES),
        CrossoverKind::SimulatedBinary if pruning => Some(EQUAL_LENGTHS),
        CrossoverKind::SimulatedBinary if integer_loci => Some(REAL_GENES),
        CrossoverKind::VariableLength if alignment != 1 && alignment != width => Some(ALIGNMENT),
        _ => None,
    }
}

fn alignments(width: usize) -> Vec<usize> {
    let mut out = vec![1, width, width + 1];
    out.dedup();
    out
}

fn assert_rejected(result: anyhow::Result<()>, needle: &str, case: &str) {
    let error = format!("{:#}", result.expect_err(case));
    assert!(error.contains(needle), "{case}: {error}");
    assert!(
        error.contains("operators.crossover") && error.contains("use operators."),
        "{case}: message must name the option and an alternative: {error}"
    );
}

#[test]
fn numeric_operator_matrix_runs_every_accepted_combination() {
    let problems = [
        (ProblemKind::Sf2, None, 1),
        (ProblemKind::Kur, Some(3), 3),
        (ProblemKind::Okk, None, 7),
        (ProblemKind::Zdt6, Some(4), 4),
    ];
    let mut case = 0;
    let (mut accepted, mut rejected) = (0, 0);
    for (kind, dimensions, width) in problems {
        for crossover in CROSSOVERS {
            for mutation in MUTATIONS {
                for alignment in alignments(width) {
                    case += 1;
                    let mut cfg = run_config(case, crossover, mutation);
                    cfg.problem.kind = kind;
                    cfg.problem.dimensions = dimensions;
                    cfg.operators.crossover_alignment = alignment;
                    let label =
                        format!("{kind:?} {crossover:?} {mutation:?} alignment={alignment}");
                    cfg.validate().unwrap();
                    let problem = NumericProblem::new(&cfg.problem).unwrap();
                    let validation = problem.validate_operators(&cfg.operators);
                    match expected_rejection(crossover, alignment, width, false, false, false) {
                        Some(needle) => {
                            assert_rejected(validation, needle, &label);
                            rejected += 1;
                        }
                        None => {
                            validation.unwrap_or_else(|e| panic!("{label}: {e:#}"));
                            let result = algorithms::run(&problem, &cfg)
                                .unwrap_or_else(|e| panic!("{label}: {e:#}"));
                            assert!(!result.front.is_empty(), "{label}");
                            accepted += 1;
                        }
                    }
                }
            }
        }
    }
    assert!(accepted > 0 && rejected > 0);
}

#[test]
fn learning_operator_matrix_runs_every_accepted_combination() {
    let models = [
        (ModelKind::Rule, classification(), true, true),
        (ModelKind::Instance, classification(), true, true),
        (ModelKind::Mixed, classification(), true, true),
        (ModelKind::RegressionRule, regression(), false, false),
        (
            ModelKind::RegressionRule,
            Dataset::parse_str(NOMINAL_REGRESSION).unwrap(),
            false,
            true,
        ),
        (ModelKind::RegressionLinearRule, regression(), false, false),
    ];
    let mut case = 0;
    let (mut accepted, mut rejected) = (0, 0);
    for (model, data, classification_model, integer_loci) in models {
        let training = TrainingConfig {
            model,
            initial_rules: 4,
            ..TrainingConfig::default()
        };
        let width =
            LearningProblem::new(data.clone(), training.clone(), OperatorsConfig::default())
                .unwrap()
                .rule_width();
        for crossover in CROSSOVERS {
            for mutation in MUTATIONS {
                for prune_probability in [0.0, 1.0] {
                    for alignment in alignments(width) {
                        case += 1;
                        let mut cfg = run_config(case, crossover, mutation);
                        cfg.operators.crossover_alignment = alignment;
                        cfg.training = TrainingConfig {
                            prune_probability,
                            ..training.clone()
                        };
                        let label = format!(
                            "{model:?} {crossover:?} {mutation:?} prune={prune_probability} \
                             alignment={alignment}"
                        );
                        cfg.validate().unwrap();
                        let pre_data = cfg.operators.validate_for_training(&cfg.training);
                        let problem = LearningProblem::new(
                            data.clone(),
                            cfg.training.clone(),
                            cfg.operators.clone(),
                        );
                        match expected_rejection(
                            crossover,
                            alignment,
                            width,
                            classification_model,
                            integer_loci,
                            prune_probability > 0.0,
                        ) {
                            Some(needle) => {
                                // Model-only incompatibilities are caught before data loading.
                                if needle != ALIGNMENT
                                    && (classification_model || prune_probability > 0.0)
                                {
                                    assert_rejected(pre_data, needle, &label);
                                } else {
                                    pre_data.unwrap_or_else(|e| panic!("{label}: {e:#}"));
                                }
                                assert_rejected(problem.map(|_| ()), needle, &label);
                                rejected += 1;
                            }
                            None => {
                                pre_data.unwrap_or_else(|e| panic!("{label}: {e:#}"));
                                let problem = problem.unwrap_or_else(|e| panic!("{label}: {e:#}"));
                                let result = algorithms::run(&problem, &cfg)
                                    .unwrap_or_else(|e| panic!("{label}: {e:#}"));
                                assert!(!result.front.is_empty(), "{label}");
                                problem.choose(&result.front).unwrap();
                                accepted += 1;
                            }
                        }
                    }
                }
            }
        }
    }
    assert!(accepted > 0 && rejected > 0);
}

/// Counts crossover calls whose parents differ in length.
struct CountingOperators {
    unequal: AtomicUsize,
}

impl EvolutionOperators for CountingOperators {
    fn mate(
        &self,
        population: &[Individual],
        scores: &[f64],
        config: &OperatorsConfig,
        rng: &mut Random,
    ) -> mocs_rs::Result<(usize, usize)> {
        BuiltInOperators.mate(population, scores, config, rng)
    }

    fn crossover(
        &self,
        first: &Individual,
        second: &Individual,
        config: &OperatorsConfig,
        rng: &mut Random,
        child_id: u64,
    ) -> mocs_rs::Result<(Individual, Individual)> {
        if first.genome().values.len() != second.genome().values.len() {
            self.unequal.fetch_add(1, Ordering::Relaxed);
        }
        BuiltInOperators.crossover(first, second, config, rng, child_id)
    }

    fn mutate(
        &self,
        individual: &mut Individual,
        config: &OperatorsConfig,
        rng: &mut Random,
    ) -> mocs_rs::Result<bool> {
        BuiltInOperators.mutate(individual, config, rng)
    }
}

#[test]
fn default_k_point_trains_pruned_unequal_length_genomes_for_every_model() {
    for (model, data) in [
        (ModelKind::Rule, classification()),
        (ModelKind::Instance, classification()),
        (ModelKind::Mixed, classification()),
        (
            ModelKind::RegressionRule,
            Dataset::parse_str(WIDER_REGRESSION).unwrap(),
        ),
        (
            ModelKind::RegressionLinearRule,
            Dataset::parse_str(WIDER_REGRESSION).unwrap(),
        ),
    ] {
        let mut cfg = RunConfig {
            population: 16,
            offspring: 16,
            generations: 20,
            threads: 1,
            ..RunConfig::default()
        };
        cfg.training.model = model;
        cfg.training.prune_probability = 1.0;
        assert_eq!(cfg.operators.crossover, CrossoverKind::KPoint);
        let problem =
            LearningProblem::new(data, cfg.training.clone(), cfg.operators.clone()).unwrap();
        let operators = CountingOperators {
            unequal: AtomicUsize::new(0),
        };
        let result = algorithms::run_with_operators(&problem, &cfg, &operators)
            .unwrap_or_else(|e| panic!("{model:?}: {e:#}"));
        assert!(
            operators.unequal.load(Ordering::Relaxed) > 0,
            "{model:?}: pruning never produced unequal-length parents"
        );
        let width = problem.rule_width();
        for individual in &result.front {
            assert!(individual.genome().values.len().is_multiple_of(width));
        }
        let chosen = problem.choose(&result.front).unwrap();
        problem
            .score_dataset(chosen.genome(), problem.dataset())
            .unwrap();
    }
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "mocs-operator-compat-{}-{name}",
        std::process::id()
    ));
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn train(dir: &std::path::Path, config: &str, data: &str) -> std::process::Output {
    let config_path = dir.join("config.toml");
    fs::write(&config_path, config).unwrap();
    Command::new(env!("CARGO_BIN_EXE_mocs"))
        .arg("train")
        .arg("--config")
        .arg(&config_path)
        .arg("--data")
        .arg(root_file(data))
        .arg("--model-out")
        .arg(dir.join("model.json"))
        .arg("--output")
        .arg(dir.join("run.jsonl"))
        .output()
        .unwrap()
}

#[test]
fn cli_train_with_default_crossover_and_pruning_succeeds() {
    for (model, data) in [
        ("rule", "examples/data/classification.arff"),
        ("instance", "examples/data/classification.arff"),
        ("mixed", "examples/data/classification.arff"),
        ("regression_rule", "tests/fixtures/regression.arff"),
    ] {
        let dir = scratch(model);
        let output = train(
            &dir,
            &format!(
                "population = 16\noffspring = 16\ngenerations = 20\nthreads = 1\n\n\
                 [training]\nmodel = \"{model}\"\nprune_probability = 1.0\n"
            ),
            data,
        );
        assert!(
            output.status.success(),
            "{model}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(dir.join("model.json").exists());
        fs::remove_dir_all(dir).unwrap();
    }
}

#[test]
fn cli_rejects_incompatible_crossover_before_loading_data() {
    let dir = scratch("sbx");
    let output = train(
        &dir,
        "population = 4\noffspring = 4\ngenerations = 1\nthreads = 1\n\n\
         [operators]\ncrossover = \"simulated_binary\"\n\n[training]\nmodel = \"rule\"\n",
        "tests/fixtures/does-not-exist.arff",
    );
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains(REAL_GENES), "{stderr}");
    assert!(!dir.join("run.jsonl").exists());

    let output = train(
        &dir,
        "population = 4\noffspring = 4\ngenerations = 1\nthreads = 1\n\n\
         [operators]\ncrossover = \"variable_length\"\ncrossover_alignment = 2\n\n\
         [training]\nmodel = \"rule\"\n",
        "examples/data/classification.arff",
    );
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains(ALIGNMENT), "{stderr}");
    assert!(!dir.join("run.jsonl").exists());
    fs::remove_dir_all(dir).unwrap();
}
