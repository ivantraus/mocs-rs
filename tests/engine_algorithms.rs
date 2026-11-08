#![allow(clippy::field_reassign_with_default)]

use mocs_rs::{
    Result,
    algorithms::run,
    config::{AlgorithmKind, RunConfig},
    evaluation::{EvaluationReport, Problem},
    genome::{Domain, GeneValue, Genome, GenomeLayout, Individual},
    operators::{BuiltInOperators, EvolutionOperators},
    rng::Random,
};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

struct Line;
impl Problem for Line {
    fn initialize(&self, rng: &mut Random) -> Result<Genome> {
        let layout = Arc::new(GenomeLayout::new(vec![Domain::Real {
            min: 0.0,
            max: 1.0,
        }])?);
        let mut g = Genome::new(vec![GeneValue::Real(rng.uniform())], layout)?;
        g.deviation = 0.2;
        g.deviations[0] = 0.2;
        Ok(g)
    }
    fn evaluate(&self, g: &Genome) -> Result<EvaluationReport> {
        let x = g.values[0].numeric();
        Ok(EvaluationReport {
            objectives: vec![x, 1.0 - x],
            loss: x,
            coverage: 1.0,
            conditional_loss: x,
            generality: 1.0,
            size: 1.0,
            rows: 1,
            ..Default::default()
        })
    }
}

struct PrunableLine(Line);
impl Problem for PrunableLine {
    fn initialize(&self, rng: &mut Random) -> Result<Genome> {
        self.0.initialize(rng)
    }
    fn evaluate(&self, g: &Genome) -> Result<EvaluationReport> {
        self.0.evaluate(g)
    }
    fn prune(&self, g: &mut Genome, _report: &EvaluationReport) -> Result<bool> {
        g.set(0, GeneValue::Real(0.5))?;
        Ok(true)
    }
}

struct ChangingObjectiveCount {
    initialized: AtomicUsize,
    evaluated: AtomicUsize,
}

struct CountingOperators {
    inner: BuiltInOperators,
    mates: AtomicUsize,
    crossovers: AtomicUsize,
    mutations: AtomicUsize,
}
impl EvolutionOperators for CountingOperators {
    fn mate(
        &self,
        population: &[Individual],
        scores: &[f64],
        config: &mocs_rs::config::OperatorsConfig,
        rng: &mut Random,
    ) -> Result<(usize, usize)> {
        self.mates.fetch_add(1, Ordering::Relaxed);
        self.inner.mate(population, scores, config, rng)
    }
    fn crossover(
        &self,
        first: &Individual,
        second: &Individual,
        config: &mocs_rs::config::OperatorsConfig,
        rng: &mut Random,
        child_id: u64,
    ) -> Result<(Individual, Individual)> {
        self.crossovers.fetch_add(1, Ordering::Relaxed);
        self.inner.crossover(first, second, config, rng, child_id)
    }
    fn mutate(
        &self,
        individual: &mut Individual,
        config: &mocs_rs::config::OperatorsConfig,
        rng: &mut Random,
    ) -> Result<bool> {
        self.mutations.fetch_add(1, Ordering::Relaxed);
        self.inner.mutate(individual, config, rng)
    }
}
impl Problem for ChangingObjectiveCount {
    fn initialize(&self, _rng: &mut Random) -> Result<Genome> {
        let value = self.initialized.fetch_add(1, Ordering::Relaxed) as f64 / 10.0;
        let layout = Arc::new(GenomeLayout::new(vec![Domain::Real {
            min: 0.0,
            max: 1.0,
        }])?);
        let mut genome = Genome::new(vec![GeneValue::Real(value)], layout)?;
        genome.deviation = 1.0;
        genome.deviations[0] = 1.0;
        Ok(genome)
    }
    fn evaluate(&self, genome: &Genome) -> Result<EvaluationReport> {
        let i = self.evaluated.fetch_add(1, Ordering::Relaxed);
        Ok(EvaluationReport {
            objectives: if i < 4 {
                vec![genome.values[0].numeric(), 1.0]
            } else {
                vec![genome.values[0].numeric()]
            },
            ..EvaluationReport::default()
        })
    }
}

#[test]
fn every_algorithm_runs_distinct_survivor_path_with_finite_nondominated_result() {
    for kind in [
        AlgorithmKind::Moga,
        AlgorithmKind::Moes,
        AlgorithmKind::Ngga,
        AlgorithmKind::Nsga2,
        AlgorithmKind::Spea2,
    ] {
        let mut c = RunConfig::default();
        c.seed = 7;
        c.population = 10;
        c.offspring = 6;
        c.generations = 3;
        c.algorithm.kind = kind;
        if matches!(kind, AlgorithmKind::Moga | AlgorithmKind::Moes) {
            c.offspring = c.population;
        }
        c.operators.per_locus_probability = Some(0.5);
        let r = run(&Line, &c).unwrap();
        assert_eq!(r.generations, 3, "{kind:?}");
        assert!(!r.front.is_empty());
        assert!(r.front.iter().all(|x| {
            x.evaluation()
                .unwrap()
                .objectives
                .iter()
                .all(|v| v.is_finite())
        }));
        assert!(r.evaluations >= c.population);
        assert!(r.evaluations <= c.population + c.generations * c.offspring);
    }
}
#[test]
fn fixed_seed_repeats_scientific_records() {
    let mut c = RunConfig::default();
    c.seed = 83;
    c.population = 8;
    c.offspring = 4;
    c.generations = 2;
    let a = run(&Line, &c).unwrap();
    let b = run(&Line, &c).unwrap();
    assert_eq!(a.evaluations, b.evaluations);
    assert_eq!(
        a.history.iter().map(|g| &g.front).collect::<Vec<_>>(),
        b.history.iter().map(|g| &g.front).collect::<Vec<_>>()
    );
}

#[test]
fn evaluation_budget_rejects_initialization_prune_overshoot() {
    let mut c = RunConfig::default();
    c.population = 4;
    c.offspring = 2;
    c.generations = 0;
    c.max_evaluations = Some(4);
    c.training.prune_probability = 1.0;
    let err = run(&PrunableLine(Line), &c).unwrap_err().to_string();
    assert!(err.contains("initialization used"));
}

#[test]
fn spea2_k_cannot_exceed_configured_population() {
    let mut config = RunConfig::default();
    config.population = 4;
    config.algorithm.spea2_k = Some(5);
    assert!(
        config
            .validate()
            .unwrap_err()
            .to_string()
            .contains("spea2_k")
    );
}

#[test]
fn budget_mode_stops_at_completed_generation_boundary() {
    let mut c = RunConfig::default();
    c.population = 4;
    c.offspring = 2;
    c.generations = 0;
    c.max_evaluations = Some(7);
    c.operators.crossover_probability = 0.0;
    c.operators.mutation_probability = 0.0;
    let r = run(&Line, &c).unwrap();
    assert_eq!(r.evaluations, 4);
    assert_eq!(r.generations, 1);
}

#[test]
fn objective_schema_change_is_reported_before_survivor_comparison() {
    let problem = ChangingObjectiveCount {
        initialized: AtomicUsize::new(0),
        evaluated: AtomicUsize::new(0),
    };
    let mut c = RunConfig::default();
    c.threads = 1;
    c.population = 4;
    c.offspring = 2;
    c.generations = 1;
    c.operators.crossover_probability = 0.0;
    c.operators.mutation_probability = 1.0;
    let err = run(&problem, &c).unwrap_err().to_string();
    assert!(err.contains("expected 2"));
}

#[test]
fn explicit_per_locus_probability_initializes_numeric_mutation_state() {
    let mut c = RunConfig::default();
    c.seed = 21;
    c.population = 8;
    c.offspring = 8;
    c.generations = 1;
    c.operators.crossover_probability = 0.0;
    c.operators.mutation_probability = 1.0;
    c.operators.mutation = mocs_rs::config::MutationKind::Uniform;
    c.operators.per_locus_probability = Some(0.0);
    let frozen = run(&Line, &c).unwrap();
    c.operators.per_locus_probability = Some(1.0);
    let mutating = run(&Line, &c).unwrap();
    assert_eq!(frozen.evaluations, c.population);
    assert!(mutating.evaluations > frozen.evaluations);
}

#[test]
fn custom_evolution_operator_extension_is_invoked_by_run() {
    let mut c = RunConfig::default();
    c.population = 8;
    c.offspring = 8;
    c.generations = 1;
    c.operators.crossover_probability = 1.0;
    c.operators.mutation_probability = 1.0;
    let operators = CountingOperators {
        inner: BuiltInOperators,
        mates: AtomicUsize::new(0),
        crossovers: AtomicUsize::new(0),
        mutations: AtomicUsize::new(0),
    };
    mocs_rs::algorithms::run_with_operators(&Line, &c, &operators).unwrap();
    assert_eq!(operators.mates.load(Ordering::Relaxed), c.offspring);
    assert_eq!(operators.crossovers.load(Ordering::Relaxed), c.offspring);
    assert_eq!(operators.mutations.load(Ordering::Relaxed), c.offspring);
}

#[cfg(feature = "parallel")]
#[test]
fn parallel_problem_evaluation_preserves_serial_scientific_records() {
    let mut c = RunConfig::default();
    c.threads = 1;
    c.seed = 991;
    c.population = 12;
    c.offspring = 8;
    c.generations = 4;
    c.training.prune_probability = 0.37;
    let serial = run(&Line, &c).unwrap();
    c.threads = 3;
    let parallel = run(&Line, &c).unwrap();
    assert_eq!(serial.evaluations, parallel.evaluations);
    assert_eq!(serial.generations, parallel.generations);
    assert_eq!(
        serial
            .history
            .iter()
            .map(|r| (&r.front, &r.population_ids))
            .collect::<Vec<_>>(),
        parallel
            .history
            .iter()
            .map(|r| (&r.front, &r.population_ids))
            .collect::<Vec<_>>()
    );
}
