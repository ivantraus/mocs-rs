use mocs_rs::{
    algorithms::{RunResult, run_with_operators},
    config::{AlgorithmKind, RunConfig},
    evaluation::{EvaluationReport, Problem},
    genome::{Domain, GeneValue, Genome, GenomeLayout, Individual},
    operators::EvolutionOperators,
    rng::Random,
};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

const PARENTS: [[f64; 2]; 3] = [[0.0, 9.0], [3.0, 5.0], [8.0, 1.0]];
const CHILDREN: [[f64; 2]; 4] = [[1.0, 8.0], [4.0, 4.0], [6.0, 3.0], [3.0, 10.0]];

struct TradeoffProblem {
    next: AtomicUsize,
    layout: Arc<GenomeLayout>,
}

impl TradeoffProblem {
    fn new() -> Self {
        Self {
            next: AtomicUsize::new(0),
            layout: Arc::new(
                GenomeLayout::new(vec![
                    Domain::Real {
                        min: 0.0,
                        max: 10.0,
                    },
                    Domain::Real {
                        min: 0.0,
                        max: 10.0,
                    },
                ])
                .unwrap(),
            ),
        }
    }

    fn genome(&self, point: [f64; 2]) -> Genome {
        Genome::new(
            point.into_iter().map(GeneValue::Real).collect(),
            self.layout.clone(),
        )
        .unwrap()
    }
}

impl Problem for TradeoffProblem {
    fn initialize(&self, _: &mut Random) -> mocs_rs::Result<Genome> {
        let index = self.next.fetch_add(1, Ordering::Relaxed);
        Ok(self.genome(PARENTS[index]))
    }

    fn evaluate(&self, genome: &Genome) -> mocs_rs::Result<EvaluationReport> {
        Ok(EvaluationReport {
            objectives: genome.values.iter().map(|value| value.numeric()).collect(),
            ..EvaluationReport::default()
        })
    }
}

struct FixedChildren {
    layout: Arc<GenomeLayout>,
    points: [[f64; 2]; 4],
}

impl EvolutionOperators for FixedChildren {
    fn mate(
        &self,
        _: &[Individual],
        _: &[f64],
        _: &mocs_rs::config::OperatorsConfig,
        _: &mut Random,
    ) -> mocs_rs::Result<(usize, usize)> {
        Ok((0, 1))
    }

    fn crossover(
        &self,
        _: &Individual,
        _: &Individual,
        _: &mocs_rs::config::OperatorsConfig,
        _: &mut Random,
        child_id: u64,
    ) -> mocs_rs::Result<(Individual, Individual)> {
        let index =
            usize::try_from((child_id - PARENTS.len() as u64) / 2).unwrap() % self.points.len();
        let point = self.points[index];
        let genome = Genome::new(
            point.into_iter().map(GeneValue::Real).collect(),
            self.layout.clone(),
        )?;
        Ok((
            Individual::new(child_id, genome.clone()),
            Individual::new(child_id + 1, genome),
        ))
    }

    fn mutate(
        &self,
        _: &mut Individual,
        _: &mocs_rs::config::OperatorsConfig,
        _: &mut Random,
    ) -> mocs_rs::Result<bool> {
        Ok(false)
    }
}

fn fixture(kind: AlgorithmKind) -> RunResult {
    let problem = TradeoffProblem::new();
    let operators = FixedChildren {
        layout: problem.layout.clone(),
        points: CHILDREN,
    };
    let mut config = RunConfig {
        seed: 19,
        population: PARENTS.len(),
        offspring: CHILDREN.len(),
        generations: 1,
        ..RunConfig::default()
    };
    config.algorithm.kind = kind;
    config.operators.crossover_probability = 1.0;
    config.operators.mutation_probability = 0.0;
    if kind == AlgorithmKind::Ngga {
        // Isolate the documented equal-score ID tie break in steady-state
        // replacement while retaining distinct, independently evaluated points.
        config.algorithm.ngga_n1 = Some(0);
        config.algorithm.ngga_n2 = Some(0);
    }
    run_with_operators(&problem, &config, &operators).unwrap()
}

fn fixture_with_old_best(kind: AlgorithmKind) -> RunResult {
    let problem = TradeoffProblem::new();
    let operators = FixedChildren {
        layout: problem.layout.clone(),
        points: [[10.0, 10.0]; 4],
    };
    let mut config = RunConfig {
        seed: 23,
        population: PARENTS.len(),
        offspring: CHILDREN.len(),
        generations: 2,
        ..RunConfig::default()
    };
    config.algorithm.kind = kind;
    config.algorithm.best = 1;
    config.operators.crossover_probability = 1.0;
    config.operators.mutation_probability = 0.0;
    run_with_operators(&problem, &config, &operators).unwrap()
}

fn assert_generation(result: &RunResult, ids: &[u64], points: &[[f64; 2]]) {
    assert_eq!(result.evaluations, 7);
    assert_eq!(result.generations, 1);
    assert_eq!(result.history[1].population_ids, ids);
    let observed = result.history[1]
        .front
        .iter()
        .map(|point| [point[0], point[1]])
        .collect::<Vec<_>>();
    assert_eq!(observed, points);
}

#[test]
fn one_generation_survivors_match_hand_worked_multiobjective_fixtures() {
    // The offspring include three useful trade-offs and (3,10), which is
    // dominated by the initial (0,9). Expected membership is derived from the
    // stated selection rules, not from a recorded implementation output.
    let nsga = fixture(AlgorithmKind::Nsga2);
    assert_generation(&nsga, &[0, 2, 1], &[[0.0, 9.0], [8.0, 1.0], [3.0, 5.0]]);

    // SPEA2 keeps the spread-out nondominated set under iterative nearest
    // neighbor truncation; the closest interior pair is removed first.
    let spea = fixture(AlgorithmKind::Spea2);
    assert_generation(&spea, &[0, 1, 2], &[[0.0, 9.0], [3.0, 5.0], [8.0, 1.0]]);

    // MOGA and MOES replace the old generation with offspring, then rank by
    // nondominated count; the dominated fourth child is discarded.
    for kind in [AlgorithmKind::Moga, AlgorithmKind::Moes] {
        let result = fixture(kind);
        assert_generation(&result, &[3, 5, 7], &[[1.0, 8.0], [4.0, 4.0], [6.0, 3.0]]);
    }

    // `best` feeds previous elites into survivor selection instead of
    // appending them after the configured population has already been chosen.
    for kind in [AlgorithmKind::Moga, AlgorithmKind::Moes] {
        let result = fixture_with_old_best(kind);
        for generation in &result.history[1..] {
            assert_eq!(generation.population_ids.len(), PARENTS.len());
            assert_eq!(generation.population_ids, [0, 1, 2]);
        }
    }

    // With NGGA's neighborhood sample counts set to zero, all replacement
    // scores tie and the documented newest-ID worst tie break is exercised.
    let ngga = fixture(AlgorithmKind::Ngga);
    assert_generation(&ngga, &[0, 1, 9], &[[0.0, 9.0], [3.0, 5.0]]);
}

fn nsga_family_fixture(
    kind: AlgorithmKind,
    points: [[f64; 2]; 4],
    elitism: Option<bool>,
    best: usize,
    archived_add_best: bool,
    generations: usize,
) -> RunResult {
    let problem = TradeoffProblem::new();
    let operators = FixedChildren {
        layout: problem.layout.clone(),
        points,
    };
    let mut config = RunConfig {
        seed: 29,
        population: PARENTS.len(),
        offspring: CHILDREN.len(),
        generations,
        ..RunConfig::default()
    };
    config.algorithm.kind = kind;
    config.algorithm.elitism = elitism;
    config.algorithm.best = best;
    config.algorithm.archived_add_best = archived_add_best;
    config.operators.crossover_probability = 1.0;
    config.operators.mutation_probability = 0.0;
    run_with_operators(&problem, &config, &operators).unwrap()
}

fn sorted(ids: &[u64]) -> Vec<u64> {
    let mut ids = ids.to_vec();
    ids.sort_unstable();
    ids
}

#[test]
fn non_elitist_nsga2_and_spea2_select_from_offspring_only() {
    // Children 3, 5, 7 are mutually nondominated; child 9 = (3,10) is
    // dominated. Without elitism the old (0,9) etc. cannot survive.
    for kind in [AlgorithmKind::Nsga2, AlgorithmKind::Spea2] {
        let result = nsga_family_fixture(kind, CHILDREN, Some(false), 0, false, 1);
        assert_eq!(sorted(&result.history[1].population_ids), [3, 5, 7]);
        // The elitist default keeps the old extremes instead.
        let elitist = nsga_family_fixture(kind, CHILDREN, None, 0, false, 1);
        assert!(elitist.history[1].population_ids.contains(&0));
    }
}

#[test]
fn modern_best_feeds_old_front_into_non_elitist_nsga2_and_spea2_selection() {
    // Every child is dominated, so the old front introduced by `best` wins
    // survival and the configured population size is kept.
    for kind in [AlgorithmKind::Nsga2, AlgorithmKind::Spea2] {
        let result = nsga_family_fixture(kind, [[10.0, 10.0]; 4], Some(false), 1, false, 2);
        for generation in &result.history[1..] {
            assert_eq!(sorted(&generation.population_ids), [0, 1, 2]);
        }
        let without_best = nsga_family_fixture(kind, [[10.0, 10.0]; 4], Some(false), 0, false, 1);
        assert!(
            without_best.history[1]
                .population_ids
                .iter()
                .all(|&id| id >= PARENTS.len() as u64)
        );
    }
}

#[test]
fn archived_add_best_appends_old_front_and_best_after_nsga2_and_spea2_truncation() {
    for kind in [AlgorithmKind::Nsga2, AlgorithmKind::Spea2] {
        let result = nsga_family_fixture(kind, CHILDREN, None, 1, true, 2);
        let first = &result.history[1].population_ids;
        // Three truncated survivors, three unique old front copies, one best
        // copy by first objective; copies receive fresh IDs after children 3..=10.
        assert_eq!(first.len(), PARENTS.len() + 3 + 1, "{kind:?}");
        assert_eq!(&first[PARENTS.len()..], [11, 12, 13, 14]);
        assert!(first[..PARENTS.len()].iter().all(|&id| id < 11));
        // The enlarged population competes in the next generation and is
        // truncated back before the next append.
        assert!(result.history[2].population_ids.len() > PARENTS.len());
    }
}
