use mocs_rs::{
    algorithms::run,
    config::{AlgorithmKind, CrossoverKind, MatingKind, OperatorsConfig, RunConfig},
    evaluation::{EvaluationReport, Problem},
    genome::{Domain, GeneValue, Genome, GenomeLayout, Individual},
    objectives::{crowding, dominates, fronts, nondominated_indices},
    operators::{crossover, evaluate_dirty, mate, mutate},
    rng::Random,
};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

#[test]
fn dominance_fronts_and_crowding_match_hand_worked_minimization_fixture() {
    let points = vec![
        vec![1.0, 4.0], // front 0 boundary
        vec![2.0, 3.0], // front 0 interior
        vec![3.0, 2.0], // front 0 interior
        vec![4.0, 1.0], // front 0 boundary
        vec![3.0, 4.0], // front 1
        vec![4.0, 4.0], // front 2
        vec![2.0, 3.0], // duplicate remains nondominated
    ];
    assert!(!dominates(&points[1], &points[6]));
    assert_eq!(fronts(&points), vec![vec![0, 1, 2, 3, 6], vec![4], vec![5]]);
    assert_eq!(nondominated_indices(&points), vec![0, 1, 2, 3, 6]);

    let (crowd, boundary) = crowding(&points, &[0, 1, 2, 3, 6], false).unwrap();
    assert_eq!(boundary, vec![true, false, false, true, false]);
    // Duplicate ties are ordered by candidate index, so the two copies have
    // distinct neighbor gaps even though their objective vectors are equal.
    assert_eq!(crowd, vec![0.0, 2.0 / 3.0, 4.0 / 3.0, 0.0, 2.0 / 3.0]);
}

#[test]
fn constant_objective_does_not_create_arbitrary_crowding_boundaries() {
    let points = vec![vec![2.0, 7.0], vec![2.0, 7.0], vec![2.0, 7.0]];
    let (crowd, boundary) = crowding(&points, &[0, 1, 2], false).unwrap();
    assert_eq!(crowd, vec![0.0; 3]);
    assert_eq!(boundary, vec![false; 3]);
}

#[test]
fn rng_contract_known_answer_and_bernoulli_frequency() {
    let mut rng = Random::new(0);
    let first = rng.uniform();
    let second = rng.uniform();
    assert_eq!(first.to_bits(), 0x3fe6_b0be_ecf4_f347); // ChaCha8 / rand 0.9 fixture
    assert_eq!(second.to_bits(), 0x3fdd_d1a9_57ee_b630); // ChaCha8 / rand 0.9 fixture

    let mut rng = Random::new(0x5eed);
    for _ in 0..1000 {
        assert!(!rng.bernoulli(0.0));
        assert!(rng.bernoulli(1.0));
    }
    let trials = 100_000usize;
    let successes = (0..trials).filter(|_| rng.bernoulli(0.37)).count();
    let expected = trials as f64 * 0.37;
    let six_sigma = 6.0 * (trials as f64 * 0.37 * 0.63).sqrt();
    assert!((successes as f64 - expected).abs() <= six_sigma);
}

fn pair() -> (Individual, Individual) {
    let layout = Arc::new(
        GenomeLayout::new(vec![
            Domain::Real {
                min: -2.0,
                max: 2.0,
            },
            Domain::Integer { min: 0, max: 4 },
            Domain::Real { min: 1.0, max: 3.0 },
        ])
        .unwrap(),
    );
    let a = Genome::new(
        vec![
            GeneValue::Real(-1.5),
            GeneValue::Integer(1),
            GeneValue::Real(1.2),
        ],
        layout.clone(),
    )
    .unwrap();
    let b = Genome::new(
        vec![
            GeneValue::Real(1.5),
            GeneValue::Integer(3),
            GeneValue::Real(2.8),
        ],
        layout,
    )
    .unwrap();
    (Individual::new(10, a), Individual::new(11, b))
}

#[test]
fn crossover_and_adaptive_mutation_preserve_domain_and_state_invariants() {
    let (a, b) = pair();
    let mut cfg = OperatorsConfig {
        crossover_points: 2,
        crossover_alignment: 1,
        ..OperatorsConfig::default()
    };
    let mut rng = Random::new(91);
    for child_id in 0..200 {
        let (left, right) = crossover(&a, &b, &cfg, &mut rng, child_id * 2).unwrap();
        for child in [left, right] {
            child.genome().validate().unwrap();
            for (value, domain) in child
                .genome()
                .values
                .iter()
                .zip(&child.genome().layout.domains)
            {
                let (lo, hi) = domain.bounds();
                assert!((lo..=hi).contains(&value.numeric()));
            }
        }
    }

    cfg.mutation = mocs_rs::config::MutationKind::Adaptive;
    cfg.per_locus_probability = Some(1.0);
    for seed in 0..500 {
        let mut child = a.clone();
        let mut rng = Random::new(seed);
        mutate(&mut child, &cfg, &mut rng).unwrap();
        let g = child.genome();
        g.validate().unwrap();
        assert!(g.deviation.is_finite() && g.deviation > 0.0);
        assert!(g.deviations.iter().all(|d| d.is_finite() && *d >= 0.0));
        for (value, domain) in g.values.iter().zip(&g.layout.domains) {
            let (lo, hi) = domain.bounds();
            assert!((lo..=hi).contains(&value.numeric()));
        }
    }
}

#[test]
fn variable_length_crossover_and_sbx_preserve_layout_and_domains() {
    let (one, two) = pair();
    let width = one.genome().layout.domains.len();
    let many_values = (0..5)
        .flat_map(|_| two.genome().values.iter().copied())
        .collect::<Vec<_>>();
    let many = Individual::new(
        12,
        Genome::new(many_values, one.genome().layout.clone()).unwrap(),
    );
    let mut cfg = OperatorsConfig {
        crossover: CrossoverKind::VariableLength,
        crossover_points: 5,
        crossover_alignment: width,
        ..OperatorsConfig::default()
    };
    let mut rng = Random::new(0xabc);
    for id in 0..1000 {
        let (a, b) = crossover(&one, &many, &cfg, &mut rng, id * 2).unwrap();
        for child in [a, b] {
            assert!(child.genome().values.len().is_multiple_of(width));
            assert_eq!(child.genome().deviations.len(), child.genome().values.len());
            child.genome().validate().unwrap();
        }
    }

    let sbx_layout =
        Arc::new(GenomeLayout::new(vec![Domain::Real { min: 0.0, max: 2.0 }]).unwrap());
    let sbx_parent = |id, value, deviation| {
        let mut genome = Genome::new(vec![GeneValue::Real(value)], sbx_layout.clone()).unwrap();
        genome.deviation = deviation;
        genome.deviations[0] = deviation;
        Individual::new(id, genome)
    };
    let sbx_one = sbx_parent(20, 0.2, 0.1);
    let sbx_two = sbx_parent(21, 1.8, 1.0);
    cfg.crossover = CrossoverKind::SimulatedBinary;
    cfg.crossover_alignment = 1;
    for id in 0..500 {
        let mut rng = Random::new(id);
        let (a, b) = crossover(&sbx_one, &sbx_two, &cfg, &mut rng, id * 2).unwrap();
        for child in [a, b] {
            child.genome().validate().unwrap();
            assert!(child.genome().deviation.is_finite() && child.genome().deviation > 0.0);
            assert!(child.genome().deviations[0].is_finite());
            for (value, domain) in child
                .genome()
                .values
                .iter()
                .zip(&child.genome().layout.domains)
            {
                let (lo, hi) = domain.bounds();
                assert!((lo..=hi).contains(&value.numeric()));
            }
        }
    }
}

struct GateProblem(AtomicUsize);
impl Problem for GateProblem {
    fn initialize(&self, _: &mut Random) -> mocs_rs::Result<Genome> {
        Ok(pair().0.genome().clone())
    }
    fn evaluate(&self, _: &Genome) -> mocs_rs::Result<EvaluationReport> {
        Ok(EvaluationReport {
            objectives: vec![1.0],
            loss: 1.0,
            coverage: 1.0,
            conditional_loss: 1.0,
            ..EvaluationReport::default()
        })
    }
    fn prune(&self, _: &mut Genome, _: &EvaluationReport) -> mocs_rs::Result<bool> {
        self.0.fetch_add(1, Ordering::Relaxed);
        Ok(false)
    }
}

#[test]
fn prune_gate_and_tournament_frequencies_follow_the_declared_probabilities() {
    let make_items = |count: usize| {
        let genome = pair().0.genome().clone();
        (0..count)
            .map(|id| Individual::new(id as u64, genome.clone()))
            .collect::<Vec<_>>()
    };
    let count = 100_000usize;
    let mut items = make_items(count);
    let problem = GateProblem(AtomicUsize::new(0));
    let mut rng = Random::new(0x51a7);
    let mut evaluations = 0;
    evaluate_dirty(&problem, &mut items, &mut rng, 0.0, &mut evaluations).unwrap();
    assert_eq!(problem.0.load(Ordering::Relaxed), 0);
    assert_eq!(evaluations, count);

    let mut items = make_items(count);
    let problem = GateProblem(AtomicUsize::new(0));
    let mut rng = Random::new(0x51a7);
    let mut evaluations = 0;
    evaluate_dirty(&problem, &mut items, &mut rng, 1.0, &mut evaluations).unwrap();
    assert_eq!(problem.0.load(Ordering::Relaxed), count);
    assert_eq!(evaluations, count);

    let mut items = make_items(count);
    let problem = GateProblem(AtomicUsize::new(0));
    let mut rng = Random::new(0x51a7);
    let mut evaluations = 0;
    evaluate_dirty(&problem, &mut items, &mut rng, 0.37, &mut evaluations).unwrap();
    let observed = problem.0.load(Ordering::Relaxed);
    let expected = count as f64 * 0.37;
    let six_sigma = 6.0 * (count as f64 * 0.37 * 0.63).sqrt();
    assert!((observed as f64 - expected).abs() <= six_sigma);

    let mut cfg = OperatorsConfig {
        mating: MatingKind::Tournament,
        mating_tournament_size: 2,
        mating_winners: 1,
        fitness_tournament: true,
        ..OperatorsConfig::default()
    };
    cfg.mating_tournament_size = 2;
    let (a, b) = pair();
    let population = vec![a, b];
    let mut rng = Random::new(0x7a11);
    let selected = (0..count)
        .filter(|_| mate(&population, &[0.0, 1.0], &cfg, &mut rng).unwrap().0 == 0)
        .count();
    let expected = count as f64 * 0.75;
    let six_sigma = 6.0 * (count as f64 * 0.75 * 0.25).sqrt();
    assert!((selected as f64 - expected).abs() <= six_sigma);

    let mut rng = Random::new(0xb4a1);
    let crossover_attempts = (0..count).filter(|_| rng.bernoulli(0.8)).count();
    let expected = count as f64 * 0.8;
    let six_sigma = 6.0 * (count as f64 * 0.8 * 0.2).sqrt();
    assert!((crossover_attempts as f64 - expected).abs() <= six_sigma);

    let mut rng = Random::new(0x6d75);
    let mutation_attempts = (0..count).filter(|_| rng.bernoulli(0.35)).count();
    let expected = count as f64 * 0.35;
    let six_sigma = 6.0 * (count as f64 * 0.35 * 0.65).sqrt();
    assert!((mutation_attempts as f64 - expected).abs() <= six_sigma);
}

struct FlatProblem {
    layout: Arc<GenomeLayout>,
}
impl Problem for FlatProblem {
    fn initialize(&self, _: &mut Random) -> mocs_rs::Result<Genome> {
        Genome::new(vec![GeneValue::Real(0.5)], self.layout.clone())
    }
    fn evaluate(&self, _: &Genome) -> mocs_rs::Result<EvaluationReport> {
        Ok(EvaluationReport {
            objectives: vec![1.0],
            loss: 1.0,
            coverage: 1.0,
            conditional_loss: 1.0,
            ..EvaluationReport::default()
        })
    }
}

#[test]
fn one_generation_constant_fixture_keeps_oldest_equal_candidates_without_extra_evaluations() {
    let problem = FlatProblem {
        layout: Arc::new(GenomeLayout::new(vec![Domain::Real { min: 0.0, max: 1.0 }]).unwrap()),
    };
    for kind in [
        AlgorithmKind::Moga,
        AlgorithmKind::Moes,
        AlgorithmKind::Ngga,
        AlgorithmKind::Nsga2,
        AlgorithmKind::Spea2,
    ] {
        let config = RunConfig {
            seed: 0,
            population: 4,
            offspring: 4,
            generations: 1,
            ..RunConfig::default()
        };
        let mut config = config;
        config.algorithm.kind = kind;
        config.operators.crossover_probability = 0.0;
        config.operators.mutation_probability = 0.0;
        config.training.prune_probability = 0.0;
        let result = run(&problem, &config).unwrap();
        assert_eq!(result.history.len(), 2, "{kind:?}");
        let expected = match kind {
            AlgorithmKind::Moga | AlgorithmKind::Moes => vec![4, 5, 6, 7],
            AlgorithmKind::Ngga => vec![0, 7, 5, 6],
            AlgorithmKind::Nsga2 | AlgorithmKind::Spea2 => vec![0, 1, 2, 3],
        };
        assert_eq!(result.history[1].population_ids, expected, "{kind:?}");
        assert_eq!(result.evaluations, 4, "{kind:?}");
    }
}
