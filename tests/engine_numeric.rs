use mocs_rs::{
    config::{ProblemConfig, ProblemKind, RunConfig},
    evaluation::Problem,
    genome::{Domain, GeneValue, Genome, GenomeLayout},
    problems::NumericProblem,
    rng::Random,
};
use std::sync::Arc;

fn problem(kind: ProblemKind, dimensions: Option<usize>) -> NumericProblem {
    NumericProblem::new(&ProblemConfig {
        kind,
        dimensions,
        min: None,
        max: None,
    })
    .unwrap()
}
fn genome(values: &[f64], domains: &[Domain]) -> Genome {
    Genome::new(
        values.iter().copied().map(GeneValue::Real).collect(),
        Arc::new(GenomeLayout::new(domains.to_vec()).unwrap()),
    )
    .unwrap()
}

#[test]
fn sf2_is_the_source_two_objective_formula_and_allows_one_dimension() {
    let p = problem(ProblemKind::Sf2, Some(1));
    let g = genome(
        &[3.0],
        &[Domain::Real {
            min: -6.0,
            max: 6.0,
        }],
    );
    let r = p.evaluate(&g).unwrap();
    assert_eq!(r.objectives, vec![9.0, 1.0]);
}
#[test]
fn kur_uses_adjacent_pairs_and_all_coordinates() {
    let p = problem(ProblemKind::Kur, None);
    let g = genome(
        &[0.0, 0.0, 0.0],
        &vec![
            Domain::Real {
                min: -5.0,
                max: 5.0
            };
            3
        ],
    );
    let r = p.evaluate(&g).unwrap();
    assert_eq!(r.objectives, vec![-20.0, 0.0]);
}
#[test]
fn zdt6_at_zero_has_finite_expected_objectives() {
    let p = problem(ProblemKind::Zdt6, None);
    let g = genome(&[0.0; 10], &vec![Domain::Real { min: 0.0, max: 1.0 }; 10]);
    let r = p.evaluate(&g).unwrap();
    assert_eq!(r.objectives, vec![1.0, 0.0]);
}
#[test]
fn okk_infeasible_geometry_gets_the_source_penalty() {
    let p = problem(ProblemKind::Okk, None);
    let ds = [
        (10.0, 150.0),
        (10.0, 150.0),
        (100.0, 200.0),
        (0.0, 50.0),
        (10.0, 150.0),
        (100.0, 300.0),
        (1.0, std::f64::consts::PI),
    ]
    .map(|(min, max)| Domain::Real { min, max });
    let g = genome(&[10.0, 10.0, 100.0, 0.0, 10.0, 100.0, 1.0], &ds);
    let r = p.evaluate(&g).unwrap();
    assert_eq!(r.objectives, vec![1000.0, 1000.0]);
}

#[test]
fn okk_feasible_geometry_matches_analytic_fixture() {
    let p = problem(ProblemKind::Okk, None);
    // The historical domain endpoint is decimal 3.14.
    #[allow(clippy::approx_constant)]
    const OKK_D_MAX: f64 = 3.14;
    let ds = [
        (10.0, 150.0),
        (10.0, 150.0),
        (100.0, 200.0),
        (0.0, 50.0),
        (10.0, 150.0),
        (100.0, 300.0),
        (1.0, OKK_D_MAX),
    ]
    .map(|(min, max)| Domain::Real { min, max });
    // With A=B=100 and E=0, beta is acos((150-z)/200), so the force is
    // (100*B/C)*sin(beta) and increases across the original z scan. The
    // original C++ constraints give yz0=115.30645587661496 in [100,150],
    // yzmax=3.017166689202952 in [0,50], and positive remaining margins
    // 17500, 50, and 2500. The analytic 1,000-point oracle gives the
    // objectives below. Incremental C++ z stepping differs by about 1e-13
    // and is compared separately by scripts/kernel_parity.py.
    let g = genome(&[100.0, 100.0, 150.0, 0.0, 140.0, 150.0, 3.0], &ds);
    let objectives = p.evaluate(&g).unwrap().objectives;
    let expected = [20.44525144109236, 2.267786838055363];
    for (actual, expected) in objectives.iter().zip(expected) {
        assert!(
            (actual - expected).abs() <= 1e-9,
            "actual {actual:.17}, expected {expected:.17}"
        );
    }
}
#[test]
fn numeric_initializers_use_domain_and_nonzero_mutation_state() {
    let p = problem(ProblemKind::Zdt6, None);
    let mut rng = Random::new(0);
    let g = p.initialize(&mut rng).unwrap();
    assert_eq!(g.values.len(), 10);
    assert!(g.values.iter().all(|v| (0.0..=1.0).contains(&v.numeric())));
    assert!(g.deviation > 0.0 && g.deviations.iter().all(|d| *d > 0.0));
}

#[test]
fn real_domains_reject_overflowing_spans_at_domain_problem_and_config_boundaries() {
    let domain = Domain::Real {
        min: -1.0e308,
        max: 1.0e308,
    };
    assert!(domain.validate().unwrap_err().to_string().contains("span"));
    assert!(
        GenomeLayout::new(vec![domain])
            .unwrap_err()
            .to_string()
            .contains("span")
    );

    let problem = ProblemConfig {
        kind: ProblemKind::Sf2,
        dimensions: Some(1),
        min: Some(-1.0e308),
        max: Some(1.0e308),
    };
    assert!(
        NumericProblem::new(&problem)
            .unwrap_err()
            .to_string()
            .contains("span")
    );

    let mut config = RunConfig::default();
    config.problem.min = Some(-1.0e308);
    config.problem.max = Some(1.0e308);
    assert!(config.validate().unwrap_err().to_string().contains("span"));
}

#[test]
fn adaptive_epsilon_cannot_exceed_its_clamp_upper_bound() {
    let mut config = RunConfig::default();
    config.operators.adaptive_epsilon = 2.0;
    assert!(
        config
            .validate()
            .unwrap_err()
            .to_string()
            .contains("adaptive_epsilon must be <= 1")
    );
}
