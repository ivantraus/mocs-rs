use mocs_rs::{
    config::{ProblemConfig, ProblemKind},
    evaluation::Problem,
    genome::{Domain, GeneValue, Genome, GenomeLayout},
    problems::NumericProblem,
};
use std::sync::Arc;

fn evaluate(kind: ProblemKind, bounds: (f64, f64), values: &[f64]) -> Vec<f64> {
    let config = ProblemConfig {
        kind,
        dimensions: Some(values.len()),
        min: Some(bounds.0),
        max: Some(bounds.1),
    };
    let problem = NumericProblem::new(&config).unwrap();
    let layout = Arc::new(
        GenomeLayout::new(vec![
            Domain::Real {
                min: bounds.0,
                max: bounds.1,
            };
            values.len()
        ])
        .unwrap(),
    );
    let genome = Genome::new(
        values.iter().copied().map(GeneValue::Real).collect(),
        layout,
    )
    .unwrap();
    problem.evaluate(&genome).unwrap().objectives
}

fn evaluate_okk(values: &[f64]) -> Vec<f64> {
    let config = ProblemConfig {
        kind: ProblemKind::Okk,
        dimensions: None,
        min: None,
        max: None,
    };
    let problem = NumericProblem::new(&config).unwrap();
    // The historical OKK domain endpoint is decimal 3.14.
    #[allow(clippy::approx_constant)]
    const OKK_D_MAX: f64 = 3.14;
    let domains = [
        (10.0, 150.0),
        (10.0, 150.0),
        (100.0, 200.0),
        (0.0, 50.0),
        (10.0, 150.0),
        (100.0, 300.0),
        (1.0, OKK_D_MAX),
    ]
    .map(|(min, max)| Domain::Real { min, max });
    let genome = Genome::new(
        values.iter().copied().map(GeneValue::Real).collect(),
        Arc::new(GenomeLayout::new(domains.to_vec()).unwrap()),
    )
    .unwrap();
    problem.evaluate(&genome).unwrap().objectives
}

#[test]
fn emit_original_kernel_parity_fixtures() {
    if std::env::var_os("MOCS_KERNEL_PARITY").is_none() {
        return;
    }
    let fixtures = [
        ("sf2", evaluate(ProblemKind::Sf2, (-6.0, 6.0), &[0.25])),
        (
            "kur",
            evaluate(ProblemKind::Kur, (-5.0, 5.0), &[-1.0, 0.5, 2.0]),
        ),
        (
            "okk",
            evaluate_okk(&[100.0, 100.0, 150.0, 0.0, 140.0, 150.0, 3.0]),
        ),
        (
            "zdt6",
            evaluate(
                ProblemKind::Zdt6,
                (0.0, 1.0),
                &[0.25, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            ),
        ),
    ];
    for (name, objectives) in fixtures {
        println!(
            "RUST_PARITY {name} {:.17e} {:.17e}",
            objectives[0], objectives[1]
        );
    }
}
