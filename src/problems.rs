//! Built-in numeric benchmark problems (SF2, KUR, OKK, ZDT6), all two-objective
//! and minimized.

use crate::{
    Result,
    config::{OperatorsConfig, ProblemConfig, ProblemKind},
    evaluation::{EvaluationReport, Problem},
    genome::{Domain, GeneValue, Genome, GenomeLayout},
    rng::Random,
};
use std::sync::Arc;

/// A two-objective numeric benchmark selected by [`ProblemKind`].
///
/// Defaults per kind (dimensions, bounds): SF2 (1, `[-6, 6]`), KUR (3, `[-5, 5]`),
/// OKK (7 fixed per-variable bounds; `min`/`max` ignored), ZDT6 (10, `[0, 1]`).
#[derive(Debug, Clone)]
pub struct NumericProblem {
    kind: ProblemKind,
    domains: Vec<Domain>,
}
impl NumericProblem {
    /// Build the problem from [`ProblemConfig`], applying the per-kind defaults.
    /// Errors on nonfinite or empty bounds and on a dimension count the kind does not
    /// support (SF2 exactly 1, OKK exactly 7, KUR and ZDT6 at least 2).
    pub fn new(config: &ProblemConfig) -> Result<Self> {
        let default = match config.kind {
            ProblemKind::Sf2 => (1, -6.0, 6.0),
            ProblemKind::Kur => (3, -5.0, 5.0),
            ProblemKind::Okk => (7, 0.0, 1.0),
            ProblemKind::Zdt6 => (10, 0.0, 1.0),
        };
        let n = config.dimensions.unwrap_or(default.0);
        anyhow::ensure!(n > 0, "numeric problem dimension must be positive");
        let domains = if config.kind == ProblemKind::Okk {
            vec![
                Domain::Real {
                    min: 10.0,
                    max: 150.0,
                },
                Domain::Real {
                    min: 10.0,
                    max: 150.0,
                },
                Domain::Real {
                    min: 100.0,
                    max: 200.0,
                },
                Domain::Real {
                    min: 0.0,
                    max: 50.0,
                },
                Domain::Real {
                    min: 10.0,
                    max: 150.0,
                },
                Domain::Real {
                    min: 100.0,
                    max: 300.0,
                },
                Domain::Real {
                    min: 1.0,
                    #[allow(clippy::approx_constant)]
                    max: 3.14,
                },
            ]
        } else {
            let min = config.min.unwrap_or(default.1);
            let max = config.max.unwrap_or(default.2);
            anyhow::ensure!(
                min.is_finite() && max.is_finite() && min < max,
                "numeric bounds must be finite and min < max"
            );
            anyhow::ensure!(
                (max - min).is_finite(),
                "numeric bounds must have a finite span"
            );
            vec![Domain::Real { min, max }; n]
        };
        if config.kind == ProblemKind::Kur {
            anyhow::ensure!(n >= 2, "KUR requires at least two dimensions");
        }
        if config.kind == ProblemKind::Zdt6 {
            anyhow::ensure!(n >= 2, "ZDT6 requires at least two dimensions");
        }
        if config.kind == ProblemKind::Okk {
            anyhow::ensure!(n == 7, "OKK has exactly seven design variables");
        }
        if config.kind == ProblemKind::Sf2 {
            anyhow::ensure!(n == 1, "SF2 has exactly one design variable");
        }
        Ok(Self {
            kind: config.kind,
            domains,
        })
    }
    /// The configured benchmark.
    pub fn kind(&self) -> ProblemKind {
        self.kind
    }
    /// Check that the configured crossover can run on this problem's fixed-length genome.
    pub fn validate_operators(&self, operators: &OperatorsConfig) -> Result<()> {
        operators.validate_for_layout(&GenomeLayout::new(self.domains.clone())?, false)
    }
    fn values(&self, g: &Genome) -> Result<Vec<f64>> {
        g.validate()?;
        anyhow::ensure!(
            g.values.len() == self.domains.len(),
            "genome has {} values; {} required for {:?}",
            g.values.len(),
            self.domains.len(),
            self.kind
        );
        let xs: Vec<f64> = g.values.iter().map(|v| v.numeric()).collect();
        anyhow::ensure!(
            xs.iter().all(|x| x.is_finite()),
            "genome contains nonfinite values"
        );
        Ok(xs)
    }
    fn report(&self, objectives: Vec<f64>) -> Result<EvaluationReport> {
        anyhow::ensure!(
            objectives.iter().all(|v| v.is_finite()),
            "problem produced a nonfinite objective"
        );
        let r = EvaluationReport {
            loss: objectives[0],
            coverage: 1.0,
            conditional_loss: objectives[0],
            objectives,
            ..EvaluationReport::default()
        };
        r.validate()?;
        Ok(r)
    }
}
impl Problem for NumericProblem {
    fn initialize(&self, rng: &mut Random) -> Result<Genome> {
        let values = self
            .domains
            .iter()
            .map(|d| {
                let (a, b) = d.bounds();
                GeneValue::Real(rng.range(a, b))
            })
            .collect();
        let mut g = Genome::new(values, Arc::new(GenomeLayout::new(self.domains.clone())?))?;
        g.deviation = 1.0 / g.values.len() as f64;
        g.deviations = vec![g.deviation; g.values.len()];
        Ok(g)
    }
    fn evaluate(&self, g: &Genome) -> Result<EvaluationReport> {
        let x = self.values(g)?;
        let f = match self.kind {
            ProblemKind::Sf2 => vec![x[0] * x[0], (x[0] - 2.0).powi(2)],
            ProblemKind::Kur => {
                let mut f1 = 0.0;
                for pair in x.windows(2) {
                    f1 += -10.0 * (-0.2 * (pair[0] * pair[0] + pair[1] * pair[1]).sqrt()).exp();
                }
                let f2 = x
                    .iter()
                    .map(|v| v.abs().powf(0.8) + 5.0 * v.sin().powi(3))
                    .sum();
                vec![f1, f2]
            }
            ProblemKind::Zdt6 => {
                let n = x.len();
                let f1 =
                    1.0 - (-4.0 * x[0]).exp() * (6.0 * std::f64::consts::PI * x[0]).sin().powi(6);
                let g =
                    1.0 + (n - 1) as f64 * (x[1..].iter().sum::<f64>() / (n - 1) as f64).powf(0.25);
                vec![f1, g * (1.0 - (f1 / g).powi(2))]
            }
            ProblemKind::Okk => {
                let (a, b, c, e, f, l, d) = (x[0], x[1], x[2], x[3], x[4], x[5], x[6]);
                if !okk_constraint(a, b, c, e, f, l, d) {
                    vec![1000.0, 1000.0]
                } else {
                    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
                    for i in 0..1000 {
                        let z = 100.0 * (i as f64) / 1000.0;
                        let y = okk_function(a, b, c, e, l, z).unwrap_or(f64::NAN);
                        if !y.is_finite() {
                            lo = f64::NAN;
                            break;
                        }
                        lo = lo.min(y);
                        hi = hi.max(y);
                    }
                    if !lo.is_finite() || !hi.is_finite() {
                        vec![1000.0, 1000.0]
                    } else {
                        let f1 = hi - lo;
                        let f2 = 100.0 / lo.min(hi);
                        if f1 > 100.0 || f2 < 0.8 {
                            vec![1000.0, 1000.0]
                        } else {
                            vec![f1, f2]
                        }
                    }
                }
            }
        };
        self.report(f)
    }

    fn estimated_evaluation_work(&self, _genome: &Genome) -> Option<usize> {
        let dimensions = self.domains.len();
        Some(match self.kind {
            ProblemKind::Sf2 => 8,
            ProblemKind::Kur => dimensions.saturating_mul(32),
            ProblemKind::Zdt6 => dimensions.saturating_add(32),
            // Each feasible OKK evaluation scans 1,000 samples and computes
            // trigonometric and distance terms; infeasible candidates still
            // pay validation and constraint-check costs.
            ProblemKind::Okk => 32_000,
        })
    }
}
fn angle_cos(arg: f64) -> Option<f64> {
    if arg.is_finite() && (-1.0 - 1e-12..=1.0 + 1e-12).contains(&arg) {
        Some(arg.clamp(-1.0, 1.0).acos())
    } else {
        None
    }
}
fn okk_function(a: f64, b: f64, c: f64, e: f64, l: f64, z: f64) -> Option<f64> {
    let dx = l - z;
    let phi = (e / dx).atan();
    let g = dx.hypot(e);
    let alpha = angle_cos((a * a + g * g - b * b) / (2.0 * a * g))? + phi;
    let beta = angle_cos((b * b + g * g - a * a) / (2.0 * b * g))? - phi;
    Some((100.0 * b * (alpha + beta).sin()) / (2.0 * c * alpha.cos()))
}
fn okk_constraint(a: f64, b: f64, c: f64, e: f64, f: f64, l: f64, d: f64) -> bool {
    let phi = (e / l).atan();
    let g = l.hypot(e);
    let Some(beta0) = angle_cos((b * b + g * g - a * a) / (2.0 * b * g)).map(|v| v - phi) else {
        return false;
    };
    let yz0 = 2.0 * (e + f + c * (beta0 + d).sin());
    let dx = l - 100.0;
    let phi = (e / dx).atan();
    let g = dx.hypot(e);
    let Some(beta1) = angle_cos((b * b + g * g - a * a) / (2.0 * b * g)).map(|v| v - phi) else {
        return false;
    };
    let yzmax = 2.0 * (e + f + c * (beta1 + d).sin());
    50.0 - yzmax >= 0.0
        && yzmax >= 0.0
        && yz0 - 100.0 >= 0.0
        && 150.0 - yz0 >= 0.0
        && (a + b).powi(2) - l * l - e * e >= 0.0
        && l - 100.0 >= 0.0
        && (l - 100.0).powi(2) + (a - e).powi(2) - b * b >= 0.0
}
