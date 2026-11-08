//! The [`Problem`] trait implemented by optimization and learning problems, and
//! the [`EvaluationReport`] each evaluation returns.

use serde::{Deserialize, Serialize};

use crate::{Result, genome::Genome, rng::Random};

/// Result of evaluating one genome. Selection uses only [`objectives`](Self::objectives),
/// all of which are minimized; the other fields are diagnostics for reports.
/// Numeric problems set `loss` and `conditional_loss` to the first objective and
/// `coverage` to 1.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct EvaluationReport {
    /// Objective vector, all minimized. Its length must be the same for every candidate in a run.
    pub objectives: Vec<f64>,
    /// Training-style loss: classification error rate with abstentions counted as errors, or regression mean squared error.
    pub loss: f64,
    /// Fraction of rows that received a prediction (did not abstain), in `[0, 1]`.
    pub coverage: f64,
    /// Loss over covered rows only; 1 when no row is covered.
    pub conditional_loss: f64,
    /// Mean over used rules of one minus the rule's specificity; 1 for instance models or when no rule is used.
    pub generality: f64,
    /// Model size: rule count divided by the training-row count.
    pub size: f64,
    /// Number of rules represented by the evaluated genome.
    #[serde(default)]
    pub rule_count: usize,
    /// Fraction of rules that did not match any training example.
    #[serde(default)]
    pub unused_rule_fraction: f64,
    /// Per-rule count of rows the rule decided (or, for regression, contributed to).
    pub rule_usage: Vec<usize>,
    /// Rows for which the model made no prediction.
    pub abstentions: usize,
    /// Regression rows that received the training-target mean because no rule applied.
    pub fallbacks: usize,
    /// Rows with a numeric predictor outside its fitted training range.
    pub out_of_range: usize,
    /// Number of rows scored.
    pub rows: usize,
}
impl EvaluationReport {
    /// Check that objectives are present and finite, all metrics are finite, and
    /// `coverage` and `unused_rule_fraction` lie in `[0, 1]`.
    pub fn validate(&self) -> Result<()> {
        anyhow::ensure!(
            !self.objectives.is_empty(),
            "evaluation must contain at least one objective"
        );
        for (i, v) in self.objectives.iter().enumerate() {
            anyhow::ensure!(v.is_finite(), "objective {i} is not finite");
        }
        for (name, v) in [
            ("loss", self.loss),
            ("coverage", self.coverage),
            ("conditional_loss", self.conditional_loss),
            ("generality", self.generality),
            ("size", self.size),
            ("unused_rule_fraction", self.unused_rule_fraction),
        ] {
            anyhow::ensure!(v.is_finite(), "evaluation {name} is not finite");
        }
        anyhow::ensure!(
            (0.0..=1.0).contains(&self.coverage),
            "coverage must be in [0,1]"
        );
        anyhow::ensure!(
            (0.0..=1.0).contains(&self.unused_rule_fraction),
            "unused_rule_fraction must be in [0,1]"
        );
        Ok(())
    }
}

/// An optimization problem: creates random genomes and evaluates them.
///
/// Implementations must be deterministic for a given genome so that runs are
/// reproducible; `Sync` is required because evaluation may run in parallel.
pub trait Problem: Sync {
    /// Create a random initial genome using the run's RNG stream.
    fn initialize(&self, rng: &mut Random) -> Result<Genome>;
    /// Evaluate a genome. Returning an error aborts the run.
    fn evaluate(&self, genome: &Genome) -> Result<EvaluationReport>;
    /// Advisory estimate of primitive evaluation work for scheduling. `None`
    /// means the implementation has no useful estimate and uses the normal
    /// parallel dispatch policy. Estimates must not influence scientific
    /// results; built-ins use deterministic, input-derived units.
    fn estimated_evaluation_work(&self, _genome: &Genome) -> Option<usize> {
        None
    }
    /// Optionally simplify `genome` using its evaluation `report` (e.g. drop unused
    /// rules). Return `true` if it changed, so it is re-evaluated. The default does nothing.
    fn prune(&self, _genome: &mut Genome, _report: &EvaluationReport) -> Result<bool> {
        Ok(false)
    }
}
