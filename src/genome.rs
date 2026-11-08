//! Genome representation: typed gene values, per-locus domains, and the
//! [`Individual`] wrapper that pairs a genome with an ID and cached evaluation.

use crate::{Result, evaluation::EvaluationReport};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// A single gene value.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum GeneValue {
    /// Integer gene (nominal categories, class labels and integer loci).
    Integer(i64),
    /// Real-valued gene.
    Real(f64),
}
impl GeneValue {
    /// Value as `f64`.
    pub fn numeric(self) -> f64 {
        match self {
            Self::Integer(v) => v as f64,
            Self::Real(v) => v,
        }
    }
}
/// Inclusive value range of one locus.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Domain {
    /// Integers in `min..=max`.
    Integer {
        /// Smallest allowed value.
        min: i64,
        /// Largest allowed value.
        max: i64,
    },
    /// Finite reals in `[min, max]`.
    Real {
        /// Smallest allowed value.
        min: f64,
        /// Largest allowed value.
        max: f64,
    },
}
impl Domain {
    /// Clamp `value` into this domain. Integer values are converted to reals for a
    /// real domain; errors on NaN, a real value for an integer domain, or an invalid domain.
    pub fn clamp(&self, value: GeneValue) -> Result<GeneValue> {
        self.validate()?;
        match (self, value) {
            (Self::Integer { min, max }, GeneValue::Integer(v)) => {
                Ok(GeneValue::Integer(v.clamp(*min, *max)))
            }
            (Self::Real { min, max }, GeneValue::Real(v)) if !v.is_nan() => {
                Ok(GeneValue::Real(v.clamp(*min, *max)))
            }
            (Self::Real { min, max }, GeneValue::Integer(v)) => {
                Ok(GeneValue::Real((v as f64).clamp(*min, *max)))
            }
            (_, GeneValue::Real(v)) if v.is_nan() => anyhow::bail!("gene value must not be NaN"),
            _ => anyhow::bail!("gene value type does not match domain"),
        }
    }
    /// `(min, max)` as `f64`.
    pub fn bounds(&self) -> (f64, f64) {
        match self {
            Self::Integer { min, max } => (*min as f64, *max as f64),
            Self::Real { min, max } => (*min, *max),
        }
    }
    /// Check that `min <= max` and, for real domains, that bounds and span are finite.
    pub fn validate(&self) -> Result<()> {
        match self {
            Self::Integer { min, max } => {
                anyhow::ensure!(min <= max, "integer domain min exceeds max")
            }
            Self::Real { min, max } => {
                anyhow::ensure!(
                    min.is_finite() && max.is_finite() && min <= max,
                    "real domain must be finite with min <= max"
                );
                anyhow::ensure!((*max - *min).is_finite(), "real domain span must be finite");
            }
        };
        Ok(())
    }
}
/// Domains of one repeated block of loci. A genome is one or more consecutive
/// blocks, so locus `i` uses `domains[i % domains.len()]`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GenomeLayout {
    /// Domain of each locus in the block.
    pub domains: Vec<Domain>,
}
impl GenomeLayout {
    /// Create a layout; errors if `domains` is empty or any domain is invalid.
    pub fn new(domains: Vec<Domain>) -> Result<Self> {
        anyhow::ensure!(!domains.is_empty(), "genome layout cannot be empty");
        for d in &domains {
            d.validate()?;
        }
        Ok(Self { domains })
    }
}
/// A variable-length sequence of gene values with self-adaptive mutation state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Genome {
    /// Gene values; the length is a positive multiple of the layout width.
    pub values: Vec<GeneValue>,
    /// Per-locus mutation rates/step sizes (used by uniform and normal mutation).
    pub deviations: Vec<f64>,
    /// Genome-wide mutation rate (used by polynomial and adaptive mutation).
    pub deviation: f64,
    /// Shared block layout giving each locus its domain.
    pub layout: Arc<GenomeLayout>,
}
impl Genome {
    /// Create a genome with zero deviations; errors if it fails [`Genome::validate`].
    pub fn new(values: Vec<GeneValue>, layout: Arc<GenomeLayout>) -> Result<Self> {
        let n = values.len();
        let genome = Self {
            values,
            deviations: vec![0.0; n],
            deviation: 0.0,
            layout,
        };
        genome.validate()?;
        Ok(genome)
    }
    /// Check shape (nonempty, whole blocks, matching deviation count), that every
    /// value lies in its domain and that deviations are finite and nonnegative.
    pub fn validate(&self) -> Result<()> {
        anyhow::ensure!(!self.values.is_empty(), "genome cannot be empty");
        anyhow::ensure!(
            !self.layout.domains.is_empty(),
            "genome layout cannot be empty"
        );
        for (i, domain) in self.layout.domains.iter().enumerate() {
            domain
                .validate()
                .map_err(|e| anyhow::anyhow!("domain {i}: {e}"))?;
        }
        anyhow::ensure!(
            !self.layout.domains.is_empty()
                && self.values.len().is_multiple_of(self.layout.domains.len()),
            "genome length must be a positive multiple of layout width"
        );
        anyhow::ensure!(
            self.deviations.len() == self.values.len(),
            "per-locus deviations must match genome length"
        );
        anyhow::ensure!(
            self.deviation.is_finite() && self.deviation >= 0.0,
            "genome deviation must be finite and nonnegative"
        );
        for (i, v) in self.values.iter().enumerate() {
            let d = &self.layout.domains[i % self.layout.domains.len()];
            let c = d.clamp(*v).map_err(|e| anyhow::anyhow!("gene {i}: {e}"))?;
            anyhow::ensure!(c == *v, "gene {i} is outside its domain");
        }
        for (i, v) in self.deviations.iter().enumerate() {
            anyhow::ensure!(
                v.is_finite() && *v >= 0.0,
                "deviation {i} must be finite and nonnegative"
            );
        }
        Ok(())
    }
    /// Set gene `index`, clamping the value into its domain. Errors if the index is
    /// out of range or the value type does not fit the domain.
    pub fn set(&mut self, index: usize, value: GeneValue) -> Result<()> {
        anyhow::ensure!(
            index < self.values.len(),
            "gene index {index} is out of range"
        );
        anyhow::ensure!(
            !self.layout.domains.is_empty(),
            "genome layout cannot be empty"
        );
        let domain = self
            .layout
            .domains
            .get(index % self.layout.domains.len())
            .unwrap();
        self.values[index] = domain.clamp(value)?;
        Ok(())
    }
}

/// A candidate solution: a genome, a unique ID within the run, and its cached
/// evaluation. Changing the genome clears the evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Individual {
    /// Run-unique candidate ID, also used for deterministic tie-breaks.
    pub id: u64,
    genome: Genome,
    evaluation: Option<EvaluationReport>,
}
impl Individual {
    /// Create an unevaluated individual.
    pub fn new(id: u64, genome: Genome) -> Self {
        Self {
            id,
            genome,
            evaluation: None,
        }
    }
    /// The genome.
    pub fn genome(&self) -> &Genome {
        &self.genome
    }
    /// Mutable genome access; clears the cached evaluation.
    pub fn genome_mut(&mut self) -> &mut Genome {
        self.evaluation = None;
        &mut self.genome
    }
    /// Replace the genome after validating it, clearing the cached evaluation.
    pub fn replace_genome(&mut self, genome: Genome) -> Result<()> {
        genome.validate()?;
        self.genome = genome;
        self.evaluation = None;
        Ok(())
    }
    /// Cached evaluation, or `None` if the genome has not been evaluated since it last changed.
    pub fn evaluation(&self) -> Option<&EvaluationReport> {
        self.evaluation.as_ref()
    }
    /// Store an evaluation after [`EvaluationReport::validate`].
    pub fn set_evaluation(&mut self, report: EvaluationReport) -> Result<()> {
        report.validate()?;
        self.evaluation = Some(report);
        Ok(())
    }
    /// Discard the cached evaluation.
    pub fn clear_evaluation(&mut self) {
        self.evaluation = None
    }
}
