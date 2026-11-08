//! Immutable-data learning problems and versioned saved predictors.
//!
//! [`LearningProblem`] turns an ARFF [`Dataset`] into an evolutionary problem for
//! rule, instance, mixed or regression-rule models (see [`ModelKind`]);
//! [`SavedModel`] loads a chosen model for prediction and held-out scoring.

use serde::{Deserialize, Serialize};
use std::{borrow::Cow, cmp::Ordering, fs, path::Path, sync::Arc};

use crate::{
    Result,
    config::{DecisionKind, FitnessKind, ModelKind, OperatorsConfig, TrainingConfig},
    data::{AttributeKind, Dataset, Row, Schema, Value},
    evaluation::{EvaluationReport, Problem},
    genome::{Domain, GeneValue, Genome, GenomeLayout, Individual},
    rng::{RNG_CONTRACT_VERSION, Random},
};

/// Semantics version for model distances, rule matching and prediction.
pub const MODEL_SEMANTICS_VERSION: u32 = 1;
const LINEAR_REGRESSION_SEMANTICS_VERSION: u32 = 2;
const SAVED_MODEL_VERSION: u32 = 1;

/// Classification or regression as an evolutionary [`Problem`]: each genome is an
/// ordered list of rules (or instances) over the dataset's predictors, evaluated on
/// the training rows according to [`TrainingConfig`].
#[derive(Debug, Clone)]
pub struct LearningProblem {
    dataset: Dataset,
    config: TrainingConfig,
    operators: OperatorsConfig,
    layout: Arc<GenomeLayout>,
    block: Vec<Locus>,
    nominal_training: Option<PreparedNominalRows>,
}

#[derive(Debug, Clone)]
enum Locus {
    NumericPoint { attribute: usize },
    NumericLow { attribute: usize },
    NumericHigh { attribute: usize },
    NominalCategory { attribute: usize, wildcard: bool },
    NumericTarget,
    LinearIntercept,
    LinearSlope,
    NominalTarget,
}

/// Model output for one row.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Prediction {
    /// Zero-based source row index ([`Row::source_index`]).
    pub row: usize,
    /// Predicted class or numeric value; `None` when the model abstained.
    pub value: Option<Value>,
    /// The model made no prediction for the row (classification only).
    pub abstained: bool,
    /// Regression row with no usable predictor evidence, predicted with the training-target mean.
    pub fallback: bool,
    /// Distance from the row to the deciding rule or instance, when one was computed.
    pub distance: Option<f64>,
    /// The row has a numeric predictor outside its fitted training range (not clipped).
    pub out_of_range: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SavedModelRecord {
    format_version: u32,
    semantics_version: u32,
    rng_contract_version: u32,
    schema: Schema,
    config: TrainingConfig,
    operators: OperatorsConfig,
    genome: Genome,
}

/// A selected model loaded from a saved JSON file, ready to predict or score data
/// without the training rows.
#[derive(Debug, Clone)]
pub struct SavedModel {
    record: SavedModelRecord,
}

impl LearningProblem {
    /// Build a learning problem from training data and settings.
    ///
    /// Errors if the dataset is empty, the model does not fit the target type
    /// (classification models need a nominal target, regression models a numeric
    /// one), training settings are out of range, `regression_linear_rule` does not have
    /// exactly one observed nonconstant numeric predictor, or the crossover cannot run
    /// on the resulting genome (see [`OperatorsConfig::validate_for_layout`]).
    pub fn new(
        dataset: Dataset,
        config: TrainingConfig,
        operators: OperatorsConfig,
    ) -> Result<Self> {
        let problem = Self::new_with_empty(dataset, config, operators, false)?;
        // Operator compatibility depends on the model and on the data-derived
        // layout, so finish it here, after loading data and before any evolution.
        problem.operators.validate_for_training(&problem.config)?;
        problem
            .operators
            .validate_for_layout(&problem.layout, problem.config.prune_probability > 0.0)?;
        Ok(problem)
    }
    fn new_with_empty(
        dataset: Dataset,
        config: TrainingConfig,
        operators: OperatorsConfig,
        allow_empty: bool,
    ) -> Result<Self> {
        anyhow::ensure!(
            allow_empty || !dataset.is_empty(),
            "training dataset is empty"
        );
        anyhow::ensure!(
            config.initial_rules > 0,
            "training.initial_rules must be positive"
        );
        anyhow::ensure!(
            config.neighbours > 0,
            "training.neighbours must be positive"
        );
        anyhow::ensure!(
            config.temperature.is_finite() && config.temperature >= 0.0,
            "training.temperature must be finite and nonnegative"
        );
        anyhow::ensure!(
            (0.0..=1.0).contains(&config.initial_generality),
            "training.initial_generality must be in [0,1]"
        );
        anyhow::ensure!(
            (0.0..=1.0).contains(&config.wildcard_probability),
            "training.wildcard_probability must be in [0,1]"
        );
        anyhow::ensure!(
            dataset.is_classification()
                || matches!(
                    config.model,
                    ModelKind::RegressionRule | ModelKind::RegressionLinearRule
                ),
            "numeric targets require a regression model"
        );
        anyhow::ensure!(
            !dataset.is_classification()
                || !matches!(
                    config.model,
                    ModelKind::RegressionRule | ModelKind::RegressionLinearRule
                ),
            "regression models require a numeric target"
        );
        anyhow::ensure!(
            config.fitness != FitnessKind::ArchivedLossSize || dataset.is_classification(),
            "archived_loss_size fitness requires a classification target"
        );
        if config.model == ModelKind::RegressionLinearRule {
            let predictors = dataset.schema().predictors().collect::<Vec<_>>();
            anyhow::ensure!(
                predictors.len() == 1,
                "regression_linear_rule requires exactly one predictor"
            );
            anyhow::ensure!(
                predictors[0].1.is_numeric(),
                "regression_linear_rule requires a numeric or integer predictor"
            );
            anyhow::ensure!(
                predictors[0].1.observed_in_training
                    && predictors[0].1.fitted_range.is_some_and(|(lo, hi)| lo < hi),
                "regression_linear_rule requires a nonconstant observed predictor"
            );
        }
        let mut block = Vec::new();
        let mut domains = Vec::new();
        for (index, attribute) in dataset.schema().predictors() {
            if !attribute.observed_in_training {
                continue;
            }
            match attribute.kind {
                AttributeKind::Numeric | AttributeKind::Integer => {
                    let (lo, hi) = attribute.fitted_range.ok_or_else(|| {
                        anyhow::anyhow!(
                            "predictor {:?} has no observed training values",
                            attribute.name
                        )
                    })?;
                    anyhow::ensure!(
                        lo.is_finite() && hi.is_finite(),
                        "predictor {:?} has nonfinite training range",
                        attribute.name
                    );
                    if lo == hi {
                        continue;
                    }
                    let bound = Domain::Real { min: lo, max: hi };
                    if config.model == ModelKind::Instance {
                        block.push(Locus::NumericPoint { attribute: index });
                        domains.push(bound);
                    } else {
                        block.push(Locus::NumericLow { attribute: index });
                        domains.push(bound.clone());
                        block.push(Locus::NumericHigh { attribute: index });
                        domains.push(bound);
                    }
                }
                AttributeKind::Nominal(ref labels) => {
                    let (min, max) = if config.model == ModelKind::Instance {
                        (0, labels.len() as i64 - 1)
                    } else {
                        (
                            wildcard_min(labels.len(), config.wildcard_probability),
                            labels.len() as i64 - 1,
                        )
                    };
                    block.push(Locus::NominalCategory {
                        attribute: index,
                        wildcard: config.model != ModelKind::Instance,
                    });
                    domains.push(Domain::Integer { min, max });
                }
            }
        }
        match &dataset.target_attribute().kind {
            AttributeKind::Nominal(labels) => {
                block.push(Locus::NominalTarget);
                domains.push(Domain::Integer {
                    min: 0,
                    max: labels.len() as i64 - 1,
                });
            }
            AttributeKind::Numeric => {
                let (lo, hi) = dataset
                    .target_attribute()
                    .fitted_range
                    .ok_or_else(|| anyhow::anyhow!("numeric target has no observed values"))?;
                if config.model != ModelKind::RegressionLinearRule {
                    block.push(Locus::NumericTarget);
                    domains.push(Domain::Real { min: lo, max: hi });
                }
            }
            AttributeKind::Integer => {
                let (lo, hi) = dataset
                    .target_attribute()
                    .fitted_range
                    .ok_or_else(|| anyhow::anyhow!("integer target has no observed values"))?;
                if config.model != ModelKind::RegressionLinearRule {
                    block.push(Locus::NumericTarget);
                    domains.push(Domain::Real { min: lo, max: hi });
                }
            }
        }
        if config.model == ModelKind::RegressionLinearRule {
            block.push(Locus::LinearIntercept);
            domains.push(Domain::Real {
                min: -1.0,
                max: 2.0,
            });
            block.push(Locus::LinearSlope);
            domains.push(Domain::Real {
                min: -4.0,
                max: 4.0,
            });
        }
        anyhow::ensure!(!block.is_empty(), "model has no genome loci");
        let layout = Arc::new(GenomeLayout::new(domains)?);
        let nominal_training = if config.model == ModelKind::Rule {
            nominal_rule_conditions(&block)
                .map(|conditions| PreparedNominalRows::new(&dataset, conditions))
        } else {
            None
        };
        Ok(Self {
            dataset,
            config,
            operators,
            layout,
            block,
            nominal_training,
        })
    }

    /// The training dataset.
    pub fn dataset(&self) -> &Dataset {
        &self.dataset
    }
    /// Training settings.
    pub fn config(&self) -> &TrainingConfig {
        &self.config
    }
    /// Operator settings (stored in saved models).
    pub fn operators(&self) -> &OperatorsConfig {
        &self.operators
    }
    /// Number of loci per rule block.
    pub fn rule_width(&self) -> usize {
        self.block.len()
    }

    /// Pick the final model from a front according to `training.decision`:
    /// `best_accuracy` takes the lowest training loss, then size; `compromise` takes the
    /// member closest to the origin in objective space. Ties break by ID. Errors on an
    /// empty front or an unevaluated member.
    pub fn choose<'a>(&self, front: &'a [Individual]) -> Result<&'a Individual> {
        anyhow::ensure!(!front.is_empty(), "cannot choose from an empty front");
        let mut eligible = Vec::with_capacity(front.len());
        for ind in front {
            let report = ind
                .evaluation()
                .ok_or_else(|| anyhow::anyhow!("candidate {} has no evaluation", ind.id))?;
            eligible.push((ind, report));
        }
        let chosen = match self.config.decision {
            DecisionKind::BestAccuracy => eligible.into_iter().min_by(|(a, ar), (b, br)| {
                cmp_f64(ar.loss, br.loss)
                    .then_with(|| cmp_f64(ar.size, br.size))
                    .then_with(|| lex_cmp(&ar.objectives, &br.objectives))
                    .then_with(|| a.id.cmp(&b.id))
            }),
            DecisionKind::Compromise => eligible.into_iter().min_by(|(a, ar), (b, br)| {
                let comparison = if self.config.fitness == FitnessKind::ArchivedLossSize {
                    l2_cmp(&[ar.loss, ar.size], &[br.loss, br.size])
                } else {
                    l2_cmp(&ar.objectives, &br.objectives)
                };
                comparison.then_with(|| a.id.cmp(&b.id))
            }),
        }
        .expect("nonempty input");
        Ok(chosen.0)
    }

    /// Save `genome` with the schema and settings as a versioned JSON model file.
    /// Errors if the genome does not fit this problem or the file cannot be written.
    pub fn save_model(&self, genome: &Genome, path: impl AsRef<Path>) -> Result<()> {
        self.validate_genome(genome)?;
        let record = SavedModelRecord {
            format_version: SAVED_MODEL_VERSION,
            semantics_version: model_semantics_version(self.config.model),
            rng_contract_version: RNG_CONTRACT_VERSION,
            schema: self.dataset.schema().clone(),
            config: self.config.clone(),
            operators: self.operators.clone(),
            genome: genome.clone(),
        };
        let data = serde_json::to_vec_pretty(&record)?;
        fs::write(path.as_ref(), data)
            .map_err(|e| anyhow::anyhow!("write model {}: {e}", path.as_ref().display()))?;
        Ok(())
    }

    /// Score an ARFF file (target required) against the fitted training schema.
    pub fn score_file(&self, genome: &Genome, path: impl AsRef<Path>) -> Result<EvaluationReport> {
        self.validate_genome(genome)?;
        let input = fs::read_to_string(path.as_ref())
            .map_err(|e| anyhow::anyhow!("read scored data {}: {e}", path.as_ref().display()))?;
        let scored = Dataset::parse_str_for_prediction(&input, self.dataset.schema())?;
        evaluate_dataset(
            &scored,
            genome,
            &self.config,
            &self.block,
            &self.layout,
            self.dataset.target_mean(),
        )
    }

    /// Score a dataset after re-interpreting it with the fitted training schema.
    /// Errors if its columns do not match the training schema.
    pub fn score_dataset(&self, genome: &Genome, dataset: &Dataset) -> Result<EvaluationReport> {
        self.validate_genome(genome)?;
        let scored = dataset.with_fitted_schema(self.dataset.schema())?;
        evaluate_dataset(
            &scored,
            genome,
            &self.config,
            &self.block,
            &self.layout,
            self.dataset.target_mean(),
        )
    }

    /// Predict rows using this problem's fitted schema, keeping preprocessing
    /// identical to training and making pre/post-save checks straightforward.
    pub fn predict_dataset(&self, genome: &Genome, dataset: &Dataset) -> Result<Vec<Prediction>> {
        self.validate_genome(genome)?;
        let scored = dataset.with_fitted_schema(self.dataset.schema())?;
        predict_rows(
            &scored,
            genome,
            &self.config,
            &self.block,
            &self.layout,
            self.dataset.target_mean(),
        )
    }

    fn validate_genome(&self, genome: &Genome) -> Result<()> {
        genome.validate()?;
        anyhow::ensure!(
            Arc::ptr_eq(&genome.layout, &self.layout)
                || genome.layout.domains == self.layout.domains,
            "genome layout does not match learning problem"
        );
        anyhow::ensure!(
            genome.values.len().is_multiple_of(self.block.len()),
            "genome has a partial rule"
        );
        anyhow::ensure!(
            !genome.values.is_empty(),
            "model genome must contain at least one rule"
        );
        Ok(())
    }
}

impl Problem for LearningProblem {
    fn initialize(&self, rng: &mut Random) -> Result<Genome> {
        let mut values = Vec::with_capacity(self.config.initial_rules * self.block.len());
        for _ in 0..self.config.initial_rules {
            let mut j = 0;
            while j < self.block.len() {
                let locus = &self.block[j];
                let domain = &self.layout.domains[j];
                match locus {
                    Locus::NumericPoint { attribute } => {
                        let (lo, hi) = self.dataset.schema().attributes[*attribute]
                            .fitted_range
                            .unwrap();
                        values.push(GeneValue::Real(rng.range(lo, hi)));
                        j += 1;
                    }
                    Locus::NumericLow { attribute }
                        if self.block.get(j + 1).is_some_and(
                            |x| matches!(x,Locus::NumericHigh{attribute:a} if a==attribute),
                        ) =>
                    {
                        let (lo, hi) = self.dataset.schema().attributes[*attribute]
                            .fitted_range
                            .unwrap();
                        let (a, b) = if rng.bernoulli(self.config.initial_generality) {
                            (lo, hi)
                        } else {
                            let x = rng.uniform();
                            let lower = lo + (1.0 - x) * rng.uniform() * (hi - lo);
                            (lower, lower + x * (hi - lo))
                        };
                        values.push(GeneValue::Real(a));
                        values.push(GeneValue::Real(b));
                        j += 2;
                    }
                    Locus::NumericLow { attribute } | Locus::NumericHigh { attribute } => {
                        let (lo, hi) = self.dataset.schema().attributes[*attribute]
                            .fitted_range
                            .unwrap();
                        values.push(GeneValue::Real(rng.range(lo, hi)));
                        j += 1;
                    }
                    Locus::NominalCategory { wildcard: true, .. } => {
                        let max = domain.bounds().1 as i64;
                        // A zero mutation wildcard probability removes the wildcard
                        // sentinel from the domain, so preserve that valid boundary
                        // even when initial_generality is positive. When the
                        // sentinel is available, initial generality controls the
                        // initial wildcard rate independently of mutation settings.
                        let initial_wildcard_probability = if domain.bounds().0 < 0.0 {
                            self.config.initial_generality
                        } else {
                            0.0
                        };
                        if rng.bernoulli(initial_wildcard_probability) {
                            values.push(GeneValue::Integer(-1));
                        } else {
                            values.push(GeneValue::Integer(rng.integer(0, max)));
                        }
                        j += 1;
                    }
                    Locus::NominalCategory { .. } | Locus::NominalTarget => {
                        let (lo, hi) = domain.bounds();
                        values.push(GeneValue::Integer(rng.integer(lo as i64, hi as i64)));
                        j += 1;
                    }
                    Locus::NumericTarget => {
                        let (lo, hi) = domain.bounds();
                        values.push(GeneValue::Real(rng.range(lo, hi)));
                        j += 1;
                    }
                    Locus::LinearIntercept => {
                        values.push(GeneValue::Real(rng.range(-1.0, 2.0)));
                        j += 1;
                    }
                    Locus::LinearSlope => {
                        values.push(GeneValue::Real(rng.range(-4.0, 4.0)));
                        j += 1;
                    }
                }
            }
        }
        let mut genome = Genome::new(values, self.layout.clone())?;
        genome.deviation = self
            .operators
            .per_locus_probability
            .unwrap_or(1.0 / self.block.len() as f64);
        genome.deviations.fill(genome.deviation);
        Ok(genome)
    }

    fn evaluate(&self, genome: &Genome) -> Result<EvaluationReport> {
        self.validate_genome(genome)?;
        evaluate_dataset_with_nominal(
            &self.dataset,
            genome,
            &self.config,
            &self.block,
            &self.layout,
            self.dataset.target_mean(),
            self.nominal_training.as_ref(),
        )
    }

    fn estimated_evaluation_work(&self, genome: &Genome) -> Option<usize> {
        let rules = genome.values.len() / self.block.len().max(1);
        Some(
            self.dataset
                .len()
                .saturating_mul(rules)
                .saturating_mul(self.dataset.schema().feature_count().max(1)),
        )
    }

    fn prune(&self, genome: &mut Genome, report: &EvaluationReport) -> Result<bool> {
        let width = self.block.len();
        anyhow::ensure!(
            genome.values.len().is_multiple_of(width),
            "genome has a partial rule"
        );
        let count = genome.values.len() / width;
        anyhow::ensure!(count > 0, "genome has no rules");
        let mut keep = Vec::new();
        for r in 0..count {
            if report.rule_usage.get(r).copied().unwrap_or(0) > 0 {
                keep.push(r);
            }
        }
        if keep.is_empty() {
            keep.push(0);
        }
        if keep.len() == count {
            return Ok(false);
        }
        let mut values = Vec::with_capacity(keep.len() * width);
        let mut deviations = Vec::with_capacity(keep.len() * width);
        for r in keep {
            let a = r * width;
            values.extend_from_slice(&genome.values[a..a + width]);
            deviations.extend_from_slice(&genome.deviations[a..a + width]);
        }
        genome.values = values;
        genome.deviations = deviations;
        Ok(true)
    }
}

impl Dataset {
    fn target_mean(&self) -> Option<f64> {
        self.schema().attributes[self.target_index()].fitted_mean
    }
}

impl SavedModel {
    /// Load a saved model, checking format, semantics and RNG-contract versions, the
    /// schema and the genome. Errors if the file is unreadable, malformed or from an
    /// incompatible version.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let input = fs::read_to_string(path.as_ref())
            .map_err(|e| anyhow::anyhow!("read model {}: {e}", path.as_ref().display()))?;
        let record: SavedModelRecord =
            serde_json::from_str(&input).map_err(|e| anyhow::anyhow!("decode saved model: {e}"))?;
        anyhow::ensure!(
            record.format_version == SAVED_MODEL_VERSION,
            "unsupported saved-model format version {}",
            record.format_version
        );
        anyhow::ensure!(
            record.semantics_version == model_semantics_version(record.config.model),
            "saved model semantics version {} does not match model {:?} (expected {})",
            record.semantics_version,
            record.config.model,
            model_semantics_version(record.config.model)
        );
        anyhow::ensure!(
            record.rng_contract_version == RNG_CONTRACT_VERSION,
            "unsupported RNG contract version {}",
            record.rng_contract_version
        );
        anyhow::ensure!(
            record.schema.target_index < record.schema.attributes.len(),
            "invalid saved schema target index"
        );
        record.schema.validate()?;
        anyhow::ensure!(
            record.schema.training_rows > 0,
            "saved schema has no training rows"
        );
        anyhow::ensure!(
            record.config.fitness != FitnessKind::ArchivedLossSize
                || record.schema.is_classification(),
            "archived_loss_size fitness requires a classification target"
        );
        let mut names = std::collections::BTreeSet::new();
        for (i, attr) in record.schema.attributes.iter().enumerate() {
            anyhow::ensure!(
                names.insert(attr.name.clone()),
                "saved schema has duplicate attribute {:?}",
                attr.name
            );
            if let Some((lo, hi)) = attr.fitted_range {
                anyhow::ensure!(
                    lo.is_finite() && hi.is_finite() && lo <= hi,
                    "saved schema attribute {i} has invalid fitted range"
                );
            }
            if let Some(mean) = attr.fitted_mean {
                anyhow::ensure!(
                    mean.is_finite(),
                    "saved schema attribute {i} has nonfinite fitted mean"
                );
            }
            if let Some((lo, hi)) = attr.declared_bounds {
                anyhow::ensure!(
                    lo <= hi,
                    "saved schema attribute {i} has invalid declared bounds"
                );
            }
            if let AttributeKind::Nominal(labels) = &attr.kind {
                anyhow::ensure!(
                    !labels.is_empty(),
                    "saved nominal attribute {i} has no labels"
                );
                let distinct = labels.iter().collect::<std::collections::BTreeSet<_>>();
                anyhow::ensure!(
                    distinct.len() == labels.len(),
                    "saved nominal attribute {i} has duplicate labels"
                );
            }
        }
        record.genome.validate()?;
        let problem = Self::problem_for_record(&record)?;
        problem.validate_genome(&record.genome)?;
        Ok(Self { record })
    }

    fn problem_for_record(record: &SavedModelRecord) -> Result<LearningProblem> {
        let empty = Dataset::schema_only(record.schema.clone());
        LearningProblem::new_with_empty(
            empty,
            record.config.clone(),
            record.operators.clone(),
            true,
        )
    }

    /// Training schema stored with the model.
    pub fn schema(&self) -> &Schema {
        &self.record.schema
    }
    /// Predict every row of an ARFF file; see [`SavedModel::predict_str`].
    pub fn predict_file(&self, path: impl AsRef<Path>) -> Result<Vec<Prediction>> {
        let input = fs::read_to_string(path.as_ref()).map_err(|e| {
            anyhow::anyhow!("read prediction data {}: {e}", path.as_ref().display())
        })?;
        self.predict_str(&input)
    }
    /// Predict every row of ARFF text. The target column may be absent or missing;
    /// other columns must match the saved schema.
    pub fn predict_str(&self, input: &str) -> Result<Vec<Prediction>> {
        let data = Dataset::parse_str_for_prediction(input, &self.record.schema)?;
        let problem = Self::problem_for_record(&self.record)?;
        predict_rows(
            &data,
            &self.record.genome,
            &self.record.config,
            &problem.block,
            &problem.layout,
            self.record.schema_target_mean(),
        )
    }
    /// Score an ARFF file (target required) with the saved model.
    pub fn score_file(&self, path: impl AsRef<Path>) -> Result<EvaluationReport> {
        let input = fs::read_to_string(path.as_ref())
            .map_err(|e| anyhow::anyhow!("read scored data {}: {e}", path.as_ref().display()))?;
        let data = Dataset::parse_str_for_prediction(&input, &self.record.schema)?;
        let problem = Self::problem_for_record(&self.record)?;
        evaluate_dataset(
            &data,
            &self.record.genome,
            &self.record.config,
            &problem.block,
            &problem.layout,
            self.record.schema_target_mean(),
        )
    }
}

/// Saved-model (prediction) semantics version for a model kind: 2 for
/// `regression_linear_rule`, otherwise [`MODEL_SEMANTICS_VERSION`].
pub fn model_semantics_version(model: ModelKind) -> u32 {
    if model == ModelKind::RegressionLinearRule {
        LINEAR_REGRESSION_SEMANTICS_VERSION
    } else {
        MODEL_SEMANTICS_VERSION
    }
}

/// Version the training semantics recorded for a learning run.
///
/// Saved-model semantics remain separately versioned by
/// [`model_semantics_version`], since this initialization correction does not
/// change prediction behavior for an already-saved genome.
pub fn learning_run_semantics_version(model: ModelKind) -> u32 {
    if matches!(
        model,
        ModelKind::Rule | ModelKind::Mixed | ModelKind::RegressionRule
    ) {
        3
    } else {
        model_semantics_version(model)
    }
}

impl SavedModelRecord {
    fn schema_target_mean(&self) -> Option<f64> {
        self.schema.attributes[self.schema.target_index].fitted_mean
    }
}

fn wildcard_min(categories: usize, probability: f64) -> i64 {
    if probability <= 0.0 {
        return 0;
    }
    // A signed sentinel band keeps uniform integer mutation near the configured wildcard mass.
    let count = if probability >= 1.0 {
        1_000_000_000.0
    } else {
        (probability * categories as f64 / (1.0 - probability))
            .ceil()
            .max(1.0)
    };
    -(count.min(1_000_000_000.0) as i64)
}

fn evaluate_dataset(
    data: &Dataset,
    genome: &Genome,
    cfg: &TrainingConfig,
    block: &[Locus],
    layout: &Arc<GenomeLayout>,
    target_mean: Option<f64>,
) -> Result<EvaluationReport> {
    evaluate_dataset_with_nominal(data, genome, cfg, block, layout, target_mean, None)
}

fn evaluate_dataset_with_nominal(
    data: &Dataset,
    genome: &Genome,
    cfg: &TrainingConfig,
    block: &[Locus],
    layout: &Arc<GenomeLayout>,
    target_mean: Option<f64>,
    nominal_training: Option<&PreparedNominalRows>,
) -> Result<EvaluationReport> {
    genome.validate()?;
    anyhow::ensure!(
        genome.values.len().is_multiple_of(block.len()) && !genome.values.is_empty(),
        "invalid model genome shape"
    );
    let (predictions, usage) = predict_rows_with_nominal(
        data,
        genome,
        cfg,
        block,
        layout,
        target_mean,
        nominal_training,
    )?;
    let n = data.len();
    let mut loss = 0.0_f64;
    let mut covered = 0usize;
    let mut conditional_sum = 0.0_f64;
    let mut abstentions = 0;
    let mut fallbacks = 0;
    let mut out_of_range = 0;
    for (row, pred) in data.rows().iter().zip(&predictions) {
        if pred.out_of_range {
            out_of_range += 1;
        }
        if pred.fallback {
            fallbacks += 1;
        }
        if pred.abstained {
            abstentions += 1;
        }
        let actual = row.values[data.target_index()].as_ref().ok_or_else(|| {
            anyhow::anyhow!("scored row {} has missing target", row.source_index + 1)
        })?;
        if pred.abstained {
            let next_loss = loss + 1.0;
            anyhow::ensure!(
                next_loss.is_finite(),
                "scored row {}: loss accumulation overflowed",
                row.source_index + 1
            );
            loss = next_loss;
            continue;
        }
        let l = match (actual, pred.value.as_ref()) {
            (Value::Nominal(a), Some(Value::Nominal(b))) => {
                if a == b {
                    0.0
                } else {
                    1.0
                }
            }
            (a, Some(b)) if a.numeric().is_some() && b.numeric().is_some() => {
                let d = a.numeric().unwrap() - b.numeric().unwrap();
                anyhow::ensure!(
                    d.is_finite(),
                    "scored row {}: regression prediction error overflowed",
                    row.source_index + 1
                );
                let squared = d * d;
                anyhow::ensure!(
                    squared.is_finite(),
                    "scored row {}: squared regression error overflowed",
                    row.source_index + 1
                );
                squared
            }
            _ => anyhow::bail!("target type mismatch at row {}", row.source_index + 1),
        };
        let next_loss = loss + l;
        anyhow::ensure!(
            next_loss.is_finite(),
            "scored row {}: loss accumulation overflowed",
            row.source_index + 1
        );
        loss = next_loss;
        covered += 1;
        let next_conditional = conditional_sum + l;
        anyhow::ensure!(
            next_conditional.is_finite(),
            "scored row {}: conditional loss accumulation overflowed",
            row.source_index + 1
        );
        conditional_sum = next_conditional;
    }
    let generality = if cfg.model == ModelKind::Instance {
        1.0
    } else {
        compute_generality(genome, block, layout, &usage, data.schema())
    };
    let size = (genome.values.len() / block.len()) as f64 / data.schema().training_rows as f64;
    let raw_loss = loss / n as f64;
    let conditional_loss = if covered == 0 {
        1.0
    } else {
        conditional_sum / covered as f64
    };
    let coverage = covered as f64 / n as f64;
    let rule_count = usage.len();
    let unused_rule_fraction = if rule_count == 0 {
        0.0
    } else {
        usage.iter().filter(|&&count| count == 0).count() as f64 / rule_count as f64
    };
    let objectives = match cfg.fitness {
        FitnessKind::PenalizedLoss => vec![raw_loss + size],
        FitnessKind::LossSize => vec![raw_loss, size],
        FitnessKind::ArchivedLossSize if rule_count <= data.schema().training_rows => {
            vec![raw_loss, size]
        }
        FitnessKind::ArchivedLossSize => vec![1.0, 1.0],
        FitnessKind::LossGeneralitySize => vec![raw_loss, generality, size],
        FitnessKind::CoverageSize => vec![conditional_loss, 1.0 - coverage, size],
    };
    let report = EvaluationReport {
        objectives,
        loss: raw_loss,
        coverage,
        conditional_loss,
        generality,
        size,
        rule_count,
        unused_rule_fraction,
        rule_usage: usage,
        abstentions,
        fallbacks,
        out_of_range,
        rows: n,
    };
    report.validate()?;
    Ok(report)
}

fn predict_rows(
    data: &Dataset,
    genome: &Genome,
    cfg: &TrainingConfig,
    block: &[Locus],
    layout: &Arc<GenomeLayout>,
    mean: Option<f64>,
) -> Result<Vec<Prediction>> {
    Ok(predict_rows_with_usage(data, genome, cfg, block, layout, mean)?.0)
}
fn predict_rows_with_usage(
    data: &Dataset,
    genome: &Genome,
    cfg: &TrainingConfig,
    block: &[Locus],
    layout: &Arc<GenomeLayout>,
    mean: Option<f64>,
) -> Result<(Vec<Prediction>, Vec<usize>)> {
    predict_rows_with_nominal(data, genome, cfg, block, layout, mean, None)
}

fn predict_rows_with_nominal(
    data: &Dataset,
    genome: &Genome,
    cfg: &TrainingConfig,
    block: &[Locus],
    layout: &Arc<GenomeLayout>,
    mean: Option<f64>,
    nominal_training: Option<&PreparedNominalRows>,
) -> Result<(Vec<Prediction>, Vec<usize>)> {
    genome.validate()?;
    anyhow::ensure!(
        !block.is_empty() && genome.values.len().is_multiple_of(block.len()),
        "invalid model genome shape"
    );
    let rule_count = genome.values.len() / block.len();
    anyhow::ensure!(rule_count > 0, "model requires at least one rule");
    // Prepare the all-nominal Rule matching path once per prediction batch.
    // Numeric and mixed Rule layouts retain the general full-distance behavior.
    let owned_nominal_conditions = (nominal_training.is_none() && cfg.model == ModelKind::Rule)
        .then(|| nominal_rule_conditions(block))
        .flatten();
    let nominal_conditions = nominal_training
        .map(|cache| cache.conditions.as_slice())
        .or(owned_nominal_conditions.as_deref());
    let mut usage = vec![0; rule_count];
    let mut results = Vec::with_capacity(data.len());
    for (row_index, row) in data.rows().iter().enumerate() {
        let outside = is_out_of_range(row, data.schema());
        let nominal_values: Option<Cow<'_, [NominalObservation]>> = if let Some(cache) =
            nominal_training
        {
            Some(Cow::Borrowed(cache.row(row_index)))
        } else {
            nominal_conditions.map(|conditions| {
                Cow::Owned(
                    conditions
                        .iter()
                        .map(|condition| match row.values[condition.attribute].as_ref() {
                            None => NominalObservation::Missing,
                            Some(Value::Nominal(value)) => data.schema().attributes
                                [condition.attribute]
                                .nominal_values()
                                .and_then(|labels| labels.iter().position(|label| label == value))
                                .map_or(
                                    NominalObservation::UnknownNominal,
                                    NominalObservation::Known,
                                ),
                            Some(Value::Numeric(_) | Value::Integer(_)) => {
                                NominalObservation::InvalidType
                            }
                        })
                        .collect::<Vec<_>>(),
                )
            })
        };
        if cfg.model == ModelKind::RegressionLinearRule {
            let mut ds = Vec::with_capacity(rule_count);
            for r in 0..rule_count {
                ds.push((
                    r,
                    rule_distance(row, genome, r, block, layout, data.schema())?,
                ));
            }
            if ds.iter().all(|(_, d)| d.is_none()) {
                let v = mean.ok_or_else(|| {
                    anyhow::anyhow!("no training-target mean available for fallback")
                })?;
                results.push(Prediction {
                    row: row.source_index,
                    value: Some(Value::Numeric(v)),
                    abstained: false,
                    fallback: true,
                    distance: None,
                    out_of_range: outside,
                });
                continue;
            }
            let mut ds = ds
                .into_iter()
                .filter_map(|(r, d)| d.map(|x| (r, x)))
                .collect::<Vec<_>>();
            ds.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
            let k = cfg.neighbours.min(ds.len());
            let base = ds[0].1;
            let logits = ds[..k]
                .iter()
                .map(|(_, d)| -cfg.temperature * (d - base))
                .collect::<Vec<_>>();
            let max_logit = logits.iter().copied().max_by(f64::total_cmp).unwrap();
            let weights = logits
                .iter()
                .map(|x| (x - max_logit).exp())
                .collect::<Vec<_>>();
            let total = weights.iter().sum::<f64>();
            let predictor_index = data
                .schema()
                .predictors()
                .next()
                .ok_or_else(|| anyhow::anyhow!("linear regression predictor is missing"))?
                .0;
            let predictor = data.schema().attributes[predictor_index]
                .fitted_range
                .ok_or_else(|| anyhow::anyhow!("linear regression predictor range is missing"))?;
            let target = data
                .schema()
                .target()
                .fitted_range
                .ok_or_else(|| anyhow::anyhow!("linear regression target range is missing"))?;
            let x = row.values[predictor_index]
                .as_ref()
                .and_then(Value::numeric)
                .ok_or_else(|| anyhow::anyhow!("linear regression row has missing predictor"))?;
            let z = if predictor.1 == predictor.0 {
                0.0
            } else {
                (x - predictor.0) / (predictor.1 - predictor.0)
            };
            anyhow::ensure!(
                z.is_finite(),
                "row {}: linear regression normalized predictor is nonfinite",
                row.source_index + 1
            );
            let intercept_index = block
                .iter()
                .position(|l| matches!(l, Locus::LinearIntercept))
                .ok_or_else(|| anyhow::anyhow!("linear regression intercept is missing"))?;
            let slope_index = block
                .iter()
                .position(|l| matches!(l, Locus::LinearSlope))
                .ok_or_else(|| anyhow::anyhow!("linear regression slope is missing"))?;
            let mut rule_values = Vec::with_capacity(k);
            for (r, _) in &ds[..k] {
                usage[*r] += 1;
                let value_r = if target.1 == target.0 {
                    target.0
                } else {
                    let a = genome.values[*r * block.len() + intercept_index].numeric();
                    let b = genome.values[*r * block.len() + slope_index].numeric();
                    let linear = a + b * z;
                    let scaled = (target.1 - target.0) * linear;
                    target.0 + scaled
                };
                anyhow::ensure!(
                    value_r.is_finite(),
                    "row {}: linear regression prediction is nonfinite",
                    row.source_index + 1
                );
                rule_values.push(value_r);
            }
            let value = if rule_values
                .iter()
                .all(|v| v.to_bits() == rule_values[0].to_bits())
            {
                rule_values[0]
            } else {
                rule_values
                    .iter()
                    .zip(&weights)
                    .map(|(v, weight)| v * (weight / total))
                    .sum::<f64>()
            };
            anyhow::ensure!(
                value.is_finite(),
                "row {}: linear regression prediction is nonfinite",
                row.source_index + 1
            );
            results.push(Prediction {
                row: row.source_index,
                value: Some(Value::Numeric(value)),
                abstained: false,
                fallback: false,
                distance: Some(base),
                out_of_range: outside,
            });
        } else if cfg.model == ModelKind::RegressionRule {
            let mut ds = Vec::with_capacity(rule_count);
            for r in 0..rule_count {
                ds.push((
                    r,
                    rule_distance(row, genome, r, block, layout, data.schema())?,
                ));
            }
            if ds.iter().all(|(_, d)| d.is_none()) {
                let v = mean.ok_or_else(|| {
                    anyhow::anyhow!("no training-target mean available for fallback")
                })?;
                results.push(Prediction {
                    row: row.source_index,
                    value: Some(Value::Numeric(v)),
                    abstained: false,
                    fallback: true,
                    distance: None,
                    out_of_range: outside,
                });
                continue;
            }
            let mut ds = ds
                .into_iter()
                .filter_map(|(r, d)| d.map(|x| (r, x)))
                .collect::<Vec<_>>();
            ds.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
            let k = cfg.neighbours.min(ds.len());
            let base = ds[0].1;
            let mut maxw = f64::NEG_INFINITY;
            let weights = ds[..k]
                .iter()
                .map(|(_, d)| -cfg.temperature * (d - base))
                .collect::<Vec<_>>();
            if let Some(m) = weights.iter().copied().max_by(f64::total_cmp) {
                maxw = m;
            }
            let weights = weights.iter().map(|x| (x - maxw).exp()).collect::<Vec<_>>();
            let total = weights.iter().sum::<f64>();
            let target_idx = block
                .iter()
                .position(|l| matches!(l, Locus::NumericTarget))
                .unwrap();
            let mut target_values = Vec::with_capacity(k);
            for (r, _) in &ds[..k] {
                usage[*r] += 1;
                target_values.push(genome.values[*r * block.len() + target_idx].numeric());
            }
            // Preserve an exactly shared target bit pattern (including signed zero) across averaging.
            let value = if target_values
                .iter()
                .all(|v| v.to_bits() == target_values[0].to_bits())
            {
                target_values[0]
            } else {
                target_values
                    .iter()
                    .zip(&weights)
                    .map(|(v, w)| v * w / total)
                    .sum::<f64>()
            };
            anyhow::ensure!(value.is_finite(), "regression prediction is nonfinite");
            results.push(Prediction {
                row: row.source_index,
                value: Some(Value::Numeric(value)),
                abstained: false,
                fallback: false,
                distance: Some(base),
                out_of_range: outside,
            });
        } else {
            let mut matched = None;
            match cfg.model {
                ModelKind::Rule => {
                    for r in 0..rule_count {
                        let is_match = if let (Some(conditions), Some(values)) =
                            (&nominal_conditions, &nominal_values)
                        {
                            nominal_rule_matches(genome, r, conditions, values) == Some(true)
                        } else {
                            rule_distance(row, genome, r, block, layout, data.schema())?
                                .is_some_and(|distance| distance == 0.0)
                        };
                        if is_match {
                            matched = Some((r, 0.0));
                            break;
                        }
                    }
                }
                ModelKind::Instance | ModelKind::Mixed => {
                    for r in 0..rule_count {
                        if let Some(d) =
                            rule_distance(row, genome, r, block, layout, data.schema())?
                            && matched.is_none_or(|(_, best)| d < best)
                        {
                            matched = Some((r, d));
                        }
                    }
                }
                ModelKind::RegressionRule => unreachable!(),
                ModelKind::RegressionLinearRule => unreachable!(),
            }
            if let Some((r, d)) = matched {
                usage[r] += 1;
                let t = block
                    .iter()
                    .position(|l| matches!(l, Locus::NominalTarget))
                    .ok_or_else(|| anyhow::anyhow!("classification genome has no class locus"))?;
                let idx = genome.values[r * block.len() + t].numeric() as usize;
                let labels = data.schema().target().nominal_values().unwrap();
                results.push(Prediction {
                    row: row.source_index,
                    value: Some(Value::Nominal(labels[idx].clone())),
                    abstained: false,
                    fallback: false,
                    distance: Some(d),
                    out_of_range: outside,
                });
            } else {
                results.push(Prediction {
                    row: row.source_index,
                    value: None,
                    abstained: true,
                    fallback: false,
                    distance: None,
                    out_of_range: outside,
                });
            }
        }
    }
    Ok((results, usage))
}

fn rule_distance(
    row: &Row,
    genome: &Genome,
    r: usize,
    block: &[Locus],
    _layout: &GenomeLayout,
    schema: &Schema,
) -> Result<Option<f64>> {
    let mut sum = 0.0;
    let mut observed = 0usize;
    let start = r * block.len();
    for (j, locus) in block.iter().enumerate() {
        let attribute = match locus {
            Locus::NumericPoint { attribute }
            | Locus::NumericLow { attribute }
            | Locus::NumericHigh { attribute }
            | Locus::NominalCategory { attribute, .. } => *attribute,
            _ => continue,
        };
        let Some(value) = row.values[attribute].as_ref() else {
            continue;
        };
        let attr = &schema.attributes[attribute];
        match (locus, value) {
            (Locus::NumericPoint { .. }, v) if v.numeric().is_some() => {
                if let Some((lo, hi)) = attr.fitted_range
                    && hi > lo
                {
                    let d = (genome.values[start + j].numeric()
                        - v.numeric().expect("guarded numeric value"))
                        / (hi - lo);
                    let contribution = d * d;
                    anyhow::ensure!(
                        d.is_finite()
                            && contribution.is_finite()
                            && (sum + contribution).is_finite(),
                        "row {}, attribute {:?}: normalized point distance overflowed",
                        row.source_index + 1,
                        attr.name
                    );
                    sum += contribution;
                    observed += 1;
                }
            }
            (Locus::NumericLow { .. }, v) if v.numeric().is_some() => {
                let high_j = j + 1;
                let a = genome.values[start + j].numeric();
                let b = genome.values[start + high_j].numeric();
                let lo = a.min(b);
                let hi = a.max(b);
                if let Some((rlo, rhi)) = attr.fitted_range
                    && rhi > rlo
                {
                    let x = v.numeric().expect("guarded numeric value");
                    let delta = if x < lo {
                        lo - x
                    } else if x > hi {
                        x - hi
                    } else {
                        0.0
                    };
                    let d = delta / (rhi - rlo);
                    let contribution = d * d;
                    anyhow::ensure!(
                        d.is_finite()
                            && contribution.is_finite()
                            && (sum + contribution).is_finite(),
                        "row {}, attribute {:?}: normalized interval distance overflowed",
                        row.source_index + 1,
                        attr.name
                    );
                    sum += contribution;
                    observed += 1;
                }
            }
            (Locus::NumericHigh { .. }, v) if v.numeric().is_some() => {}
            (Locus::NominalCategory { wildcard, .. }, Value::Nominal(x)) => {
                let gene = genome.values[start + j].numeric() as i64;
                observed += 1;
                if !(*wildcard && gene < 0) {
                    let labels = attr.nominal_values().unwrap();
                    if labels.get(gene as usize).is_none_or(|v| v != x) {
                        sum += 1.0;
                    }
                }
            }
            _ => return Ok(None),
        }
    }
    if observed == 0 {
        return Ok(None);
    }
    let distance = sum.sqrt();
    anyhow::ensure!(
        distance.is_finite(),
        "row {}: model distance is nonfinite",
        row.source_index + 1
    );
    Ok(Some(distance))
}

/// Test Rule-model coverage without calculating a distance that can only be zero or nonzero.
/// Missing predictor values are ignored, and `None` means the row supplied no rule evidence.
#[derive(Debug, Clone, Copy)]
struct NominalRuleCondition {
    attribute: usize,
    locus: usize,
    wildcard: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NominalObservation {
    Missing,
    Known(usize),
    UnknownNominal,
    InvalidType,
}

#[derive(Debug, Clone)]
struct PreparedNominalRows {
    conditions: Vec<NominalRuleCondition>,
    values: Vec<NominalObservation>,
    width: usize,
}

impl PreparedNominalRows {
    fn new(data: &Dataset, conditions: Vec<NominalRuleCondition>) -> Self {
        let width = conditions.len();
        let mut values = Vec::with_capacity(data.len() * width);
        for row in data.rows() {
            values.extend(conditions.iter().map(|condition| {
                match row.values[condition.attribute].as_ref() {
                    None => NominalObservation::Missing,
                    Some(Value::Nominal(value)) => data.schema().attributes[condition.attribute]
                        .nominal_values()
                        .and_then(|labels| labels.iter().position(|label| label == value))
                        .map_or(
                            NominalObservation::UnknownNominal,
                            NominalObservation::Known,
                        ),
                    Some(Value::Numeric(_) | Value::Integer(_)) => NominalObservation::InvalidType,
                }
            }));
        }
        Self {
            conditions,
            values,
            width,
        }
    }

    fn row(&self, index: usize) -> &[NominalObservation] {
        let start = index * self.width;
        &self.values[start..start + self.width]
    }
}

fn nominal_rule_conditions(block: &[Locus]) -> Option<Vec<NominalRuleCondition>> {
    let mut conditions = Vec::new();
    for (locus, item) in block.iter().enumerate() {
        match item {
            Locus::NominalCategory {
                attribute,
                wildcard,
            } => conditions.push(NominalRuleCondition {
                attribute: *attribute,
                locus,
                wildcard: *wildcard,
            }),
            Locus::NominalTarget => {}
            _ => return None,
        }
    }
    (!conditions.is_empty()).then_some(conditions)
}

/// Evaluate precomputed nominal predictor positions against one rule. `None`
/// preserves abstention when the row provides no predictor evidence.
fn nominal_rule_matches(
    genome: &Genome,
    rule: usize,
    conditions: &[NominalRuleCondition],
    observed_values: &[NominalObservation],
) -> Option<bool> {
    let mut observed = false;
    let start = rule * genome.layout.domains.len();
    for (condition, value) in conditions.iter().zip(observed_values) {
        let value = match value {
            NominalObservation::Missing => continue,
            NominalObservation::InvalidType => return Some(false),
            NominalObservation::UnknownNominal => {
                observed = true;
                let gene = genome.values[start + condition.locus].numeric() as i64;
                if !condition.wildcard || gene >= 0 {
                    return Some(false);
                }
                continue;
            }
            NominalObservation::Known(value) => {
                observed = true;
                *value
            }
        };
        let gene = genome.values[start + condition.locus].numeric() as i64;
        if !condition.wildcard && gene < 0 {
            return Some(false);
        }
        if !(condition.wildcard && gene < 0) && gene as usize != value {
            return Some(false);
        }
    }
    observed.then_some(true)
}

fn compute_generality(
    genome: &Genome,
    block: &[Locus],
    _layout: &GenomeLayout,
    usage: &[usize],
    schema: &Schema,
) -> f64 {
    if usage.iter().all(|v| *v == 0) {
        return 1.0;
    }
    let mut total = 0.0;
    let mut count = 0;
    for (r, u) in usage.iter().enumerate() {
        if *u == 0 {
            continue;
        }
        let mut specificity = 1.0;
        let start = r * block.len();
        for (j, l) in block.iter().enumerate() {
            match l {
                Locus::NumericLow { attribute } => {
                    let lo = genome.values[start + j].numeric();
                    let hi = genome.values[start + j + 1].numeric();
                    if let Some((a, b)) = schema.attributes[*attribute].fitted_range
                        && b > a
                    {
                        specificity *= ((hi - lo).abs() / (b - a)).clamp(0.0, 1.0);
                    }
                }
                Locus::NominalCategory {
                    attribute,
                    wildcard,
                } => {
                    let g = genome.values[start + j].numeric();
                    if !*wildcard || g >= 0.0 {
                        let n = schema.attributes[*attribute]
                            .nominal_values()
                            .map_or(1, |v| v.len());
                        specificity *= 1.0 / n as f64;
                    }
                }
                _ => {}
            }
        }
        total += 1.0 - specificity;
        count += 1;
    }
    total / count as f64
}

fn is_out_of_range(row: &Row, schema: &Schema) -> bool {
    schema
        .predictors()
        .any(|(i, a)| match (&a.kind, &row.values[i]) {
            (AttributeKind::Numeric, Some(Value::Numeric(v))) => {
                a.fitted_range.is_some_and(|(lo, hi)| *v < lo || *v > hi)
            }
            (AttributeKind::Integer, Some(Value::Integer(v))) => a
                .fitted_range
                .is_some_and(|(lo, hi)| (*v as f64) < lo || (*v as f64) > hi),
            _ => false,
        })
}

fn cmp_f64(a: f64, b: f64) -> Ordering {
    a.total_cmp(&b)
}
fn lex_cmp(a: &[f64], b: &[f64]) -> Ordering {
    for (x, y) in a.iter().zip(b) {
        let c = x.total_cmp(y);
        if c != Ordering::Equal {
            return c;
        }
    }
    a.len().cmp(&b.len())
}
fn l2_cmp(a: &[f64], b: &[f64]) -> Ordering {
    let naive_a = naive_l2(a);
    let naive_b = naive_l2(b);
    let underflowed_a = naive_a == 0.0 && a.iter().any(|value| *value != 0.0);
    let underflowed_b = naive_b == 0.0 && b.iter().any(|value| *value != 0.0);
    if naive_a.is_finite() && naive_b.is_finite() && !underflowed_a && !underflowed_b {
        return naive_a.total_cmp(&naive_b);
    }

    let (scale_a, root_a) = scaled_l2(a);
    let (scale_b, root_b) = scaled_l2(b);
    match (scale_a == 0.0, scale_b == 0.0) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Less,
        (false, true) => Ordering::Greater,
        (false, false) if scale_a >= scale_b => root_a.total_cmp(&(scale_b / scale_a * root_b)),
        (false, false) => (scale_a / scale_b * root_a).total_cmp(&root_b),
    }
}

fn naive_l2(x: &[f64]) -> f64 {
    x.iter().map(|v| v * v).sum::<f64>().sqrt()
}

fn scaled_l2(x: &[f64]) -> (f64, f64) {
    let scale = x.iter().map(|value| value.abs()).fold(0.0, f64::max);
    if scale == 0.0 {
        return (0.0, 0.0);
    }
    let sum = x
        .iter()
        .map(|value| {
            let normalized = value / scale;
            normalized * normalized
        })
        .sum::<f64>();
    (scale, sum.sqrt())
}

#[cfg(test)]
mod model_numeric_tests {
    use super::l2_cmp;
    use std::cmp::Ordering;

    #[test]
    fn compromise_norm_orders_large_finite_objectives_without_overflow() {
        assert_eq!(l2_cmp(&[1.0e200, 0.0], &[0.0, 2.0e200]), Ordering::Less);
        assert_eq!(
            l2_cmp(&[1.0e308, 1.0e308], &[1.4e308, 0.0]),
            Ordering::Greater
        );
    }

    #[test]
    fn compromise_norm_keeps_ordinary_range_and_equal_norm_ordering() {
        assert_eq!(l2_cmp(&[3.0, 4.0], &[6.0, 8.0]), Ordering::Less);
        assert_eq!(l2_cmp(&[3.0, 4.0], &[0.0, 5.0]), Ordering::Equal);
    }

    #[test]
    fn compromise_norm_handles_tiny_values_and_extreme_scale_ratios() {
        assert_eq!(l2_cmp(&[1.0e-300], &[2.0e-300]), Ordering::Less);
        assert_eq!(
            l2_cmp(&[1.0e-300, 1.0e-300], &[1.0e300, 0.0]),
            Ordering::Less
        );
        assert_eq!(l2_cmp(&[0.0, -0.0], &[-0.0, 0.0]), Ordering::Equal);
    }
}

#[cfg(test)]
mod rule_match_tests {
    use super::{
        LearningProblem, ModelKind, learning_run_semantics_version, model_semantics_version,
        nominal_rule_conditions, nominal_rule_matches, rule_distance,
    };
    use crate::{
        config::{FitnessKind, OperatorsConfig, TrainingConfig},
        data::{Dataset, Value},
        evaluation::Problem,
        genome::{GeneValue, Genome},
        rng::Random,
    };

    fn rule_problem() -> (LearningProblem, Dataset) {
        let train = Dataset::parse_str(
            "@relation rule_match\n@attribute x numeric\n@attribute color {red,blue}\n@attribute class {a,b}\n@data\n0,red,a\n10,blue,b\n",
        )
        .unwrap();
        let problem = LearningProblem::new(
            train.clone(),
            TrainingConfig {
                model: ModelKind::Rule,
                initial_rules: 2,
                wildcard_probability: 0.5,
                ..TrainingConfig::default()
            },
            OperatorsConfig::default(),
        )
        .unwrap();
        (problem, train)
    }

    fn nominal_rule_problem() -> LearningProblem {
        let train = Dataset::parse_str(
            "@relation nominal_rule\n@attribute color {red,blue}\n@attribute shape {square,circle}\n@attribute class {a,b}\n@data\nred,square,a\nblue,circle,b\n",
        )
        .unwrap();
        LearningProblem::new(
            train,
            TrainingConfig {
                model: ModelKind::Rule,
                initial_rules: 2,
                wildcard_probability: 0.5,
                ..TrainingConfig::default()
            },
            OperatorsConfig::default(),
        )
        .unwrap()
    }

    fn reference_matches(
        problem: &LearningProblem,
        genome: &Genome,
        row: &crate::data::Row,
        rule: usize,
    ) -> bool {
        rule_distance(
            row,
            genome,
            rule,
            &problem.block,
            &problem.layout,
            problem.dataset.schema(),
        )
        .unwrap()
        .is_some_and(|distance| distance == 0.0)
    }

    #[test]
    fn mixed_rule_layout_keeps_full_distance_reference_for_edge_cases() {
        let (problem, _) = rule_problem();
        let mut rng = Random::new(42);
        let initial = problem.initialize(&mut rng).unwrap();
        let mut values = initial.values;
        // Rule 0 uses a reversed interval and a nominal wildcard; rule 1 overlaps
        // it and has a concrete blue condition. Targets are a then b.
        values[..4].copy_from_slice(&[
            GeneValue::Real(8.0),
            GeneValue::Real(2.0),
            GeneValue::Integer(-1),
            GeneValue::Integer(0),
        ]);
        values[4..8].copy_from_slice(&[
            GeneValue::Real(0.0),
            GeneValue::Real(10.0),
            GeneValue::Integer(1),
            GeneValue::Integer(1),
        ]);
        let genome = Genome::new(values, problem.layout.clone()).unwrap();
        assert!(nominal_rule_conditions(&problem.block).is_none());
        let heldout = Dataset::parse_str_for_prediction(
            "@relation rule_match_test\n@attribute x numeric\n@attribute color {red,blue}\n@attribute class {a,b}\n@data\n5,red,?\n2,red,?\n8,blue,?\n1,blue,?\n9,red,?\n?,red,?\n?,?,?\n11,blue,?\n",
            problem.dataset.schema(),
        )
        .unwrap();
        let predictions = problem.predict_dataset(&genome, &heldout).unwrap();
        for (row, prediction) in heldout.rows().iter().zip(&predictions) {
            let expected = (0..2).find(|rule| reference_matches(&problem, &genome, row, *rule));
            let expected_value = expected.map(|rule| {
                let target_index = problem.block.len() - 1;
                let class =
                    genome.values[rule * problem.block.len() + target_index].numeric() as usize;
                Value::Nominal(
                    problem.dataset.schema().target().nominal_values().unwrap()[class].clone(),
                )
            });
            assert_eq!(
                prediction.value, expected_value,
                "source row {}",
                row.source_index
            );
            assert_eq!(prediction.abstained, expected.is_none());
        }
        assert_eq!(predictions[0].value, Some(Value::Nominal("a".into())));
        assert_eq!(predictions[1].value, Some(Value::Nominal("a".into()))); // inclusive low boundary
        assert_eq!(predictions[2].value, Some(Value::Nominal("a".into()))); // first overlapping rule wins
        assert_eq!(predictions[3].value, Some(Value::Nominal("b".into())));
        assert!(predictions[4].abstained);
        assert!(!predictions[5].abstained); // missing numeric value is ignored; color is evidence
        assert!(predictions[6].abstained); // all predictors missing gives no evidence
        assert!(predictions[7].abstained);
        assert!(predictions[7].out_of_range);
    }

    #[test]
    fn all_nominal_fast_path_matches_full_distance_with_missing_and_wildcards() {
        let problem = nominal_rule_problem();
        let mut rng = Random::new(91);
        let initial = problem.initialize(&mut rng).unwrap();
        let mut values = initial.values;
        // Block: color, shape, target. Rule 0 covers red and blue squares;
        // rule 1 covers blue of either shape. The first rule wins overlap.
        values[..3].copy_from_slice(&[
            GeneValue::Integer(-1),
            GeneValue::Integer(0),
            GeneValue::Integer(0),
        ]);
        values[3..6].copy_from_slice(&[
            GeneValue::Integer(1),
            GeneValue::Integer(-1),
            GeneValue::Integer(1),
        ]);
        let genome = Genome::new(values, problem.layout.clone()).unwrap();
        let conditions = nominal_rule_conditions(&problem.block).unwrap();
        let heldout = Dataset::parse_str_for_prediction(
            "@relation nominal_test\n@attribute color {red,blue}\n@attribute shape {square,circle}\n@attribute class {a,b}\n@data\nred,square,?\nblue,square,?\nred,circle,?\n?,circle,?\n?,?,?\n",
            problem.dataset.schema(),
        )
        .unwrap();
        for row in heldout.rows() {
            let observed = conditions
                .iter()
                .map(|condition| {
                    row.values[condition.attribute]
                        .as_ref()
                        .map(|value| match value {
                            Value::Nominal(value) => problem.dataset.schema().attributes
                                [condition.attribute]
                                .nominal_values()
                                .and_then(|labels| labels.iter().position(|label| label == value))
                                .map_or(
                                    super::NominalObservation::UnknownNominal,
                                    super::NominalObservation::Known,
                                ),
                            Value::Numeric(_) | Value::Integer(_) => {
                                super::NominalObservation::InvalidType
                            }
                        })
                        .unwrap_or(super::NominalObservation::Missing)
                })
                .collect::<Vec<_>>();
            for rule in 0..2 {
                let reference = reference_matches(&problem, &genome, row, rule);
                let fast =
                    nominal_rule_matches(&genome, rule, &conditions, &observed) == Some(true);
                assert_eq!(fast, reference, "rule {rule}, row {}", row.source_index);
            }
        }
        let predictions = problem.predict_dataset(&genome, &heldout).unwrap();
        assert_eq!(predictions[0].value, Some(Value::Nominal("a".into())));
        assert_eq!(predictions[1].value, Some(Value::Nominal("a".into())));
        assert!(predictions[2].abstained);
        assert_eq!(predictions[3].value, Some(Value::Nominal("b".into())));
        assert!(predictions[4].abstained);
    }

    #[test]
    fn prepared_nominal_training_matches_reference_reports_for_seeded_genomes() {
        let train = Dataset::parse_str(
            "@relation nominal_cache\n@attribute color {red,blue}\n@attribute shape {square,circle}\n@attribute class {a,b}\n@data\nred,square,a\nblue,circle,b\n?,square,a\nblue,?,b\n",
        )
        .unwrap();
        let fitnesses = [
            FitnessKind::PenalizedLoss,
            FitnessKind::LossSize,
            FitnessKind::ArchivedLossSize,
            FitnessKind::LossGeneralitySize,
            FitnessKind::CoverageSize,
        ];
        for fitness in fitnesses {
            let problem = LearningProblem::new(
                train.clone(),
                TrainingConfig {
                    model: ModelKind::Rule,
                    fitness,
                    initial_rules: 4,
                    wildcard_probability: 0.5,
                    ..TrainingConfig::default()
                },
                OperatorsConfig::default(),
            )
            .unwrap();
            let mut rng = Random::new(0x6142_7073);
            for ordinal in 0..200 {
                let mut genome = problem.initialize(&mut rng).unwrap();
                // Include deliberately overlapping rules, concrete conditions,
                // and wildcard sentinels alongside initialized variation.
                for (index, value) in genome.values.iter_mut().enumerate() {
                    if index % problem.block.len() < problem.block.len() - 1 {
                        *value = GeneValue::Integer(match (ordinal + index) % 3 {
                            0 => -1,
                            1 => 0,
                            _ => 1,
                        });
                    }
                }
                let optimized = problem.evaluate(&genome).unwrap();
                let reference = problem.score_dataset(&genome, &train).unwrap();
                assert_eq!(
                    optimized, reference,
                    "fitness {fitness:?}, genome {ordinal}"
                );
            }
        }
    }

    #[test]
    fn numeric_rule_layout_preserves_distance_underflow_and_overflow_behavior() {
        let train = Dataset::parse_str(
            "@relation numeric_rule\n@attribute x numeric\n@attribute class {a,b}\n@data\n0,a\n1e308,b\n",
        )
        .unwrap();
        let problem = LearningProblem::new(
            train.clone(),
            TrainingConfig {
                model: ModelKind::Rule,
                initial_rules: 1,
                ..TrainingConfig::default()
            },
            OperatorsConfig::default(),
        )
        .unwrap();
        let mut rng = Random::new(7);
        let _initial = problem.initialize(&mut rng).unwrap();
        let genome = Genome::new(
            vec![
                GeneValue::Real(0.0),
                GeneValue::Real(0.0),
                GeneValue::Integer(1),
            ],
            problem.layout.clone(),
        )
        .unwrap();
        let tiny_outside = Dataset::parse_str_for_prediction(
            "@relation tiny\n@attribute x numeric\n@attribute class {a,b}\n@data\n1e-200,?\n",
            train.schema(),
        )
        .unwrap();
        let row = &tiny_outside.rows()[0];
        assert_eq!(
            rule_distance(
                row,
                &genome,
                0,
                &problem.block,
                &problem.layout,
                problem.dataset.schema()
            )
            .unwrap(),
            Some(0.0)
        );
        assert_eq!(
            problem.predict_dataset(&genome, &tiny_outside).unwrap()[0].value,
            Some(Value::Nominal("b".into()))
        );

        let mixed_train = Dataset::parse_str(
            "@relation mixed_overflow\n@attribute color {red,blue}\n@attribute x numeric\n@attribute class {a,b}\n@data\nred,0,a\nblue,1,b\n",
        )
        .unwrap();
        let mixed = LearningProblem::new(
            mixed_train.clone(),
            TrainingConfig {
                model: ModelKind::Rule,
                initial_rules: 1,
                ..TrainingConfig::default()
            },
            OperatorsConfig::default(),
        )
        .unwrap();
        assert!(nominal_rule_conditions(&mixed.block).is_none());
        let mut rng = Random::new(8);
        let mut values = mixed.initialize(&mut rng).unwrap().values;
        values[0] = GeneValue::Integer(0); // mismatch on color precedes the numeric overflow
        values[1] = GeneValue::Real(0.0);
        values[2] = GeneValue::Real(1.0);
        values[3] = GeneValue::Integer(0);
        let mixed_genome = Genome::new(values, mixed.layout.clone()).unwrap();
        let far = Dataset::parse_str_for_prediction(
            "@relation far\n@attribute color {red,blue}\n@attribute x numeric\n@attribute class {a,b}\n@data\nblue,1e308,?\n",
            mixed_train.schema(),
        )
        .unwrap();
        let err = mixed
            .predict_dataset(&mixed_genome, &far)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("normalized interval distance overflowed"),
            "{err}"
        );
    }

    fn dataset_with_mutated_cell(mutator: impl FnOnce(&mut serde_json::Value)) -> Dataset {
        let valid = Dataset::parse_str_for_prediction(
            "@relation nominal_test\n@attribute color {red,blue}\n@attribute shape {square,circle}\n@attribute class {a,b}\n@data\nred,square,?\n",
            nominal_rule_problem().dataset.schema(),
        )
        .unwrap();
        let mut serialized = serde_json::to_value(valid).unwrap();
        mutator(&mut serialized);
        serde_json::from_value(serialized).unwrap()
    }

    fn set_cell(row: &mut serde_json::Value, attribute: usize, value: serde_json::Value) {
        row["rows"][0]["values"][attribute] = value;
    }

    fn serialized_value(kind: &str, value: serde_json::Value) -> serde_json::Value {
        serde_json::json!({ "kind": kind, "value": value })
    }

    #[test]
    fn serde_unknown_nominals_count_as_evidence_and_match_wildcards_only() {
        let problem = nominal_rule_problem();
        let mut rng = Random::new(92);
        let mut values = problem.initialize(&mut rng).unwrap().values;
        values[..3].copy_from_slice(&[
            GeneValue::Integer(-1), // wildcard color
            GeneValue::Integer(0),
            GeneValue::Integer(0),
        ]);
        values[3..6].copy_from_slice(&[
            GeneValue::Integer(0), // concrete red
            GeneValue::Integer(0),
            GeneValue::Integer(1),
        ]);
        let genome = Genome::new(values, problem.layout.clone()).unwrap();
        let unknown = dataset_with_mutated_cell(|json| {
            set_cell(
                json,
                0,
                serialized_value("nominal", serde_json::json!("green")),
            );
            set_cell(json, 1, serde_json::Value::Null);
        });
        let row = &unknown.rows()[0];
        assert!(reference_matches(&problem, &genome, row, 0));
        assert!(!reference_matches(&problem, &genome, row, 1));
        assert_eq!(
            problem.predict_dataset(&genome, &unknown).unwrap()[0].value,
            Some(Value::Nominal("a".into()))
        );
    }

    #[test]
    fn serde_unknown_nominal_does_not_hide_a_known_feature_mismatch() {
        let problem = nominal_rule_problem();
        let mut rng = Random::new(93);
        let mut values = problem.initialize(&mut rng).unwrap().values;
        values[..3].copy_from_slice(&[
            GeneValue::Integer(-1), // wildcard color
            GeneValue::Integer(1),  // concrete circle, but row is square
            GeneValue::Integer(0),
        ]);
        values[3..6].copy_from_slice(&[
            GeneValue::Integer(0),
            GeneValue::Integer(0),
            GeneValue::Integer(1),
        ]);
        let genome = Genome::new(values, problem.layout.clone()).unwrap();
        let unknown_and_square = dataset_with_mutated_cell(|json| {
            set_cell(
                json,
                0,
                serialized_value("nominal", serde_json::json!("green")),
            );
        });
        let row = &unknown_and_square.rows()[0];
        assert!(!reference_matches(&problem, &genome, row, 0));
        assert!(!reference_matches(&problem, &genome, row, 1));
        assert!(
            problem
                .predict_dataset(&genome, &unknown_and_square)
                .unwrap()[0]
                .abstained
        );
    }

    #[test]
    fn serde_wrong_types_at_multiple_attributes_never_match_a_rule() {
        let problem = nominal_rule_problem();
        let mut rng = Random::new(94);
        let mut values = problem.initialize(&mut rng).unwrap().values;
        values[..3].copy_from_slice(&[
            GeneValue::Integer(-1),
            GeneValue::Integer(-1),
            GeneValue::Integer(0),
        ]);
        values[3..6].copy_from_slice(&[
            GeneValue::Integer(0),
            GeneValue::Integer(0),
            GeneValue::Integer(1),
        ]);
        let genome = Genome::new(values, problem.layout.clone()).unwrap();

        for (attribute, invalid) in [
            (0, serialized_value("numeric", serde_json::json!(1.0))),
            (1, serialized_value("integer", serde_json::json!(1))),
        ] {
            let invalid_dataset = dataset_with_mutated_cell(|json| {
                set_cell(json, attribute, invalid);
            });
            let row = &invalid_dataset.rows()[0];
            for rule in 0..2 {
                assert!(
                    !reference_matches(&problem, &genome, row, rule),
                    "wrong type at attribute {attribute}, rule {rule}"
                );
            }
            assert!(problem.predict_dataset(&genome, &invalid_dataset).unwrap()[0].abstained);
        }
    }

    fn initialization_problem(
        wildcard_probability: f64,
        initial_generality: f64,
        rules: usize,
    ) -> LearningProblem {
        let train = Dataset::parse_str(
            "@relation nominal_init\n@attribute color {red,blue}\n@attribute shape {square,circle}\n@attribute class {a,b}\n@data\nred,square,a\nblue,circle,b\n",
        )
        .unwrap();
        LearningProblem::new(
            train,
            TrainingConfig {
                model: ModelKind::Rule,
                initial_rules: rules,
                initial_generality,
                wildcard_probability,
                ..TrainingConfig::default()
            },
            OperatorsConfig::default(),
        )
        .unwrap()
    }

    #[test]
    fn initial_nominal_wildcards_follow_generality_independent_of_mutation_setting() {
        let initialized = [0.25, 0.5, 1.0].map(|mutation_wildcard_probability| {
            let problem = initialization_problem(mutation_wildcard_probability, 0.4, 2_000);
            problem.initialize(&mut Random::new(502)).unwrap()
        });
        assert_eq!(initialized[0].values, initialized[1].values);
        assert_eq!(initialized[1].values, initialized[2].values);

        let wildcard_count = initialized[0]
            .values
            .iter()
            .enumerate()
            .filter(|(index, gene)| index % 3 < 2 && gene.numeric() == -1.0)
            .count();
        let wildcard_rate = wildcard_count as f64 / (2 * 2_000) as f64;
        assert!((wildcard_rate - 0.4).abs() < 0.025, "rate={wildcard_rate}");

        for initial_generality in [0.0, 1.0] {
            let problem = initialization_problem(0.5, initial_generality, 128);
            let genome = problem.initialize(&mut Random::new(503)).unwrap();
            let wildcard_count = genome
                .values
                .iter()
                .enumerate()
                .filter(|(index, gene)| index % 3 < 2 && gene.numeric() == -1.0)
                .count();
            assert_eq!(wildcard_count, (initial_generality * 256.0) as usize);
        }
    }

    #[test]
    fn zero_wildcard_domain_remains_valid_when_generality_is_one() {
        let problem = initialization_problem(0.0, 1.0, 128);
        let genome = problem.initialize(&mut Random::new(504)).unwrap();
        genome.validate().unwrap();
        assert!(
            genome
                .values
                .iter()
                .enumerate()
                .filter(|(index, _)| index % 3 < 2)
                .all(|(_, gene)| gene.numeric() >= 0.0)
        );
    }

    #[test]
    fn learning_run_semantics_version_tracks_initialization_without_changing_saved_model_version() {
        assert_eq!(learning_run_semantics_version(ModelKind::Rule), 3);
        assert_eq!(learning_run_semantics_version(ModelKind::Mixed), 3);
        assert_eq!(learning_run_semantics_version(ModelKind::RegressionRule), 3);
        assert_eq!(learning_run_semantics_version(ModelKind::Instance), 1);
        assert_eq!(
            learning_run_semantics_version(ModelKind::RegressionLinearRule),
            model_semantics_version(ModelKind::RegressionLinearRule)
        );
        assert_eq!(model_semantics_version(ModelKind::Rule), 1);
    }
}
