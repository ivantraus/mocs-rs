//! Run configuration, deserialized from TOML.
//!
//! [`RunConfig`] is the root: its top-level keys (`seed`, `population`, ...) sit at
//! the top of the file and the nested structs are the `[algorithm]`, `[operators]`,
//! `[problem]` and `[training]` tables. Every field is optional and defaults to the
//! values documented here; unknown keys are rejected. Enum values are written in
//! snake case (for example `kind = "nsga2"`). [`RunConfig::load`] merges several
//! files and validates the result.

use std::{
    fs,
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};

use crate::{
    Result,
    genome::{Domain, GenomeLayout},
};

/// Evolutionary algorithm (`algorithm.kind`, default `nsga2`).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AlgorithmKind {
    /// `moga`: Fonseca–Fleming Pareto ranking with fitness sharing and tournament parent selection.
    Moga,
    /// `moes`: multi-objective evolution strategy; MOGA ranking and sharing, but the whole
    /// population reproduces without parent tournaments.
    Moes,
    /// `ngga`: steady-state non-generational GA; each offspring replaces the member with the
    /// worst weighted dominance/sharing value.
    Ngga,
    /// `nsga2`: NSGA-II nondominated sorting with crowding distance.
    #[default]
    Nsga2,
    /// `spea2`: SPEA2 strength fitness with k-nearest-neighbour density.
    Spea2,
}

/// The `[algorithm]` table: algorithm choice and selection settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AlgorithmConfig {
    /// `algorithm.kind` (default `nsga2`).
    pub kind: AlgorithmKind,
    /// `algorithm.raw_crowding` (default `false`): NSGA-II crowding uses raw objective
    /// differences instead of differences divided by each objective's front range.
    pub raw_crowding: bool,
    /// `algorithm.sharing` (default `0.1`, nonnegative): niche radius, as Euclidean
    /// distance in objective space, for MOGA/MOES fitness sharing and NGGA sharing.
    pub sharing: f64,
    /// `algorithm.elitism`: whether old members compete with offspring for survival.
    /// Absent (the default) means the algorithm default: true for nsga2 and spea2,
    /// false for moga and moes. NGGA rejects an explicit value.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub elitism: Option<bool>,
    /// `algorithm.best` (default `0`): when positive, the old nondominated members and
    /// the `best` old members by first objective are offered to survivor selection.
    /// Must be below `population`; not allowed for NGGA, nor with elitism unless
    /// `archived_add_best` is set.
    pub best: usize,
    /// `algorithm.archived_add_best` (default `false`): reproduce the upstream
    /// post-selection addBest retention path, appending old front and `best`
    /// members after truncation (so the population may exceed its size).
    pub archived_add_best: bool,
    /// `algorithm.archived_moes_tournament_preselection` (default `false`): for MOES,
    /// reproduce the upstream tournament preselection of an offspring-sized
    /// reproductive pool before mating.
    pub archived_moes_tournament_preselection: bool,
    /// `algorithm.tournament_size` (default `2`, in `1..=population`): size of the
    /// tournaments that draw the reproductive pool from the population (not used by
    /// NGGA, nor by MOES without archived preselection).
    pub tournament_size: usize,
    /// `algorithm.ngga_n1`: random comparisons per member when (re)initializing NGGA
    /// dominance and sharing counts. Default: population / 3, rounded down.
    pub ngga_n1: Option<usize>,
    /// `algorithm.ngga_n2`: random pair comparisons that update NGGA counts before each
    /// offspring. Default: population / 3, rounded down.
    pub ngga_n2: Option<usize>,
    /// `algorithm.ngga_decay_dominance` (default `0`, in `[0, 1]`): fraction of an NGGA
    /// dominance count lost at each comparison.
    pub ngga_decay_dominance: f64,
    /// `algorithm.ngga_decay_sharing` (in `[0, 1]`): fraction of an NGGA sharing count
    /// lost at each comparison. Default: 1 / population.
    pub ngga_decay_sharing: Option<f64>,
    /// `algorithm.ngga_dominance_weight` (default `10`, nonnegative): weight of the
    /// dominance count in the NGGA replacement value.
    pub ngga_dominance_weight: f64,
    /// `algorithm.ngga_sharing_weight` (default `0.05`, nonnegative): weight of the
    /// sharing count in the NGGA replacement value.
    pub ngga_sharing_weight: f64,
    /// `algorithm.spea2_k` (in `1..=population`): neighbour rank used for SPEA2
    /// density, counting the member itself. Default: `floor(sqrt(2 * population))`.
    pub spea2_k: Option<usize>,
    /// `algorithm.push_equal` (default `false`): NSGA-II keeps one member per identical
    /// objective vector in each front; extra copies are used only after every front.
    pub push_equal: bool,
    /// `algorithm.push_more_than_two_equal` (default `false`): like `push_equal`, but
    /// keeps two members per identical objective vector (takes precedence).
    pub push_more_than_two_equal: bool,
}
impl Default for AlgorithmConfig {
    fn default() -> Self {
        Self {
            kind: AlgorithmKind::Nsga2,
            raw_crowding: false,
            sharing: 0.1,
            elitism: None,
            best: 0,
            archived_add_best: false,
            archived_moes_tournament_preselection: false,
            tournament_size: 2,
            ngga_n1: None,
            ngga_n2: None,
            ngga_decay_dominance: 0.0,
            ngga_decay_sharing: None,
            ngga_dominance_weight: 10.0,
            ngga_sharing_weight: 0.05,
            spea2_k: None,
            push_equal: false,
            push_more_than_two_equal: false,
        }
    }
}

impl AlgorithmConfig {
    /// Effective survivor elitism after applying the algorithm default.
    /// NSGA-II and SPEA2 default to merging parents and offspring; MOGA, MOES
    /// and NGGA default to false (NGGA has no survivor-pool elitism setting).
    pub fn effective_elitism(&self) -> bool {
        self.elitism.unwrap_or(matches!(
            self.kind,
            AlgorithmKind::Nsga2 | AlgorithmKind::Spea2
        ))
    }
}

/// Built-in numeric benchmark (`problem.kind`, default `sf2`). All have two objectives.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProblemKind {
    /// `sf2`: Schaffer's F2, one variable in `[-6, 6]` by default.
    #[default]
    Sf2,
    /// `kur`: Kursawe, three variables in `[-5, 5]` by default.
    Kur,
    /// `okk`: constrained seven-variable robot-gripper design with fixed variable bounds.
    Okk,
    /// `zdt6`: ZDT6, ten variables in `[0, 1]` by default.
    Zdt6,
}
/// The `[problem]` table, used by numeric optimization runs.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ProblemConfig {
    /// `problem.kind` (default `sf2`).
    pub kind: ProblemKind,
    /// `problem.dimensions`: number of variables. Default per kind: SF2 1, KUR 3, OKK 7,
    /// ZDT6 10; SF2 and OKK accept only their default, KUR and ZDT6 need at least 2.
    pub dimensions: Option<usize>,
    /// `problem.min`: lower bound of every variable (default per kind; ignored by OKK).
    pub min: Option<f64>,
    /// `problem.max`: upper bound of every variable (default per kind; ignored by OKK).
    /// Must exceed `min`.
    pub max: Option<f64>,
}
impl Default for ProblemConfig {
    fn default() -> Self {
        Self {
            kind: ProblemKind::Sf2,
            dimensions: None,
            min: None,
            max: None,
        }
    }
}

/// Learning model (`training.model`, default `rule`).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelKind {
    /// `rule`: ordered interval/category rules; a row takes the class of the first
    /// matching rule, or abstains.
    #[default]
    Rule,
    /// `instance`: labelled prototypes; a row takes the class of the nearest one.
    Instance,
    /// `mixed`: rules used as nearest-rule classifiers (distance 0 inside a rule).
    Mixed,
    /// `regression_rule`: rules with constant numeric consequences, blended over the
    /// nearest `training.neighbours` rules.
    RegressionRule,
    /// `regression_linear_rule`: experimental affine consequences for exactly one
    /// numeric predictor.
    RegressionLinearRule,
}
/// Objectives minimized during learning (`training.fitness`, default `loss_size`).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FitnessKind {
    /// `penalized_loss`: one objective, loss + size.
    PenalizedLoss,
    /// `loss_size`: loss and size.
    #[default]
    LossSize,
    /// `archived_loss_size`: loss and size, but `[1, 1]` when the rule count exceeds the
    /// training-row count (classification only).
    ArchivedLossSize,
    /// `loss_generality_size`: loss, generality and size.
    LossGeneralitySize,
    /// `coverage_size`: conditional loss, 1 − coverage, and size.
    CoverageSize,
}
/// How the final model is chosen from the front (`training.decision`, default `best_accuracy`).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DecisionKind {
    /// `best_accuracy`: lowest training loss, then smallest size.
    #[default]
    BestAccuracy,
    /// `compromise`: front member closest to the origin in objective space.
    Compromise,
}
/// The `[training]` table, used by classification and regression runs.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TrainingConfig {
    /// `training.model` (default `rule`).
    pub model: ModelKind,
    /// `training.fitness` (default `loss_size`).
    pub fitness: FitnessKind,
    /// `training.decision` (default `best_accuracy`).
    pub decision: DecisionKind,
    /// `training.target`: name of the target attribute, passed by the CLI to
    /// [`Dataset::load`](crate::data::Dataset::load). Default: the ARFF `@target`
    /// attribute if declared, else the last attribute.
    pub target: Option<String>,
    /// `training.initial_rules` (default `15`, positive): rules (or instances) per initial genome.
    pub initial_rules: usize,
    /// `training.prune_probability` (default `0`, in `[0, 1]`): probability that a newly
    /// evaluated genome has its unused rules removed and is re-evaluated.
    pub prune_probability: f64,
    /// `training.initial_generality` (default `0.75`, in `[0, 1]`): probability that an
    /// initial numeric interval spans the full training range, or that an initial
    /// nominal condition is a wildcard (where wildcards are enabled).
    pub initial_generality: f64,
    /// `training.wildcard_probability` (default `0.5`, in `[0, 1]`): approximate
    /// probability that resampling a nominal condition yields a wildcard; zero
    /// disables nominal wildcards.
    pub wildcard_probability: f64,
    /// `training.neighbours` (default `2`, positive): nearest rules blended in a
    /// regression prediction (capped by the rule count).
    pub neighbours: usize,
    /// `training.temperature` (default `1`, nonnegative): inverse temperature of the
    /// softmax weighting over regression rule distances; 0 weights them equally.
    pub temperature: f64,
}
impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            model: ModelKind::Rule,
            fitness: FitnessKind::LossSize,
            decision: DecisionKind::BestAccuracy,
            target: None,
            initial_rules: 15,
            prune_probability: 0.0,
            initial_generality: 0.75,
            wildcard_probability: 0.5,
            neighbours: 2,
            temperature: 1.0,
        }
    }
}

/// Parent-pair selection (`operators.mating`, default `random`).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MatingKind {
    /// `random`: two distinct uniformly chosen members.
    #[default]
    Random,
    /// `tournament`: tournaments of `mating_tournament_size` candidates (see
    /// `mating_winners` and `fitness_tournament`).
    Tournament,
    /// `nearest_neighbour`: a random member and the nearest (in objective space) of
    /// `operators.neighbours` sampled candidates.
    NearestNeighbour,
    /// `fitness_proportionate`: two-pointer stochastic universal sampling that favours
    /// lower selection scores.
    FitnessProportionate,
}
/// Crossover operator (`operators.crossover`, default `k_point`).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CrossoverKind {
    /// `k_point`: swap segments between `crossover_points` shared cut points.
    #[default]
    KPoint,
    /// `variable_length`: independent cut points per parent, aligned to rule blocks, so
    /// child lengths can change.
    VariableLength,
    /// `simulated_binary`: SBX for equal-length, all-real genomes.
    SimulatedBinary,
}
/// Mutation operator (`operators.mutation`, default `uniform`).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MutationKind {
    /// `uniform`: resample each locus uniformly with its per-locus probability.
    #[default]
    Uniform,
    /// `normal`: self-adaptive Gaussian steps with per-locus step sizes.
    Normal,
    /// `polynomial`: polynomial mutation with index `mutation_eta` at the genome-wide rate.
    Polynomial,
    /// `adaptive`: polynomial mutation whose genome-wide rate self-adapts in
    /// `[adaptive_epsilon, 1]`.
    Adaptive,
}
/// The `[operators]` table: mating, crossover and mutation settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct OperatorsConfig {
    /// `operators.mating` (default `random`).
    pub mating: MatingKind,
    /// `operators.crossover` (default `k_point`).
    pub crossover: CrossoverKind,
    /// `operators.mutation` (default `uniform`).
    pub mutation: MutationKind,
    /// `operators.crossover_probability` (default `0.8`, in `[0, 1]`): probability that
    /// an offspring comes from crossover rather than copying one member.
    pub crossover_probability: f64,
    /// `operators.mutation_probability` (default `1`, in `[0, 1]`): probability that an
    /// offspring is passed to mutation.
    pub mutation_probability: f64,
    /// `operators.archived_evaluated_copy_uniform_mutation_gate` (default `false`):
    /// reproduce upstream uniform mutation's unconditional mutation of evaluated
    /// copies. Only for uniform mutation under moga or moes.
    pub archived_evaluated_copy_uniform_mutation_gate: bool,
    /// `operators.per_locus_probability` (in `[0, 1]`): initial mutation rate per locus
    /// and per genome. Default: 1 / number of loci in one genome block.
    pub per_locus_probability: Option<f64>,
    /// `operators.crossover_points` (default `1`, positive): number of cut points.
    pub crossover_points: usize,
    /// `operators.crossover_alignment` (default `1`, positive): block width for
    /// `variable_length` crossover; 1 derives it from the genome, any other value must
    /// equal it.
    pub crossover_alignment: usize,
    /// `operators.sbx_eta` (default `15`, positive): SBX distribution index.
    pub sbx_eta: f64,
    /// `operators.sbx_shared_beta` (default `false`): draw one SBX spread factor per
    /// crossover instead of one per locus.
    pub sbx_shared_beta: bool,
    /// `operators.mutation_eta` (default `20`, positive): polynomial-mutation distribution index.
    pub mutation_eta: f64,
    /// `operators.normal_epsilon` (default `1e-10`, positive): minimum step size of
    /// normal mutation.
    pub normal_epsilon: f64,
    /// `operators.adaptive_epsilon` (default `1e-10`, in `(0, 1]`): minimum adaptive
    /// mutation rate, also the floor for SBX-blended deviations.
    pub adaptive_epsilon: f64,
    /// `operators.mating_tournament_size` (default `4`, positive): candidates per
    /// mating tournament.
    pub mating_tournament_size: usize,
    /// `operators.mating_winners` (default `2`): 2 takes both parents from one
    /// tournament, 1 runs two independent tournaments.
    pub mating_winners: usize,
    /// `operators.neighbours` (default `10`, positive): candidates sampled by
    /// nearest-neighbour mating.
    pub neighbours: usize,
    /// `operators.fitness_tournament` (default `true`): tournaments compare selection
    /// scores only; `false` compares Pareto dominance first.
    pub fitness_tournament: bool,
}
impl Default for OperatorsConfig {
    fn default() -> Self {
        Self {
            mating: MatingKind::Random,
            crossover: CrossoverKind::KPoint,
            mutation: MutationKind::Uniform,
            crossover_probability: 0.8,
            mutation_probability: 1.0,
            archived_evaluated_copy_uniform_mutation_gate: false,
            per_locus_probability: None,
            crossover_points: 1,
            crossover_alignment: 1,
            sbx_eta: 15.0,
            sbx_shared_beta: false,
            mutation_eta: 20.0,
            normal_epsilon: 1e-10,
            adaptive_epsilon: 1e-10,
            mating_tournament_size: 4,
            mating_winners: 2,
            neighbours: 10,
            fitness_tournament: true,
        }
    }
}

impl OperatorsConfig {
    /// Reject crossover settings that cannot run on any genome of the configured
    /// learning model. This needs only the configuration, so CLIs call it before
    /// loading data; [`OperatorsConfig::validate_for_layout`] completes the check
    /// once the data-derived genome layout is known.
    pub fn validate_for_training(&self, training: &TrainingConfig) -> Result<()> {
        if self.crossover != CrossoverKind::SimulatedBinary {
            return Ok(());
        }
        let model = match training.model {
            ModelKind::Rule => Some("rule"),
            ModelKind::Instance => Some("instance"),
            ModelKind::Mixed => Some("mixed"),
            ModelKind::RegressionRule | ModelKind::RegressionLinearRule => None,
        };
        if let Some(model) = model {
            anyhow::bail!(
                "operators.crossover = \"simulated_binary\" requires real-valued genes, but \
                 training.model = \"{model}\" encodes the class label as an integer locus; \
                 use operators.crossover = \"k_point\" or \"variable_length\""
            );
        }
        anyhow::ensure!(
            training.prune_probability == 0.0,
            "operators.crossover = \"simulated_binary\" requires equal-length genomes, but \
             training.prune_probability > 0 removes unused rules; set \
             training.prune_probability = 0 or use operators.crossover = \"k_point\" or \
             \"variable_length\""
        );
        Ok(())
    }

    /// Reject crossover settings that the concrete genome layout cannot support.
    /// `lengths_vary` is true when genomes may hold different numbers of layout
    /// blocks during a run, as pruned learning genomes do.
    pub fn validate_for_layout(&self, layout: &GenomeLayout, lengths_vary: bool) -> Result<()> {
        let width = layout.domains.len();
        match self.crossover {
            CrossoverKind::SimulatedBinary => {
                anyhow::ensure!(
                    layout
                        .domains
                        .iter()
                        .all(|domain| matches!(domain, Domain::Real { .. })),
                    "operators.crossover = \"simulated_binary\" requires real-valued genes, but \
                     this genome has integer loci (nominal predictors or class labels); use \
                     operators.crossover = \"k_point\" or \"variable_length\""
                );
                anyhow::ensure!(
                    !lengths_vary,
                    "operators.crossover = \"simulated_binary\" requires equal-length genomes, \
                     but training.prune_probability > 0 removes unused rules; set \
                     training.prune_probability = 0 or use operators.crossover = \"k_point\" \
                     or \"variable_length\""
                );
            }
            CrossoverKind::VariableLength => {
                anyhow::ensure!(
                    self.crossover_alignment == 1 || self.crossover_alignment == width,
                    "operators.crossover_alignment = {} does not match the genome block width \
                     {width} required by operators.crossover = \"variable_length\"; use \
                     operators.crossover_alignment = 1 (derive the width) or {width}",
                    self.crossover_alignment
                );
            }
            CrossoverKind::KPoint => {}
        }
        Ok(())
    }
}

/// Complete configuration of a run: the root of a TOML config file.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RunConfig {
    /// `seed` (default `0`): seed of the run's random stream.
    pub seed: u64,
    /// `population` (default `100`, at least 2): survivors kept each generation.
    pub population: usize,
    /// `offspring` (default `100`, positive): offspring per generation (NGGA: steady-state
    /// replacements per generation). Without elitism, must be at least `population`.
    pub offspring: usize,
    /// `generations` (default `100`): generation limit; 0 is allowed only with
    /// `max_evaluations`, which then bounds the run.
    pub generations: usize,
    /// `max_evaluations` (default none): evaluation budget. No generation starts once
    /// it is reached, so the final count may exceed it. Must cover the initial population.
    pub max_evaluations: Option<usize>,
    /// `threads` (positive): evaluation threads; results do not depend on it. Default:
    /// available parallelism with the `parallel` feature, otherwise 1 (the only
    /// allowed value without that feature).
    pub threads: usize,
    /// The `[algorithm]` table.
    pub algorithm: AlgorithmConfig,
    /// The `[operators]` table.
    pub operators: OperatorsConfig,
    /// The `[problem]` table.
    pub problem: ProblemConfig,
    /// The `[training]` table.
    pub training: TrainingConfig,
}
impl Default for RunConfig {
    fn default() -> Self {
        Self {
            seed: 0,
            population: 100,
            offspring: 100,
            generations: 100,
            max_evaluations: None,
            threads: default_threads(),
            algorithm: AlgorithmConfig::default(),
            operators: OperatorsConfig::default(),
            problem: ProblemConfig::default(),
            training: TrainingConfig::default(),
        }
    }
}
impl RunConfig {
    /// Copy with algorithm-dependent defaults (currently `algorithm.elitism`) made
    /// explicit, for run records.
    pub fn resolved(&self) -> Self {
        let mut out = self.clone();
        if out.algorithm.kind != AlgorithmKind::Ngga {
            out.algorithm.elitism = Some(out.algorithm.effective_elitism());
        }
        out
    }
    /// Check ranges and cross-field constraints, returning a descriptive error that
    /// names the offending key. [`RunConfig::load`] calls this; callers that change
    /// fields afterwards should call it again (the [`run`](crate::algorithms::run)
    /// entry points do).
    pub fn validate(&self) -> Result<()> {
        anyhow::ensure!(self.population >= 2, "population must be at least 2");
        anyhow::ensure!(self.offspring > 0, "offspring must be positive");
        anyhow::ensure!(
            self.generations > 0 || self.max_evaluations.is_some(),
            "generations must be positive unless an evaluation budget is set"
        );
        if let Some(cap) = self.max_evaluations {
            anyhow::ensure!(
                cap >= self.population,
                "max_evaluations must cover the initial population evaluation count"
            );
        }
        anyhow::ensure!(self.threads > 0, "threads must be positive");
        #[cfg(not(feature = "parallel"))]
        anyhow::ensure!(
            self.threads == 1,
            "threads > 1 require rebuilding with --features parallel"
        );
        anyhow::ensure!(
            self.algorithm.sharing.is_finite() && self.algorithm.sharing >= 0.0,
            "algorithm.sharing must be finite and nonnegative"
        );
        for (name, value) in [
            (
                "algorithm.ngga_decay_dominance",
                self.algorithm.ngga_decay_dominance,
            ),
            (
                "algorithm.ngga_dominance_weight",
                self.algorithm.ngga_dominance_weight,
            ),
            (
                "algorithm.ngga_sharing_weight",
                self.algorithm.ngga_sharing_weight,
            ),
        ] {
            anyhow::ensure!(
                value.is_finite() && value >= 0.0,
                "{name} must be finite and nonnegative"
            );
        }
        anyhow::ensure!(
            self.operators.adaptive_epsilon <= 1.0,
            "operators.adaptive_epsilon must be <= 1"
        );
        if let Some(value) = self.algorithm.ngga_decay_sharing {
            anyhow::ensure!(
                value.is_finite() && (0.0..=1.0).contains(&value),
                "algorithm.ngga_decay_sharing must be finite and in [0,1]"
            );
        }
        anyhow::ensure!(
            self.algorithm.ngga_decay_dominance <= 1.0,
            "NGGA decays must be in [0,1]"
        );
        if let Some(k) = self.algorithm.spea2_k {
            anyhow::ensure!(
                k > 0 && k <= self.population,
                "algorithm.spea2_k must be in 1..=population"
            );
        }
        anyhow::ensure!(
            self.algorithm.best < self.population,
            "algorithm.best must be smaller than population"
        );
        let kind = self.algorithm.kind;
        let elitism = self.algorithm.effective_elitism();
        if kind == AlgorithmKind::Ngga {
            anyhow::ensure!(
                self.algorithm.elitism.is_none(),
                "algorithm.elitism does not apply to ngga (steady-state replacement of the worst member); remove it"
            );
            anyhow::ensure!(
                self.algorithm.best == 0,
                "algorithm.best does not apply to ngga; remove it or set it to 0"
            );
            anyhow::ensure!(
                !self.algorithm.archived_add_best,
                "algorithm.archived_add_best is supported only for moga, moes, nsga2 and spea2"
            );
        }
        anyhow::ensure!(
            !self.algorithm.archived_add_best
                || matches!(kind, AlgorithmKind::Moga | AlgorithmKind::Moes)
                || self.algorithm.best > 0,
            "algorithm.archived_add_best for {} requires algorithm.best > 0",
            kind_name(kind)
        );
        anyhow::ensure!(
            kind == AlgorithmKind::Ngga
                || self.algorithm.best == 0
                || !elitism
                || self.algorithm.archived_add_best,
            "algorithm.best has no effect for {} with elitism, because every previous member already competes for survival; set algorithm.elitism = false or algorithm.archived_add_best = true",
            kind_name(kind)
        );
        anyhow::ensure!(
            !self.algorithm.archived_moes_tournament_preselection
                || self.algorithm.kind == AlgorithmKind::Moes,
            "algorithm.archived_moes_tournament_preselection is supported only for moes"
        );
        anyhow::ensure!(
            !self.algorithm.archived_moes_tournament_preselection
                || self.algorithm.archived_add_best
                || self.algorithm.best == 0,
            "MOES archived tournament preselection with algorithm.best > 0 requires algorithm.archived_add_best"
        );
        anyhow::ensure!(
            !self.operators.archived_evaluated_copy_uniform_mutation_gate
                || self.operators.mutation == MutationKind::Uniform,
            "operators.archived_evaluated_copy_uniform_mutation_gate requires uniform mutation"
        );
        anyhow::ensure!(
            !self.operators.archived_evaluated_copy_uniform_mutation_gate
                || matches!(
                    self.algorithm.kind,
                    AlgorithmKind::Moga | AlgorithmKind::Moes
                ),
            "operators.archived_evaluated_copy_uniform_mutation_gate is supported only for moga and moes"
        );
        if kind != AlgorithmKind::Ngga && !elitism {
            anyhow::ensure!(
                self.offspring >= self.population,
                "{} without elitism requires offspring >= population",
                kind_name(kind)
            );
        }
        anyhow::ensure!(
            self.algorithm.tournament_size > 0 && self.algorithm.tournament_size <= self.population,
            "algorithm.tournament_size must be in 1..=population"
        );
        for (name, p) in [
            (
                "operators.crossover_probability",
                self.operators.crossover_probability,
            ),
            (
                "operators.mutation_probability",
                self.operators.mutation_probability,
            ),
            (
                "training.prune_probability",
                self.training.prune_probability,
            ),
            (
                "training.initial_generality",
                self.training.initial_generality,
            ),
            (
                "training.wildcard_probability",
                self.training.wildcard_probability,
            ),
        ] {
            anyhow::ensure!(
                p.is_finite() && (0.0..=1.0).contains(&p),
                "{name} must be finite and in [0,1]"
            );
        }
        if let Some(p) = self.operators.per_locus_probability {
            anyhow::ensure!(
                p.is_finite() && (0.0..=1.0).contains(&p),
                "operators.per_locus_probability must be in [0,1]"
            );
        }
        anyhow::ensure!(
            self.operators.crossover_points > 0,
            "operators.crossover_points must be positive"
        );
        anyhow::ensure!(
            self.operators.crossover_alignment > 0,
            "operators.crossover_alignment must be positive"
        );
        anyhow::ensure!(
            self.operators.mating_tournament_size > 0,
            "operators.mating_tournament_size must be positive"
        );
        anyhow::ensure!(
            self.operators.mating_winners > 0
                && self.operators.mating_winners <= self.operators.mating_tournament_size,
            "operators.mating_winners must be in 1..=mating_tournament_size"
        );
        anyhow::ensure!(
            self.operators.mating_winners <= 2,
            "operators.mating_winners must be 1 (two independent tournaments) or 2 (both parents from one tournament)"
        );
        anyhow::ensure!(
            self.operators.neighbours > 0,
            "operators.neighbours must be positive"
        );
        anyhow::ensure!(
            self.operators.sbx_eta.is_finite() && self.operators.sbx_eta > 0.0,
            "operators.sbx_eta must be positive"
        );
        anyhow::ensure!(
            self.operators.mutation_eta.is_finite() && self.operators.mutation_eta > 0.0,
            "operators.mutation_eta must be positive"
        );
        for (name, value) in [
            ("operators.normal_epsilon", self.operators.normal_epsilon),
            (
                "operators.adaptive_epsilon",
                self.operators.adaptive_epsilon,
            ),
        ] {
            anyhow::ensure!(
                value.is_finite() && value > 0.0,
                "{name} must be finite and positive"
            );
        }
        anyhow::ensure!(
            self.training.initial_rules > 0,
            "training.initial_rules must be positive"
        );
        anyhow::ensure!(
            self.training.neighbours > 0,
            "training.neighbours must be positive"
        );
        anyhow::ensure!(
            self.training.temperature.is_finite() && self.training.temperature >= 0.0,
            "training.temperature must be finite and nonnegative"
        );
        if let Some(d) = self.problem.dimensions {
            let min = if self.problem.kind == ProblemKind::Sf2 {
                1
            } else {
                2
            };
            anyhow::ensure!(
                d >= min,
                "problem.dimensions must be at least {min} for {:?}",
                self.problem.kind
            );
        }
        for (name, value) in [
            ("problem.min", self.problem.min),
            ("problem.max", self.problem.max),
        ] {
            if let Some(value) = value {
                anyhow::ensure!(value.is_finite(), "{name} must be finite");
            }
        }
        if let (Some(a), Some(b)) = (self.problem.min, self.problem.max) {
            anyhow::ensure!(
                a.is_finite() && b.is_finite() && a < b,
                "problem bounds must be finite and min < max"
            );
            anyhow::ensure!(
                (b - a).is_finite(),
                "problem bounds must have a finite span"
            );
        }
        Ok(())
    }
    /// Load native TOML files in order. Tables merge field by field; changing an enum tag replaces that table.
    pub fn load(paths: &[PathBuf]) -> Result<Self> {
        let cfg = Self::load_unvalidated(paths)?;
        cfg.validate()?;
        Ok(cfg)
    }

    /// Decode and merge files without checking cross-field constraints. This is
    /// for CLIs that apply explicit overrides before validating the final config.
    pub fn load_unvalidated(paths: &[PathBuf]) -> Result<Self> {
        let mut merged = toml::Value::Table(Default::default());
        for path in paths {
            let src = fs::read_to_string(path)
                .map_err(|e| anyhow::anyhow!("read config {}: {e}", path.display()))?;
            let mut next: toml::Value = toml::from_str(&src)
                .map_err(|e| anyhow::anyhow!("parse config {}: {e}", path.display()))?;
            resolve_declared_paths(&mut next, path.parent().unwrap_or(Path::new(".")));
            merge(&mut merged, next);
        }
        let cfg: Self = merged
            .try_into()
            .map_err(|e| anyhow::anyhow!("decode merged config: {e}"))?;
        Ok(cfg)
    }
}
fn kind_name(kind: AlgorithmKind) -> &'static str {
    match kind {
        AlgorithmKind::Moga => "moga",
        AlgorithmKind::Moes => "moes",
        AlgorithmKind::Ngga => "ngga",
        AlgorithmKind::Nsga2 => "nsga2",
        AlgorithmKind::Spea2 => "spea2",
    }
}
fn default_threads() -> usize {
    #[cfg(feature = "parallel")]
    {
        std::thread::available_parallelism().map_or(1, usize::from)
    }
    #[cfg(not(feature = "parallel"))]
    {
        1
    }
}
fn resolve_declared_paths(_v: &mut toml::Value, _base: &Path) { /* path fields are added by workflow config extensions */
}
fn merge(base: &mut toml::Value, next: toml::Value) {
    match (base, next) {
        (toml::Value::Table(dst), toml::Value::Table(src)) => {
            for (k, v) in src {
                if let Some(old) = dst.get_mut(&k) {
                    let enum_changed = [
                        "kind",
                        "model",
                        "fitness",
                        "decision",
                        "mating",
                        "crossover",
                        "mutation",
                    ]
                    .iter()
                    .any(|tag| old.get(*tag).zip(v.get(*tag)).is_some_and(|(a, b)| a != b));
                    if enum_changed {
                        *old = v;
                    } else {
                        merge(old, v);
                    }
                } else {
                    dst.insert(k, v);
                }
            }
        }
        (slot, value) => *slot = value,
    }
}

#[cfg(test)]
mod execution_defaults_tests {
    use super::RunConfig;

    #[test]
    fn thread_default_tracks_build_feature_and_explicit_serial_is_preserved() {
        let mut config = RunConfig::default();
        #[cfg(feature = "parallel")]
        assert_eq!(
            config.threads,
            std::thread::available_parallelism().map_or(1, usize::from)
        );
        #[cfg(not(feature = "parallel"))]
        assert_eq!(config.threads, 1);
        config.threads = 1;
        assert_eq!(config.threads, 1);
        assert!(config.validate().is_ok());
    }

    #[cfg(not(feature = "parallel"))]
    #[test]
    fn thread_count_above_one_is_rejected_without_parallel_feature() {
        let config = RunConfig {
            threads: 2,
            ..RunConfig::default()
        };
        assert!(
            config
                .validate()
                .unwrap_err()
                .to_string()
                .contains("--features parallel")
        );
    }
}
