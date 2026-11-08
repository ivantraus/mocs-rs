//! Evolutionary multi-objective algorithms (MOGA, MOES, NGGA, NSGA-II, SPEA2) and
//! the [`run`] entry points that drive a [`Problem`] to a final Pareto front.
//!
//! A run is fully determined by its [`RunConfig`] (including the seed); thread count
//! does not change results.

use crate::{
    Result,
    config::{AlgorithmKind, RunConfig},
    evaluation::Problem,
    execution::ExecutionContext,
    genome::Individual,
    objectives::{crowding, dominates, fronts, nondominated_indices},
    operators::{
        BuiltInOperators, EvolutionOperators, evaluate_dirty_with_context,
        evaluate_dirty_with_threads,
    },
    rng::Random,
};
use anyhow::Context;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Outcome of an evolutionary run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunResult {
    /// Nondominated members of the final population, sorted by objectives then ID.
    pub front: Vec<Individual>,
    /// One record per completed generation, starting with the initial population (generation 0).
    pub history: Vec<GenerationRecord>,
    /// Total problem evaluations, including re-evaluations after pruning.
    pub evaluations: usize,
    /// Number of generations completed (may be below `generations` when `max_evaluations` stops the run).
    pub generations: usize,
}
/// Snapshot of the population after one generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationRecord {
    /// Generation number; 0 is the initial population.
    pub generation: usize,
    /// Cumulative evaluations at the end of this generation.
    pub evaluations: usize,
    /// Objective vectors of the population's nondominated members.
    pub front: Vec<Vec<f64>>,
    /// Candidate IDs in survivor order, retained for deterministic audit traces.
    pub population_ids: Vec<u64>,
}

/// Run the configured algorithm with the built-in operators.
///
/// Validates `config` (see [`RunConfig::validate`]) and creates an
/// [`ExecutionContext`] with `config.threads` workers. Errors on invalid
/// configuration, problem evaluation errors, inconsistent objective counts, or an
/// initial population that already exceeds `max_evaluations`.
pub fn run<P: Problem>(problem: &P, config: &RunConfig) -> Result<RunResult> {
    run_with_operators(problem, config, &BuiltInOperators)
}

/// Run with custom mating/crossover/mutation behavior while retaining the same
/// deterministic algorithm, evaluation, pruning, and survivor-selection paths.
pub fn run_with_operators<P, O>(problem: &P, config: &RunConfig, operators: &O) -> Result<RunResult>
where
    P: Problem,
    O: EvolutionOperators + ?Sized,
{
    config.validate()?;
    let context = ExecutionContext::new(config.threads)?;
    run_with_operators_context(problem, config, operators, &context)
}

/// Like [`run`], but reuses a caller-owned execution context, e.g. across the
/// folds of an experiment. Errors if `context.threads()` differs from `config.threads`.
pub fn run_with_context<P: Problem>(
    problem: &P,
    config: &RunConfig,
    context: &ExecutionContext,
) -> Result<RunResult> {
    config.validate()?;
    anyhow::ensure!(
        context.threads() == config.threads,
        "execution context has {} threads but run config requests {}",
        context.threads(),
        config.threads
    );
    run_with_operators_context(problem, config, &BuiltInOperators, context)
}

fn run_with_operators_context<P, O>(
    problem: &P,
    config: &RunConfig,
    operators: &O,
    context: &ExecutionContext,
) -> Result<RunResult>
where
    P: Problem,
    O: EvolutionOperators + ?Sized,
{
    config.validate()?;
    let mut rng = Random::new(config.seed);
    let mut id = 0u64;
    let mut evaluations = 0usize;
    let mut pop = Vec::with_capacity(config.population);
    for _ in 0..config.population {
        let mut genome = problem.initialize(&mut rng)?;
        if let Some(probability) = config.operators.per_locus_probability {
            genome.deviation = probability;
            genome.deviations.fill(probability);
            genome.validate()?;
        }
        pop.push(Individual::new(id, genome));
        id += 1;
    }
    evaluate_dirty_with_context(
        problem,
        &mut pop,
        &mut rng,
        config.training.prune_probability,
        &mut evaluations,
        context,
    )?;
    validate_objective_dimensions(&pop, None)?;
    let mut archived_scores = if archived_rank_mode(config) {
        assign_initial_archived_ranks(&mut pop, config)?
    } else {
        HashMap::new()
    };
    if let Some(cap) = config.max_evaluations {
        anyhow::ensure!(
            evaluations <= cap,
            "initialization used {evaluations} evaluations, exceeding max_evaluations {cap}"
        );
    }
    let mut ngga_state = if config.algorithm.kind == AlgorithmKind::Ngga {
        Some(NggaState::initialize(&pop, config, &mut rng)?)
    } else {
        None
    };
    let mut history = vec![record(0, &pop, evaluations)];
    let mut completed = 0;
    let generation_limit = if config.generations == 0 {
        config.max_evaluations.unwrap_or(0).saturating_add(1)
    } else {
        config.generations
    };
    for generation in 1..=generation_limit {
        if config.max_evaluations.is_some_and(|cap| evaluations >= cap) {
            break;
        }
        let before = evaluations;
        if config.algorithm.kind == AlgorithmKind::Ngga {
            ngga_generation(
                problem,
                &mut pop,
                ngga_state.as_mut().unwrap(),
                config,
                operators,
                &mut rng,
                &mut id,
                &mut evaluations,
            )?;
        } else {
            generational(
                problem,
                &mut pop,
                config,
                operators,
                &mut rng,
                &mut id,
                &mut evaluations,
                &mut archived_scores,
                context,
            )?;
        }
        completed = generation;
        history.push(record(generation, &pop, evaluations));
        if config.max_evaluations.is_some() && evaluations == before {
            break;
        }
    }
    let points: Vec<Vec<f64>> = pop
        .iter()
        .map(|x| {
            x.evaluation()
                .expect("population evaluated")
                .objectives
                .clone()
        })
        .collect();
    let keep = nondominated_indices(&points);
    let mut front: Vec<_> = keep.into_iter().map(|i| pop[i].clone()).collect();
    front.sort_by(|a, b| lex_objectives(a, b).then(a.id.cmp(&b.id)));
    Ok(RunResult {
        front,
        history,
        evaluations,
        generations: completed,
    })
}
fn record(generation: usize, pop: &[Individual], evaluations: usize) -> GenerationRecord {
    let pts: Vec<_> = pop
        .iter()
        .filter_map(|x| x.evaluation().map(|r| r.objectives.clone()))
        .collect();
    let nd = nondominated_indices(&pts);
    GenerationRecord {
        generation,
        evaluations,
        front: nd.into_iter().map(|i| pts[i].clone()).collect(),
        population_ids: pop.iter().map(|item| item.id).collect(),
    }
}
fn objectives(x: &Individual) -> &[f64] {
    &x.evaluation()
        .expect("selection requires evaluated candidate")
        .objectives
}
fn validate_objective_dimensions(items: &[Individual], expected: Option<usize>) -> Result<usize> {
    let dimensions = expected.or_else(|| {
        items
            .first()
            .and_then(|item| item.evaluation())
            .map(|report| report.objectives.len())
    });
    let dimensions = dimensions.ok_or_else(|| anyhow::anyhow!("no evaluated candidates"))?;
    anyhow::ensure!(dimensions > 0, "objective vectors cannot be empty");
    for item in items {
        let report = item
            .evaluation()
            .ok_or_else(|| anyhow::anyhow!("selection requires evaluated candidates"))?;
        anyhow::ensure!(
            report.objectives.len() == dimensions,
            "problem returned {} objectives; expected {dimensions}",
            report.objectives.len()
        );
    }
    Ok(dimensions)
}
fn lex_objectives(a: &Individual, b: &Individual) -> std::cmp::Ordering {
    for (x, y) in objectives(a).iter().zip(objectives(b)) {
        let c = x.total_cmp(y);
        if c != std::cmp::Ordering::Equal {
            return c;
        }
    }
    a.id.cmp(&b.id)
}
fn quality_scores(pop: &[Individual], raw: bool) -> Result<Vec<f64>> {
    let pts: Vec<_> = pop.iter().map(|x| objectives(x).to_vec()).collect();
    let fs = fronts(&pts);
    let mut keys = vec![(usize::MAX, false, 0.0); pop.len()];
    for (rank, f) in fs.iter().enumerate() {
        let (crowd, boundary) = crowding(&pts, f, raw).with_context(|| {
            format!(
                "NSGA-II crowding for candidate IDs {:?}",
                f.iter().map(|&i| pop[i].id).collect::<Vec<_>>()
            )
        })?;
        for (k, &i) in f.iter().enumerate() {
            keys[i] = (rank, boundary[k], crowd[k]);
        }
    }
    let mut order: Vec<_> = (0..pop.len()).collect();
    order.sort_by(|&a, &b| {
        keys[a]
            .0
            .cmp(&keys[b].0)
            .then(keys[b].1.cmp(&keys[a].1))
            .then(keys[b].2.total_cmp(&keys[a].2))
            .then(pop[a].id.cmp(&pop[b].id))
    });
    let mut scores = vec![0.0; pop.len()];
    for (ordinal, index) in order.into_iter().enumerate() {
        scores[index] = ordinal as f64;
    }
    Ok(scores)
}
fn select_parents(
    pop: &[Individual],
    scores: &[f64],
    tournament: usize,
    count: usize,
    cfg: &crate::config::OperatorsConfig,
    rng: &mut Random,
) -> Vec<Individual> {
    (0..count)
        .map(|_| {
            let group: Vec<_> = (0..tournament).map(|_| rng.usize(pop.len())).collect();
            let mut best = group[0];
            for &candidate in &group[1..] {
                if candidate_better(pop, scores, candidate, best, cfg.fitness_tournament) {
                    best = candidate;
                }
            }
            pop[best].clone()
        })
        .collect()
}
fn candidate_better(pop: &[Individual], scores: &[f64], a: usize, b: usize, fitness: bool) -> bool {
    if !fitness {
        if dominates(objectives(&pop[a]), objectives(&pop[b])) {
            return true;
        }
        if dominates(objectives(&pop[b]), objectives(&pop[a])) {
            return false;
        }
    }
    scores[a]
        .total_cmp(&scores[b])
        .then(pop[a].id.cmp(&pop[b].id))
        .is_lt()
}
// Keep transition state explicit here: problem, population, config, operators,
// RNG, ID allocator, evaluation counter, and archived ranks are independently
// owned run state and collapsing them would obscure which path mutates each.
#[allow(clippy::too_many_arguments)]
fn generational<P: Problem, O: EvolutionOperators + ?Sized>(
    problem: &P,
    pop: &mut Vec<Individual>,
    cfg: &RunConfig,
    operators: &O,
    rng: &mut Random,
    next_id: &mut u64,
    evaluations: &mut usize,
    archived_scores: &mut HashMap<u64, f64>,
    context: &ExecutionContext,
) -> Result<()> {
    let previous = pop.clone();
    let archived_rank_mode = archived_rank_mode(cfg);
    let scores = match cfg.algorithm.kind {
        _ if archived_rank_mode => pop
            .iter()
            .map(|item| {
                archived_scores.get(&item.id).copied().ok_or_else(|| {
                    anyhow::anyhow!("missing archived selection rank for ID {}", item.id)
                })
            })
            .collect::<Result<Vec<_>>>()?,
        AlgorithmKind::Moga | AlgorithmKind::Moes => moga_scores(pop, cfg.algorithm.sharing)?,
        AlgorithmKind::Spea2 => spea2_scores(pop, spea2_k(cfg))?,
        _ => quality_scores(pop, cfg.algorithm.raw_crowding)?,
    };
    let reproductive = if cfg.algorithm.kind == AlgorithmKind::Moes {
        if cfg.algorithm.archived_moes_tournament_preselection {
            archived_tournament_select(
                pop,
                &scores,
                cfg.algorithm.tournament_size,
                1,
                cfg.operators.fitness_tournament,
                cfg.offspring,
                rng,
            )
        } else {
            pop.clone()
        }
    } else {
        if archived_rank_mode {
            archived_tournament_select(
                pop,
                &scores,
                cfg.algorithm.tournament_size,
                1,
                cfg.operators.fitness_tournament,
                cfg.offspring,
                rng,
            )
        } else {
            select_parents(
                pop,
                &scores,
                cfg.algorithm.tournament_size,
                cfg.population,
                &cfg.operators,
                rng,
            )
        }
    };
    let parent_scores = match cfg.algorithm.kind {
        _ if archived_rank_mode => reproductive
            .iter()
            .map(|item| {
                archived_scores.get(&item.id).copied().ok_or_else(|| {
                    anyhow::anyhow!(
                        "missing archived selection rank for reproductive ID {}",
                        item.id
                    )
                })
            })
            .collect::<Result<Vec<_>>>()?,
        AlgorithmKind::Moga | AlgorithmKind::Moes => {
            moga_scores(&reproductive, cfg.algorithm.sharing)?
        }
        AlgorithmKind::Spea2 => spea2_scores(&reproductive, spea2_k(cfg))?,
        _ => quality_scores(&reproductive, cfg.algorithm.raw_crowding)?,
    };
    let mut children = Vec::with_capacity(cfg.offspring);
    let mut evaluated_copies = Vec::with_capacity(cfg.offspring);
    let archived_uniform_gate = cfg.operators.archived_evaluated_copy_uniform_mutation_gate
        && cfg.operators.mutation == crate::config::MutationKind::Uniform;
    for _ in 0..cfg.offspring {
        let (parent, evaluated_copy) = if rng.bernoulli(cfg.operators.crossover_probability) {
            let (a, b) = operators
                .mate(&reproductive, &parent_scores, &cfg.operators, rng)
                .with_context(|| "generational mating selection")?;
            let (c, _) = operators.crossover(
                &reproductive[a],
                &reproductive[b],
                &cfg.operators,
                rng,
                *next_id,
            )?;
            *next_id += 2;
            (c, false)
        } else {
            let mut c = reproductive[rng.usize(reproductive.len())].clone();
            c.id = *next_id;
            *next_id += 1;
            (c, true)
        };
        let mut child = parent;
        if !archived_uniform_gate && rng.bernoulli(cfg.operators.mutation_probability) {
            operators.mutate(&mut child, &cfg.operators, rng)?;
        }
        children.push(child);
        evaluated_copies.push(evaluated_copy);
    }
    if archived_uniform_gate {
        // The C++ operator first creates the whole offspring population, then
        // applies its evaluated-copy gate across that population.
        apply_archived_uniform_mutation(&mut children, &evaluated_copies, cfg, operators, rng)?;
    }
    evaluate_dirty_with_context(
        problem,
        &mut children,
        rng,
        cfg.training.prune_probability,
        evaluations,
        context,
    )?;
    validate_objective_dimensions(&children, Some(objectives(&previous[0]).len()))?;
    match cfg.algorithm.kind {
        AlgorithmKind::Nsga2 | AlgorithmKind::Spea2 => {
            // C++ NSGAII and SPEA2 run MOGA::operator(): elitism merges the old
            // population into the offspring before truncation; otherwise only
            // offspring (plus modern `best` candidates) compete for survival.
            let mut pool = if cfg.algorithm.effective_elitism() {
                let mut pool = std::mem::take(pop);
                pool.append(&mut children);
                pool
            } else {
                children
            };
            if !cfg.algorithm.archived_add_best {
                introduce_old_best(&mut pool, &previous, cfg.algorithm.best);
            }
            // Upstream assignFitness sorts survivors by its fitness value and
            // stores the ordinal; addBest then appends rank-zero copies.
            let pool_scores: Option<HashMap<u64, f64>> = if archived_rank_mode {
                let scores = if cfg.algorithm.kind == AlgorithmKind::Nsga2 {
                    quality_scores(&pool, cfg.algorithm.raw_crowding)?
                } else {
                    spea2_scores(&pool, spea2_k(cfg))?
                };
                Some(pool.iter().map(|item| item.id).zip(scores).collect())
            } else {
                None
            };
            let mut selected = if cfg.algorithm.kind == AlgorithmKind::Nsga2 {
                nsga2_select(&pool, cfg)?
            } else {
                spea2_select(pool, cfg.population, spea2_k(cfg))?
            };
            if let Some(by_id) = pool_scores {
                selected
                    .sort_by(|a, b| by_id[&a.id].total_cmp(&by_id[&b.id]).then(a.id.cmp(&b.id)));
                let mut next_scores: HashMap<u64, f64> = selected
                    .iter()
                    .enumerate()
                    .map(|(rank, item)| (item.id, rank as f64))
                    .collect();
                append_archived_best(
                    &mut selected,
                    &previous,
                    cfg.algorithm.best,
                    next_id,
                    &mut next_scores,
                )?;
                *archived_scores = next_scores;
            }
            *pop = selected;
        }
        AlgorithmKind::Moga | AlgorithmKind::Moes => {
            if archived_rank_mode {
                let mut pool = children;
                if cfg.algorithm.effective_elitism() {
                    pool.extend(previous.iter().cloned());
                }
                let (mut selected, mut next_scores) =
                    moga_select_with_ranks(pool, cfg.population, cfg.algorithm.sharing)?;
                if cfg.algorithm.archived_add_best {
                    append_archived_best(
                        &mut selected,
                        &previous,
                        cfg.algorithm.best,
                        next_id,
                        &mut next_scores,
                    )?;
                }
                *archived_scores = next_scores;
                *pop = selected;
            } else {
                *pop = children;
                if cfg.algorithm.effective_elitism() {
                    pop.extend(previous.iter().cloned());
                }
                introduce_old_best(pop, &previous, cfg.algorithm.best);
                let pool = std::mem::take(pop);
                *pop = moga_select(pool, cfg.population, cfg.algorithm.sharing)?;
            }
        }
        AlgorithmKind::Ngga => unreachable!(),
    }
    Ok(())
}
fn archived_rank_mode(cfg: &RunConfig) -> bool {
    match cfg.algorithm.kind {
        AlgorithmKind::Moga | AlgorithmKind::Moes => {
            cfg.algorithm.archived_add_best || cfg.algorithm.archived_moes_tournament_preselection
        }
        AlgorithmKind::Nsga2 | AlgorithmKind::Spea2 => cfg.algorithm.archived_add_best,
        AlgorithmKind::Ngga => false,
    }
}
fn spea2_k(cfg: &RunConfig) -> usize {
    cfg.algorithm
        .spea2_k
        .unwrap_or_else(|| ((2 * cfg.population) as f64).sqrt() as usize)
}
/// Modern `best`: offer distinct old nondominated members, then the `best`
/// old members by first objective, to survivor selection (deduplicated by ID).
fn introduce_old_best(pool: &mut Vec<Individual>, previous: &[Individual], best: usize) {
    if best == 0 {
        return;
    }
    let old_points: Vec<_> = previous.iter().map(|x| objectives(x).to_vec()).collect();
    let mut retained = nondominated_indices(&old_points);
    let mut best_order: Vec<_> = (0..previous.len()).collect();
    best_order.sort_by(|&a, &b| {
        objectives(&previous[a])[0]
            .total_cmp(&objectives(&previous[b])[0])
            .then(previous[a].id.cmp(&previous[b].id))
    });
    retained.extend(best_order.into_iter().take(best));
    let mut ids: std::collections::BTreeSet<_> = pool.iter().map(|item| item.id).collect();
    for i in retained {
        if ids.insert(previous[i].id) {
            pool.push(previous[i].clone());
        }
    }
}
fn nsga2_select(pop: &[Individual], cfg: &RunConfig) -> Result<Vec<Individual>> {
    let pts: Vec<_> = pop.iter().map(|x| objectives(x).to_vec()).collect();
    let fs = fronts(&pts);
    let mut chosen = Vec::new();
    let mut duplicate_fallback = Vec::new();
    for mut f in fs {
        if cfg.algorithm.push_equal || cfg.algorithm.push_more_than_two_equal {
            let mut counts = std::collections::BTreeMap::<Vec<u64>, usize>::new();
            let mut filtered = Vec::new();
            for i in f.drain(..) {
                let key: Vec<u64> = pts[i]
                    .iter()
                    .map(|v| if *v == 0.0 { 0 } else { v.to_bits() })
                    .collect();
                let count = counts.entry(key).or_default();
                let limit = if cfg.algorithm.push_more_than_two_equal {
                    2
                } else {
                    1
                };
                if *count < limit {
                    filtered.push(i);
                    *count += 1
                } else {
                    duplicate_fallback.push(i)
                }
            }
            f = filtered;
        }
        if chosen.len() + f.len() <= cfg.population {
            chosen.extend(f.into_iter().map(|i| pop[i].clone()));
        } else {
            let (c, b) = crowding(&pts, &f, cfg.algorithm.raw_crowding).with_context(|| {
                format!(
                    "NSGA-II survivor crowding for candidate IDs {:?}",
                    f.iter().map(|&i| pop[i].id).collect::<Vec<_>>()
                )
            })?;
            let mut order: Vec<_> = (0..f.len()).collect();
            order.sort_by(|&i, &j| {
                b[j].cmp(&b[i])
                    .then_with(|| c[j].total_cmp(&c[i]))
                    .then(pop[f[i]].id.cmp(&pop[f[j]].id))
            });
            for k in order.into_iter().take(cfg.population - chosen.len()) {
                chosen.push(pop[f[k]].clone());
            }
            break;
        }
    }
    if chosen.len() < cfg.population {
        duplicate_fallback.sort_by_key(|&i| pop[i].id);
        for i in duplicate_fallback
            .into_iter()
            .take(cfg.population - chosen.len())
        {
            chosen.push(pop[i].clone());
        }
    }
    Ok(chosen)
}
fn moga_scores(pop: &[Individual], sharing: f64) -> Result<Vec<f64>> {
    let n = pop.len();
    if n < 2 {
        return Ok(vec![0.0; n]);
    }
    let pts: Vec<_> = pop.iter().map(|x| objectives(x).to_vec()).collect();
    let mut counts = vec![1usize; n];
    for i in 0..n {
        for j in 0..n {
            if i != j && dominates(&pts[j], &pts[i]) {
                counts[i] += 1
            }
        }
    }
    let mut order: Vec<_> = (0..n).collect();
    order.sort_by(|&a, &b| counts[a].cmp(&counts[b]).then(pop[a].id.cmp(&pop[b].id)));
    let mut fit = vec![0.0; n];
    let mut i = 0;
    while i < n {
        let mut j = i + 1;
        while j < n && counts[order[i]] == counts[order[j]] {
            j += 1
        }
        let mid = (i + j + 1) as f64 / 2.0;
        let value = (n as f64 - mid) / (n as f64 - 1.0);
        for k in i..j {
            fit[order[k]] = value;
        }
        i = j;
    }
    let mut niches = vec![1.0; n];
    // Lexicographic i<j traversal feeds each niche its lower-index neighbors
    // in ascending order, then its higher-index neighbors in ascending order,
    // matching the old per-candidate accumulation sequence.
    for i in 0..n {
        for j in (i + 1)..n {
            let d = crate::objectives::distance(&pts[i], &pts[j]).with_context(|| {
                format!(
                    "MOGA sharing distance between candidate IDs {} and {}",
                    pop[i].id, pop[j].id
                )
            })?;
            if d <= sharing {
                let contribution = if sharing > 0.0 {
                    1.0 - d / sharing
                } else {
                    1.0
                };
                niches[i] += contribution;
                niches[j] += contribution;
            }
        }
    }
    for (score, niche) in fit.iter_mut().zip(niches) {
        *score /= niche;
    }
    Ok(fit.into_iter().map(|x| -x).collect())
}
fn moga_select(pool: Vec<Individual>, n: usize, sharing: f64) -> Result<Vec<Individual>> {
    let scores = moga_scores(&pool, sharing)?;
    let mut order: Vec<_> = (0..pool.len()).collect();
    order.sort_by(|&a, &b| {
        scores[a]
            .total_cmp(&scores[b])
            .then(pool[a].id.cmp(&pool[b].id))
    });
    let out = order
        .into_iter()
        .take(n)
        .map(|i| pool[i].clone())
        .collect::<Vec<_>>();
    anyhow::ensure!(!out.is_empty(), "survivor selection returned no candidates");
    Ok(out)
}
fn assign_initial_archived_ranks(
    pop: &mut Vec<Individual>,
    cfg: &RunConfig,
) -> Result<HashMap<u64, f64>> {
    match cfg.algorithm.kind {
        AlgorithmKind::Nsga2 => {
            let scores = quality_scores(pop, cfg.algorithm.raw_crowding)?;
            rank_by_scores(pop, &scores)
        }
        AlgorithmKind::Spea2 => {
            let scores = spea2_scores(pop, spea2_k(cfg))?;
            rank_by_scores(pop, &scores)
        }
        _ => assign_archived_ranks(pop, cfg.algorithm.sharing),
    }
}
fn assign_archived_ranks(pop: &mut Vec<Individual>, sharing: f64) -> Result<HashMap<u64, f64>> {
    let scores = moga_scores(pop, sharing)?;
    rank_by_scores(pop, &scores)
}
fn rank_by_scores(pop: &mut Vec<Individual>, scores: &[f64]) -> Result<HashMap<u64, f64>> {
    let mut order: Vec<_> = (0..pop.len()).collect();
    order.sort_by(|&a, &b| {
        scores[a]
            .total_cmp(&scores[b])
            .then(pop[a].id.cmp(&pop[b].id))
    });
    let mut sorted = Vec::with_capacity(pop.len());
    let mut ranks = HashMap::with_capacity(pop.len());
    for (rank, i) in order.into_iter().enumerate() {
        ranks.insert(pop[i].id, rank as f64);
        sorted.push(pop[i].clone());
    }
    *pop = sorted;
    Ok(ranks)
}
fn moga_select_with_ranks(
    pool: Vec<Individual>,
    n: usize,
    sharing: f64,
) -> Result<(Vec<Individual>, HashMap<u64, f64>)> {
    let scores = moga_scores(&pool, sharing)?;
    let mut order: Vec<_> = (0..pool.len()).collect();
    order.sort_by(|&a, &b| {
        scores[a]
            .total_cmp(&scores[b])
            .then(pool[a].id.cmp(&pool[b].id))
    });
    let mut selected = Vec::with_capacity(n);
    let mut ranks = HashMap::with_capacity(n);
    for (rank, i) in order.into_iter().take(n).enumerate() {
        selected.push(pool[i].clone());
        ranks.insert(pool[i].id, rank as f64);
    }
    anyhow::ensure!(
        !selected.is_empty(),
        "survivor selection returned no candidates"
    );
    Ok((selected, ranks))
}
fn append_archived_best(
    selected: &mut Vec<Individual>,
    old: &[Individual],
    best: usize,
    next_id: &mut u64,
    scores: &mut HashMap<u64, f64>,
) -> Result<()> {
    if best == 0 {
        return Ok(());
    }
    if best < old.len() {
        let points: Vec<_> = old.iter().map(|item| objectives(item).to_vec()).collect();
        let nondominated: std::collections::BTreeSet<_> =
            nondominated_indices(&points).into_iter().collect();
        let mut retained_points: Vec<Vec<f64>> = Vec::new();
        for index in 0..old.len() {
            if !nondominated.contains(&index) {
                continue;
            }
            let point = &points[index];
            if retained_points.iter().any(|candidate| candidate == point) {
                continue;
            }
            retained_points.push(point.clone());
            let mut copy = old[index].clone();
            copy.id = *next_id;
            *next_id += 1;
            scores.insert(copy.id, 0.0);
            selected.push(copy);
        }
    }
    let mut best_order: Vec<_> = (0..old.len()).collect();
    if best < old.len() {
        best_order.sort_by(|&a, &b| {
            objectives(&old[a])[0]
                .total_cmp(&objectives(&old[b])[0])
                .then(old[a].id.cmp(&old[b].id))
        });
    }
    for index in best_order.into_iter().take(best) {
        let mut copy = old[index].clone();
        copy.id = *next_id;
        *next_id += 1;
        scores.insert(copy.id, 0.0);
        selected.push(copy);
    }
    Ok(())
}
fn archived_tournament_select(
    pop: &[Individual],
    scores: &[f64],
    tournament: usize,
    winners: usize,
    fitness_tournament: bool,
    count: usize,
    rng: &mut Random,
) -> Vec<Individual> {
    let mut selected = Vec::with_capacity(count);
    while selected.len() < count {
        let mut pool: Vec<_> = (0..tournament).map(|_| rng.usize(pop.len())).collect();
        for k in 0..winners.min(count - selected.len()) {
            let best = archived_tournament_winner(pop, scores, &pool, k, fitness_tournament);
            pool.swap(k, best);
            selected.push(pop[pool[k]].clone());
        }
    }
    selected
}
fn archived_tournament_winner(
    pop: &[Individual],
    scores: &[f64],
    pool: &[usize],
    start: usize,
    fitness_tournament: bool,
) -> usize {
    let mut best = start;
    for i in start + 1..pool.len() {
        let candidate = pool[i];
        let incumbent = pool[best];
        let candidate_dominates = !fitness_tournament
            && dominates(objectives(&pop[candidate]), objectives(&pop[incumbent]));
        let incumbent_dominates = !fitness_tournament
            && dominates(objectives(&pop[incumbent]), objectives(&pop[candidate]));
        if candidate_dominates || (!incumbent_dominates && scores[candidate] < scores[incumbent]) {
            best = i;
        }
    }
    best
}
fn apply_archived_uniform_mutation<O: EvolutionOperators + ?Sized>(
    children: &mut [Individual],
    evaluated_copies: &[bool],
    cfg: &RunConfig,
    operators: &O,
    rng: &mut Random,
) -> Result<()> {
    anyhow::ensure!(
        children.len() == evaluated_copies.len(),
        "child origins do not match offspring"
    );
    for (child, &evaluated_copy) in children.iter_mut().zip(evaluated_copies) {
        if !evaluated_copy {
            // C++ crossover always creates an unevaluated individual, even when
            // the resulting genome happens to equal a parent.
            child.clear_evaluation();
        }
        if evaluated_copy || rng.bernoulli(cfg.operators.mutation_probability) {
            // C++ marks candidates dirty after invoking uniform mutation,
            // including when no locus changes.
            child.clear_evaluation();
            operators.mutate(child, &cfg.operators, rng)?;
        }
    }
    Ok(())
}
fn spea2_scores(pop: &[Individual], k: usize) -> Result<Vec<f64>> {
    anyhow::ensure!(
        k > 0 && k <= pop.len(),
        "SPEA2 k must be in 1..=population size"
    );
    let pts: Vec<_> = pop.iter().map(|x| objectives(x).to_vec()).collect();
    let m = pop.len();
    let mut strength = vec![0.0; m];
    let mut raw = vec![0.0; m];
    let mut dists = vec![vec![0.0; m]; m];
    for i in 0..m {
        for j in i + 1..m {
            let d = crate::objectives::distance(&pts[i], &pts[j]).with_context(|| {
                format!(
                    "SPEA2 fitness distance between candidate IDs {} and {}",
                    pop[i].id, pop[j].id
                )
            })?;
            dists[i][j] = d;
            dists[j][i] = d;
            if dominates(&pts[i], &pts[j]) {
                strength[i] += 1.0
            } else if dominates(&pts[j], &pts[i]) {
                strength[j] += 1.0
            }
        }
    }
    for i in 0..m {
        for j in 0..m {
            if dominates(&pts[j], &pts[i]) {
                raw[i] += strength[j];
            }
        }
    }
    let mut fit = vec![0.0; m];
    for i in 0..m {
        let mut ds: Vec<_> = (0..m).filter(|&j| j != i).map(|j| dists[i][j]).collect();
        ds.sort_by(f64::total_cmp);
        let kth = if k == 1 { 0.0 } else { ds[k - 2] };
        fit[i] = raw[i] + 1.0 / (kth + 2.0);
    }
    Ok(fit)
}
fn spea2_select(mut pool: Vec<Individual>, n: usize, k_override: usize) -> Result<Vec<Individual>> {
    let pts: Vec<_> = pool.iter().map(|x| objectives(x).to_vec()).collect();
    let m = pool.len();
    let mut strength = vec![0.0; m];
    let mut raw = vec![0.0; m];
    let mut dists = vec![vec![0.0; m]; m];
    for i in 0..m {
        for j in i + 1..m {
            let d = crate::objectives::distance(&pts[i], &pts[j]).with_context(|| {
                format!(
                    "SPEA2 selection distance between candidate IDs {} and {}",
                    pool[i].id, pool[j].id
                )
            })?;
            dists[i][j] = d;
            dists[j][i] = d;
            if dominates(&pts[i], &pts[j]) {
                strength[i] += 1.0
            } else if dominates(&pts[j], &pts[i]) {
                strength[j] += 1.0
            }
        }
    }
    for i in 0..m {
        for j in 0..m {
            if dominates(&pts[j], &pts[i]) {
                raw[i] += strength[j];
            }
        }
    }
    anyhow::ensure!(
        k_override > 0 && k_override <= m,
        "SPEA2 k must be in 1..=candidate pool size"
    );
    let k = k_override;
    let mut fit = vec![0.0; m];
    for i in 0..m {
        let mut ds: Vec<_> = (0..m).filter(|&j| j != i).map(|j| dists[i][j]).collect();
        ds.sort_by(f64::total_cmp);
        let kth = if k == 1 { 0.0 } else { ds[k - 2] };
        fit[i] = raw[i] + 1.0 / (kth + 2.0);
    }
    let mut active: Vec<usize> = (0..m).filter(|&i| raw[i] == 0.0).collect();
    if active.len() > n {
        let mut is_active = vec![false; m];
        for &i in &active {
            is_active[i] = true;
        }
        let mut sorted_neighbors = vec![Vec::new(); m];
        for &i in &active {
            let neighbors = &mut sorted_neighbors[i];
            neighbors.extend(active.iter().copied().filter(|&j| j != i));
            neighbors.sort_by(|&a, &b| dists[i][a].total_cmp(&dists[i][b]));
        }
        while active.len() > n {
            let rem = *active
                .iter()
                .min_by(|&&a, &&b| {
                    compare_spea2_neighbor_streams(a, b, &sorted_neighbors, &is_active, &dists)
                        .then_with(|| pool[b].id.cmp(&pool[a].id))
                })
                .unwrap();
            is_active[rem] = false;
            active.retain(|&i| i != rem);
        }
    }
    if active.len() < n {
        let mut rest: Vec<_> = (0..m).filter(|i| !active.contains(i)).collect();
        rest.sort_by(|&a, &b| fit[a].total_cmp(&fit[b]).then(pool[a].id.cmp(&pool[b].id)));
        active.extend(rest.into_iter().take(n - active.len()));
    }
    let out = active.into_iter().map(|i| pool[i].clone()).collect();
    pool.clear();
    Ok(out)
}

fn compare_spea2_neighbor_streams(
    a: usize,
    b: usize,
    sorted_neighbors: &[Vec<usize>],
    is_active: &[bool],
    dists: &[Vec<f64>],
) -> std::cmp::Ordering {
    let mut a_neighbors = sorted_neighbors[a]
        .iter()
        .filter(|&&j| is_active[j])
        .map(|&j| dists[a][j]);
    let mut b_neighbors = sorted_neighbors[b]
        .iter()
        .filter(|&&j| is_active[j])
        .map(|&j| dists[b][j]);
    loop {
        match (a_neighbors.next(), b_neighbors.next()) {
            (Some(x), Some(y)) => {
                let ordering = x.total_cmp(&y);
                if ordering != std::cmp::Ordering::Equal {
                    return ordering;
                }
            }
            (None, None) => return std::cmp::Ordering::Equal,
            (None, Some(_)) => return std::cmp::Ordering::Less,
            (Some(_), None) => return std::cmp::Ordering::Greater,
        }
    }
}

#[cfg(test)]
fn spea2_select_reference(
    mut pool: Vec<Individual>,
    n: usize,
    k_override: usize,
) -> Result<Vec<Individual>> {
    let pts: Vec<_> = pool.iter().map(|x| objectives(x).to_vec()).collect();
    let m = pool.len();
    let mut strength = vec![0.0; m];
    let mut raw = vec![0.0; m];
    let mut dists = vec![vec![0.0; m]; m];
    for i in 0..m {
        for j in i + 1..m {
            let d = crate::objectives::distance(&pts[i], &pts[j]).with_context(|| {
                format!(
                    "SPEA2 selection distance between candidate IDs {} and {}",
                    pool[i].id, pool[j].id
                )
            })?;
            dists[i][j] = d;
            dists[j][i] = d;
            if dominates(&pts[i], &pts[j]) {
                strength[i] += 1.0;
            } else if dominates(&pts[j], &pts[i]) {
                strength[j] += 1.0;
            }
        }
    }
    for i in 0..m {
        for j in 0..m {
            if dominates(&pts[j], &pts[i]) {
                raw[i] += strength[j];
            }
        }
    }
    anyhow::ensure!(
        k_override > 0 && k_override <= m,
        "SPEA2 k must be in 1..=candidate pool size"
    );
    let k = k_override;
    let mut fit = vec![0.0; m];
    for i in 0..m {
        let mut ds: Vec<_> = (0..m).filter(|&j| j != i).map(|j| dists[i][j]).collect();
        ds.sort_by(f64::total_cmp);
        let kth = if k == 1 { 0.0 } else { ds[k - 2] };
        fit[i] = raw[i] + 1.0 / (kth + 2.0);
    }
    let mut active: Vec<usize> = (0..m).filter(|&i| raw[i] == 0.0).collect();
    while active.len() > n {
        let rem = *active
            .iter()
            .min_by(|&&a, &&b| {
                let mut da: Vec<_> = active
                    .iter()
                    .filter(|&&j| j != a)
                    .map(|&j| dists[a][j])
                    .collect();
                let mut db: Vec<_> = active
                    .iter()
                    .filter(|&&j| j != b)
                    .map(|&j| dists[b][j])
                    .collect();
                da.sort_by(f64::total_cmp);
                db.sort_by(f64::total_cmp);
                da.iter()
                    .zip(&db)
                    .find_map(|(x, y)| {
                        let ordering = x.total_cmp(y);
                        (ordering != std::cmp::Ordering::Equal).then_some(ordering)
                    })
                    .unwrap_or_else(|| pool[b].id.cmp(&pool[a].id))
            })
            .unwrap();
        active.retain(|&i| i != rem);
    }
    if active.len() < n {
        let mut rest: Vec<_> = (0..m).filter(|i| !active.contains(i)).collect();
        rest.sort_by(|&a, &b| fit[a].total_cmp(&fit[b]).then(pool[a].id.cmp(&pool[b].id)));
        active.extend(rest.into_iter().take(n - active.len()));
    }
    let out = active.into_iter().map(|i| pool[i].clone()).collect();
    pool.clear();
    Ok(out)
}

struct NggaState {
    dominance: Vec<f64>,
    sharing: Vec<f64>,
}
impl NggaState {
    fn initialize(pop: &[Individual], cfg: &RunConfig, rng: &mut Random) -> Result<Self> {
        let mut state = Self {
            dominance: vec![0.0; pop.len()],
            sharing: vec![0.0; pop.len()],
        };
        let n1 = cfg.algorithm.ngga_n1.unwrap_or(pop.len() / 3);
        for i in 0..pop.len() {
            state.initialize_one(i, pop, n1, cfg, rng)?;
        }
        Ok(state)
    }
    fn initialize_one(
        &mut self,
        i: usize,
        pop: &[Individual],
        n1: usize,
        cfg: &RunConfig,
        rng: &mut Random,
    ) -> Result<()> {
        self.dominance[i] = 0.0;
        self.sharing[i] = 0.0;
        for _ in 0..n1 {
            let j = random_other(rng, pop.len(), i);
            if dominates(objectives(&pop[i]), objectives(&pop[j])) {
                self.dominance[j] += 1.0;
            } else if dominates(objectives(&pop[j]), objectives(&pop[i])) {
                self.dominance[i] += 1.0;
            }
            let d = crate::objectives::distance(objectives(&pop[i]), objectives(&pop[j]))
                .with_context(|| {
                    format!(
                        "NGGA initialization distance between candidate IDs {} and {}",
                        pop[i].id, pop[j].id
                    )
                })?;
            if d <= cfg.algorithm.sharing {
                let v = if cfg.algorithm.sharing > 0.0 {
                    1.0 - d / cfg.algorithm.sharing
                } else {
                    1.0
                };
                self.sharing[i] += v;
                self.sharing[j] += v;
            }
        }
        Ok(())
    }
    fn value(&self, i: usize, cfg: &RunConfig) -> f64 {
        cfg.algorithm.ngga_sharing_weight * self.sharing[i]
            + cfg.algorithm.ngga_dominance_weight * self.dominance[i]
    }
    fn update(
        &mut self,
        pop: &[Individual],
        n2: usize,
        cfg: &RunConfig,
        rng: &mut Random,
    ) -> Result<()> {
        for _ in 0..n2 {
            let a = rng.usize(pop.len());
            let b = random_other(rng, pop.len(), a);
            self.dominance[a] *= 1.0 - cfg.algorithm.ngga_decay_dominance;
            self.dominance[b] *= 1.0 - cfg.algorithm.ngga_decay_dominance;
            if dominates(objectives(&pop[a]), objectives(&pop[b])) {
                self.dominance[b] += 1.0;
            } else if dominates(objectives(&pop[b]), objectives(&pop[a])) {
                self.dominance[a] += 1.0;
            }
            let sharing_decay = cfg
                .algorithm
                .ngga_decay_sharing
                .unwrap_or(1.0 / pop.len() as f64);
            self.sharing[a] *= 1.0 - sharing_decay;
            self.sharing[b] *= 1.0 - sharing_decay;
            let d = crate::objectives::distance(objectives(&pop[a]), objectives(&pop[b]))
                .with_context(|| {
                    format!(
                        "NGGA update distance between candidate IDs {} and {}",
                        pop[a].id, pop[b].id
                    )
                })?;
            if d <= cfg.algorithm.sharing {
                let v = if cfg.algorithm.sharing > 0.0 {
                    1.0 - d / cfg.algorithm.sharing
                } else {
                    1.0
                };
                self.sharing[a] += v;
                self.sharing[b] += v;
            }
        }
        Ok(())
    }
}
fn random_other(rng: &mut Random, population: usize, excluded: usize) -> usize {
    let mut index = rng.usize(population - 1);
    if index >= excluded {
        index += 1;
    }
    index
}
#[allow(clippy::too_many_arguments)]
fn ngga_generation<P: Problem, O: EvolutionOperators + ?Sized>(
    problem: &P,
    pop: &mut [Individual],
    state: &mut NggaState,
    cfg: &RunConfig,
    operators: &O,
    rng: &mut Random,
    next_id: &mut u64,
    evaluations: &mut usize,
) -> Result<()> {
    let n = pop.len();
    let n1 = cfg.algorithm.ngga_n1.unwrap_or(n / 3);
    let n2 = cfg.algorithm.ngga_n2.unwrap_or(n / 3);
    for _ in 0..cfg.offspring {
        state.update(pop, n2, cfg, rng)?;
        let scores: Vec<_> = (0..n).map(|i| state.value(i, cfg)).collect();
        let (a, b) = operators
            .mate(pop, &scores, &cfg.operators, rng)
            .with_context(|| format!("NGGA mating at generation offspring {}", *next_id))?;
        let mut child = if rng.bernoulli(cfg.operators.crossover_probability) {
            let (c, _) = operators.crossover(&pop[a], &pop[b], &cfg.operators, rng, *next_id)?;
            *next_id += 2;
            c
        } else {
            let mut c = pop[a].clone();
            c.id = *next_id;
            *next_id += 1;
            c
        };
        if rng.bernoulli(cfg.operators.mutation_probability) {
            operators.mutate(&mut child, &cfg.operators, rng)?;
        }
        evaluate_dirty_with_threads(
            problem,
            std::slice::from_mut(&mut child),
            rng,
            cfg.training.prune_probability,
            evaluations,
            1,
        )?;
        validate_objective_dimensions(
            std::slice::from_ref(&child),
            Some(objectives(&pop[0]).len()),
        )?;
        let worst = (0..n)
            .max_by(|&i, &j| {
                state
                    .value(i, cfg)
                    .total_cmp(&state.value(j, cfg))
                    .then(pop[i].id.cmp(&pop[j].id))
            })
            .unwrap();
        pop[worst] = child;
        state.initialize_one(worst, pop, n1, cfg, rng)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "parallel")]
    use super::run_with_context;
    use super::{
        NggaState, append_archived_best, apply_archived_uniform_mutation,
        archived_tournament_winner, assign_archived_ranks, quality_scores, run_with_operators,
        spea2_select, spea2_select_reference,
    };
    #[cfg(feature = "parallel")]
    use crate::execution::ExecutionContext;
    use crate::{
        config::{AlgorithmKind, ProblemConfig, ProblemKind, RunConfig},
        evaluation::{EvaluationReport, Problem},
        genome::{Domain, GeneValue, Genome, GenomeLayout, Individual},
        operators::EvolutionOperators,
        problems::NumericProblem,
        rng::Random,
    };
    use anyhow::Context as _;
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };

    #[cfg(feature = "parallel")]
    #[test]
    fn serial_and_parallel_runs_keep_full_learning_results_with_pruning() {
        use crate::{config::TrainingConfig, data::Dataset, models::LearningProblem};
        use std::collections::HashSet;
        struct ParallelHint {
            inner: LearningProblem,
            workers: Arc<std::sync::Mutex<HashSet<std::thread::ThreadId>>>,
        }
        impl crate::evaluation::Problem for ParallelHint {
            fn initialize(&self, rng: &mut Random) -> crate::Result<Genome> {
                self.inner.initialize(rng)
            }
            fn evaluate(&self, genome: &Genome) -> crate::Result<EvaluationReport> {
                self.workers
                    .lock()
                    .unwrap()
                    .insert(std::thread::current().id());
                std::thread::sleep(std::time::Duration::from_millis(2));
                self.inner.evaluate(genome)
            }
            fn estimated_evaluation_work(&self, _: &Genome) -> Option<usize> {
                Some(10_000)
            }
            fn prune(&self, genome: &mut Genome, report: &EvaluationReport) -> crate::Result<bool> {
                self.inner.prune(genome, report)
            }
        }
        const DATA: &str = "@relation tiny\n@attribute x numeric\n@attribute y {a,b}\n@data\n0,a\n1,a\n2,b\n3,b\n4,a\n5,b\n";
        let dataset = Dataset::parse_str(DATA).unwrap();
        let problem = ParallelHint {
            inner: LearningProblem::new(
                dataset,
                TrainingConfig {
                    initial_rules: 4,
                    ..TrainingConfig::default()
                },
                Default::default(),
            )
            .unwrap(),
            workers: Arc::new(std::sync::Mutex::new(HashSet::new())),
        };
        for kind in [
            AlgorithmKind::Nsga2,
            AlgorithmKind::Moga,
            AlgorithmKind::Moes,
        ] {
            let mut serial = RunConfig {
                threads: 1,
                population: 12,
                offspring: 12,
                generations: 3,
                seed: 71,
                ..RunConfig::default()
            };
            serial.algorithm.kind = kind;
            serial.algorithm.archived_add_best =
                matches!(kind, AlgorithmKind::Moga | AlgorithmKind::Moes);
            serial.training.prune_probability = 1.0;
            serial.operators.crossover = crate::config::CrossoverKind::VariableLength;
            let mut parallel = serial.clone();
            parallel.threads = 4;
            let serial_result =
                run_with_context(&problem, &serial, &ExecutionContext::new(1).unwrap()).unwrap();
            problem.workers.lock().unwrap().clear();
            let caller = std::thread::current().id();
            let parallel_result =
                run_with_context(&problem, &parallel, &ExecutionContext::new(4).unwrap()).unwrap();
            assert!(
                problem
                    .workers
                    .lock()
                    .unwrap()
                    .iter()
                    .filter(|worker| **worker != caller)
                    .count()
                    >= 2,
                "algorithm {kind:?} parity run did not use multiple workers"
            );
            assert_eq!(
                serde_json::to_value(serial_result).unwrap(),
                serde_json::to_value(parallel_result).unwrap(),
                "algorithm {kind:?} diverged"
            );
        }
    }

    fn candidate(id: u64, objectives: Vec<f64>) -> Individual {
        let layout =
            Arc::new(GenomeLayout::new(vec![Domain::Real { min: 0.0, max: 1.0 }]).unwrap());
        let genome = Genome::new(vec![GeneValue::Real(0.5)], layout).unwrap();
        let mut individual = Individual::new(id, genome);
        individual
            .set_evaluation(EvaluationReport {
                objectives,
                ..EvaluationReport::default()
            })
            .unwrap();
        individual
    }

    fn moga_scores_reference(pop: &[Individual], sharing: f64) -> crate::Result<Vec<f64>> {
        let n = pop.len();
        if n < 2 {
            return Ok(vec![0.0; n]);
        }
        let points: Vec<_> = pop
            .iter()
            .map(|item| super::objectives(item).to_vec())
            .collect();
        let mut counts = vec![1usize; n];
        for i in 0..n {
            for j in 0..n {
                if i != j && crate::objectives::dominates(&points[j], &points[i]) {
                    counts[i] += 1;
                }
            }
        }
        let mut order: Vec<_> = (0..n).collect();
        order.sort_by(|&a, &b| counts[a].cmp(&counts[b]).then(pop[a].id.cmp(&pop[b].id)));
        let mut fit = vec![0.0; n];
        let mut i = 0;
        while i < n {
            let mut j = i + 1;
            while j < n && counts[order[i]] == counts[order[j]] {
                j += 1;
            }
            let mid = (i + j + 1) as f64 / 2.0;
            let value = (n as f64 - mid) / (n as f64 - 1.0);
            for k in i..j {
                fit[order[k]] = value;
            }
            i = j;
        }
        for i in 0..n {
            let mut niche = 1.0;
            for j in 0..n {
                if i == j {
                    continue;
                }
                let d = crate::objectives::distance(&points[i], &points[j]).with_context(|| {
                    format!(
                        "MOGA sharing distance between candidate IDs {} and {}",
                        pop[i].id, pop[j].id
                    )
                })?;
                if d <= sharing {
                    niche += if sharing > 0.0 {
                        1.0 - d / sharing
                    } else {
                        1.0
                    };
                }
            }
            fit[i] /= niche;
        }
        Ok(fit.into_iter().map(|score| -score).collect())
    }

    fn assert_moga_scores_match_reference(pop: &[Individual], sharing: f64) {
        match (
            super::moga_scores(pop, sharing),
            moga_scores_reference(pop, sharing),
        ) {
            (Ok(actual), Ok(expected)) => assert_eq!(
                actual
                    .iter()
                    .map(|score| score.to_bits())
                    .collect::<Vec<_>>(),
                expected
                    .iter()
                    .map(|score| score.to_bits())
                    .collect::<Vec<_>>(),
                "sharing radius {sharing}"
            ),
            (Err(actual), Err(expected)) => {
                assert_eq!(format!("{actual:#}"), format!("{expected:#}"));
            }
            (actual, expected) => {
                panic!("optimized/reference result kinds differ: {actual:?} vs {expected:?}")
            }
        }
    }

    #[test]
    fn pairwise_moga_sharing_matches_reference_bits_and_errors() {
        let cases = vec![
            (Vec::<Vec<f64>>::new(), 0.0),
            (vec![vec![3.0, 4.0]], 1.0),
            (vec![vec![0.0, 1.0], vec![2.0, 3.0]], 0.0),
            (vec![vec![0.0, -0.0], vec![-0.0, 0.0], vec![0.0, 0.0]], 0.0),
            (vec![vec![0.0, 0.0], vec![0.0, 0.0]], -1.0),
            (vec![vec![0.0, 0.0], vec![f64::from_bits(1), 0.0]], 0.0),
            (
                vec![
                    vec![1.0, 1.0],
                    vec![1.25, 1.0],
                    vec![2.0, 2.0],
                    vec![1.0, 1.0],
                ],
                0.5,
            ),
            (
                vec![
                    vec![1e150, -1e150, 0.0, 1.0, 2.0, -3.0, 4.0, 5.0],
                    vec![0.0, 1e150, 1.0, 2.0, -3.0, 4.0, 5.0, -6.0],
                    vec![-1e150, 0.0, 2.0, -3.0, 4.0, 5.0, -6.0, 7.0],
                ],
                1e151,
            ),
            (
                vec![vec![0.0, 1e200], vec![1e200, 0.0], vec![1.0, 2.0]],
                1.0,
            ),
            (vec![vec![f64::MAX, 0.0], vec![-f64::MAX, 0.0]], 0.0),
            (vec![vec![0.0, 0.0], vec![1e154, 1e154]], 0.0),
            (vec![vec![0.0, 1e200], vec![1e200, 0.0]], 0.0),
            (vec![vec![0.0, 1e200], vec![1e200, 0.0]], -1.0),
        ];
        for (objectives, sharing) in cases {
            let pop: Vec<_> = objectives
                .into_iter()
                .enumerate()
                .map(|(i, point)| candidate(100 + i as u64, point))
                .collect();
            assert_moga_scores_match_reference(&pop, sharing);
        }

        let mut rng = Random::new(0x4d4f_4741_5348_4152);
        for case in 0..100 {
            let count = 3 + rng.usize(48);
            let dimensions = 1 + rng.usize(9);
            let exponent = rng.integer(-300, 145) as i32;
            let scale = 10.0_f64.powi(exponent);
            let mut points = Vec::<Vec<f64>>::with_capacity(count);
            for _ in 0..count {
                if !points.is_empty() && rng.bernoulli(0.08) {
                    points.push(points[rng.usize(points.len())].clone());
                    continue;
                }
                let point = (0..dimensions)
                    .map(|_| {
                        if rng.usize(32) == 0 {
                            if rng.bernoulli(0.5) { -0.0 } else { 0.0 }
                        } else {
                            rng.range(-1.0, 1.0) * scale
                        }
                    })
                    .collect();
                points.push(point);
            }
            let sharing = match case % 5 {
                0 => 0.0,
                1 => -1.0,
                2 => scale,
                3 => scale * rng.range(0.01, 3.0),
                _ => scale * rng.range(1.0, 20.0),
            };
            let pop: Vec<_> = points
                .into_iter()
                .enumerate()
                .map(|(i, point)| candidate(1_000 + i as u64, point))
                .collect();
            assert_moga_scores_match_reference(&pop, sharing);
        }
    }

    #[test]
    fn archived_tournament_uses_saved_scalar_and_keeps_first_drawn_tie() {
        let pop = vec![
            candidate(90, vec![0.0, 10.0]),
            candidate(1, vec![10.0, 0.0]),
        ];
        let scores = vec![8.0, 2.0];
        assert_eq!(
            archived_tournament_winner(&pop, &scores, &[0, 1], 0, true),
            1
        );

        let tied = vec![
            candidate(90, vec![0.0, 10.0]),
            candidate(1, vec![10.0, 0.0]),
        ];
        assert_eq!(
            archived_tournament_winner(&tied, &[2.0, 2.0], &[0, 1], 0, true),
            0
        );

        let dominated = vec![candidate(90, vec![5.0, 5.0]), candidate(1, vec![4.0, 4.0])];
        // With fitnessTournament disabled, strict objective dominance precedes rank.
        assert_eq!(
            archived_tournament_winner(&dominated, &[0.0, 99.0], &[0, 1], 0, false),
            1
        );
    }

    #[test]
    fn archived_initial_fitness_assignment_sorts_population_and_stores_scalar_rank() {
        let mut pop = vec![candidate(2, vec![1.0, 1.0]), candidate(1, vec![0.0, 0.0])];
        let ranks = assign_archived_ranks(&mut pop, 0.1).unwrap();
        assert_eq!(
            pop.iter().map(|item| item.id).collect::<Vec<_>>(),
            vec![1, 2]
        );
        assert_eq!(ranks.get(&1), Some(&0.0));
        assert_eq!(ranks.get(&2), Some(&1.0));
    }

    struct ObserveMatePool {
        observed: AtomicUsize,
    }

    impl EvolutionOperators for ObserveMatePool {
        fn mate(
            &self,
            population: &[Individual],
            scores: &[f64],
            config: &crate::config::OperatorsConfig,
            rng: &mut Random,
        ) -> crate::Result<(usize, usize)> {
            self.observed.store(population.len(), Ordering::Relaxed);
            crate::operators::BuiltInOperators.mate(population, scores, config, rng)
        }

        fn crossover(
            &self,
            first: &Individual,
            second: &Individual,
            config: &crate::config::OperatorsConfig,
            rng: &mut Random,
            child_id: u64,
        ) -> crate::Result<(Individual, Individual)> {
            crate::operators::BuiltInOperators.crossover(first, second, config, rng, child_id)
        }

        fn mutate(
            &self,
            individual: &mut Individual,
            config: &crate::config::OperatorsConfig,
            rng: &mut Random,
        ) -> crate::Result<bool> {
            crate::operators::BuiltInOperators.mutate(individual, config, rng)
        }
    }

    #[test]
    fn archived_moga_preselection_matches_requested_offspring_pool() {
        let mut cfg = RunConfig::default();
        cfg.algorithm.kind = AlgorithmKind::Moga;
        cfg.algorithm.archived_add_best = true;
        cfg.population = 4;
        cfg.offspring = 7;
        cfg.generations = 1;
        cfg.operators.crossover_probability = 1.0;
        cfg.operators.mutation_probability = 0.0;
        let problem = NumericProblem::new(&ProblemConfig {
            kind: ProblemKind::Sf2,
            dimensions: Some(1),
            ..ProblemConfig::default()
        })
        .unwrap();
        let operators = ObserveMatePool {
            observed: AtomicUsize::new(0),
        };

        run_with_operators(&problem, &cfg, &operators).unwrap();
        assert_eq!(operators.observed.load(Ordering::Relaxed), cfg.offspring);
    }

    #[test]
    fn archived_add_best_retains_unique_old_front_then_best_with_overflow_and_zero_rank() {
        let old = vec![
            candidate(10, vec![0.1, 0.8]),
            candidate(11, vec![0.2, 0.6]),
            candidate(12, vec![0.2, 0.6]),
            candidate(13, vec![0.4, 0.9]),
        ];
        let mut selected = vec![
            candidate(20, vec![0.1, 0.2]),
            candidate(21, vec![0.2, 0.3]),
            candidate(22, vec![0.3, 0.4]),
        ];
        let mut scores = std::collections::HashMap::from([(20, 0.0), (21, 1.0), (22, 2.0)]);
        let mut next_id = 100;
        append_archived_best(&mut selected, &old, 1, &mut next_id, &mut scores).unwrap();

        let oracle: serde_json::Value = serde_json::from_str(include_str!(
            "../tests/fixtures/cpp_archived_transitions.json"
        ))
        .unwrap();
        let expected = &oracle["front_dedup_then_best_duplicate"];
        let observed = selected
            .iter()
            .map(|item| item.evaluation().unwrap().objectives.clone())
            .collect::<Vec<_>>();
        assert_eq!(serde_json::to_value(observed).unwrap(), expected["fitness"]);
        let observed_values = selected
            .iter()
            .map(|item| scores[&item.id] as usize)
            .collect::<Vec<_>>();
        assert_eq!(
            serde_json::to_value(observed_values).unwrap(),
            expected["values"]
        );
        assert_eq!(selected.len(), 6);
        let retained = &selected[3..];
        assert_eq!(
            retained.iter().map(|item| item.id).collect::<Vec<_>>(),
            vec![100, 101, 102]
        );
        assert!(
            retained
                .iter()
                .all(|item| scores.get(&item.id) == Some(&0.0))
        );
        assert_eq!(retained[0].evaluation().unwrap().objectives, vec![0.1, 0.8]);
        assert_eq!(retained[2].evaluation().unwrap().objectives, vec![0.1, 0.8]);
        assert!(selected.len() > 1);
    }

    #[test]
    fn archived_add_best_preserves_old_front_order_and_best_equal_to_old_size_behavior() {
        let oracle: serde_json::Value = serde_json::from_str(include_str!(
            "../tests/fixtures/cpp_archived_transitions.json"
        ))
        .unwrap();
        let expected = &oracle["best_equals_old_size_skips_front_block"];
        let old = vec![
            candidate(40, vec![0.3, 0.2]),
            candidate(41, vec![0.1, 0.4]),
            candidate(42, vec![0.2, 0.3]),
        ];
        let mut selected = vec![candidate(30, vec![0.1, 0.2]), candidate(31, vec![0.2, 0.3])];
        let mut scores = std::collections::HashMap::from([(30, 0.0), (31, 1.0)]);
        let mut next_id = 100;
        scores.insert(31, 1.0);
        append_archived_best(&mut selected, &old, old.len(), &mut next_id, &mut scores).unwrap();
        let observed = selected
            .iter()
            .map(|item| item.evaluation().unwrap().objectives.clone())
            .collect::<Vec<_>>();
        assert_eq!(serde_json::to_value(observed).unwrap(), expected["fitness"]);
        let observed_values = selected
            .iter()
            .map(|item| scores[&item.id] as usize)
            .collect::<Vec<_>>();
        assert_eq!(
            serde_json::to_value(observed_values).unwrap(),
            expected["values"]
        );
    }

    struct CountMutation(AtomicUsize);
    impl EvolutionOperators for CountMutation {
        fn mate(
            &self,
            _: &[Individual],
            _: &[f64],
            _: &crate::config::OperatorsConfig,
            _: &mut Random,
        ) -> crate::Result<(usize, usize)> {
            Ok((0, 0))
        }
        fn crossover(
            &self,
            _: &Individual,
            _: &Individual,
            _: &crate::config::OperatorsConfig,
            _: &mut Random,
            _: u64,
        ) -> crate::Result<(Individual, Individual)> {
            unreachable!()
        }
        fn mutate(
            &self,
            _: &mut Individual,
            _: &crate::config::OperatorsConfig,
            _: &mut Random,
        ) -> crate::Result<bool> {
            self.0.fetch_add(1, Ordering::Relaxed);
            Ok(false)
        }
    }

    #[test]
    fn archived_uniform_mutation_tracks_copy_origin_and_dirties_unchanged_children() {
        let mut cfg = RunConfig::default();
        cfg.operators.mutation_probability = 0.0;
        let operators = CountMutation(AtomicUsize::new(0));
        let mut children = vec![candidate(1, vec![0.0]), candidate(2, vec![1.0])];
        apply_archived_uniform_mutation(
            &mut children,
            &[true, false],
            &cfg,
            &operators,
            &mut Random::new(8),
        )
        .unwrap();
        assert_eq!(operators.0.load(Ordering::Relaxed), 1);
        assert!(children.iter().all(|child| child.evaluation().is_none()));
    }

    #[test]
    fn nsga_parent_order_uses_rank_boundary_raw_crowding_then_id() {
        let pop = vec![
            candidate(10, vec![0.0, 10.0]),
            candidate(11, vec![1.0, 8.0]),
            candidate(12, vec![2.0, 7.0]),
            candidate(13, vec![8.0, 6.0]),
            candidate(14, vec![10.0, 0.0]),
            candidate(15, vec![10.0, 11.0]),
        ];
        let scores = quality_scores(&pop, true).unwrap();
        assert_eq!(scores, vec![0.0, 4.0, 3.0, 2.0, 1.0, 5.0]);
    }

    #[test]
    fn spea2_truncation_keeps_oldest_when_all_neighbour_distances_tie() {
        let pool = (0..5).map(|id| candidate(id, vec![1.0, 1.0])).collect();
        let survivors = spea2_select(pool, 2, 2).unwrap();
        assert_eq!(
            survivors.iter().map(|item| item.id).collect::<Vec<_>>(),
            vec![0, 1]
        );
    }

    #[test]
    fn spea2_cached_truncation_matches_v2_reference_for_deterministic_pools() {
        for case in 0..12 {
            let size = 4 + case;
            let dimensions = if case % 2 == 0 { 2 } else { 3 };
            let pool: Vec<_> = (0..size)
                .map(|index| {
                    let base = if index > 0 && index % 6 == 0 {
                        index - 1
                    } else {
                        index
                    };
                    let x = ((base * 7 + case * 3) % 13) as f64;
                    let y = ((base * 5 + case * 11) % 17) as f64;
                    let mut point = if dimensions == 2 {
                        vec![x, y]
                    } else {
                        vec![x, y, ((base * 3 + case) % 9) as f64]
                    };
                    if index % 4 == 0 {
                        point[0] = if index % 8 == 0 { -0.0 } else { 0.0 };
                    }
                    candidate(index as u64, point)
                })
                .collect();

            let mut capacities = vec![1, size / 2, size - 1, size];
            capacities.sort_unstable();
            capacities.dedup();
            for capacity in capacities {
                for k in 1..=size {
                    let expected = spea2_select_reference(pool.clone(), capacity, k).unwrap();
                    let actual = spea2_select(pool.clone(), capacity, k).unwrap();
                    assert_eq!(
                        actual.iter().map(|item| item.id).collect::<Vec<_>>(),
                        expected.iter().map(|item| item.id).collect::<Vec<_>>(),
                        "case={case}, capacity={capacity}, k={k}"
                    );
                }
            }
            for invalid_k in [0, size + 1] {
                assert!(spea2_select_reference(pool.clone(), 1, invalid_k).is_err());
                assert!(spea2_select(pool.clone(), 1, invalid_k).is_err());
            }
        }
    }

    #[test]
    fn spea2_k_uses_the_self_inclusive_source_convention() {
        let pop = [0.0, 1.0, 3.0, 7.0]
            .into_iter()
            .enumerate()
            .map(|(id, x)| candidate(id as u64, vec![x, 10.0 - x]))
            .collect::<Vec<_>>();

        let k1 = super::spea2_scores(&pop, 1).unwrap();
        assert_eq!(k1, vec![0.5; 4]);

        let k2 = super::spea2_scores(&pop, 2).unwrap();
        let nearest: [f64; 4] = [1.0, 1.0, 2.0, 4.0];
        for (score, dx) in k2.iter().zip(nearest) {
            let distance = (dx * dx + dx * dx).sqrt();
            let expected = 1.0 / (distance + 2.0);
            assert_eq!(score.to_bits(), expected.to_bits());
        }

        let km = super::spea2_scores(&pop, pop.len()).unwrap();
        let farthest: [f64; 4] = [7.0, 6.0, 4.0, 7.0];
        for (score, dx) in km.iter().zip(farthest) {
            let distance = (dx * dx + dx * dx).sqrt();
            let expected = 1.0 / (distance + 2.0);
            assert_eq!(score.to_bits(), expected.to_bits());
        }
        assert!(super::spea2_scores(&pop, pop.len() + 1).is_err());
        assert!(spea2_select(pop, 2, 5).is_err());
    }

    #[test]
    fn objective_overflow_errors_include_algorithm_stage_and_candidate_ids() {
        let pop = vec![
            candidate(41, vec![-1e308, 1.0]),
            candidate(42, vec![0.0, 0.0]),
            candidate(43, vec![1e308, -1.0]),
        ];
        let error = format!("{:#}", quality_scores(&pop, false).unwrap_err());
        assert!(error.contains("NSGA-II crowding"), "{error}");
        assert!(
            error.contains("41") && error.contains("42") && error.contains("43"),
            "{error}"
        );
        assert!(error.contains("objective coordinate 0"), "{error}");

        let pop = vec![
            candidate(51, vec![0.0, 1e308]),
            candidate(52, vec![5e307, 5e307]),
            candidate(53, vec![1e308, 0.0]),
        ];
        let error = format!("{:#}", quality_scores(&pop, true).unwrap_err());
        assert!(error.contains("candidate IDs [51, 52, 53]"), "{error}");
        assert!(error.contains("accumulation overflow"), "{error}");
    }

    #[test]
    fn moga_and_ngga_distance_errors_retain_algorithm_context() {
        let pop = vec![
            candidate(71, vec![0.0, 1e200]),
            candidate(72, vec![1e200, 0.0]),
        ];
        let moga = format!("{:#}", super::moga_scores(&pop, 1.0).unwrap_err());
        assert!(moga.contains("MOGA sharing distance"), "{moga}");
        assert!(moga.contains("71") && moga.contains("72"), "{moga}");
        assert!(moga.contains("objective coordinate"), "{moga}");

        let mut config = RunConfig::default();
        config.algorithm.ngga_n1 = Some(1);
        let mut rng = Random::new(99);
        let ngga = format!(
            "{:#}",
            NggaState::initialize(&pop, &config, &mut rng)
                .err()
                .unwrap()
        );
        assert!(ngga.contains("NGGA initialization distance"), "{ngga}");
        assert!(ngga.contains("71") && ngga.contains("72"), "{ngga}");
        assert!(ngga.contains("objective coordinate"), "{ngga}");
    }

    #[test]
    fn spea2_reports_context_for_large_finite_sf2_objectives() {
        let problem = NumericProblem::new(&ProblemConfig {
            kind: ProblemKind::Sf2,
            dimensions: Some(1),
            min: Some(0.0),
            max: Some(1e153),
        })
        .unwrap();
        let mut rng = Random::new(3);
        let mut a = problem.initialize(&mut rng).unwrap();
        let mut b = problem.initialize(&mut rng).unwrap();
        a.values[0] = GeneValue::Real(0.0);
        b.values[0] = GeneValue::Real(1e153);
        let mut first = Individual::new(61, a.clone());
        first.set_evaluation(problem.evaluate(&a).unwrap()).unwrap();
        let mut second = Individual::new(62, b.clone());
        second
            .set_evaluation(problem.evaluate(&b).unwrap())
            .unwrap();
        let error = format!(
            "{:#}",
            super::spea2_scores(&[first, second], 1).unwrap_err()
        );
        assert!(error.contains("SPEA2 fitness distance"), "{error}");
        assert!(error.contains("61") && error.contains("62"), "{error}");
        assert!(error.contains("objective coordinate"), "{error}");
    }

    #[test]
    fn ngga_samples_distinct_candidates_and_keeps_decaying_state() {
        let pop = vec![candidate(0, vec![0.0, 2.0]), candidate(1, vec![1.0, 1.0])];
        let contribution = 1.0 - 2.0_f64.sqrt() / 3.0;
        let mut cfg = RunConfig::default();
        cfg.algorithm.sharing = 3.0;
        cfg.algorithm.ngga_decay_sharing = None;
        cfg.algorithm.ngga_n1 = Some(1);
        cfg.algorithm.ngga_n2 = Some(1);
        let mut rng = Random::new(7);
        let mut state = NggaState::initialize(&pop, &cfg, &mut rng).unwrap();
        assert!(
            (state.sharing[0] - contribution * 2.0).abs() < 1e-14,
            "{:?}",
            state.sharing
        );
        assert!(
            (state.sharing[1] - contribution).abs() < 1e-14,
            "{:?}",
            state.sharing
        );
        state.update(&pop, 1, &cfg, &mut rng).unwrap();
        assert!((state.sharing[0] - contribution * 2.0).abs() < 1e-14);
        assert!((state.sharing[1] - contribution * 1.5).abs() < 1e-14);
        state.initialize_one(0, &pop, 1, &cfg, &mut rng).unwrap();
        assert_eq!(state.sharing[0], contribution);
        assert!((state.sharing[1] - contribution * 2.5).abs() < 1e-14);
    }
}
