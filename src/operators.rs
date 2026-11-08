//! Variation operators (mating selection, crossover, mutation) and batch
//! evaluation with optional pruning.
//!
//! Selection scores passed to mating are "lower is better"; ties break by candidate ID.

use crate::{
    Result,
    config::{CrossoverKind, MatingKind, MutationKind, OperatorsConfig},
    evaluation::Problem,
    execution::ExecutionContext,
    genome::{Domain, GeneValue, Genome, GenomeLayout, Individual},
    objectives::distance,
    rng::Random,
};
use anyhow::Context;
use std::sync::Arc;

/// Extension point for mating and genome variation. Implementations receive the
/// serial run RNG so custom operators remain reproducible and auditable.
pub trait EvolutionOperators: Sync {
    /// Choose two parent indices from `population`. `scores` holds one selection
    /// score per candidate, lower being better.
    fn mate(
        &self,
        population: &[Individual],
        scores: &[f64],
        config: &OperatorsConfig,
        rng: &mut Random,
    ) -> Result<(usize, usize)>;
    /// Recombine two parents into two children, which should take IDs `child_id`
    /// and `child_id + 1` (both are reserved). The algorithms use the first child.
    fn crossover(
        &self,
        first: &Individual,
        second: &Individual,
        config: &OperatorsConfig,
        rng: &mut Random,
        child_id: u64,
    ) -> Result<(Individual, Individual)>;
    /// Mutate `individual` in place and return whether its genome changed. Changes
    /// must go through [`Individual`] methods that clear the cached evaluation.
    fn mutate(
        &self,
        individual: &mut Individual,
        config: &OperatorsConfig,
        rng: &mut Random,
    ) -> Result<bool>;
}

/// The built-in operators selected by [`OperatorsConfig`]; delegates to [`mate`],
/// [`crossover`] and [`mutate`].
#[derive(Debug, Default, Clone, Copy)]
pub struct BuiltInOperators;

impl EvolutionOperators for BuiltInOperators {
    fn mate(
        &self,
        population: &[Individual],
        scores: &[f64],
        config: &OperatorsConfig,
        rng: &mut Random,
    ) -> Result<(usize, usize)> {
        mate(population, scores, config, rng)
    }

    fn crossover(
        &self,
        first: &Individual,
        second: &Individual,
        config: &OperatorsConfig,
        rng: &mut Random,
        child_id: u64,
    ) -> Result<(Individual, Individual)> {
        crossover(first, second, config, rng, child_id)
    }

    fn mutate(
        &self,
        individual: &mut Individual,
        config: &OperatorsConfig,
        rng: &mut Random,
    ) -> Result<bool> {
        mutate(individual, config, rng)
    }
}

/// Apply the configured crossover (`k_point`, `variable_length` or
/// `simulated_binary`) to parents `a` and `b`, producing children with IDs
/// `child_id` and `child_id + 1`.
///
/// Errors if the parents have different layouts, or the genomes do not suit the
/// operator (SBX needs equal-length, all-real genomes; variable-length crossover
/// needs whole blocks and a matching `crossover_alignment`).
pub fn crossover(
    a: &Individual,
    b: &Individual,
    cfg: &OperatorsConfig,
    rng: &mut Random,
    child_id: u64,
) -> Result<(Individual, Individual)> {
    let ga = a.genome();
    let gb = b.genome();
    anyhow::ensure!(
        ga.layout == gb.layout,
        "parents have incompatible genome layouts"
    );
    let width = ga.layout.domains.len();
    let (mut av, mut bv) = (ga.values.clone(), gb.values.clone());
    let (mut ad, mut bd) = (ga.deviations.clone(), gb.deviations.clone());
    if cfg.crossover == CrossoverKind::SimulatedBinary {
        anyhow::ensure!(av.len() == bv.len(), "SBX requires equal genome lengths");
        anyhow::ensure!(
            ga.layout
                .domains
                .iter()
                .all(|domain| matches!(domain, Domain::Real { .. })),
            "simulated binary crossover requires real-valued genes"
        );
        let shared_beta = cfg.sbx_shared_beta.then(|| sbx_beta(rng, cfg.sbx_eta));
        for i in 0..av.len() {
            let beta = shared_beta.unwrap_or_else(|| sbx_beta(rng, cfg.sbx_eta));
            let x = ga.values[i].numeric();
            let y = gb.values[i].numeric();
            let d = &ga.layout.domains[i % width];
            av[i] = set_numeric(d, 0.5 * ((1.0 + beta) * x + (1.0 - beta) * y))?;
            bv[i] = set_numeric(d, 0.5 * ((1.0 - beta) * x + (1.0 + beta) * y))?;
            ad[i] = (0.5 * ((1.0 + beta) * ga.deviations[i] + (1.0 - beta) * gb.deviations[i]))
                .abs()
                .max(cfg.adaptive_epsilon);
            bd[i] = (0.5 * ((1.0 - beta) * ga.deviations[i] + (1.0 + beta) * gb.deviations[i]))
                .abs()
                .max(cfg.adaptive_epsilon);
        }
    } else if cfg.crossover == CrossoverKind::KPoint {
        // Upstream UniformCrossover alternates parents between sorted cut points
        // and lets the final segment run to the end of the genome. Cuts are drawn
        // over the common prefix, so pruned parents of unequal lengths are
        // supported: the final segment carries each parent's own tail, which
        // exchanges the tails when the cut count is odd. Both parents share one
        // repeated layout, so every common locus has the same typed domain and a
        // cut inside a rule block keeps whole blocks in both children. Equal-length
        // parents draw and swap exactly as before.
        let common = av.len().min(bv.len());
        let mut cuts: Vec<usize> = (0..cfg.crossover_points)
            .map(|_| rng.usize(common))
            .collect();
        cuts.sort_unstable();
        let mut swap = false;
        let mut c = 0;
        for i in 0..common {
            while c < cuts.len() && cuts[c] <= i {
                swap = !swap;
                c += 1;
            }
            if swap {
                std::mem::swap(&mut av[i], &mut bv[i]);
                std::mem::swap(&mut ad[i], &mut bd[i]);
            }
        }
        if cuts.len() % 2 == 1 && av.len() != bv.len() {
            let (a_tail, b_tail) = (av.split_off(common), bv.split_off(common));
            let (a_dev_tail, b_dev_tail) = (ad.split_off(common), bd.split_off(common));
            av.extend(b_tail);
            bv.extend(a_tail);
            ad.extend(b_dev_tail);
            bd.extend(a_dev_tail);
        }
    } else {
        // Corresponding cut points may fall inside a repeated rule block.
        anyhow::ensure!(
            av.len().is_multiple_of(width) && bv.len().is_multiple_of(width),
            "variable-length crossover requires whole layout blocks"
        );
        let alignment = if cfg.crossover_alignment == 1 {
            width
        } else {
            cfg.crossover_alignment
        };
        anyhow::ensure!(
            alignment == width,
            "variable-length crossover alignment must equal genome layout width ({width})"
        );
        let na = av.len() / alignment;
        let nb = bv.len() / alignment;
        anyhow::ensure!(na > 0 && nb > 0, "variable-length parents cannot be empty");
        let mut ca: Vec<_> = (0..cfg.crossover_points)
            .map(|_| rng.usize(na) * alignment)
            .collect();
        let mut cb: Vec<_> = (0..cfg.crossover_points)
            .map(|_| rng.usize(nb) * alignment)
            .collect();
        let mut offsets: Vec<_> = (0..cfg.crossover_points)
            .map(|_| rng.usize(alignment))
            .collect();
        ca.sort_unstable();
        cb.sort_unstable();
        loop {
            let inversion = (1..cfg.crossover_points).find(|&i| {
                (ca[i - 1] == ca[i] || cb[i - 1] == cb[i]) && offsets[i - 1] > offsets[i]
            });
            if let Some(i) = inversion {
                offsets.swap(i - 1, i);
            } else {
                break;
            }
        }
        for i in 0..cfg.crossover_points {
            ca[i] += offsets[i];
            cb[i] += offsets[i];
        }
        let (mut xa, mut xb, mut da, mut db) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let (mut pa, mut pb) = (0, 0);
        let mut take_b = false;
        for k in 0..ca.len() {
            if take_b {
                xa.extend_from_slice(&gb.values[pb..cb[k]]);
                da.extend_from_slice(&gb.deviations[pb..cb[k]]);
                xb.extend_from_slice(&ga.values[pa..ca[k]]);
                db.extend_from_slice(&ga.deviations[pa..ca[k]]);
            } else {
                xa.extend_from_slice(&ga.values[pa..ca[k]]);
                da.extend_from_slice(&ga.deviations[pa..ca[k]]);
                xb.extend_from_slice(&gb.values[pb..cb[k]]);
                db.extend_from_slice(&gb.deviations[pb..cb[k]]);
            }
            pa = ca[k];
            pb = cb[k];
            take_b = !take_b;
        }
        if take_b {
            xa.extend_from_slice(&gb.values[pb..]);
            da.extend_from_slice(&gb.deviations[pb..]);
            xb.extend_from_slice(&ga.values[pa..]);
            db.extend_from_slice(&ga.deviations[pa..]);
        } else {
            xa.extend_from_slice(&ga.values[pa..]);
            da.extend_from_slice(&ga.deviations[pa..]);
            xb.extend_from_slice(&gb.values[pb..]);
            db.extend_from_slice(&gb.deviations[pb..]);
        }
        av = xa;
        bv = xb;
        ad = da;
        bd = db;
    }
    let mut ga2 = ga.clone();
    ga2.values = av;
    ga2.deviation = (ga.deviation + gb.deviation) / 2.0;
    ga2.deviations = ad;
    let mut gb2 = gb.clone();
    gb2.values = bv;
    gb2.deviation = (ga.deviation + gb.deviation) / 2.0;
    gb2.deviations = bd;
    let c = if ga2.values.is_empty()
        || (ga2.values == ga.values
            && ga2.deviations == ga.deviations
            && ga2.deviation == ga.deviation)
    {
        copy_with_id(a, child_id)?
    } else {
        ga2.validate()?;
        Individual::new(child_id, ga2)
    };
    let d = if gb2.values.is_empty()
        || (gb2.values == gb.values
            && gb2.deviations == gb.deviations
            && gb2.deviation == gb.deviation)
    {
        copy_with_id(b, child_id + 1)?
    } else {
        gb2.validate()?;
        Individual::new(child_id + 1, gb2)
    };
    Ok((c, d))
}
fn sbx_beta(rng: &mut Random, eta: f64) -> f64 {
    let u = rng.uniform();
    if u <= 0.5 {
        (2.0 * u).powf(1.0 / (eta + 1.0))
    } else {
        (1.0 / (2.0 * (1.0 - u))).powf(1.0 / (eta + 1.0))
    }
}
fn copy_with_id(parent: &Individual, id: u64) -> Result<Individual> {
    let mut out = parent.clone();
    out.id = id;
    if let Some(e) = parent.evaluation() {
        out.set_evaluation(e.clone())?;
    }
    Ok(out)
}
fn set_numeric(d: &Domain, x: f64) -> Result<GeneValue> {
    match d {
        Domain::Real { .. } => d.clamp(GeneValue::Real(x)),
        Domain::Integer { .. } => d.clamp(GeneValue::Integer(x.round() as i64)),
    }
}

/// Apply the configured mutation (`uniform`, `normal`, `polynomial` or `adaptive`)
/// to `ind`, updating its self-adaptive deviations. Returns whether the genome
/// changed; a change clears the cached evaluation.
pub fn mutate(ind: &mut Individual, cfg: &OperatorsConfig, rng: &mut Random) -> Result<bool> {
    let mut genome = ind.genome().clone();
    let n = genome.values.len();
    if cfg.mutation == MutationKind::Adaptive {
        genome.deviation = (genome.deviation.max(cfg.adaptive_epsilon)
            * ((rng.normal() / (n as f64).sqrt()).exp()))
        .clamp(cfg.adaptive_epsilon, 1.0);
    }
    let global_n = (cfg.mutation == MutationKind::Normal).then(|| rng.normal());
    let tau = 1.0 / (2.0 * (n as f64).sqrt()).sqrt();
    let tau2 = 1.0 / (2.0 * n as f64).sqrt();
    for i in 0..n {
        let d = genome.layout.domains[i % genome.layout.domains.len()].clone();
        let old = genome.values[i];
        let (lo, hi) = d.bounds();
        let span = hi - lo;
        let val = match cfg.mutation {
            MutationKind::Normal => {
                let sigma = (genome.deviations[i]
                    * ((tau2 * global_n.unwrap() + tau * rng.normal()).exp()))
                .max(cfg.normal_epsilon);
                genome.deviations[i] = sigma;
                if !rng.bernoulli(sigma.min(1.0)) {
                    old
                } else if let Domain::Integer { min, max } = d {
                    if rng.bernoulli(sigma.min(1.0)) {
                        GeneValue::Integer(rng.integer(min, max))
                    } else {
                        old
                    }
                } else {
                    set_numeric(&d, old.numeric() + sigma * rng.normal() * span)?
                }
            }
            MutationKind::Polynomial | MutationKind::Adaptive => {
                if !rng.bernoulli(genome.deviation.min(1.0)) {
                    continue;
                }
                genome.deviations[i] = 1.0;
                if let Domain::Integer { min, max } = d {
                    GeneValue::Integer(rng.integer(min, max))
                } else {
                    let u = rng.uniform();
                    let delta = if u < 0.5 {
                        (2.0 * u).powf(1.0 / (cfg.mutation_eta + 1.0)) - 1.0
                    } else {
                        1.0 - (2.0 * (1.0 - u)).powf(1.0 / (cfg.mutation_eta + 1.0))
                    };
                    set_numeric(&d, old.numeric() + delta * span)?
                }
            }
            MutationKind::Uniform => {
                if !rng.bernoulli(genome.deviations[i].min(1.0)) {
                    continue;
                }
                match d {
                    Domain::Integer { min, max } => GeneValue::Integer(rng.integer(min, max)),
                    Domain::Real { min, max } => GeneValue::Real(rng.range(min, max)),
                }
            }
        };
        genome.values[i] = val;
    }
    genome.validate()?;
    let changed = genome.values != ind.genome().values
        || genome.deviations != ind.genome().deviations
        || genome.deviation != ind.genome().deviation;
    if changed {
        ind.replace_genome(genome)?;
    }
    Ok(changed)
}

/// Pick a parent pair with the configured mating scheme (`random`,
/// `tournament`, `nearest_neighbour` or `fitness_proportionate`).
///
/// `scores` are per-candidate selection scores, lower being better. Errors if the
/// pool is empty, `scores` has the wrong length, or an objective distance
/// overflows (nearest-neighbour mating). Candidates must be evaluated.
pub fn mate(
    pop: &[Individual],
    scores: &[f64],
    cfg: &OperatorsConfig,
    rng: &mut Random,
) -> Result<(usize, usize)> {
    let n = pop.len();
    anyhow::ensure!(n > 0, "mating requires a nonempty reproductive pool");
    anyhow::ensure!(
        scores.len() == n,
        "mating received {} scores for {n} candidates",
        scores.len()
    );
    if cfg.mating == MatingKind::Tournament {
        return Ok(tournament_pair(pop, scores, cfg, rng));
    }
    if cfg.mating == MatingKind::FitnessProportionate {
        let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let weights: Vec<_> = scores.iter().map(|s| (max - *s).max(0.0) + 1e-12).collect();
        let total: f64 = weights.iter().sum();
        if total <= 0.0 {
            return Ok((rng.usize(n), rng.usize(n)));
        }
        let x = rng.uniform() * total;
        let mut y = x + total / 2.0;
        if y >= total {
            y -= total
        }
        return Ok((select_weight(&weights, x), select_weight(&weights, y)));
    }
    let a = match cfg.mating {
        MatingKind::Random => rng.usize(n),
        MatingKind::NearestNeighbour => rng.usize(n),
        _ => unreachable!(),
    };
    let b = match cfg.mating {
        // Like the upstream random mating, a single-member pool mates with itself.
        MatingKind::Random if n == 1 => a,
        MatingKind::Random => {
            let mut j = rng.usize(n - 1);
            if j >= a {
                j += 1
            }
            j
        }
        MatingKind::NearestNeighbour => {
            let mut b = rng.usize(n);
            let mut best = distance(objectives(pop, a), objectives(pop, b)).with_context(|| {
                format!(
                    "nearest-neighbour mating distance between candidate IDs {} and {}",
                    pop[a].id, pop[b].id
                )
            })?;
            for _ in 1..cfg.neighbours {
                let c = rng.usize(n);
                let d = distance(objectives(pop, a), objectives(pop, c)).with_context(|| {
                    format!(
                        "nearest-neighbour mating distance between candidate IDs {} and {}",
                        pop[a].id, pop[c].id
                    )
                })?;
                if (d < best && d > 0.0) || best == 0.0 {
                    b = c;
                    best = d;
                }
            }
            b
        }
        _ => unreachable!(),
    };
    Ok((a, b))
}
fn objectives(pop: &[Individual], i: usize) -> &[f64] {
    &pop[i]
        .evaluation()
        .expect("mating requires evaluated candidates")
        .objectives
}
fn tournament_pair(
    pop: &[Individual],
    scores: &[f64],
    cfg: &OperatorsConfig,
    rng: &mut Random,
) -> (usize, usize) {
    let pick_group = |rng: &mut Random| -> Vec<usize> {
        (0..cfg.mating_tournament_size)
            .map(|_| rng.usize(scores.len()))
            .collect()
    };
    let winner = |group: &[usize], excluded: Option<usize>| -> Option<usize> {
        let mut best: Option<(usize, usize)> = None;
        for (position, &candidate) in group.iter().enumerate() {
            if Some(position) == excluded {
                continue;
            }
            if best.is_none_or(|(_, best_idx)| {
                candidate_better(pop, scores, candidate, best_idx, cfg.fitness_tournament)
            }) {
                best = Some((position, candidate));
            }
        }
        best.map(|(_, idx)| idx)
    };
    let first = pick_group(rng);
    let a = winner(&first, None).unwrap();
    if cfg.mating_winners == 1 {
        (a, winner(&pick_group(rng), None).unwrap())
    } else {
        let first_position = first.iter().position(|&x| x == a).unwrap();
        (a, winner(&first, Some(first_position)).unwrap_or(a))
    }
}

fn candidate_better(pop: &[Individual], scores: &[f64], a: usize, b: usize, fitness: bool) -> bool {
    if !fitness {
        if crate::objectives::dominates(objectives(pop, a), objectives(pop, b)) {
            return true;
        }
        if crate::objectives::dominates(objectives(pop, b), objectives(pop, a)) {
            return false;
        }
    }
    scores[a]
        .total_cmp(&scores[b])
        .then(pop[a].id.cmp(&pop[b].id))
        .is_lt()
}
fn select_weight(w: &[f64], mut x: f64) -> usize {
    for (i, v) in w.iter().enumerate() {
        if x < *v {
            return i;
        }
        x -= *v
    }
    w.len() - 1
}

/// Single-threaded [`evaluate_dirty_with_threads`].
pub fn evaluate_dirty<P: Problem>(
    problem: &P,
    items: &mut [Individual],
    rng: &mut Random,
    prune_probability: f64,
    evaluations: &mut usize,
) -> Result<()> {
    evaluate_dirty_with_threads(problem, items, rng, prune_probability, evaluations, 1)
}

/// Evaluate every unevaluated item, in ID order, adding to `evaluations`.
///
/// Each evaluated item is pruned with probability `prune_probability` (one RNG draw
/// per item, taken before evaluation); a pruned genome that changes is re-evaluated
/// and counted again. Results do not depend on `threads`. Errors on evaluation or
/// pruning failure, an invalid report, or an objective count that differs between
/// candidates.
pub fn evaluate_dirty_with_threads<P: Problem>(
    problem: &P,
    items: &mut [Individual],
    rng: &mut Random,
    prune_probability: f64,
    evaluations: &mut usize,
    threads: usize,
) -> Result<()> {
    let context = ExecutionContext::new(threads)?;
    evaluate_dirty_with_context(
        problem,
        items,
        rng,
        prune_probability,
        evaluations,
        &context,
    )
}

pub(crate) fn evaluate_dirty_with_context<P: Problem>(
    problem: &P,
    items: &mut [Individual],
    rng: &mut Random,
    prune_probability: f64,
    evaluations: &mut usize,
    _context: &ExecutionContext,
) -> Result<()> {
    let mut objective_count = items
        .iter()
        .find_map(|item| item.evaluation().map(|report| report.objectives.len()));
    let mut order: Vec<usize> = (0..items.len())
        .filter(|&i| items[i].evaluation().is_none())
        .collect();
    order.sort_by_key(|&i| items[i].id);
    let gates: Vec<bool> = order
        .iter()
        .map(|_| rng.uniform() < prune_probability)
        .collect();
    #[cfg(feature = "parallel")]
    let work_summary = if _context.is_parallel() && order.len() > 1 {
        Some(order.iter().fold((0usize, true), |(sum, known), &idx| {
            match problem.estimated_evaluation_work(items[idx].genome()) {
                Some(work) => (sum.saturating_add(work), known),
                None => (sum, false),
            }
        }))
    } else {
        None
    };
    #[cfg(feature = "parallel")]
    let should_parallel =
        work_summary.is_some_and(|(work, all_known)| !all_known || work >= 32_000);
    #[cfg(feature = "parallel")]
    let reports: Vec<_> = if should_parallel {
        use rayon::prelude::*;
        let max_chunks = order
            .len()
            .checked_div(_context.threads().saturating_mul(4).max(1))
            .unwrap_or(0)
            .max(1);
        let min_chunk_len = match work_summary {
            Some((total, true)) if !order.is_empty() => {
                let average = (total / order.len()).max(1);
                (4_000usize.saturating_add(average - 1) / average).clamp(1, max_chunks)
            }
            _ => max_chunks,
        };
        _context
            .install(|| {
                order
                    .par_iter()
                    .with_min_len(min_chunk_len)
                    .map(|&idx| problem.evaluate(items[idx].genome()))
                    .collect::<Vec<_>>()
            })?
            .into_iter()
            .collect::<Result<Vec<_>>>()?
    } else {
        order
            .iter()
            .map(|&idx| problem.evaluate(items[idx].genome()))
            .collect::<Result<Vec<_>>>()?
    };
    #[cfg(not(feature = "parallel"))]
    let reports: Vec<_> = {
        order
            .iter()
            .map(|&idx| problem.evaluate(items[idx].genome()))
            .collect::<Result<Vec<_>>>()?
    };
    for (at, (&idx, mut report)) in order.iter().zip(reports).enumerate() {
        report.validate()?;
        if let Some(expected) = objective_count {
            anyhow::ensure!(
                report.objectives.len() == expected,
                "problem returned {} objectives; expected {expected}",
                report.objectives.len()
            );
        } else {
            objective_count = Some(report.objectives.len());
        }
        *evaluations += 1;
        if gates[at] && problem.prune(items[idx].genome_mut(), &report)? {
            report = problem.evaluate(items[idx].genome())?;
            report.validate()?;
            if let Some(expected) = objective_count {
                anyhow::ensure!(
                    report.objectives.len() == expected,
                    "problem returned {} objectives; expected {expected}",
                    report.objectives.len()
                );
            }
            *evaluations += 1;
        }
        items[idx].set_evaluation(report)?;
    }
    Ok(())
}

/// Create a one-block genome with a uniform random value in each domain (integer
/// domains round the draw). Errors if `domains` is empty or invalid.
pub fn initialize_genome(domains: Vec<Domain>, rng: &mut Random) -> Result<Genome> {
    let values = domains
        .iter()
        .map(|d| {
            let (a, b) = d.bounds();
            set_numeric(d, rng.range(a, b))
        })
        .collect::<Result<Vec<_>>>()?;
    Genome::new(values, Arc::new(GenomeLayout::new(domains)?))
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "parallel")]
    use super::evaluate_dirty_with_context;
    use super::{crossover, mate};
    use crate::{
        config::{CrossoverKind, MatingKind, OperatorsConfig},
        evaluation::EvaluationReport,
        genome::{Domain, GeneValue, Genome, GenomeLayout, Individual},
        rng::Random,
    };
    #[cfg(feature = "parallel")]
    use crate::{evaluation::Problem, execution::ExecutionContext};
    use std::sync::Arc;

    #[cfg(feature = "parallel")]
    struct FailingProblem;
    #[cfg(feature = "parallel")]
    impl Problem for FailingProblem {
        fn initialize(&self, _: &mut Random) -> crate::Result<Genome> {
            unreachable!()
        }
        fn evaluate(&self, genome: &Genome) -> crate::Result<EvaluationReport> {
            let id = genome.values[0].numeric() as u64;
            if id == 10 {
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
            anyhow::bail!("failure candidate {id}")
        }
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn parallel_evaluation_reports_lowest_id_error_before_committing_any_result() {
        let layout = Arc::new(
            GenomeLayout::new(vec![Domain::Real {
                min: 0.0,
                max: 100.0,
            }])
            .unwrap(),
        );
        let mut items = [30, 10, 20]
            .into_iter()
            .map(|id| {
                Individual::new(
                    id,
                    Genome::new(vec![GeneValue::Real(id as f64)], layout.clone()).unwrap(),
                )
            })
            .collect::<Vec<_>>();
        let mut rng = Random::new(1);
        let mut evaluations = 0;
        let error = evaluate_dirty_with_context(
            &FailingProblem,
            &mut items,
            &mut rng,
            0.0,
            &mut evaluations,
            &ExecutionContext::new(2).unwrap(),
        )
        .unwrap_err();
        assert!(format!("{error:#}").contains("failure candidate 10"));
        assert_eq!(evaluations, 0);
        assert!(items.iter().all(|item| item.evaluation().is_none()));
    }

    #[test]
    fn nearest_mating_returns_contextual_distance_overflow() {
        let layout =
            Arc::new(GenomeLayout::new(vec![Domain::Real { min: 0.0, max: 1.0 }]).unwrap());
        let mut population = Vec::new();
        for (id, objectives) in [(81, vec![0.0, 0.0]), (82, vec![1e200, 0.0])] {
            let genome = Genome::new(vec![GeneValue::Real(0.5)], layout.clone()).unwrap();
            let mut individual = Individual::new(id, genome);
            individual
                .set_evaluation(EvaluationReport {
                    objectives,
                    ..EvaluationReport::default()
                })
                .unwrap();
            population.push(individual);
        }
        let config = OperatorsConfig {
            mating: MatingKind::NearestNeighbour,
            neighbours: 2,
            ..OperatorsConfig::default()
        };
        let error = (0..100)
            .find_map(|seed| mate(&population, &[0.0, 1.0], &config, &mut Random::new(seed)).err())
            .map(|error| format!("{error:#}"))
            .expect("one nearest-neighbour sample should compare the distinct candidates");
        assert!(
            error.contains("nearest-neighbour mating distance"),
            "{error}"
        );
        assert!(error.contains("81") && error.contains("82"), "{error}");
        assert!(error.contains("objective coordinate"), "{error}");
    }

    /// Parents over a typed repeated block `[Real, Integer, Real]` whose loci are
    /// distinguishable by value: `a` uses position `i`, `b` uses `100 + i`.
    fn typed_parent(id: u64, blocks: usize, offset: i64) -> Individual {
        let layout = Arc::new(
            GenomeLayout::new(vec![
                Domain::Real {
                    min: 0.0,
                    max: 1000.0,
                },
                Domain::Integer { min: 0, max: 1000 },
                Domain::Real {
                    min: 0.0,
                    max: 1000.0,
                },
            ])
            .unwrap(),
        );
        let values = (0..blocks * 3)
            .map(|i| {
                let v = offset + i as i64;
                if i % 3 == 1 {
                    GeneValue::Integer(v)
                } else {
                    GeneValue::Real(v as f64)
                }
            })
            .collect();
        let mut genome = Genome::new(values, layout).unwrap();
        genome.deviations = (0..blocks * 3)
            .map(|i| (offset + i as i64) as f64)
            .collect();
        Individual::new(id, genome)
    }

    #[test]
    fn k_point_crosses_unequal_lengths_over_common_prefix_and_keeps_typed_blocks() {
        for (a_blocks, b_blocks) in [(2, 5), (5, 2), (1, 4), (3, 3)] {
            let a = typed_parent(1, a_blocks, 0);
            let b = typed_parent(2, b_blocks, 100);
            let (la, lb) = (a_blocks * 3, b_blocks * 3);
            let common = la.min(lb);
            for points in 1..=4 {
                let cfg = OperatorsConfig {
                    crossover: CrossoverKind::KPoint,
                    crossover_points: points,
                    ..OperatorsConfig::default()
                };
                for seed in 0..200 {
                    let mut rng = Random::new(seed);
                    let (c, d) = crossover(&a, &b, &cfg, &mut rng, 10).unwrap();
                    let (cg, dg) = (c.genome(), d.genome());
                    cg.validate().unwrap();
                    dg.validate().unwrap();
                    assert_eq!(cg.deviations.len(), cg.values.len());
                    assert_eq!(dg.deviations.len(), dg.values.len());
                    // Each common locus comes from one parent and its partner gets the other.
                    for i in 0..common {
                        let pair = [cg.values[i], dg.values[i]];
                        let parents = [a.genome().values[i], b.genome().values[i]];
                        assert!(
                            pair == parents || pair == [parents[1], parents[0]],
                            "locus {i}: {pair:?} vs {parents:?}"
                        );
                        assert_eq!(cg.values[i].numeric(), cg.deviations[i]);
                        assert_eq!(dg.values[i].numeric(), dg.deviations[i]);
                    }
                    // The final segment runs to each parent's own end: an odd cut
                    // count exchanges the tails, an even count keeps them.
                    let (c_len, d_len) = if points % 2 == 1 { (lb, la) } else { (la, lb) };
                    assert_eq!(cg.values.len(), c_len);
                    assert_eq!(dg.values.len(), d_len);
                    let longer = if la >= lb { a.genome() } else { b.genome() };
                    let tail_child = if cg.values.len() > common { cg } else { dg };
                    assert_eq!(tail_child.values[common..], longer.values[common..]);
                    assert_eq!(tail_child.deviations[common..], longer.deviations[common..]);
                }
            }
        }
    }

    #[test]
    fn k_point_equal_lengths_keep_the_upstream_cut_sequence() {
        let a = typed_parent(1, 4, 0);
        let b = typed_parent(2, 4, 100);
        let n = a.genome().values.len();
        for points in 1..=3 {
            let cfg = OperatorsConfig {
                crossover: CrossoverKind::KPoint,
                crossover_points: points,
                ..OperatorsConfig::default()
            };
            for seed in 0..100 {
                let (c, d) = crossover(&a, &b, &cfg, &mut Random::new(seed), 10).unwrap();
                let mut reference = Random::new(seed);
                let mut cuts: Vec<usize> = (0..points).map(|_| reference.usize(n)).collect();
                cuts.sort_unstable();
                for i in 0..n {
                    let swapped = cuts.iter().filter(|&&cut| cut <= i).count() % 2 == 1;
                    let (from_c, from_d) = if swapped { (&b, &a) } else { (&a, &b) };
                    assert_eq!(c.genome().values[i], from_c.genome().values[i]);
                    assert_eq!(d.genome().values[i], from_d.genome().values[i]);
                }
            }
        }
    }
}
