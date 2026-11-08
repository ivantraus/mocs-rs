#![allow(clippy::field_reassign_with_default)]

use mocs_rs::{
    algorithms::run,
    config::{AlgorithmKind, CrossoverKind, MatingKind, MutationKind, RunConfig},
    problems::NumericProblem,
};
use std::collections::BTreeSet;

const KINDS: [AlgorithmKind; 5] = [
    AlgorithmKind::Moga,
    AlgorithmKind::Moes,
    AlgorithmKind::Ngga,
    AlgorithmKind::Nsga2,
    AlgorithmKind::Spea2,
];
const MATINGS: [MatingKind; 4] = [
    MatingKind::Random,
    MatingKind::Tournament,
    MatingKind::NearestNeighbour,
    MatingKind::FitnessProportionate,
];

fn sbx_config(kind: AlgorithmKind, population: usize, offspring: usize) -> RunConfig {
    let mut cfg = RunConfig {
        seed: 0,
        population,
        offspring,
        generations: 3,
        threads: 1,
        ..RunConfig::default()
    };
    cfg.algorithm.kind = kind;
    cfg.operators.crossover = CrossoverKind::SimulatedBinary;
    cfg.operators.mutation = MutationKind::Polynomial;
    cfg.operators.crossover_probability = 1.0;
    cfg
}

#[test]
fn archived_moga_with_single_offspring_pool_mates_member_with_itself() {
    let mut cfg = sbx_config(AlgorithmKind::Moga, 4, 1);
    cfg.algorithm.elitism = Some(true);
    cfg.algorithm.archived_add_best = true;
    cfg.algorithm.best = 0;
    cfg.operators.mating = MatingKind::Random;
    let problem = NumericProblem::new(&cfg.problem).unwrap();
    let result = run(&problem, &cfg).unwrap();
    assert_eq!(result.generations, 3);
}

/// Every configuration that validates must run to completion with tiny
/// population/offspring sizes; anything else must be rejected up front.
#[test]
fn tiny_pool_matrix_never_panics_for_any_algorithm_and_mating() {
    let mut ran = BTreeSet::new();
    let mut rejected = 0usize;
    for kind in KINDS {
        for mating in MATINGS {
            for (population, offspring) in [(1, 1), (2, 1), (2, 2), (2, 3), (3, 1), (4, 1), (5, 3)]
            {
                for elitism in [None, Some(false), Some(true)] {
                    for (archived_add_best, best) in [(false, 0), (false, 1), (true, 0), (true, 1)]
                    {
                        for preselection in [false, true] {
                            for crossover_probability in [1.0, 0.5] {
                                let mut cfg = sbx_config(kind, population, offspring);
                                cfg.generations = 2;
                                cfg.operators.mating = mating;
                                cfg.operators.crossover_probability = crossover_probability;
                                cfg.algorithm.elitism = elitism;
                                cfg.algorithm.archived_add_best = archived_add_best;
                                cfg.algorithm.best = best;
                                cfg.algorithm.archived_moes_tournament_preselection = preselection;
                                cfg.algorithm.tournament_size = 2.min(population);
                                if let Err(error) = cfg.validate() {
                                    assert!(!error.to_string().is_empty());
                                    rejected += 1;
                                    continue;
                                }
                                let problem = NumericProblem::new(&cfg.problem).unwrap();
                                let result = run(&problem, &cfg).unwrap_or_else(|error| {
                                    panic!("{cfg:?} failed: {error:#}");
                                });
                                assert_eq!(result.generations, 2);
                                assert!(!result.front.is_empty());
                                ran.insert((format!("{kind:?}"), offspring, population));
                            }
                        }
                    }
                }
            }
        }
    }
    assert!(rejected > 0);
    for kind in KINDS {
        for (population, offspring) in [(2, 1), (3, 1), (4, 1), (5, 3)] {
            assert!(
                ran.contains(&(format!("{kind:?}"), offspring, population)),
                "{kind:?} never ran with population {population}, offspring {offspring}"
            );
        }
    }
}

#[test]
fn population_one_is_rejected_with_a_clear_message() {
    for kind in KINDS {
        let cfg = sbx_config(kind, 1, 1);
        let error = cfg.validate().unwrap_err().to_string();
        assert!(error.contains("population must be at least 2"), "{error}");
    }
}

#[test]
fn elitism_defaults_depend_on_algorithm_and_explicit_true_matches_default() {
    for kind in [AlgorithmKind::Nsga2, AlgorithmKind::Spea2] {
        let mut cfg = sbx_config(kind, 6, 6);
        assert!(cfg.algorithm.effective_elitism());
        assert_eq!(cfg.resolved().algorithm.elitism, Some(true));
        let problem = NumericProblem::new(&cfg.problem).unwrap();
        let default = run(&problem, &cfg).unwrap();
        cfg.algorithm.elitism = Some(true);
        let explicit = run(&problem, &cfg).unwrap();
        assert_eq!(
            serde_json::to_value(default).unwrap(),
            serde_json::to_value(explicit).unwrap()
        );
    }
    for kind in [AlgorithmKind::Moga, AlgorithmKind::Moes] {
        let cfg = sbx_config(kind, 6, 6);
        assert!(!cfg.algorithm.effective_elitism());
        assert_eq!(cfg.resolved().algorithm.elitism, Some(false));
    }
    let ngga = sbx_config(AlgorithmKind::Ngga, 6, 6);
    assert_eq!(ngga.resolved().algorithm.elitism, None);
    let record = serde_json::to_value(ngga.resolved()).unwrap();
    assert!(record["algorithm"].get("elitism").is_none());
}

#[test]
fn elitism_toml_values_parse_and_absent_means_algorithm_default() {
    let parse = |src: &str| -> RunConfig { toml::from_str(src).unwrap() };
    let cfg = parse("[algorithm]\nkind = \"nsga2\"\nelitism = false\n");
    assert_eq!(cfg.algorithm.elitism, Some(false));
    assert!(!cfg.algorithm.effective_elitism());
    let cfg = parse("[algorithm]\nkind = \"moga\"\nelitism = true\n");
    assert!(cfg.algorithm.effective_elitism());
    let cfg = parse("[algorithm]\nkind = \"spea2\"\n");
    assert_eq!(cfg.algorithm.elitism, None);
    assert!(cfg.algorithm.effective_elitism());
}

#[test]
fn non_elitist_nsga2_and_spea2_keep_only_offspring() {
    for kind in [AlgorithmKind::Nsga2, AlgorithmKind::Spea2] {
        let mut cfg = sbx_config(kind, 6, 8);
        cfg.generations = 4;
        cfg.algorithm.elitism = Some(false);
        let problem = NumericProblem::new(&cfg.problem).unwrap();
        let result = run(&problem, &cfg).unwrap();
        for pair in result.history.windows(2) {
            let old: BTreeSet<_> = pair[0].population_ids.iter().collect();
            assert_eq!(pair[1].population_ids.len(), cfg.population);
            assert!(
                pair[1].population_ids.iter().all(|id| !old.contains(id)),
                "{kind:?} retained a parent without elitism"
            );
        }
    }
}

#[test]
fn validation_rejects_settings_without_meaning() {
    let message = |cfg: RunConfig| cfg.validate().unwrap_err().to_string();

    let mut cfg = sbx_config(AlgorithmKind::Ngga, 6, 6);
    cfg.algorithm.elitism = Some(false);
    assert!(message(cfg).contains("algorithm.elitism does not apply to ngga"));

    let mut cfg = sbx_config(AlgorithmKind::Ngga, 6, 6);
    cfg.algorithm.best = 1;
    assert!(message(cfg).contains("algorithm.best does not apply to ngga"));

    let mut cfg = sbx_config(AlgorithmKind::Ngga, 6, 6);
    cfg.algorithm.archived_add_best = true;
    assert!(message(cfg).contains("archived_add_best"));

    for kind in [
        AlgorithmKind::Nsga2,
        AlgorithmKind::Spea2,
        AlgorithmKind::Moga,
        AlgorithmKind::Moes,
    ] {
        let mut cfg = sbx_config(kind, 6, 6);
        cfg.algorithm.elitism = Some(true);
        cfg.algorithm.best = 1;
        assert!(message(cfg.clone()).contains("algorithm.best has no effect"));
        cfg.algorithm.elitism = Some(false);
        cfg.validate().unwrap();
        cfg.offspring = 5;
        assert!(message(cfg).contains("without elitism requires offspring >= population"));
    }

    for kind in [AlgorithmKind::Nsga2, AlgorithmKind::Spea2] {
        let mut cfg = sbx_config(kind, 6, 6);
        cfg.algorithm.archived_add_best = true;
        assert!(message(cfg.clone()).contains("requires algorithm.best > 0"));
        cfg.algorithm.best = 1;
        cfg.validate().unwrap();
    }

    let mut cfg = sbx_config(AlgorithmKind::Nsga2, 6, 6);
    cfg.operators.mating = MatingKind::Tournament;
    cfg.operators.mating_tournament_size = 4;
    cfg.operators.mating_winners = 3;
    assert!(message(cfg).contains("operators.mating_winners must be 1"));
}
