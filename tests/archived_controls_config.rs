use mocs_rs::config::{AlgorithmKind, MutationKind, RunConfig};

#[test]
fn archived_controls_default_off_and_validate_their_algorithm_scope() {
    let config = RunConfig::default();
    assert!(!config.algorithm.archived_add_best);
    assert!(!config.algorithm.archived_moes_tournament_preselection);
    assert!(
        !config
            .operators
            .archived_evaluated_copy_uniform_mutation_gate
    );
    config.validate().unwrap();

    let mut invalid = RunConfig::default();
    invalid.algorithm.archived_add_best = true;
    assert!(invalid.validate().is_err());

    let mut invalid = RunConfig::default();
    invalid.algorithm.kind = AlgorithmKind::Moga;
    invalid.algorithm.archived_moes_tournament_preselection = true;
    assert!(invalid.validate().is_err());

    let mut invalid = RunConfig::default();
    invalid.algorithm.kind = AlgorithmKind::Moes;
    invalid.algorithm.archived_moes_tournament_preselection = true;
    invalid.algorithm.best = 1;
    assert!(invalid.validate().is_err());

    let mut valid = RunConfig::default();
    valid.algorithm.kind = AlgorithmKind::Moes;
    valid.algorithm.archived_moes_tournament_preselection = true;
    valid.algorithm.archived_add_best = true;
    valid.algorithm.best = 1;
    valid.validate().unwrap();

    let mut invalid = RunConfig::default();
    invalid
        .operators
        .archived_evaluated_copy_uniform_mutation_gate = true;
    assert!(invalid.validate().is_err());

    let mut invalid = RunConfig::default();
    invalid.algorithm.kind = AlgorithmKind::Moga;
    invalid
        .operators
        .archived_evaluated_copy_uniform_mutation_gate = true;
    invalid.operators.mutation = MutationKind::Normal;
    assert!(invalid.validate().is_err());
}
