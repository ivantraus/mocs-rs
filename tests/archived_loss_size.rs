use mocs_rs::{
    config::{DecisionKind, FitnessKind, ModelKind, OperatorsConfig, RunConfig, TrainingConfig},
    data::{Dataset, Value},
    evaluation::Problem,
    genome::{GeneValue, Genome, Individual},
    models::{LearningProblem, SavedModel},
    rng::Random,
};

const CLASSIFICATION: &str =
    "@relation archived\n@attribute x {a,b}\n@attribute class {yes,no}\n@data\na,yes\nb,no\n";

fn problem(fitness: FitnessKind, decision: DecisionKind) -> LearningProblem {
    let data = Dataset::parse_str(CLASSIFICATION).unwrap();
    LearningProblem::new(
        data,
        TrainingConfig {
            model: ModelKind::Rule,
            fitness,
            decision,
            initial_rules: 1,
            wildcard_probability: 0.0,
            ..TrainingConfig::default()
        },
        OperatorsConfig::default(),
    )
    .unwrap()
}

fn rule_genome(problem: &LearningProblem, rules: &[(i64, i64)]) -> Genome {
    assert_eq!(problem.rule_width(), 2);
    let layout = problem.initialize(&mut Random::new(771)).unwrap().layout;
    let values = rules
        .iter()
        .flat_map(|(condition, target)| {
            [GeneValue::Integer(*condition), GeneValue::Integer(*target)]
        })
        .collect();
    Genome::new(values, layout).unwrap()
}

#[test]
fn archived_fitness_caps_objectives_above_training_row_threshold_but_keeps_raw_reports() {
    let archived = problem(FitnessKind::ArchivedLossSize, DecisionKind::BestAccuracy);
    let base = problem(FitnessKind::LossSize, DecisionKind::BestAccuracy);
    let cases = [
        (vec![(0, 0)], vec![0.5, 0.5]), // below: one rule, two training rows
        (vec![(0, 0), (1, 1)], vec![0.0, 1.0]), // equal: perfect raw accuracy
        (vec![(0, 0), (1, 1), (0, 0)], vec![1.0, 1.0]), // above: archived cap
    ];
    for (rules, expected_objectives) in cases {
        let genome = rule_genome(&archived, &rules);
        let report = archived.evaluate(&genome).unwrap();
        assert_eq!(report.objectives, expected_objectives);
        assert_eq!(report.rule_count, rules.len());

        let raw_report = base.evaluate(&genome).unwrap();
        assert_eq!(
            raw_report.objectives,
            vec![raw_report.loss, raw_report.size]
        );
        assert_eq!(report.loss, raw_report.loss);
        assert_eq!(report.size, raw_report.size);
        assert_eq!(report.coverage, raw_report.coverage);
        assert_eq!(report.rule_usage, raw_report.rule_usage);
        assert_eq!(
            archived
                .predict_dataset(&genome, archived.dataset())
                .unwrap(),
            base.predict_dataset(&genome, base.dataset()).unwrap()
        );
    }
    let oversized_perfect = rule_genome(&archived, &[(0, 0), (1, 1), (0, 0)]);
    assert_eq!(archived.evaluate(&oversized_perfect).unwrap().loss, 0.0);
}

#[test]
fn archived_compromise_selects_using_raw_loss_and_size_among_the_given_front() {
    let problem = problem(FitnessKind::ArchivedLossSize, DecisionKind::Compromise);
    let bad = rule_genome(&problem, &[(0, 1), (0, 1), (0, 1)]);
    let good = rule_genome(&problem, &[(0, 0), (1, 1), (0, 0)]);
    let bad_report = problem.evaluate(&bad).unwrap();
    let good_report = problem.evaluate(&good).unwrap();
    assert_eq!(bad_report.objectives, vec![1.0, 1.0]);
    assert_eq!(good_report.objectives, vec![1.0, 1.0]);
    assert!(good_report.loss < bad_report.loss);

    let mut bad_individual = Individual::new(0, bad);
    bad_individual.set_evaluation(bad_report).unwrap();
    let mut good_individual = Individual::new(1, good);
    good_individual.set_evaluation(good_report).unwrap();
    let front = vec![bad_individual, good_individual];
    assert_eq!(problem.choose(&front).unwrap().id, 1);
}

#[test]
fn archived_policy_roundtrips_by_name_without_changing_predictions() {
    let problem = problem(FitnessKind::ArchivedLossSize, DecisionKind::BestAccuracy);
    let genome = rule_genome(&problem, &[(0, 0), (1, 1), (0, 0)]);
    let before = problem.predict_dataset(&genome, problem.dataset()).unwrap();
    let path = std::env::temp_dir().join(format!("mocs-archived-{}.json", std::process::id()));
    problem.save_model(&genome, &path).unwrap();
    let serialized: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    assert_eq!(serialized["config"]["fitness"], "archived_loss_size");
    let loaded = SavedModel::load(&path).unwrap();
    let after = loaded.predict_str(CLASSIFICATION).unwrap();
    std::fs::remove_file(path).unwrap();
    assert_eq!(before, after);
    assert_eq!(after[0].value, Some(Value::Nominal("yes".into())));
    assert_eq!(after[1].value, Some(Value::Nominal("no".into())));
}

#[test]
fn archived_threshold_uses_training_rows_when_scoring_smaller_heldout_data() {
    let problem = problem(FitnessKind::ArchivedLossSize, DecisionKind::BestAccuracy);
    let genome = rule_genome(&problem, &[(0, 0), (1, 1)]);
    let heldout = Dataset::parse_str_for_prediction(
        "@relation heldout\n@attribute x {a,b}\n@attribute class {yes,no}\n@data\na,yes\n",
        problem.dataset().schema(),
    )
    .unwrap();
    let report = problem.score_dataset(&genome, &heldout).unwrap();
    assert_eq!(report.rows, 1);
    assert_eq!(report.rule_count, problem.dataset().schema().training_rows);
    assert_eq!(report.objectives, vec![0.0, 1.0]);

    let model_path = std::env::temp_dir().join(format!(
        "mocs-archived-heldout-model-{}.json",
        std::process::id()
    ));
    let heldout_path = std::env::temp_dir().join(format!(
        "mocs-archived-heldout-data-{}.arff",
        std::process::id()
    ));
    problem.save_model(&genome, &model_path).unwrap();
    std::fs::write(
        &heldout_path,
        "@relation heldout\n@attribute x {a,b}\n@attribute class {yes,no}\n@data\na,yes\n",
    )
    .unwrap();
    let loaded = SavedModel::load(&model_path).unwrap();
    let saved_report = loaded.score_file(&heldout_path).unwrap();
    std::fs::remove_file(model_path).unwrap();
    std::fs::remove_file(heldout_path).unwrap();
    assert_eq!(saved_report.rows, 1);
    assert_eq!(saved_report.objectives, vec![0.0, 1.0]);
}

#[test]
fn archived_fitness_rejects_numeric_targets_for_training_and_saved_models() {
    let numeric = Dataset::parse_str(
        "@relation regression\n@attribute x numeric\n@attribute y numeric\n@data\n0,0\n1,1\n",
    )
    .unwrap();
    let archived_regression = TrainingConfig {
        model: ModelKind::RegressionRule,
        fitness: FitnessKind::ArchivedLossSize,
        ..TrainingConfig::default()
    };
    let error = LearningProblem::new(
        numeric.clone(),
        archived_regression,
        OperatorsConfig::default(),
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("archived_loss_size fitness requires a classification target"));

    let base = LearningProblem::new(
        numeric,
        TrainingConfig {
            model: ModelKind::RegressionRule,
            ..TrainingConfig::default()
        },
        OperatorsConfig::default(),
    )
    .unwrap();
    let genome = base.initialize(&mut Random::new(772)).unwrap();
    let path =
        std::env::temp_dir().join(format!("mocs-archived-invalid-{}.json", std::process::id()));
    base.save_model(&genome, &path).unwrap();
    let mut serialized: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    serialized["config"]["fitness"] = serde_json::json!("archived_loss_size");
    std::fs::write(&path, serde_json::to_vec(&serialized).unwrap()).unwrap();
    let error = SavedModel::load(&path).unwrap_err().to_string();
    std::fs::remove_file(path).unwrap();
    assert!(error.contains("archived_loss_size fitness requires a classification target"));
}

#[test]
fn compatibility_learning_profiles_load_the_archived_fitness_mode() {
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    for profile in [
        "ga-iris.toml",
        "es-iris.toml",
        "ga-primary-tumour.toml",
        "es-primary-tumour.toml",
    ] {
        let path = root.join("examples/compatibility/modern").join(profile);
        let loaded = RunConfig::load(std::slice::from_ref(&path)).unwrap();
        assert_eq!(
            loaded.training.fitness,
            FitnessKind::ArchivedLossSize,
            "{profile}"
        );
    }
}
