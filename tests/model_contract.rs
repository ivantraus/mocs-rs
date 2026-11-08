use mocs_rs::{
    config::{ModelKind, OperatorsConfig, TrainingConfig},
    data::{Dataset, Value},
    evaluation::Problem,
    genome::{Domain, GeneValue, GenomeLayout},
    models::{LearningProblem, SavedModel},
    rng::Random,
};

#[test]
fn quoted_nominal_whitespace_and_quoted_question_mark_are_preserved() {
    let data = Dataset::parse_str(
        "@relation labels\n@attribute 'color name' {' blue ',blue,'?'}\n@attribute class {' yes ','no'}\n@data\n' blue ',' yes '\nblue,no\n'?',no\n",
    ).unwrap();
    assert_eq!(
        data.schema().attributes[0].nominal_values().unwrap(),
        &[" blue ", "blue", "?"]
    );
    assert_eq!(
        data.rows()[0].values[0],
        Some(Value::Nominal(" blue ".into()))
    );
    assert_eq!(
        data.rows()[0].values[1],
        Some(Value::Nominal(" yes ".into()))
    );
    assert_eq!(data.rows()[2].values[0], Some(Value::Nominal("?".into())));

    let error = Dataset::parse_str(
        "@relation labels\n@attribute color {' blue ',blue}\n@attribute class {yes,no}\n@data\n'blue ',yes\n",
    ).unwrap_err();
    assert!(error.to_string().contains("unknown nominal label"));
}

#[test]
fn all_missing_predictors_abstain_and_saved_model_preserves_predictions() {
    let train = Dataset::parse_str(
        "@relation binary\n@attribute x numeric\n@attribute class {a,b}\n@data\n0,a\n1,b\n",
    )
    .unwrap();
    let config = TrainingConfig {
        model: ModelKind::Instance,
        ..TrainingConfig::default()
    };
    let problem = LearningProblem::new(train.clone(), config, OperatorsConfig::default()).unwrap();
    let scored = Dataset::parse_str_for_prediction(
        "@relation heldout\n@attribute x numeric\n@attribute class {a,b}\n@data\n0.25,?\n?,a\n",
        train.schema(),
    )
    .unwrap();
    assert_eq!(scored.rows()[1].values[0], None);
    assert_eq!(problem.dataset().schema().target_index, 1);
    let mut rng = Random::new(37);
    let genome = problem.initialize(&mut rng).unwrap();
    let before = problem.predict_dataset(&genome, &scored).unwrap();
    assert!(!before[0].abstained);
    assert!(before[1].abstained);

    let path = std::env::temp_dir().join(format!("mocs-contract-{}.json", std::process::id()));
    problem.save_model(&genome, &path).unwrap();
    let loaded = SavedModel::load(&path).unwrap();
    let after = loaded
        .predict_str(
            "@relation heldout\n@attribute x numeric\n@attribute class {a,b}\n@data\n0.25,?\n?,a\n",
        )
        .unwrap();
    std::fs::remove_file(path).unwrap();
    assert_eq!(before, after);
}

#[test]
fn overflowing_finite_input_distance_is_reported_with_row_context() {
    let train = Dataset::parse_str(
        "@relation binary\n@attribute x numeric\n@attribute class {a,b}\n@data\n0,a\n1,b\n",
    )
    .unwrap();
    let config = TrainingConfig {
        model: ModelKind::Instance,
        ..TrainingConfig::default()
    };
    let problem = LearningProblem::new(train, config, OperatorsConfig::default()).unwrap();
    let mut rng = Random::new(4);
    let genome = problem.initialize(&mut rng).unwrap();
    let far = Dataset::parse_str_for_prediction(
        "@relation far\n@attribute x numeric\n@attribute class {a,b}\n@data\n1e308,a\n",
        problem.dataset().schema(),
    )
    .unwrap();
    let error = problem
        .score_dataset(&genome, &far)
        .unwrap_err()
        .to_string();
    assert!(error.contains("row 1"), "{error}");
    assert!(error.contains("x"), "{error}");
    assert!(error.contains("overflow"), "{error}");
}

#[test]
fn regression_scoring_rejects_overflowing_error_and_accumulation() {
    let train = Dataset::parse_str(
        "@relation regression\n@attribute x numeric\n@attribute y numeric\n@data\n0,0\n1,1\n",
    )
    .unwrap();
    let config = TrainingConfig {
        model: ModelKind::RegressionRule,
        ..TrainingConfig::default()
    };
    let problem = LearningProblem::new(train.clone(), config, OperatorsConfig::default()).unwrap();
    let mut rng = Random::new(12);
    let genome = problem.initialize(&mut rng).unwrap();

    let one_overflow = Dataset::parse_str_for_prediction(
        "@relation heldout\n@attribute x numeric\n@attribute y numeric\n@data\n0.5,1e308\n",
        train.schema(),
    )
    .unwrap();
    let error = problem
        .score_dataset(&genome, &one_overflow)
        .unwrap_err()
        .to_string();
    assert!(error.contains("row 1"), "{error}");
    assert!(
        error.contains("squared regression error overflowed"),
        "{error}"
    );

    let accumulated_overflow = Dataset::parse_str_for_prediction(
        "@relation heldout\n@attribute x numeric\n@attribute y numeric\n@data\n0.25,1e154\n0.75,1e154\n",
        train.schema(),
    )
    .unwrap();
    let error = problem
        .score_dataset(&genome, &accumulated_overflow)
        .unwrap_err()
        .to_string();
    assert!(error.contains("row 2"), "{error}");
    assert!(error.contains("loss accumulation overflowed"), "{error}");
}

#[test]
fn linear_regression_rule_uses_affine_consequent_and_roundtrips_exactly() {
    let train = Dataset::parse_str(
        "@relation line\n@attribute x numeric\n@attribute y numeric\n@data\n0,0\n10,20\n",
    )
    .unwrap();
    let config = TrainingConfig {
        model: ModelKind::RegressionLinearRule,
        initial_rules: 1,
        ..TrainingConfig::default()
    };
    assert_eq!(
        serde_json::to_string(&ModelKind::RegressionLinearRule).unwrap(),
        "\"regression_linear_rule\""
    );
    let problem = LearningProblem::new(train, config, OperatorsConfig::default()).unwrap();
    let layout = std::sync::Arc::new(
        GenomeLayout::new(vec![
            Domain::Real {
                min: 0.0,
                max: 10.0,
            },
            Domain::Real {
                min: 0.0,
                max: 10.0,
            },
            Domain::Real {
                min: -1.0,
                max: 2.0,
            },
            Domain::Real {
                min: -4.0,
                max: 4.0,
            },
        ])
        .unwrap(),
    );
    let genome = mocs_rs::genome::Genome::new(
        vec![
            GeneValue::Real(0.0),
            GeneValue::Real(10.0),
            GeneValue::Real(0.25),
            GeneValue::Real(0.5),
        ],
        layout,
    )
    .unwrap();
    let query = "@relation query\n@attribute x numeric\n@data\n5\n20\n?\n";
    let before = problem
        .predict_dataset(
            &genome,
            &Dataset::parse_str_for_prediction(query, problem.dataset().schema()).unwrap(),
        )
        .unwrap();
    assert_eq!(before[0].value, Some(Value::Numeric(10.0)));
    assert_eq!(before[1].value, Some(Value::Numeric(25.0)));
    assert!(before[1].out_of_range);
    assert_eq!(before[2].value, Some(Value::Numeric(10.0)));
    assert!(before[2].fallback);

    let path = std::env::temp_dir().join(format!(
        "mocs-linear-regression-{}.json",
        std::process::id()
    ));
    problem.save_model(&genome, &path).unwrap();
    let encoded = std::fs::read_to_string(&path).unwrap();
    assert!(encoded.contains("\"semantics_version\": 2"));
    let loaded = SavedModel::load(&path).unwrap();
    let after = loaded.predict_str(query).unwrap();
    std::fs::remove_file(path).unwrap();
    assert_eq!(before, after);
}

#[test]
fn linear_regression_rule_averages_multiple_affine_consequents() {
    let train = Dataset::parse_str(
        "@relation line\n@attribute x numeric\n@attribute y numeric\n@data\n0,0\n10,20\n",
    )
    .unwrap();
    let problem = LearningProblem::new(
        train,
        TrainingConfig {
            model: ModelKind::RegressionLinearRule,
            initial_rules: 2,
            neighbours: 2,
            temperature: 0.0,
            ..TrainingConfig::default()
        },
        OperatorsConfig::default(),
    )
    .unwrap();
    let mut init_rng = Random::new(30);
    let initialized = problem.initialize(&mut init_rng).unwrap();
    assert_eq!(initialized.deviation, 1.0 / 4.0);
    assert!(initialized.deviations.iter().all(|v| *v == 1.0 / 4.0));
    let layout = std::sync::Arc::new(
        GenomeLayout::new(vec![
            Domain::Real {
                min: 0.0,
                max: 10.0,
            },
            Domain::Real {
                min: 0.0,
                max: 10.0,
            },
            Domain::Real {
                min: -1.0,
                max: 2.0,
            },
            Domain::Real {
                min: -4.0,
                max: 4.0,
            },
        ])
        .unwrap(),
    );
    let genome = mocs_rs::genome::Genome::new(
        vec![
            GeneValue::Real(0.0),
            GeneValue::Real(0.0),
            GeneValue::Real(0.25),
            GeneValue::Real(0.5),
            GeneValue::Real(10.0),
            GeneValue::Real(10.0),
            GeneValue::Real(0.0),
            GeneValue::Real(0.0),
        ],
        layout,
    )
    .unwrap();
    let scored = Dataset::parse_str_for_prediction(
        "@relation query\n@attribute x numeric\n@data\n5\n",
        problem.dataset().schema(),
    )
    .unwrap();
    let predictions = problem.predict_dataset(&genome, &scored).unwrap();
    assert_eq!(predictions[0].value, Some(Value::Numeric(5.0)));
    assert_eq!(predictions[0].distance, Some(0.5));
}

#[test]
fn linear_regression_rule_rejects_unsupported_predictors_and_bad_semantics_tag() {
    let config = TrainingConfig {
        model: ModelKind::RegressionLinearRule,
        ..TrainingConfig::default()
    };
    let nominal = Dataset::parse_str(
        "@relation nominal\n@attribute x {a,b}\n@attribute y numeric\n@data\na,1\nb,2\n",
    )
    .unwrap();
    let error = LearningProblem::new(nominal, config.clone(), OperatorsConfig::default())
        .unwrap_err()
        .to_string();
    assert!(error.contains("numeric or integer predictor"), "{error}");

    let multiple = Dataset::parse_str(
        "@relation multiple\n@attribute x numeric\n@attribute z integer\n@attribute y numeric\n@data\n0,0,1\n1,1,2\n",
    )
    .unwrap();
    let error = LearningProblem::new(multiple, config.clone(), OperatorsConfig::default())
        .unwrap_err()
        .to_string();
    assert!(error.contains("exactly one predictor"), "{error}");

    let train = Dataset::parse_str(
        "@relation line\n@attribute x integer\n@attribute y numeric\n@data\n0,2\n10,4\n",
    )
    .unwrap();
    let problem = LearningProblem::new(train, config, OperatorsConfig::default()).unwrap();
    let mut rng = Random::new(29);
    let genome = problem.initialize(&mut rng).unwrap();
    let path = std::env::temp_dir().join(format!(
        "mocs-linear-regression-tag-{}.json",
        std::process::id()
    ));
    problem.save_model(&genome, &path).unwrap();
    let mut record: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    record["semantics_version"] = serde_json::Value::from(1);
    std::fs::write(&path, serde_json::to_vec(&record).unwrap()).unwrap();
    let error = SavedModel::load(&path).unwrap_err().to_string();
    assert!(error.contains("does not match model"), "{error}");

    record["semantics_version"] = serde_json::Value::from(2);
    record["genome"]["values"].as_array_mut().unwrap().pop();
    record["genome"]["deviations"].as_array_mut().unwrap().pop();
    std::fs::write(&path, serde_json::to_vec(&record).unwrap()).unwrap();
    let error = SavedModel::load(&path).unwrap_err().to_string();
    std::fs::remove_file(path).unwrap();
    assert!(
        error.contains("genome length must be a positive multiple"),
        "{error}"
    );
}

#[test]
fn linear_regression_rule_predicts_constant_targets_exactly_and_checks_overflow() {
    let train = Dataset::parse_str(
        "@relation constant\n@attribute x numeric\n@attribute y numeric\n@data\n0,0.1\n1,0.1\n",
    )
    .unwrap();
    let problem = LearningProblem::new(
        train.clone(),
        TrainingConfig {
            model: ModelKind::RegressionLinearRule,
            initial_rules: 2,
            neighbours: 2,
            temperature: 50.0,
            ..TrainingConfig::default()
        },
        OperatorsConfig::default(),
    )
    .unwrap();
    let layout = std::sync::Arc::new(
        GenomeLayout::new(vec![
            Domain::Real { min: 0.0, max: 1.0 },
            Domain::Real { min: 0.0, max: 1.0 },
            Domain::Real {
                min: -1.0,
                max: 2.0,
            },
            Domain::Real {
                min: -4.0,
                max: 4.0,
            },
        ])
        .unwrap(),
    );
    let genome = mocs_rs::genome::Genome::new(
        vec![
            GeneValue::Real(0.0),
            GeneValue::Real(0.0),
            GeneValue::Real(0.0),
            GeneValue::Real(0.0),
            GeneValue::Real(1.0),
            GeneValue::Real(1.0),
            GeneValue::Real(2.0),
            GeneValue::Real(4.0),
        ],
        layout,
    )
    .unwrap();
    let predictions = problem
        .predict_dataset(
            &genome,
            &Dataset::parse_str_for_prediction(
                "@relation query\n@attribute x numeric\n@data\n0.25\n",
                problem.dataset().schema(),
            )
            .unwrap(),
        )
        .unwrap();
    assert_eq!(
        predictions[0]
            .value
            .as_ref()
            .unwrap()
            .numeric()
            .unwrap()
            .to_bits(),
        0.1_f64.to_bits()
    );

    let negative_zero_train = Dataset::parse_str(
        "@relation negative_zero\n@attribute x numeric\n@attribute y numeric\n@data\n0,-0\n1,-0\n",
    )
    .unwrap();
    let negative_zero_problem = LearningProblem::new(
        negative_zero_train,
        TrainingConfig {
            model: ModelKind::RegressionLinearRule,
            initial_rules: 2,
            neighbours: 2,
            ..TrainingConfig::default()
        },
        OperatorsConfig::default(),
    )
    .unwrap();
    let layout = std::sync::Arc::new(
        GenomeLayout::new(vec![
            Domain::Real { min: 0.0, max: 1.0 },
            Domain::Real { min: 0.0, max: 1.0 },
            Domain::Real {
                min: -1.0,
                max: 2.0,
            },
            Domain::Real {
                min: -4.0,
                max: 4.0,
            },
        ])
        .unwrap(),
    );
    let negative_zero_genome = mocs_rs::genome::Genome::new(
        vec![
            GeneValue::Real(0.0),
            GeneValue::Real(0.0),
            GeneValue::Real(0.0),
            GeneValue::Real(0.0),
            GeneValue::Real(1.0),
            GeneValue::Real(1.0),
            GeneValue::Real(2.0),
            GeneValue::Real(4.0),
        ],
        layout,
    )
    .unwrap();
    let negative_zero_query = "@relation query\n@attribute x numeric\n@data\n0.25\n";
    let negative_zero_before = negative_zero_problem
        .predict_dataset(
            &negative_zero_genome,
            &Dataset::parse_str_for_prediction(
                negative_zero_query,
                negative_zero_problem.dataset().schema(),
            )
            .unwrap(),
        )
        .unwrap();
    assert_eq!(
        negative_zero_before[0]
            .value
            .as_ref()
            .unwrap()
            .numeric()
            .unwrap()
            .to_bits(),
        (-0.0_f64).to_bits()
    );
    let negative_zero_path = std::env::temp_dir().join(format!(
        "mocs-linear-regression-negative-zero-{}.json",
        std::process::id()
    ));
    negative_zero_problem
        .save_model(&negative_zero_genome, &negative_zero_path)
        .unwrap();
    let negative_zero_after = SavedModel::load(&negative_zero_path)
        .unwrap()
        .predict_str(negative_zero_query)
        .unwrap();
    std::fs::remove_file(negative_zero_path).unwrap();
    assert_eq!(negative_zero_before, negative_zero_after);
    assert_eq!(
        negative_zero_after[0]
            .value
            .as_ref()
            .unwrap()
            .numeric()
            .unwrap()
            .to_bits(),
        (-0.0_f64).to_bits()
    );

    let varying = Dataset::parse_str(
        "@relation varying\n@attribute x numeric\n@attribute y numeric\n@data\n0,0\n1,1e308\n",
    )
    .unwrap();
    let problem = LearningProblem::new(
        varying.clone(),
        TrainingConfig {
            model: ModelKind::RegressionLinearRule,
            initial_rules: 1,
            ..TrainingConfig::default()
        },
        OperatorsConfig::default(),
    )
    .unwrap();
    let mut rng = Random::new(32);
    let mut genome = problem.initialize(&mut rng).unwrap();
    *genome.values.last_mut().unwrap() = GeneValue::Real(4.0);
    let far = Dataset::parse_str_for_prediction(
        "@relation far\n@attribute x numeric\n@attribute y numeric\n@data\n1,0\n",
        varying.schema(),
    )
    .unwrap();
    let error = problem
        .predict_dataset(&genome, &far)
        .unwrap_err()
        .to_string();
    assert!(error.contains("row 1"), "{error}");
    assert!(error.contains("prediction is nonfinite"), "{error}");
}

#[test]
fn score_file_rejects_unrelated_genome_layout_before_label_indexing() {
    let train = Dataset::parse_str(
        "@relation binary\n@attribute x numeric\n@attribute class {a,b}\n@data\n0,a\n1,b\n",
    )
    .unwrap();
    let problem = LearningProblem::new(
        train,
        TrainingConfig {
            model: ModelKind::Rule,
            ..TrainingConfig::default()
        },
        OperatorsConfig::default(),
    )
    .unwrap();
    let mut rng = Random::new(43);
    let mut invalid = problem.initialize(&mut rng).unwrap();
    let mut domains = invalid.layout.domains.clone();
    let label_domain = domains.last_mut().unwrap();
    *label_domain = Domain::Integer { min: 0, max: 99 };
    invalid.layout = std::sync::Arc::new(GenomeLayout::new(domains).unwrap());
    *invalid.values.last_mut().unwrap() = GeneValue::Integer(99);

    let path = std::env::temp_dir().join(format!(
        "mocs-invalid-genome-score-{}.arff",
        std::process::id()
    ));
    std::fs::write(
        &path,
        "@relation heldout\n@attribute x numeric\n@attribute class {a,b}\n@data\n0.5,a\n",
    )
    .unwrap();
    let error = problem.score_file(&invalid, &path).unwrap_err().to_string();
    std::fs::remove_file(path).unwrap();
    assert!(
        error.contains("genome layout does not match learning problem"),
        "{error}"
    );
}
