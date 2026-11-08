use std::{fs, sync::Arc};

use mocs_rs::{
    config::{FitnessKind, ModelKind, OperatorsConfig, TrainingConfig},
    data::{AttributeKind, Dataset, Value},
    evaluation::Problem,
    genome::{Domain, GeneValue, Genome, GenomeLayout},
    models::{LearningProblem, SavedModel},
};

fn file(name: &str) -> String {
    format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"))
}

fn training(model: ModelKind) -> TrainingConfig {
    TrainingConfig {
        model,
        initial_rules: 2,
        wildcard_probability: 0.5,
        ..TrainingConfig::default()
    }
}

#[test]
fn arff_header_domains_quotes_integer_bounds_and_target_by_name() {
    let data = Dataset::load(file("models_mixed.arff"), Some("class")).unwrap();
    assert_eq!(data.schema().relation.as_deref(), Some("mixed, fixture"));
    assert_eq!(data.target_index(), 4);
    assert_eq!(
        data.schema().attributes[0].fitted_range,
        Some((0.0, 1000.0))
    );
    assert_eq!(data.schema().attributes[1].fitted_range, Some((0.0, 1.0)));
    assert_eq!(data.schema().attributes[3].declared_bounds, Some((1, 12)));
    assert_eq!(
        data.schema().attributes[2].nominal_values().unwrap()[0],
        "light blue"
    );
    assert_eq!(
        data.schema().attributes[2].nominal_values().unwrap()[2],
        "?"
    );
    assert_eq!(
        data.rows()[0].values[4],
        Some(Value::Nominal("class, one".into()))
    );
    assert_eq!(data.rows()[2].values[2], None);
}

#[test]
fn instance_distance_uses_each_training_range() {
    let train = Dataset::parse_str(
        "@relation ranges\n@attribute x numeric\n@attribute y numeric\n@attribute class {a,b}\n@data\n0,0,a\n1000,1,b\n",
    ).unwrap();
    let mut cfg = training(ModelKind::Instance);
    cfg.wildcard_probability = 0.0;
    let problem = LearningProblem::new(train, cfg, OperatorsConfig::default()).unwrap();
    let layout = Arc::new(
        GenomeLayout::new(vec![
            Domain::Real {
                min: 0.0,
                max: 1000.0,
            },
            Domain::Real { min: 0.0, max: 1.0 },
            Domain::Integer { min: 0, max: 1 },
        ])
        .unwrap(),
    );
    let genome = Genome::new(
        vec![
            GeneValue::Real(0.0),
            GeneValue::Real(0.0),
            GeneValue::Integer(0),
            GeneValue::Real(1000.0),
            GeneValue::Real(1.0),
            GeneValue::Integer(1),
        ],
        layout,
    )
    .unwrap();
    let heldout = Dataset::parse_str_for_prediction(
        "@relation heldout\n@attribute x numeric\n@attribute y numeric\n@attribute class {a,b}\n@data\n900,0,a\n",
        problem.dataset().schema(),
    ).unwrap();
    let report = problem.score_dataset(&genome, &heldout).unwrap();
    assert_eq!(report.loss, 0.0);
    assert_eq!(report.rule_usage, vec![1, 0]);
    assert_eq!(report.rule_count, 2);
    assert_eq!(report.unused_rule_fraction, 0.5);
}

#[test]
fn rule_boundaries_are_inclusive_and_unmatched_rows_abstain() {
    let train = Dataset::parse_str(
        "@relation rules\n@attribute x numeric\n@attribute color {red,blue}\n@attribute class {no,yes}\n@data\n0,red,yes\n10,blue,yes\n5,red,no\n",
    ).unwrap();
    let mut cfg = training(ModelKind::Rule);
    cfg.fitness = FitnessKind::LossGeneralitySize;
    let problem = LearningProblem::new(train, cfg, OperatorsConfig::default()).unwrap();
    let layout = Arc::new(
        GenomeLayout::new(vec![
            Domain::Real {
                min: 0.0,
                max: 10.0,
            },
            Domain::Real {
                min: 0.0,
                max: 10.0,
            },
            Domain::Integer { min: -2, max: 1 },
            Domain::Integer { min: 0, max: 1 },
        ])
        .unwrap(),
    );
    let genome = Genome::new(
        vec![
            GeneValue::Real(0.0),
            GeneValue::Real(10.0),
            GeneValue::Integer(-1),
            GeneValue::Integer(1),
        ],
        layout,
    )
    .unwrap();
    let boundary = Dataset::parse_str_for_prediction(
        "@relation boundary\n@attribute x numeric\n@attribute color {red,blue}\n@attribute class {no,yes}\n@data\n10,blue,yes\n",
        problem.dataset().schema(),
    ).unwrap();
    let report = problem.score_dataset(&genome, &boundary).unwrap();
    assert_eq!(report.loss, 0.0);
    assert_eq!(report.rule_usage, vec![1]);
    assert_eq!(report.rule_count, 1);
    assert_eq!(report.unused_rule_fraction, 0.0);
    assert_eq!(report.size, 1.0 / 3.0);
    let outside = Dataset::parse_str_for_prediction(
        "@relation outside\n@attribute x numeric\n@attribute color {red,blue}\n@attribute class {no,yes}\n@data\n15,red,no\n",
        problem.dataset().schema(),
    ).unwrap();
    let report = problem.score_dataset(&genome, &outside).unwrap();
    assert_eq!(report.abstentions, 1);
    assert_eq!(report.loss, 1.0);
    assert_eq!(report.out_of_range, 1);
    let missing = Dataset::parse_str_for_prediction(
        "@relation missing\n@attribute x numeric\n@attribute color {red,blue}\n@attribute class {no,yes}\n@data\n?,?,no\n",
        problem.dataset().schema(),
    ).unwrap();
    let report = problem.score_dataset(&genome, &missing).unwrap();
    assert_eq!(report.abstentions, 1);
    assert_eq!(report.coverage, 0.0);
    let unlabeled = Dataset::parse_str_for_prediction(
        "@relation unlabeled\n@attribute x numeric\n@attribute color {red,blue}\n@attribute class {no,yes}\n@data\n15,red,?\n",
        problem.dataset().schema(),
    ).unwrap();
    assert!(problem.score_dataset(&genome, &unlabeled).is_err());
}

#[test]
fn pruning_uses_report_and_evaluation_is_repeatable() {
    let train = Dataset::parse_str(
        "@relation prune\n@attribute x numeric\n@attribute class {a,b}\n@data\n0,a\n1,a\n10,b\n",
    )
    .unwrap();
    let problem =
        LearningProblem::new(train, training(ModelKind::Rule), OperatorsConfig::default()).unwrap();
    let layout = Arc::new(
        GenomeLayout::new(vec![
            Domain::Real {
                min: 0.0,
                max: 10.0,
            },
            Domain::Real {
                min: 0.0,
                max: 10.0,
            },
            Domain::Integer { min: 0, max: 1 },
        ])
        .unwrap(),
    );
    let genome = Genome::new(
        vec![
            GeneValue::Real(0.0),
            GeneValue::Real(1.0),
            GeneValue::Integer(0),
            GeneValue::Real(0.0),
            GeneValue::Real(1.0),
            GeneValue::Integer(0),
        ],
        layout,
    )
    .unwrap();
    let before = problem.evaluate(&genome).unwrap();
    let repeated = problem.evaluate(&genome).unwrap();
    assert_eq!(before, repeated);
    let mut pruned = genome.clone();
    assert_eq!(before.rule_usage, vec![2, 0]);
    assert!(problem.prune(&mut pruned, &before).unwrap());
    let after = problem.evaluate(&pruned).unwrap();
    assert_eq!(after.rows, before.rows);
    assert_eq!(after.rule_usage.len(), 1);
    assert_eq!(before.size, 2.0 / 3.0);
    assert_eq!(after.size, 1.0 / 3.0);
    assert_eq!(problem.evaluate(&genome).unwrap(), before);
}

#[test]
fn regression_uses_equal_boundary_weights_and_roundtrips_exactly() {
    let train = Dataset::load(file("regression.arff"), None).unwrap();
    let mut cfg = training(ModelKind::RegressionRule);
    cfg.neighbours = 2;
    cfg.temperature = 50.0;
    let problem = LearningProblem::new(train, cfg, OperatorsConfig::default()).unwrap();
    let mut rng = mocs_rs::rng::Random::new(29);
    let initialized = problem.initialize(&mut rng).unwrap();
    assert_eq!(initialized.deviation, 1.0 / 3.0);
    assert!(initialized.deviations.iter().all(|v| *v == 1.0 / 3.0));
    let layout = Arc::new(
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
                min: 0.0,
                max: 10.0,
            },
        ])
        .unwrap(),
    );
    let genome = Genome::new(
        vec![
            GeneValue::Real(0.0),
            GeneValue::Real(5.0),
            GeneValue::Real(0.0),
            GeneValue::Real(5.0),
            GeneValue::Real(10.0),
            GeneValue::Real(10.0),
        ],
        layout,
    )
    .unwrap();
    let query = "@relation query\n@attribute x numeric\n@data\n5\n?\n";
    let scored = Dataset::parse_str_for_prediction(query, problem.dataset().schema()).unwrap();
    let before = problem.predict_dataset(&genome, &scored).unwrap();
    let path = std::env::temp_dir().join(format!("mocs-model-{}.json", std::process::id()));
    problem.save_model(&genome, &path).unwrap();
    let loaded = SavedModel::load(&path).unwrap();
    let predictions = loaded.predict_str(query).unwrap();
    assert_eq!(before, predictions);
    assert_eq!(predictions[0].value, Some(Value::Numeric(5.0)));
    assert_eq!(predictions[0].distance, Some(0.0));
    assert_eq!(predictions[1].value, Some(Value::Numeric(5.0)));
    assert!(predictions[1].fallback);
    let encoded = fs::read_to_string(&path).unwrap();
    assert!(encoded.contains("\"format_version\": 1"));
    assert!(encoded.contains("\"semantics_version\": 1"));
    fs::remove_file(path).unwrap();
}

#[test]
fn unknown_nominal_and_malformed_rows_are_rejected() {
    let source = "@relation bad\n@attribute x numeric\n@attribute class {a,b}\n@data\n0,c\n";
    assert!(Dataset::parse_str(source).is_err());
    let malformed = "@relation bad\n@attribute x numeric\n@attribute class {a,b}\n@data\n0\n";
    assert!(Dataset::parse_str(malformed).is_err());
    let fixture = Dataset::load(file("models_mixed.arff"), None).unwrap();
    assert!(matches!(
        fixture.schema().attributes[0].kind,
        AttributeKind::Numeric
    ));
}
