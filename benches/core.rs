use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use mocs_rs::{
    config::{ModelKind, OperatorsConfig, TrainingConfig},
    data::Dataset,
    evaluation::Problem,
    genome::{Domain, GeneValue, Genome, GenomeLayout, Individual},
    models::LearningProblem,
    objectives::{distance, dominates, fronts},
    operators::{crossover, mutate},
    rng::Random,
};
use std::hint::black_box;
use std::sync::Arc;

fn ranking_and_distance(c: &mut Criterion) {
    let mut dominance = c.benchmark_group("dominance");
    for objectives in [2usize, 8, 32, 128] {
        let a = (0..objectives).map(|i| i as f64).collect::<Vec<_>>();
        let b = (0..objectives).map(|i| i as f64 + 0.5).collect::<Vec<_>>();
        dominance.bench_with_input(
            BenchmarkId::from_parameter(objectives),
            &objectives,
            |bencher, _| bencher.iter(|| black_box(dominates(black_box(&a), black_box(&b)))),
        );
    }
    dominance.finish();

    let mut ranking = c.benchmark_group("nondominated_fronts");
    for n in [128usize, 512, 2048] {
        let points = (0..n)
            .map(|i| {
                let x = i as f64 / n as f64;
                vec![x, (1.0 - x) + ((i * 17 % 31) as f64 / (n * 31) as f64)]
            })
            .collect::<Vec<_>>();
        ranking.bench_with_input(BenchmarkId::from_parameter(n), &points, |b, p| {
            b.iter(|| black_box(fronts(black_box(p))))
        });
    }
    ranking.finish();

    let mut distances = c.benchmark_group("euclidean_distance");
    for width in [8usize, 32, 128, 512] {
        let a = (0..width).map(|i| i as f64 * 0.25).collect::<Vec<_>>();
        let b = (0..width).map(|i| i as f64 * -0.125).collect::<Vec<_>>();
        distances.bench_with_input(BenchmarkId::from_parameter(width), &width, |bencher, _| {
            bencher.iter(|| black_box(distance(black_box(&a), black_box(&b)).unwrap()))
        });
    }
    distances.finish();
}

fn synthetic_classification_arff(rows: usize, features: usize) -> String {
    let mut input = String::from("@relation generated\n");
    for feature in 0..features {
        input.push_str(&format!("@attribute x{feature} numeric\n"));
    }
    input.push_str("@attribute class {a,b}\n@data\n");
    for row in 0..rows {
        for feature in 0..features {
            if feature > 0 {
                input.push(',');
            }
            let value = ((row * (feature + 1) + feature * 13) % 1001) as f64 / 1000.0;
            input.push_str(&value.to_string());
        }
        input.push_str(if row % 2 == 0 { ",a\n" } else { ",b\n" });
    }
    input
}

fn parsing(c: &mut Criterion) {
    let mut group = c.benchmark_group("arff_parse");
    for (rows, features) in [(128usize, 8usize), (1024, 8), (128, 32), (1024, 32)] {
        let input = synthetic_classification_arff(rows, features);
        let parsed = Dataset::parse_str(&input).unwrap();
        assert_eq!(parsed.len(), rows);
        assert_eq!(parsed.schema().feature_count(), features);
        group.bench_with_input(
            BenchmarkId::new("rows_features", format!("{rows}x{features}")),
            &input,
            |b, input| b.iter(|| black_box(Dataset::parse_str(black_box(input)).unwrap())),
        );
    }
    group.finish();
}

fn variation(c: &mut Criterion) {
    let mut group = c.benchmark_group("variation");
    for width in [16usize, 64, 256] {
        let layout = Arc::new(
            GenomeLayout::new(vec![
                Domain::Real {
                    min: -5.0,
                    max: 5.0
                };
                width
            ])
            .unwrap(),
        );
        let values_a = (0..width)
            .map(|i| GeneValue::Real((i as f64 * 0.17).sin()))
            .collect();
        let values_b = (0..width)
            .map(|i| GeneValue::Real((i as f64 * 0.21).cos()))
            .collect();
        let mut genome_a = Genome::new(values_a, layout.clone()).unwrap();
        // Mutation reads the genome's saved deviation state. Keep it nonzero so
        // this benchmark measures actual mutation work instead of a no-op scan.
        genome_a.deviations.fill(1.0);
        genome_a.deviation = 1.0;
        let a = Individual::new(1, genome_a);
        let b = Individual::new(2, Genome::new(values_b, layout).unwrap());
        let cfg = OperatorsConfig {
            per_locus_probability: Some(1.0),
            ..OperatorsConfig::default()
        };
        let mut rng = Random::new(2026);
        // Check the fixture outside Criterion's timed closure: the mutation
        // case must change a real-valued gene for the configured operator.
        let mut mutation_check = a.clone();
        mutate(&mut mutation_check, &cfg, &mut rng).unwrap();
        assert_ne!(mutation_check.genome().values, a.genome().values);
        group.bench_with_input(
            BenchmarkId::new("crossover", width),
            &width,
            |bencher, _| bencher.iter(|| black_box(crossover(&a, &b, &cfg, &mut rng, 3).unwrap())),
        );
        let mut rng = Random::new(2026);
        group.bench_with_input(BenchmarkId::new("mutation", width), &width, |bencher, _| {
            bencher.iter(|| {
                let mut child = a.clone();
                mutate(&mut child, &cfg, &mut rng).unwrap();
                black_box(child)
            })
        });
    }
    group.finish();
}

fn prediction(c: &mut Criterion) {
    let train = Dataset::parse_str(
        "@relation prediction\n@attribute x numeric\n@attribute class {a,b}\n@data\n0,a\n1,b\n",
    )
    .unwrap();
    let config = TrainingConfig {
        model: ModelKind::Instance,
        initial_rules: 8,
        ..TrainingConfig::default()
    };
    let problem = LearningProblem::new(train.clone(), config, OperatorsConfig::default()).unwrap();
    let mut rng = Random::new(19);
    let genome = problem.initialize(&mut rng).unwrap();
    let mut group = c.benchmark_group("prediction_preparsed");
    for rows in [128usize, 1024, 8192] {
        let mut input =
            String::from("@relation query\n@attribute x numeric\n@attribute class {a,b}\n@data\n");
        for i in 0..rows {
            input.push_str(&format!("{},?\n", (i % 1000) as f64 / 999.0));
        }
        let data = Dataset::parse_str_for_prediction(&input, train.schema()).unwrap();
        group.bench_with_input(BenchmarkId::from_parameter(rows), &data, |b, data| {
            b.iter(|| {
                black_box(
                    problem
                        .predict_dataset(black_box(&genome), black_box(data))
                        .unwrap(),
                )
            })
        });
    }
    group.finish();

    let mut scaled = c.benchmark_group("prediction_preparsed_rules_features_rows");
    for features in [8usize, 32] {
        let training_text = synthetic_classification_arff(64, features);
        let training = Dataset::parse_str(&training_text).unwrap();
        for rules in [8usize, 32] {
            let config = TrainingConfig {
                model: ModelKind::Instance,
                initial_rules: rules,
                ..TrainingConfig::default()
            };
            let problem =
                LearningProblem::new(training.clone(), config, OperatorsConfig::default()).unwrap();
            let mut rng = Random::new((features * 100 + rules) as u64);
            let genome = problem.initialize(&mut rng).unwrap();
            assert_eq!(genome.values.len(), rules * problem.rule_width());
            for rows in [128usize, 1024, 4096] {
                let query_text = synthetic_classification_arff(rows, features)
                    .replace(",a\n", ",?\n")
                    .replace(",b\n", ",?\n");
                let query =
                    Dataset::parse_str_for_prediction(&query_text, training.schema()).unwrap();
                assert_eq!(query.len(), rows);
                assert_eq!(query.schema().feature_count(), features);
                let fixture_predictions = problem.predict_dataset(&genome, &query).unwrap();
                assert_eq!(fixture_predictions.len(), rows);
                let id = format!("rules{rules}_features{features}_rows{rows}");
                scaled.bench_with_input(BenchmarkId::new("instance", id), &query, |b, query| {
                    b.iter(|| {
                        black_box(
                            problem
                                .predict_dataset(black_box(&genome), black_box(query))
                                .unwrap(),
                        )
                    })
                });
            }
        }
    }
    scaled.finish();
}

criterion_group!(
    benches,
    ranking_and_distance,
    parsing,
    variation,
    prediction
);
criterion_main!(benches);
