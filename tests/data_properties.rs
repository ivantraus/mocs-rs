use std::{
    panic::AssertUnwindSafe,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

use mocs_rs::{
    config::{ModelKind, OperatorsConfig, TrainingConfig},
    data::Dataset,
    genome::{Domain, GeneValue, Genome, GenomeLayout},
    models::{LearningProblem, SavedModel},
};
use proptest::prelude::*;

static NEXT_MODEL: AtomicUsize = AtomicUsize::new(0);

proptest! {
    #![proptest_config(ProptestConfig::with_cases(80))]

    #[test]
    fn bounded_unicode_arff_inputs_never_panic(chars in prop::collection::vec(any::<char>(), 0..256)) {
        let input: String = chars.into_iter().collect();
        let schema = Dataset::parse_str("@relation base\n@attribute x numeric\n@attribute class {a,b}\n@data\n0,a\n1,b\n").unwrap();
        let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
            let _ = Dataset::parse_str(&input);
            let _ = Dataset::parse_str_for_prediction(&input, schema.schema());
        }));
        prop_assert!(result.is_ok());
    }

    #[test]
    fn saved_regression_models_preserve_float_bits_and_inclusive_boundaries(
        a in -10.0f64..10.0f64,
        b in -10.0f64..10.0f64,
        target in prop_oneof![Just(-0.0f64), -10.0f64..10.0f64],
    ) {
        let lo=a.min(b);let hi=a.max(b);
        let train=Dataset::parse_str("@relation reg\n@attribute x numeric\n@attribute y numeric\n@data\n-10,-10\n10,10\n").unwrap();
        let cfg=TrainingConfig{model:ModelKind::RegressionRule,initial_rules:1,..TrainingConfig::default()};
        let problem=LearningProblem::new(train,cfg,OperatorsConfig::default()).unwrap();
        let layout=Arc::new(GenomeLayout::new(vec![
            Domain::Real{min:-10.0,max:10.0},
            Domain::Real{min:-10.0,max:10.0},
            Domain::Real{min:-10.0,max:10.0},
        ]).unwrap());
        let genome=Genome::new(vec![GeneValue::Real(lo),GeneValue::Real(hi),GeneValue::Real(target)],layout).unwrap();
        let id=NEXT_MODEL.fetch_add(1,Ordering::Relaxed);
        let path=std::env::temp_dir().join(format!("mocs-model-prop-{}-{id}.json",std::process::id()));
        problem.save_model(&genome,&path).unwrap();
        let saved=SavedModel::load(&path).unwrap();
        let query=format!("@relation edge\n@attribute x numeric\n@data\n{lo}\n{hi}\n");
        let before=saved.predict_str(&query).unwrap();
        let loaded=SavedModel::load(&path).unwrap();
        let after=loaded.predict_str(&query).unwrap();
        prop_assert_eq!(before.len(),2);
        for p in &before {
            match p.value.as_ref().unwrap() {
                mocs_rs::data::Value::Numeric(v)=>prop_assert_eq!(v.to_bits(),target.to_bits()),
                _=>prop_assert!(false,"expected numeric prediction"),
            }
            prop_assert_eq!(p.distance,Some(0.0));
        }
        prop_assert_eq!(before,after);
        std::fs::remove_file(path).unwrap();
    }
}
