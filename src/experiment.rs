//! Reproducible row-wise cross-validation split construction.

use crate::{
    data::{Dataset, Value},
    rng::Random,
};
use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// One cross-validation fold: row indices into the source dataset.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Fold {
    /// Zero-based fold number.
    pub id: usize,
    /// Sorted indices of the training rows (all rows not in the test set).
    pub train_indices: Vec<usize>,
    /// Sorted indices of the held-out rows.
    pub test_indices: Vec<usize>,
}

/// Seeded shuffled folds for regression and other nonclassification targets.
///
/// Rows are shuffled with `seed` and dealt round-robin into `count` test sets.
/// Errors unless `2 <= count <= dataset.len()`.
pub fn shuffled_folds(dataset: &Dataset, count: usize, seed: u64) -> Result<Vec<Fold>> {
    validate_count(dataset, count)?;
    let mut indices = (0..dataset.len()).collect::<Vec<_>>();
    let mut rng = Random::new(seed);
    shuffle(&mut indices, &mut rng);
    partition(dataset.len(), indices, count)
}

/// Seeded stratification for nominal classification targets. The class labels come from
/// the declared schema, while membership is assigned using observed training labels.
///
/// Each class is shuffled and dealt round-robin across folds, so class counts per
/// fold differ by at most one. Errors unless `2 <= count <= dataset.len()`, the
/// target is nominal and every row has a class label.
pub fn stratified_folds(dataset: &Dataset, count: usize, seed: u64) -> Result<Vec<Fold>> {
    validate_count(dataset, count)?;
    ensure!(
        dataset.is_classification(),
        "stratified folds require a nominal target"
    );
    let mut groups = BTreeMap::<String, Vec<usize>>::new();
    for (i, row) in dataset.rows().iter().enumerate() {
        let Some(Value::Nominal(label)) = row.values[dataset.target_index()].as_ref() else {
            anyhow::bail!("row {} has no nominal class label", row.source_index + 1);
        };
        groups.entry(label.clone()).or_default().push(i);
    }
    let mut rng = Random::new(seed);
    let mut bins = vec![Vec::new(); count];
    let mut cursor = 0;
    for members in groups.values_mut() {
        shuffle(members, &mut rng);
        for &index in members.iter() {
            bins[cursor % count].push(index);
            cursor += 1;
        }
    }
    for b in &mut bins {
        b.sort_unstable();
    }
    Ok(bins
        .into_iter()
        .enumerate()
        .map(|(id, test_indices)| {
            let is_test = test_indices
                .iter()
                .copied()
                .collect::<std::collections::BTreeSet<_>>();
            let train_indices = (0..dataset.len())
                .filter(|i| !is_test.contains(i))
                .collect();
            Fold {
                id,
                train_indices,
                test_indices,
            }
        })
        .collect())
}

fn validate_count(dataset: &Dataset, count: usize) -> Result<()> {
    ensure!(count >= 2, "fold count must be at least 2");
    ensure!(
        count <= dataset.len(),
        "fold count {count} exceeds {} rows",
        dataset.len()
    );
    Ok(())
}
fn shuffle(v: &mut [usize], rng: &mut Random) {
    for i in (1..v.len()).rev() {
        let j = rng.usize(i + 1);
        v.swap(i, j);
    }
}
fn partition(n: usize, indices: Vec<usize>, count: usize) -> Result<Vec<Fold>> {
    let mut bins = vec![Vec::new(); count];
    for (i, row) in indices.into_iter().enumerate() {
        bins[i % count].push(row);
    }
    Ok(bins
        .into_iter()
        .enumerate()
        .map(|(id, test_indices)| {
            let test = test_indices
                .iter()
                .copied()
                .collect::<std::collections::BTreeSet<_>>();
            let train_indices = (0..n).filter(|i| !test.contains(i)).collect();
            Fold {
                id,
                train_indices,
                test_indices,
            }
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    const ARFF: &str = "@relation tiny\n@attribute x numeric\n@attribute y {a,b}\n@data\n0,a\n1,a\n2,a\n3,b\n4,b\n5,b\n6,b\n7,b\n";
    #[test]
    fn stratified_is_seeded_and_balanced() {
        let d = Dataset::parse_str(ARFF).unwrap();
        let a = stratified_folds(&d, 3, 7).unwrap();
        let b = stratified_folds(&d, 3, 7).unwrap();
        assert_eq!(a, b);
        for f in a {
            let counts =
                f.test_indices
                    .iter()
                    .fold(BTreeMap::<String, usize>::new(), |mut c, &i| {
                        let l = d.rows()[i].values[1]
                            .as_ref()
                            .unwrap()
                            .nominal()
                            .unwrap()
                            .to_string();
                        *c.entry(l).or_default() += 1;
                        c
                    });
            assert!(counts.values().max().unwrap() - counts.values().min().unwrap() <= 1);
        }
    }
}
