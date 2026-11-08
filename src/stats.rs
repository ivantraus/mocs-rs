//! Paired statistical comparisons with explicit units and pairing identifiers.

use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
use statrs::distribution::{ContinuousCDF, Normal, StudentsT};
use std::collections::{BTreeMap, BTreeSet};

/// What one paired observation represents; recorded in results so analyses state their unit.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ComparisonUnit {
    /// One seeded optimization or training run.
    SeededRun,
    /// One cross-validation repetition (fold scores aggregated).
    CvRepetition,
    /// One dataset (e.g. a per-dataset mean).
    Dataset,
}

/// A value tagged with the ID used to pair it with the other sample.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Observation {
    /// Pairing key; must be nonempty and unique within a sample.
    pub pair_id: String,
    /// Observed value; must be finite.
    pub value: f64,
}

/// Outcome of a paired test of `a - b`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TestResult {
    /// Unit of the paired observations.
    pub unit: ComparisonUnit,
    /// Number of pairs.
    pub n: usize,
    /// Mean of `a - b`, when there is at least one pair.
    pub mean_difference: Option<f64>,
    /// Test statistic: Student t, or the Wilcoxon `min(W+, W-)`; `None` when untestable.
    pub statistic: Option<f64>,
    /// Two-sided p-value, or `None` when the test could not be computed.
    pub p_value: Option<f64>,
    /// Whether the test ran, or why not.
    pub status: TestStatus,
}

/// Whether a paired test produced a p-value.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TestStatus {
    /// The test ran normally.
    Tested,
    /// Fewer than two pairs (t-test only).
    InsufficientPairs,
    /// All differences are identical, so the t statistic is undefined.
    ZeroVariance,
    /// Every difference is zero (Wilcoxon); reported with p = 1.
    AllZeroDifferences,
}

/// Match observations by explicit ID; duplicate IDs, missing pairs and nonfinite values fail.
fn differences(a: &[Observation], b: &[Observation]) -> Result<Vec<f64>> {
    let mut left = BTreeMap::new();
    let mut right = BTreeMap::new();
    for (side, src, dst) in [("left", a, &mut left), ("right", b, &mut right)] {
        for x in src {
            ensure!(
                !x.pair_id.is_empty(),
                "{side} observation has empty pair_id"
            );
            ensure!(
                x.value.is_finite(),
                "{side} observation {} is nonfinite",
                x.pair_id
            );
            ensure!(
                dst.insert(x.pair_id.clone(), x.value).is_none(),
                "duplicate {side} pair_id {}",
                x.pair_id
            );
        }
    }
    let lk = left.keys().cloned().collect::<BTreeSet<_>>();
    let rk = right.keys().cloned().collect::<BTreeSet<_>>();
    ensure!(
        lk == rk,
        "paired observations have missing or extra pair IDs"
    );
    let values = lk
        .into_iter()
        .map(|k| left[&k] - right[&k])
        .collect::<Vec<_>>();
    ensure!(
        values.iter().all(|d| d.is_finite()),
        "paired difference overflowed to a nonfinite value"
    );
    Ok(values)
}

fn scaled_mean(values: &[f64]) -> f64 {
    let scale = values.iter().map(|x| x.abs()).fold(0.0, f64::max);
    if scale == 0.0 {
        0.0
    } else {
        (values.iter().map(|x| x / scale).sum::<f64>() / values.len() as f64) * scale
    }
}

/// Two-sided paired Student t-test on `a - b`, pairing observations by `pair_id`.
///
/// Returns a non-`Tested` status (with no p-value) for fewer than two pairs or zero
/// variance. Errors on empty, duplicate or unmatched pair IDs and nonfinite values.
pub fn paired_t_test(
    unit: ComparisonUnit,
    a: &[Observation],
    b: &[Observation],
) -> Result<TestResult> {
    let d = differences(a, b)?;
    let n = d.len();
    if n < 2 {
        return Ok(TestResult {
            unit,
            n,
            mean_difference: d.first().copied(),
            statistic: None,
            p_value: None,
            status: TestStatus::InsufficientPairs,
        });
    }
    let scale = d.iter().map(|x| x.abs()).fold(0.0, f64::max);
    if scale == 0.0 {
        return Ok(TestResult {
            unit,
            n,
            mean_difference: Some(0.0),
            statistic: None,
            p_value: None,
            status: TestStatus::ZeroVariance,
        });
    }
    let scaled = d.iter().map(|x| x / scale).collect::<Vec<_>>();
    let mean_scaled = scaled.iter().sum::<f64>() / n as f64;
    let mean = mean_scaled * scale;
    let variance_scaled = scaled
        .iter()
        .map(|x| (x - mean_scaled).powi(2))
        .sum::<f64>()
        / (n - 1) as f64;
    if variance_scaled == 0.0 {
        return Ok(TestResult {
            unit,
            n,
            mean_difference: Some(mean),
            statistic: None,
            p_value: None,
            status: TestStatus::ZeroVariance,
        });
    }
    let t = mean_scaled / (variance_scaled.sqrt() / (n as f64).sqrt());
    let dist = StudentsT::new(0.0, 1.0, (n - 1) as f64)?;
    let p = (2.0 * dist.sf(t.abs())).clamp(0.0, 1.0);
    Ok(TestResult {
        unit,
        n,
        mean_difference: Some(mean),
        statistic: Some(t),
        p_value: Some(p),
        status: TestStatus::Tested,
    })
}

/// Two-sided Wilcoxon signed-rank test. Exact sign enumeration by DP is used for
/// up to 50 nonzero pairs, with average tied ranks represented in doubled units.
/// Larger samples use a continuity-corrected normal approximation. Zero
/// differences are dropped. Errors as for [`paired_t_test`].
pub fn wilcoxon_signed_rank(
    unit: ComparisonUnit,
    a: &[Observation],
    b: &[Observation],
) -> Result<TestResult> {
    let d = differences(a, b)?;
    let n = d.len();
    let mean = if n == 0 { None } else { Some(scaled_mean(&d)) };
    let signs = d.iter().copied().filter(|x| *x != 0.0).collect::<Vec<_>>();
    let abs = signs.iter().copied().map(f64::abs).collect::<Vec<_>>();
    if abs.is_empty() {
        return Ok(TestResult {
            unit,
            n,
            mean_difference: mean,
            statistic: Some(0.0),
            p_value: Some(1.0),
            status: TestStatus::AllZeroDifferences,
        });
    }
    let m = abs.len();
    // Stable sorted ranks; doubled average ranks are exact integers, including ties.
    let mut ix = (0..m).collect::<Vec<_>>();
    ix.sort_by(|&i, &j| abs[i].total_cmp(&abs[j]));
    let mut ranks = vec![0usize; m];
    let mut lo = 0;
    while lo < m {
        let mut hi = lo + 1;
        while hi < m && abs[ix[hi]] == abs[ix[lo]] {
            hi += 1;
        }
        let rank_sum = (lo + 1 + hi) * (hi - lo); // 2 * sum of ranks lo+1..hi
        let doubled = rank_sum / (hi - lo);
        for k in lo..hi {
            ranks[ix[k]] = doubled;
        }
        lo = hi;
    }
    let total: usize = ranks.iter().sum();
    let wplus: usize = signs
        .iter()
        .zip(ranks.iter())
        .filter(|(x, _)| **x > 0.0)
        .map(|(_, r)| *r)
        .sum();
    let wminus = total - wplus;
    let observed = wplus.min(wminus);
    let (p, status) = if m <= 50 {
        let mut ways = vec![0f64; total + 1];
        ways[0] = 1.0;
        let mut used = 0;
        for r in &ranks {
            if *r == 0 {
                continue;
            }
            for s in (0..=used).rev() {
                let v = ways[s];
                if v != 0.0 {
                    ways[s + *r] += v;
                }
            }
            used += *r;
        }
        let extreme = ways
            .iter()
            .enumerate()
            .filter(|(s, _)| (*s).min(total - *s) <= observed)
            .map(|(_, v)| *v)
            .sum::<f64>();
        (extreme / 2f64.powi(m as i32), TestStatus::Tested)
    } else {
        let mu = total as f64 / 2.0;
        let variance = ranks.iter().map(|r| (*r as f64).powi(2)).sum::<f64>() / 4.0;
        let z = ((wplus as f64 - mu).abs() - 1.0).max(0.0) / variance.sqrt();
        let normal = Normal::new(0.0, 1.0)?;
        ((2.0 * normal.sf(z)).clamp(0.0, 1.0), TestStatus::Tested)
    };
    Ok(TestResult {
        unit,
        n,
        mean_difference: mean,
        statistic: Some(observed as f64 / 2.0),
        p_value: Some(p),
        status,
    })
}

/// Holm step-down adjustment. Undefined p-values stay null and occupy a family slot as p=1.
/// Returns adjusted p-values in input order; `results` is not modified.
pub fn holm_adjust(results: &mut [TestResult]) -> Vec<Option<f64>> {
    let m = results.len();
    let mut order = (0..m).collect::<Vec<_>>();
    order.sort_by(|&i, &j| {
        results[i]
            .p_value
            .unwrap_or(1.0)
            .total_cmp(&results[j].p_value.unwrap_or(1.0))
    });
    let mut adjusted = vec![1.0; m];
    let mut running: f64 = 0.0;
    for (rank, &i) in order.iter().enumerate() {
        running = running.max(((m - rank) as f64) * results[i].p_value.unwrap_or(1.0));
        adjusted[i] = running.min(1.0);
    }
    adjusted
        .into_iter()
        .enumerate()
        .map(|(i, p)| {
            if results[i].p_value.is_some() {
                Some(p)
            } else {
                None
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    fn obs(v: &[f64]) -> Vec<Observation> {
        v.iter()
            .enumerate()
            .map(|(i, x)| Observation {
                pair_id: i.to_string(),
                value: *x,
            })
            .collect()
    }
    #[test]
    fn t_uses_two_sided_student_tail_and_keeps_zero_pairs() {
        let t = paired_t_test(
            ComparisonUnit::SeededRun,
            &obs(&[1., 0., 3.]),
            &obs(&[0., 0., 0.]),
        )
        .unwrap();
        assert_eq!(t.n, 3);
        assert!((t.statistic.unwrap() - 4.0 / 7f64.sqrt()).abs() < 1e-12);
        assert!((t.p_value.unwrap() - (1.0 - 4.0 / 30f64.sqrt())).abs() < 1e-12);
    }
    #[test]
    fn t_zero_variance_is_degenerate() {
        let t = paired_t_test(ComparisonUnit::Dataset, &obs(&[2., 2.]), &obs(&[1., 1.])).unwrap();
        assert_eq!(t.status, TestStatus::ZeroVariance);
        assert_eq!(t.p_value, None);
    }
    #[test]
    fn exact_wilcoxon_handles_ties_and_zeroes() {
        let t = wilcoxon_signed_rank(
            ComparisonUnit::CvRepetition,
            &obs(&[1., 2., 0.]),
            &obs(&[0., 0., 0.]),
        )
        .unwrap();
        assert_eq!(t.n, 3);
        assert_eq!(t.p_value, Some(0.5));
        let z = wilcoxon_signed_rank(ComparisonUnit::Dataset, &obs(&[1., 2.]), &obs(&[1., 2.]))
            .unwrap();
        assert_eq!(z.p_value, Some(1.0));
    }
    #[test]
    fn rejects_ambiguous_pairing() {
        let a = vec![
            Observation {
                pair_id: "a".into(),
                value: 1.
            };
            2
        ];
        assert!(paired_t_test(ComparisonUnit::Dataset, &a, &a).is_err());
    }
    #[test]
    fn rejects_overflowing_finite_difference() {
        let left = [Observation {
            pair_id: "x".into(),
            value: f64::MAX,
        }];
        let right = [Observation {
            pair_id: "x".into(),
            value: -f64::MAX,
        }];
        assert!(paired_t_test(ComparisonUnit::SeededRun, &left, &right).is_err());
        assert!(wilcoxon_signed_rank(ComparisonUnit::SeededRun, &left, &right).is_err());
    }
    #[test]
    fn holm_adjusts_stepdown_and_keeps_undefined_null() {
        let mut rows = vec![
            TestResult {
                unit: ComparisonUnit::Dataset,
                n: 4,
                mean_difference: Some(1.0),
                statistic: Some(2.0),
                p_value: Some(0.01),
                status: TestStatus::Tested,
            },
            TestResult {
                unit: ComparisonUnit::Dataset,
                n: 4,
                mean_difference: Some(1.0),
                statistic: Some(1.0),
                p_value: Some(0.04),
                status: TestStatus::Tested,
            },
            TestResult {
                unit: ComparisonUnit::Dataset,
                n: 1,
                mean_difference: None,
                statistic: None,
                p_value: None,
                status: TestStatus::InsufficientPairs,
            },
        ];
        let adjusted = holm_adjust(&mut rows);
        assert_eq!(adjusted[0], Some(0.03));
        assert_eq!(adjusted[1], Some(0.08));
        assert_eq!(adjusted[2], None);
    }
}
