//! Pareto dominance, nondominated sorting, objective-space distance and NSGA-II
//! crowding. All objectives are minimized.

use crate::Result;

/// Pareto dominance for minimization. Inputs must have equal nonzero dimensions and finite values.
pub fn dominates(a: &[f64], b: &[f64]) -> bool {
    assert!(!a.is_empty() && a.len() == b.len());
    let (mut strict, mut no_worse) = (false, true);
    for (&x, &y) in a.iter().zip(b) {
        assert!(x.is_finite() && y.is_finite());
        if x > y {
            no_worse = false;
            break;
        }
        if x < y {
            strict = true
        }
    }
    no_worse && strict
}
/// Indices of the points not dominated by any other point, in input order.
///
/// # Panics
/// Panics on empty, nonfinite or mismatched-dimension points.
pub fn nondominated_indices(points: &[Vec<f64>]) -> Vec<usize> {
    for p in points {
        assert!(!p.is_empty() && p.iter().all(|x| x.is_finite()));
    }
    if let Some(p) = points.first() {
        assert!(points.iter().all(|q| q.len() == p.len()));
    }
    (0..points.len())
        .filter(|&i| !(0..points.len()).any(|j| i != j && dominates(&points[j], &points[i])))
        .collect()
}
/// Euclidean distance between two objective vectors. Errors on mismatched or
/// empty dimensions, nonfinite inputs, or overflow.
pub fn distance(a: &[f64], b: &[f64]) -> Result<f64> {
    anyhow::ensure!(
        a.len() == b.len() && !a.is_empty(),
        "distance dimensions must match and be nonempty"
    );
    let mut squared_sum = 0.0;
    for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
        anyhow::ensure!(
            x.is_finite() && y.is_finite(),
            "distance objective coordinate {i} is not finite"
        );
        let difference = x - y;
        anyhow::ensure!(
            difference.is_finite(),
            "distance overflow at objective coordinate {i} while subtracting values"
        );
        let squared = difference * difference;
        anyhow::ensure!(
            squared.is_finite(),
            "distance overflow at objective coordinate {i} while squaring difference"
        );
        squared_sum += squared;
        anyhow::ensure!(
            squared_sum.is_finite(),
            "distance overflow at objective coordinate {i} while accumulating squared differences"
        );
    }
    let result = squared_sum.sqrt();
    anyhow::ensure!(
        result.is_finite(),
        "distance is not finite after square root"
    );
    Ok(result)
}

/// Fast nondominated sort: point indices grouped into successive Pareto fronts
/// (best first), each front sorted by index.
///
/// # Panics
/// Panics on the same inputs as [`dominates`].
pub fn fronts(points: &[Vec<f64>]) -> Vec<Vec<usize>> {
    let n = points.len();
    if n == 0 {
        return vec![];
    }
    let mut dominates_set = vec![Vec::new(); n];
    let mut count = vec![0usize; n];
    let mut front = Vec::new();
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            if dominates(&points[i], &points[j]) {
                dominates_set[i].push(j)
            } else if dominates(&points[j], &points[i]) {
                count[i] += 1
            }
        }
        if count[i] == 0 {
            front.push(i)
        }
    }
    let mut result = Vec::new();
    while !front.is_empty() {
        front.sort_unstable();
        let mut next = Vec::new();
        for &i in &front {
            for &j in &dominates_set[i] {
                count[j] -= 1;
                if count[j] == 0 {
                    next.push(j)
                }
            }
        }
        result.push(front);
        front = next;
    }
    result
}

/// NSGA-II crowding distance for the points selected by `indices`, returned
/// together with a per-candidate boundary flag (`true` for the extremes of any
/// nonconstant objective, whose crowding is then not accumulated). Gaps are
/// divided by each objective's range unless `raw` is set; constant objectives
/// are skipped. Errors on out-of-range indices, mismatched dimensions, nonfinite
/// values or overflow.
pub fn crowding(
    points: &[Vec<f64>],
    indices: &[usize],
    raw: bool,
) -> Result<(Vec<f64>, Vec<bool>)> {
    let mut values = vec![0.0; indices.len()];
    let mut boundary = vec![false; indices.len()];
    if indices.is_empty() {
        return Ok((values, boundary));
    }
    let first_index = indices[0];
    anyhow::ensure!(
        first_index < points.len(),
        "crowding candidate index {first_index} is outside points length {}",
        points.len()
    );
    let dimensions = points[first_index].len();
    anyhow::ensure!(
        dimensions > 0,
        "crowding candidate index {first_index} has no objective coordinates"
    );
    for &index in indices {
        anyhow::ensure!(
            index < points.len(),
            "crowding candidate index {index} is outside points length {}",
            points.len()
        );
        anyhow::ensure!(
            points[index].len() == dimensions,
            "crowding candidate index {index} has {} objectives; expected {dimensions}",
            points[index].len()
        );
        for (objective, value) in points[index].iter().enumerate() {
            anyhow::ensure!(
                value.is_finite(),
                "crowding candidate index {index} objective coordinate {objective} is not finite"
            );
        }
    }
    for (objective, _) in points[indices[0]].iter().enumerate() {
        let mut order: Vec<usize> = (0..indices.len()).collect();
        order.sort_by(|&a, &b| {
            points[indices[a]][objective]
                .total_cmp(&points[indices[b]][objective])
                .then(indices[a].cmp(&indices[b]))
        });
        let lo = points[indices[order[0]]][objective];
        let hi = points[indices[*order.last().unwrap()]][objective];
        anyhow::ensure!(
            lo.is_finite() && hi.is_finite(),
            "crowding objective coordinate {objective} contains a nonfinite value"
        );
        let range = hi - lo;
        anyhow::ensure!(
            range.is_finite(),
            "crowding range overflow at objective coordinate {objective}"
        );
        if hi == lo {
            continue;
        }
        boundary[order[0]] = true;
        boundary[*order.last().unwrap()] = true;
        for pos in 1..order.len() - 1 {
            let slot = order[pos];
            if !boundary[slot] {
                let gap = points[indices[order[pos + 1]]][objective]
                    - points[indices[order[pos - 1]]][objective];
                anyhow::ensure!(
                    gap.is_finite(),
                    "crowding gap overflow at objective coordinate {objective}"
                );
                values[slot] += if raw { gap } else { gap / range };
                anyhow::ensure!(
                    values[slot].is_finite(),
                    "crowding accumulation overflow at objective coordinate {objective}"
                );
            }
        }
    }
    Ok((values, boundary))
}

#[cfg(test)]
mod tests {
    use super::{crowding, distance};

    #[test]
    fn distance_guards_overflow_and_preserves_ordinary_arithmetic_bits() {
        let a = [1.0, 2.0, -3.0];
        let b = [4.0, -2.0, 5.0];
        let legacy = a
            .iter()
            .zip(&b)
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f64>()
            .sqrt();
        assert_eq!(distance(&a, &b).unwrap().to_bits(), legacy.to_bits());

        let error = distance(&[0.0, 0.0], &[1e200, 0.0])
            .unwrap_err()
            .to_string();
        assert!(error.contains("objective coordinate 0"), "{error}");
    }

    #[test]
    fn crowding_guards_range_gap_and_raw_accumulation_overflow() {
        let range = vec![vec![-1e308], vec![0.0], vec![1e308]];
        let error = crowding(&range, &[0, 1, 2], false).unwrap_err().to_string();
        assert!(
            error.contains("range overflow") && error.contains("coordinate 0"),
            "{error}"
        );

        let accumulation = vec![vec![0.0, 0.0], vec![5e307, 5e307], vec![1e308, 1e308]];
        let error = crowding(&accumulation, &[0, 1, 2], true)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("accumulation overflow") && error.contains("coordinate 1"),
            "{error}"
        );

        let ordinary = vec![vec![0.0, 0.0], vec![1.0, 1.0], vec![2.0, 2.0]];
        let (values, boundary) = crowding(&ordinary, &[0, 1, 2], false).unwrap();
        assert_eq!(values[1].to_bits(), 2.0_f64.to_bits());
        assert_eq!(boundary, vec![true, false, true]);
    }

    #[test]
    fn small_fronts_mark_only_nonconstant_objective_boundaries() {
        let (values, boundary) =
            crowding(&[vec![2.0, 7.0], vec![2.0, 7.0]], &[0, 1], false).unwrap();
        assert_eq!(values, vec![0.0; 2]);
        assert_eq!(boundary, vec![false; 2]);

        let (values, boundary) =
            crowding(&[vec![2.0, 7.0], vec![3.0, 7.0]], &[0, 1], false).unwrap();
        assert_eq!(values, vec![0.0; 2]);
        assert_eq!(boundary, vec![true; 2]);
    }

    #[test]
    fn crowding_rejects_malformed_selected_points_without_panicking() {
        assert!(
            crowding(&[vec![1.0]], &[1], false)
                .unwrap_err()
                .to_string()
                .contains("candidate index 1")
        );
        assert!(
            crowding(&[vec![1.0], vec![1.0, 2.0]], &[0, 1], false)
                .unwrap_err()
                .to_string()
                .contains("expected 1")
        );
        assert!(
            crowding(&[vec![f64::NAN]], &[0], false)
                .unwrap_err()
                .to_string()
                .contains("objective coordinate 0")
        );
    }
}
