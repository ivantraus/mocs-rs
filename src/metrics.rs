//! Multiobjective front metrics. Historical `S` and `spacing` are deliberately
//! separate from standard hypervolume because their source definitions differ.

use anyhow::ensure;

/// Validate a minimization front and return its common objective dimension.
fn dimension(points: &[Vec<f64>]) -> anyhow::Result<usize> {
    let Some(first) = points.first() else {
        return Ok(0);
    };
    let n = first.len();
    ensure!(n > 0, "objective vectors must not be empty");
    for (i, p) in points.iter().enumerate() {
        ensure!(
            p.len() == n,
            "point {i} has {} objectives; expected {n}",
            p.len()
        );
        ensure!(
            p.iter().all(|x| x.is_finite()),
            "point {i} contains a nonfinite objective"
        );
    }
    Ok(n)
}

/// Keep unique points that are not dominated under minimization, preserving input order.
pub fn nondominated(points: &[Vec<f64>]) -> anyhow::Result<Vec<Vec<f64>>> {
    let n = dimension(points)?;
    if n == 0 {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    'candidate: for (i, p) in points.iter().enumerate() {
        for (j, q) in points.iter().enumerate() {
            if i == j {
                continue;
            }
            let weak = q.iter().zip(p).all(|(a, b)| a <= b);
            let strict = q.iter().zip(p).any(|(a, b)| a < b);
            if weak && strict {
                continue 'candidate;
            }
        }
        if !out.iter().any(|q: &Vec<f64>| q == p) {
            out.push(p.clone());
        }
    }
    Ok(out)
}

/// Exact dominated volume for 2D or 3D minimization fronts.
/// Points outside the reference box contribute zero and are reported separately.
pub fn hypervolume(points: &[Vec<f64>], reference: &[f64]) -> anyhow::Result<Hypervolume> {
    ensure!(
        reference.len() == 2 || reference.len() == 3,
        "reference must have 2 or 3 objectives"
    );
    ensure!(
        reference.iter().all(|x| x.is_finite()),
        "reference contains a nonfinite value"
    );
    let n = dimension(points)?;
    if n != 0 {
        ensure!(
            n == reference.len(),
            "front has {n} objectives but reference has {}",
            reference.len()
        );
    }
    let outside = points
        .iter()
        .filter(|p| p.iter().zip(reference).any(|(x, r)| x > r))
        .count();
    let mut valid = nondominated(points)?
        .into_iter()
        .filter(|p| p.iter().zip(reference).all(|(x, r)| x <= r))
        .collect::<Vec<_>>();
    if valid.is_empty() {
        return Ok(Hypervolume {
            value: 0.0,
            outside_points: outside,
        });
    }
    // Slicing along objective 0 reduces 3D volume to exact 2D unions.
    valid.sort_by(|a, b| a[0].total_cmp(&b[0]));
    let value = if reference.len() == 2 {
        let mut area = 0.0;
        let mut best_y = reference[1];
        for p in &valid {
            if p[1] < best_y {
                area += (reference[0] - p[0]) * (best_y - p[1]);
                best_y = p[1];
            }
        }
        area
    } else {
        let mut volume = 0.0;
        let mut previous_x = valid[0][0];
        let mut slice = Vec::<Vec<f64>>::new();
        let mut i = 0;
        while i < valid.len() {
            let x = valid[i][0];
            if x > previous_x {
                volume += (x - previous_x) * hv2(&slice, reference[1], reference[2]);
            }
            while i < valid.len() && valid[i][0] == x {
                slice.push(vec![valid[i][1], valid[i][2]]);
                i += 1;
            }
            previous_x = x;
        }
        volume + (reference[0] - previous_x) * hv2(&slice, reference[1], reference[2])
    };
    ensure!(
        value.is_finite() && value >= 0.0,
        "hypervolume overflowed or became negative"
    );
    Ok(Hypervolume {
        value,
        outside_points: outside,
    })
}

/// Result of [`hypervolume`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Hypervolume {
    /// Dominated volume (area in 2D) bounded by the reference point; nonnegative.
    pub value: f64,
    /// Number of input points with any objective beyond the reference point.
    pub outside_points: usize,
}

fn hv2(points: &[Vec<f64>], rx: f64, ry: f64) -> f64 {
    let mut ps = points
        .iter()
        .filter(|p| p[0] <= rx && p[1] <= ry)
        .cloned()
        .collect::<Vec<_>>();
    ps.sort_by(|a, b| a[0].total_cmp(&b[0]));
    let mut area = 0.0;
    let mut best_y = ry;
    for p in ps {
        if p[1] < best_y {
            area += (rx - p[0]) * (best_y - p[1]);
            best_y = p[1];
        }
    }
    area
}

/// Historical MOCS "S" value for a 2D front, as computed by upstream `metrics.cc`.
///
/// The front is reduced to its unique nondominated points (one copy of each
/// duplicate objective vector) and sorted by objective 0 ascending. The
/// reference point is then prepended, giving `q_0 = reference, q_1..q_k`, and
/// the value is `sum_{i=1..k} (q_{i-1}[0] - q_i[0]) * (q_{i-1}[1] - q_i[1])`.
/// Each term is the rectangle spanned by two consecutive points, not the area
/// they add to a dominated region, so this is not a hypervolume: later terms
/// are negative on a convex minimization front and the total can be negative.
/// An empty front gives 0. Use [`hypervolume`] for dominated area.
pub fn historical_s(points: &[Vec<f64>], reference: &[f64]) -> anyhow::Result<f64> {
    ensure!(
        reference.len() == 2 && reference.iter().all(|x| x.is_finite()),
        "historical S needs a finite 2D reference"
    );
    dimension(points)?;
    ensure!(
        points.iter().all(|p| p.len() == 2),
        "historical S requires 2 objectives"
    );
    let mut p = nondominated(points)?;
    // Unique nondominated 2D points have distinct objective-0 values, so this
    // order matches upstream's unstable sort.
    p.sort_by(|a, b| a[0].total_cmp(&b[0]));
    p.insert(0, reference.to_vec());
    let total = p
        .windows(2)
        .map(|w| (w[0][0] - w[1][0]) * (w[0][1] - w[1][1]))
        .fold(0.0, |sum, term| sum + term);
    ensure!(total.is_finite(), "historical S metric overflowed");
    Ok(total)
}

/// Source-compatible spacing: population standard deviation of adjacent Euclidean gaps
/// after sorting the nondominated 2D front by objective 0.
pub fn historical_spacing(points: &[Vec<f64>]) -> anyhow::Result<f64> {
    dimension(points)?;
    ensure!(
        points.iter().all(|p| p.len() == 2),
        "historical spacing requires 2 objectives"
    );
    let mut p = nondominated(points)?;
    p.sort_by(|a, b| a[0].total_cmp(&b[0]));
    if p.len() < 3 {
        return Ok(0.0);
    }
    let gaps = p
        .windows(2)
        .map(|w| ((w[0][0] - w[1][0]).powi(2) + (w[0][1] - w[1][1]).powi(2)).sqrt())
        .collect::<Vec<_>>();
    let mean = gaps.iter().sum::<f64>() / gaps.len() as f64;
    let spacing = (gaps.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / gaps.len() as f64).sqrt();
    ensure!(spacing.is_finite(), "historical spacing overflowed");
    Ok(spacing)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn hv2_exact_and_outside() {
        let h = hypervolume(&[vec![1., 3.], vec![2., 1.], vec![5., 0.]], &[4., 4.]).unwrap();
        assert!((h.value - 7.).abs() < 1e-12);
        assert_eq!(h.outside_points, 1);
    }
    #[test]
    fn hv3_exact() {
        let h = hypervolume(&[vec![1., 3., 3.], vec![2., 2., 2.]], &[4., 4., 4.]).unwrap();
        assert!((h.value - 9.).abs() < 1e-12);
    }
    #[test]
    fn filters_duplicates_and_dominated_points() {
        assert_eq!(
            nondominated(&[vec![1., 2.], vec![1., 2.], vec![2., 3.], vec![2., 1.]]).unwrap(),
            vec![vec![1., 2.], vec![2., 1.]]
        );
    }

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1e-12,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn historical_s_matches_upstream_smetric() {
        let reference = [1.0, 1.0];
        // Upstream: (1-0.2)*(1-0.8) + (0.2-0.5)*(0.8-0.3) = 0.16 - 0.15.
        assert_close(
            historical_s(&[vec![0.5, 0.3], vec![0.2, 0.8]], &reference).unwrap(),
            0.01,
        );
        // Three points: 0.9*0.1 + (-0.3)*0.5 + (-0.4)*0.3 = -0.18.
        assert_close(
            historical_s(
                &[vec![0.4, 0.4], vec![0.8, 0.1], vec![0.1, 0.9]],
                &reference,
            )
            .unwrap(),
            -0.18,
        );
        // Duplicates count once and dominated points are dropped:
        // (2-1)*(3-2) + (1-1.5)*(2-0.5) = 1 - 0.75.
        assert_close(
            historical_s(
                &[
                    vec![1.0, 2.0],
                    vec![1.5, 0.5],
                    vec![1.0, 2.0],
                    vec![1.5, 2.5],
                    vec![1.5, 0.5],
                ],
                &[2.0, 3.0],
            )
            .unwrap(),
            0.25,
        );
        // A single point is one rectangle against the reference.
        assert_close(historical_s(&[vec![0.25, 0.5]], &reference).unwrap(), 0.375);
        // Upstream does not clip points outside the reference box.
        assert_close(historical_s(&[vec![2.0, 0.5]], &reference).unwrap(), -0.5);
        assert_eq!(historical_s(&[], &reference).unwrap(), 0.0);
    }

    #[test]
    fn historical_spacing_matches_upstream_ssmetric() {
        // Gaps 0.5 and 1.0 after sorting: mean 0.75, population deviation 0.25.
        assert_close(
            historical_spacing(&[
                vec![0.9, 0.8],
                vec![0.0, 2.0],
                vec![0.3, 1.6],
                vec![0.3, 1.6],
            ])
            .unwrap(),
            0.25,
        );
        assert_eq!(
            historical_spacing(&[vec![0.0, 1.0], vec![1.0, 0.0]]).unwrap(),
            0.0
        );
        assert_eq!(historical_spacing(&[vec![0.0, 1.0]]).unwrap(), 0.0);
    }
}
