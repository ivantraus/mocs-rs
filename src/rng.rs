//! Seeded, reproducible random-number stream used by every stochastic step.
//!
//! A run owns one [`Random`] stream seeded from `RunConfig::seed`; all draws are
//! made serially so results do not depend on thread count.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal};

/// Reproducible random stream. Version 1 pins ChaCha8 and rand 0.9 sampling behavior.
pub const RNG_CONTRACT_VERSION: u32 = 1;
/// Deterministic ChaCha8 random stream with the sampling helpers used by the crate.
#[derive(Clone)]
pub struct Random {
    inner: ChaCha8Rng,
}
impl Random {
    /// Create a stream from a 64-bit seed (zero is a valid seed).
    pub fn new(seed: u64) -> Self {
        Self {
            inner: ChaCha8Rng::seed_from_u64(seed),
        }
    }
    /// Uniform draw in `[0, 1)`.
    pub fn uniform(&mut self) -> f64 {
        self.inner.random::<f64>()
    }
    /// Uniform index in `0..upper`.
    ///
    /// # Panics
    /// Panics if `upper` is zero.
    pub fn usize(&mut self, upper: usize) -> usize {
        assert!(upper > 0, "upper bound must be positive");
        self.inner.random_range(0..upper)
    }
    /// Uniform real in `[min, max)`, or exactly `min` when `min == max`.
    ///
    /// # Panics
    /// Panics unless both bounds are finite and `min <= max`.
    pub fn range(&mut self, min: f64, max: f64) -> f64 {
        assert!(min.is_finite() && max.is_finite() && min <= max);
        if min == max {
            min
        } else {
            self.inner.random_range(min..max)
        }
    }
    /// Uniform integer in the inclusive range `min..=max`.
    ///
    /// # Panics
    /// Panics if `min > max`.
    pub fn integer(&mut self, min: i64, max: i64) -> i64 {
        assert!(min <= max);
        self.inner.random_range(min..=max)
    }
    /// Standard normal draw (mean 0, standard deviation 1).
    pub fn normal(&mut self) -> f64 {
        StandardNormal.sample(&mut self.inner)
    }
    /// Return `true` with probability `p`. `p = 0` and `p = 1` consume no draw.
    ///
    /// # Panics
    /// Panics unless `p` is finite and in `[0, 1]`.
    pub fn bernoulli(&mut self, p: f64) -> bool {
        assert!(p.is_finite() && (0.0..=1.0).contains(&p));
        if p == 0.0 {
            false
        } else if p == 1.0 {
            true
        } else {
            self.uniform() < p
        }
    }
}
