//! Multi-objective evolutionary optimization, classification and regression.
//!
//! `mocs_rs` is a Rust port of the MOCS evolutionary classifier framework. It runs
//! seeded, reproducible multi-objective evolutionary searches (MOGA, MOES, NGGA,
//! NSGA-II and SPEA2) over numeric benchmark problems, and learns compact rule,
//! instance, mixed and regression models from ARFF data. All objectives are
//! minimized.
//!
//! # Modules
//!
//! - [`config`]: the TOML-backed [`RunConfig`](config::RunConfig) and its defaults.
//! - [`algorithms`]: [`run`](algorithms::run) and the evolutionary loop.
//! - [`problems`]: built-in numeric benchmarks; [`models`]: learning problems and
//!   saved models; [`evaluation`]: the [`Problem`](evaluation::Problem) trait.
//! - [`data`]: ARFF parsing and fitted schemas; [`experiment`]: cross-validation folds.
//! - [`genome`], [`operators`], [`objectives`], [`rng`], [`execution`]: building blocks
//!   for custom problems and operators.
//! - [`metrics`], [`stats`], [`output`]: front metrics, paired tests and run records.
//!
//! See `README.md` for the command-line program and configuration, and
//! `docs/SEMANTICS.md` for detailed behavior and differences from upstream MOCS.
//!
//! # Example
//!
//! ```
//! use mocs_rs::{
//!     algorithms,
//!     config::{ProblemKind, RunConfig},
//!     problems::NumericProblem,
//! };
//!
//! # fn main() -> mocs_rs::Result<()> {
//! let mut config = RunConfig::default();
//! config.problem.kind = ProblemKind::Sf2;
//! config.seed = 7;
//! config.population = 16;
//! config.offspring = 16;
//! config.generations = 5;
//! config.threads = 1;
//!
//! let problem = NumericProblem::new(&config.problem)?;
//! let result = algorithms::run(&problem, &config)?;
//! assert!(!result.front.is_empty());
//! # Ok(())
//! # }
//! ```

#![forbid(unsafe_code)]
#![warn(missing_docs)]

/// Result type used throughout the crate, carrying an [`anyhow::Error`].
pub type Result<T> = anyhow::Result<T>;

pub mod algorithms;
pub mod config;
pub mod data;
pub mod evaluation;
pub mod execution;
pub mod experiment;
pub mod genome;
pub mod metrics;
pub mod models;
pub mod objectives;
pub mod operators;
pub mod output;
pub mod problems;
pub mod rng;
pub mod stats;
