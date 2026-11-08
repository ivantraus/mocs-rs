# mocs-rs

A Rust port of [MOCS](https://github.com/ivantraus/mocs) for multi-objective
evolutionary optimization, classification, and regression.

`mocs-rs` provides a native Rust library and the `mocs` command-line program. It runs
evolutionary searches over numeric benchmark problems and learns compact rule, instance-
based, mixed, and regression models from ARFF data.

## What it includes

- Algorithms: MOGA, MOES, NGGA, NSGA-II, and SPEA2.
- Numeric problems: SF2, KUR, OKK, and ZDT6.
- Learning models: Rule, Instance, Mixed, and RegressionRule. The experimental
  RegressionLinearRule model supports exactly one numeric or integer predictor.
- Search operators: random, tournament, nearest-neighbour, and fitness-proportionate
  mating; k-point, variable-length, and simulated-binary crossover; uniform, normal,
  polynomial, and adaptive mutation.
- A typed Rust API, versioned JSON models, JSONL run records, minimization hypervolume,
  and CSV summaries.

All objectives are minimized. Learning configurations choose fitness objectives and a
final-front decision rule. See the [semantic notes](docs/SEMANTICS.md) for model
behavior and corrections relevant when comparing with MOCS.

## Install

Use the pinned Rust toolchain, 1.99.1, and build from this checkout:

```sh
cargo build --locked --release
```

To install the binary from this checkout into Cargo's configured install directory:

```sh
cargo install --path . --locked
```

The default build includes parallel execution. To build without that feature, use
`cargo build --locked --release --no-default-features`.

## Quick start

Create an output directory before running commands that write files:

```sh
mkdir -p results
```

Run a short numeric optimization, train a classifier from the bundled synthetic data,
then predict from rows with no labels:

```sh
cargo run --release --locked -- optimize \
  --config examples/sf2.toml --population 32 --offspring 32 \
  --generations 10 --threads 1 --output results/sf2.jsonl

cargo run --release --locked -- train \
  --config examples/classification.toml \
  --data examples/data/classification.arff \
  --model-out results/classifier.json --output results/training.jsonl \
  --population 32 --offspring 32 --generations 10 --threads 1

cargo run --release --locked -- predict \
  --model results/classifier.json --data examples/data/predict.arff \
  --output results/predictions.json
```

The learning config defaults to the last ARFF attribute as the target; the fixture's
last attribute is `class`. The saved model contains the fitted schema. Prediction input
must supply every predictor column using the fitted names and declared types; it may
omit the target attribute. Unknown attributes and absent predictor columns are errors.
Predictor cells may contain `?`; rows without usable evidence may abstain or use a
regression fallback. Predictions are JSON with row number, value, and abstention status.

The six commands are `optimize` (numeric search), `train` (fit and select one model),
`predict` (apply a saved model), `experiment` (seeded runs or folds), `metrics` (front
metrics), and `analyze` (CSV summaries and optional paired comparisons). Use
`mocs <command> --help` for every flag; `train --test` scores held-out rows after
selection.

## Data and models

The reader accepts dense ARFF with numeric, integer, and nominal attributes. Quoted
names and nominal values are supported, as is `?` for a missing value. It also accepts
the variations found in older ARFF files: a byte-order mark, `\r` or `\r\n` line
endings, nominal lists that span lines, integer values written as `1.0`, and nominal
values in a different letter case. Declared integer bounds are a starting range, not a
limit. Sparse `{...}` rows are not supported. Training requires a non-missing label on
every row; rows whose predictors are all missing are allowed. By default
the last attribute is the target; set `training.target` to select it by name. For
prediction, the fitted schema controls attribute types and ranges; unknown attributes
and absent predictor columns are rejected, while `?` cells may be ignored.

Classification reports error, coverage, conditional loss, and abstentions. A rule model
can abstain when no rule covers a row. Regression uses mean squared error in the
target's units squared and reports fallback and abstention counts. See
[SEMANTICS.md](docs/SEMANTICS.md) for details.

Constant-consequence regression can overfit sparse training data and perform poorly on
unseen inputs. Learning quality for the current model semantics has not been
comprehensively validated.
RegressionLinearRule remains experimental and is limited to one numeric or integer
predictor with fixed coefficient bounds. Rule models use finite inclusive intervals, so
a held-out value beyond learned intervals can produce an abstention.

Instance evolves point prototypes and predicts from the nearest prototype. Mixed evolves
interval and nominal rules, then predicts with the target of the nearest rule.
Classification and regression model kinds must match the target's declared ARFF type.

Fitness modes include `penalized_loss`, `loss_size`, `archived_loss_size`,
`loss_generality_size`, and `coverage_size`. Decision modes are `best_accuracy` and
`compromise`. Defaults are `loss_size` and `best_accuracy`; `archived_loss_size` is
classification-only. These choices affect the objectives used during search and how a
final solution is picked.

## Configuration

TOML files merge from left to right when repeated with `--config`; later files replace
earlier values at matching fields. Explicit CLI options such as `--seed`,
`--population`, `--offspring`, `--generations`, `--max-evaluations`, and `--threads`
override the merged config. Unknown fields are rejected. This is a compact example of
the native config shape:

```toml
seed = 0
population = 32
offspring = 32
generations = 10
threads = 1

[algorithm]
kind = "nsga2"

[problem]
kind = "sf2"

[training]
model = "rule"
fitness = "loss_size"
decision = "best_accuracy"
target = "class"

[operators]
mating = "random"
crossover = "k_point"
mutation = "uniform"
```

The fields shown are validated by `RunConfig`; its defaults and complete field
definitions are in [`src/config.rs`](src/config.rs). Algorithm, model, fitness,
decision, problem, and operator values use lowercase snake case. Later files merge
table fields, except changing an enum tag (`kind`, `model`, `fitness`, `decision`,
`mating`, `crossover`, or `mutation`) replaces that whole table; omitted fields then
use defaults.

You can merge a small TOML file over a project config, then override individual values
from the command line:

```sh
cargo run --release --locked -- train \
  --config examples/classification.toml --config examples/short-run.toml \
  --data examples/data/classification.arff --model-out results/model.json \
  --seed 12 --threads 1 --output results/merged-config.jsonl
```

Include only the values that need to differ from defaults. Every config section rejects
unknown keys, which helps catch misspelled options early. `--max-evaluations` sets an
evaluation budget; runs stop at generation boundaries and may finish slightly above that
count.

## Experiments and results

`experiment` creates seeded repetitions for numeric optimization, generated folds, or
existing fold files. For generated folds, pass one labeled dataset and a fold count:

```sh
cargo run --release --locked -- experiment \
  --config examples/classification.toml \
  --data examples/data/classification.arff --folds 2 \
  --seeds 0,1 --split-seed 7 --threads 2 --output results/cv.jsonl
```

Nominal targets use stratified folds; numeric targets use shuffled folds. Each fold fits
its own schema and chooses a model on training rows only, then reports held-out loss.
Optimization seeds are independent of `--split-seed`. Existing-fold patterns replace
`{fold}` with a number; the default start is `0`. For files numbered from one, pass
`--fold-start 1`:

```sh
cargo run --release --locked -- experiment --config examples/classification.toml \
  --train-pattern 'data/train{fold}.arff' --test-pattern 'data/test{fold}.arff' \
  --folds 10 --fold-start 1 --seeds 0,1 --output results/existing-folds.jsonl
```

Seed lists must be nonempty and unique. Generated fold counts must be between 2 and the row count;
stratification can leave rare classes absent from some folds.

`metrics` computes 2D or 3D hypervolume for each JSONL run record; in 2D it also writes
historical S and spacing values, computed as upstream MOCS did (historical S is not a
hypervolume and can be negative). Choose a minimization reference point worse than every
point you want included. Points outside the reference box contribute zero to the
hypervolume and are reported. For example:

```sh
cargo run --release --locked -- metrics \
  --input results/sf2.jsonl --reference 100,100 \
  --output results/sf2-metrics.csv
cargo run --release --locked -- analyze \
  --input results/sf2.jsonl --reference 100,100 \
  --output-dir results/analysis
```

`analyze` writes grouped `summary.csv` and `fronts.csv`. Repeated `--compare nsga2:moga`
options add paired t and Wilcoxon summaries with Holm adjustment. Pairing units are
`seeded_run`, `cv_repetition`, and `dataset`; CV-repetition requires `--folds` and
aggregates one complete repetition. Pairing relies on matching input and split metadata;
a shared dataset ID does not make different splits interchangeable.

Optimization, training, and experiment runs append JSON objects to their JSONL output.
Re-running with the same path adds records; choose a new file or remove the old one when
you need a fresh log. Training writes the selected model to `--model-out`, replacing an
existing file at that path. Prediction writes JSON to stdout unless `--output` is
supplied; that output file is replaced when written.

Each run record includes schema and semantic versions, RNG version, seed, resolved
configuration, input SHA-256 hashes, build metadata when available, evaluation counts,
and the final front. Learning records also include training selection and optional
held-out reports. Saved model JSON carries its format and prediction-semantics versions,
fitted schema, model configuration, operators, and evolved genome.

## Parallelism and repeatability

In parallel builds, unless TOML or the CLI sets it, the worker budget is Rust's available
core count (falling back to one). Serial-only builds default to one and reject larger
thread budgets. Both `--threads 1` and `threads = 1` request serial execution.
Pools are reused within one process, small batches stay serial, and NGGA's dependent
offspring steps remain sequential. Multiple jobs use outer parallelism with serial work
inside each job, so fewer jobs than workers can underuse cores and there are no nested
pools. At most the outer worker count of populations evolve at once; all job datasets are
materialized before execution. A single job can use the full budget for evaluations.

For memory-limited workloads, reduce `--threads`; each active learning job retains its
own population and fitted data.

With the same host, Rust toolchain, configuration, seed, and input, the run is designed
to be reproducible. Cross-host differences and identical trajectories with the C++
implementation are not promised.

## Library use

The library exports configuration, problems, algorithms, models, operators, data
handling, and metrics. This minimal example runs a numeric problem:

```rust
use mocs_rs::{
    algorithms,
    config::{ProblemKind, RunConfig},
    problems::NumericProblem,
};

fn main() -> mocs_rs::Result<()> {
    let mut config = RunConfig::default();
    config.problem.kind = ProblemKind::Sf2;
    config.seed = 7;
    config.population = 32;
    config.offspring = 32;
    config.generations = 10;
    config.threads = 1;

    let problem = NumericProblem::new(&config.problem)?;
    let result = algorithms::run(&problem, &config)?;
    println!("{} nondominated solutions", result.front.len());
    Ok(())
}
```

Generate the API documentation with `cargo doc --locked --no-deps --open`.

## Development

The repository includes unit and integration tests, a core Criterion benchmark, and
optional Python research scripts. Development checks include:

```sh
cargo fmt --all -- --check
cargo clippy --locked --all-targets --all-features -- -D warnings
cargo test --locked --all-features
cargo test --locked --no-default-features
cargo bench --locked --bench core
```

Optional developer scripts and tests require Python 3.11 or newer and use the
standard library. Run their tests with
`python3 -m unittest discover -s tests -p 'test_*.py'`.

| Tool | Purpose |
|---|---|
| `scripts/smoke.py` | Exercise a small CLI matrix using bundled synthetic fixtures. |
| `scripts/quality.py` | Measure fixed optimization, classification, and regression quality thresholds. |
| `scripts/audit_quality.py` | Audit a complete quality collection, including externally supplied Zoo folds. |
| `scripts/regression_compare.py` and `scripts/audit_regression_compare.py` | Compare constant and affine regression rules with training-mean and least-squares baselines, then verify saved artifacts. |
| `scripts/perf_study.py` and `scripts/audit_performance.py` | Collect paired timings for baseline/candidate binaries and verify identical scientific results. |
| `scripts/benchmark.py` | Compare paired JSONL run timings with bootstrap intervals. |
| `examples/profile_learning.rs` | Profile learning phases and optionally verify results against the ordinary engine. |
| `scripts/fetch_zoo.py` | Download and convert the Zoo dataset with checksum and attribution metadata. |
| `scripts/kernel_parity.py` | Optionally compare numeric kernels with an external MOCS C++ checkout. |

For a quick runtime check and a phase profile:

```sh
cargo build --locked --release
python3 scripts/smoke.py --mocs target/release/mocs
cargo run --locked --release --example profile_learning -- \
  --config examples/classification.toml --data examples/data/classification.arff \
  --generations 2 --verify-reference
```

For a regression comparison and its independent artifact audit:

```sh
python3 scripts/regression_compare.py --suite sparse --out results/regression-compare
python3 scripts/audit_regression_compare.py --out results/regression-compare
```

These studies can take substantially longer than unit tests. Quality thresholds are
measurement targets; their presence does not imply that all models pass. In particular,
the sparse constant-consequence regression case remains a known limitation.
Protocol manifests in `scripts/protocols/` pin regression run identity. Use a new
manifest for changed protocols and a fresh output directory when inputs, scripts,
binaries, or protocol identity change. Keep the binary used for collection available
until its audit finishes.

Benchmark manifests and all generated outputs belong under `results/`. For example,
`python3 scripts/perf_study.py --baseline target/release/mocs --manifest results/perf-study.json`
freezes and measures a baseline; pass `--candidate` with a separate candidate binary
to collect paired comparisons. Use `--help` on each script for its full interface.

Normal builds, tests, and the smoke matrix need no external datasets or C++ checkout.
The optional `smoke.py --diagnostics` mode accepts `--legacy-data-root` for historical
Bupa/Sonar folds. Zoo quality checks similarly require externally supplied folds;
`fetch_zoo.py` supplies the source dataset, not those historical splits. See the
[provenance record](docs/PROVENANCE.md) for data attribution and interpretation.

To run optional C++ kernel parity, build the Rust tests and set `MOCS_CPP_SOURCE` to
an upstream MOCS checkout, then run `python3 scripts/kernel_parity.py`. A C++ compiler
is required. The transition-oracle driver and its recorded outputs live in
`tests/fixtures/`; `legacy_transition_expected.json` records the original capture
metadata and regeneration instructions. Compatibility configs are in
`examples/compatibility/modern/` and `examples/compatibility/archived/`.

The crate package includes Rust sources, examples, tests, fixtures, benchmarks, and
reference documentation. Python developer tooling and CI configuration stay in the
Git repository. Inspect and verify the package with `cargo package --locked`; build
source identities can differ between a repository checkout and the packaged files.

## License and provenance

This project is licensed under Apache-2.0. See [LICENSE](LICENSE), [NOTICE](NOTICE), the
[provenance record](docs/PROVENANCE.md), and the [dependency
inventory](docs/DEPENDENCIES.tsv). Software licensing does not grant rights to third-
party datasets; check each dataset's source terms before redistributing it.
