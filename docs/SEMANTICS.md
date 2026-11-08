# Behavior and upstream compatibility

MOCS-RS provides the computational features of [MOCS](https://github.com/ivantraus/mocs)
through a Rust library, TOML configuration and a native CLI. It does not implement
the upstream command-line, configuration-file, log-format or binary-plugin interfaces.
All optimization objectives are minimized.

## ARFF input

The reader follows upstream's lenient ARFF handling where it is unambiguous:

- Keywords and the `numeric`, `real` and `integer` types are case-insensitive.
  `integer` may carry bounds as `[lo,hi]`; they are a starting range, as upstream
  treated them, so values outside them are accepted. Ranges used by models are fitted
  from training rows only.
- Integer cells may be written as integral reals (`1.0`, `1e1`); non-integral values
  such as `1.5` are errors (upstream silently stored them as reals).
- Nominal values match exactly first, then case-insensitively, and are stored with
  the declared spelling. An attribute whose declared labels differ only in case
  matches exactly only.
- `{...}` lists may span lines and may follow the attribute name without a space.
  `%` starts a comment anywhere outside quotes.
- A quote opens a quoted token only at the start of a token, so `don't` is an
  ordinary value. A UTF-8 byte-order mark is ignored, and `\n`, `\r\n` and `\r` line
  endings are accepted.
- A training row whose predictors are all missing is accepted; it abstains for
  classification and receives the target-mean fallback for regression. Training fails
  only when the dataset has no observed nonconstant predictor at all.

Sparse rows, string and date attributes, and rows with the wrong value count remain
errors. Unquoted `?` is the only missing-value marker.

## Data, prediction and fitness

Numeric bounds and regression fallback means are fitted on training rows only.
Nominal domains come from the training ARFF header. Held-out scoring and prediction
reuse the saved schema; they never extend bounds or select a model using test labels.
Unknown categories and incompatible schemas are errors.

Rules are ordered and numeric intervals are finite and inclusive. Rule classification
uses the first matching rule. Out-of-range observations are not clipped and may
produce abstentions. Instance and Mixed models use range-normalized numeric distances,
0/1 nominal mismatches and deterministic rule-order ties. Wildcards match without a
nominal distance penalty. Missing or constant numeric predictors contribute no distance.
A row without usable predictor evidence abstains for classification or receives the
saved training-target mean for regression. Every training row needs a target value,
and the training data needs at least one observed, nonconstant predictor.

Classification loss counts abstentions as errors. Reports separately expose coverage,
conditional loss, abstentions, fallbacks, out-of-range values and rule usage. Regression
loss is mean squared error in squared target units. It uses distance to intervals,
not their centers, with stable negative-distance softmax weighting over the nearest
rules. `training.temperature` is an inverse temperature and `neighbours` is capped
by the available rule count.

Fitness modes use only their active objectives. `loss_size` retains raw loss and
normalized size without an oversized-model penalty. The classification-only
`archived_loss_size` instead uses selection objectives `[1, 1]` when the rule count
exceeds the saved training-row count; reported loss and size stay raw. This is a
selection penalty, not a hard genome-size limit. The penalty is not applied to
regression MSE, and `loss_generality_size` and `coverage_size` do not apply it either
(upstream set every objective to 1 for oversized models in both of those modes).

The final `compromise` choice picks the front member closest to the origin. Under
`archived_loss_size` the distance uses raw loss and size; under every other mode it uses
the active objective vector: `[loss + size]` for `penalized_loss`, `[loss, size]`,
`[loss, generality, size]` or `[conditional_loss, 1 - coverage, size]`. Upstream used
raw loss and size for penalized loss and added raw loss as a coordinate for the
coverage mode. `best_accuracy` uses raw training loss, then size, as upstream did.
Model size is the rule count divided by the training-row count, including in
held-out reports; upstream divided held-out size by the test-row count.

## Differences from the upstream implementation

| Area | Rust behavior |
|---|---|
| Gene setters | Store clamped values instead of retaining stale or uninitialized values. |
| Objectives and sorting | Validate finite values and dimensions; use Pareto dominance as a partial relation and deterministic tie-breaks for sorting. Equal vectors never compare as strictly less. |
| Best-accuracy selection | Compare training loss, size, active objectives and stable candidate ID from the final front. |
| Evaluation state | Keep cached evaluation separate from rank, strength, crowding and density. Mutation invalidates evaluation. |
| Pruning | Evaluate, prune using local usage statistics, then reevaluate changed genomes. Count both evaluations. Every unused rule is removed, even when only one rule is used; when no rule is used, the first rule is kept. Upstream pruned only when more than one rule but not every rule was used. |
| Regression rule usage | The `neighbours` nearest rules that contribute to a regression prediction count as used, so regression genomes are pruned and have a real generality. Upstream never recorded regression usage, so it never pruned them and their generality was always 1. |
| Training bounds | Fit from training data only, avoiding upstream use of held-out numeric ranges. |
| NSGA-II crowding | Normalize by each objective's front range and skip zero ranges. Boundary flags take precedence. `algorithm.raw_crowding=true` selects raw differences; constant objectives are still skipped and give no boundary flags, whereas upstream always flagged both ends. |
| NSGA-II duplicate pushing | `push_equal` keeps one member per identical objective vector in each front and `push_more_than_two_equal` keeps two; with both set, two are kept. The extra copies are only used after every front is exhausted. Upstream applied `pushEqual` first (keeping one) and moved extra copies into the next front, where they competed on crowding. |
| Selection and mating scores | Survivor-selection scores are computed on the survivors, and mating scores are recomputed on the reproductive pool, duplicate copies included. Fitness-proportionate mating therefore weights by the raw shared-fitness or SPEA2 fitness value plus 1e-12, rather than by upstream's ordinal rank, and never degenerates when all values are equal. |
| Tournament selection | Each tournament has one winner; upstream's `winners` selection parameter is not available. Modern tournament selection draws one parent per survivor (`population`); upstream and the archived paths draw one per offspring. |
| Tournament mating | `mating_winners=2` takes the best two of one tournament; `mating_winners=1` runs two independent tournaments. Upstream indexed past the winners, so its second parent was an unselected pool member. Values other than 1 or 2 are rejected; upstream's larger values read past the pool. |
| NGGA | All partner values are recomputed after each step (upstream refreshed only the pair it touched), `ngga_n1`, `ngga_n2` and the default `ngga_decay_sharing` derive from the survivor count rather than the offspring count, and mutation uses a plain `mutation_probability` draw instead of upstream's always-mutate-uncrossed-copies gate. |
| Classifier fitness | Remove constant/padded objective coordinates. Preserve penalized loss and conditional-loss/coverage/size as distinct modes. |
| Rule usage | Use fractional unused-rule statistics rather than integer division. |
| Nominal distance | Use categorical mismatch 0/1, independent of category-domain width. |
| Regression weighting | Use stable negative-distance softmax instead of division by a signed or zero similarity sum. |
| K-point crossover | Expose the upstream operation named UniformCrossover as `k_point`. Cuts are drawn within the shorter parent and the last segment runs to each parent's own end, so pruned genomes of different lengths cross safely and children keep their parents' lengths (an odd number of cuts swaps the tails). Upstream read past the shorter parent. |
| Variable-length crossover | Use corresponding cut offsets to preserve typed rule-block alignment; cuts may fall inside rules. `crossover_alignment=1` derives block width; an explicit width must match. |
| SBX | Sample independent beta values per locus by default; `sbx_shared_beta=true` selects one shared beta. Reflect deviations with absolute value, then apply the positive epsilon floor. Require equal-length real genomes: SBX is rejected before evolution starts for rule, instance and mixed models (integer class locus), for any genome with nominal loci, and with `prune_probability > 0`. Upstream cast integer genes to real ones, which was undefined behavior. |
| Elitism and `best` | MOGA, MOES, NSGA-II and SPEA2 share upstream's MOGA generation loop. With elitism the survivor pool is the old population followed by the offspring; without it only offspring compete, and `offspring` must be at least `population`. `best > 0` adds the old nondominated members and then the `best` old members by first objective to the survivor pool, keeping the configured survivor count; with `archived_add_best` it appends them after truncation instead. Because an elitist pool already holds every old member, `best > 0` is rejected for elitist runs unless `archived_add_best` is set. NGGA has its own loop upstream, so `elitism`, `best` and `archived_add_best` are rejected for it. A single-member mating pool pairs the member with itself, as upstream did. |
| Integer normal/polynomial mutation | Preserve integer resampling and deviation gates; real-valued mutation uses range-scaled displacement. |
| Adaptive mutation | Clamp the self-adapted rate to `[adaptive_epsilon, 1]` after each update. Upstream floored it at epsilon before the update and had no upper bound, so its rate could exceed 1 and drift back. |
| SPEA2 neighborhoods | Count self at zero distance: `spea2_k=1` gives density 0.5. Validate k against the smallest scoring pool. |
| OKK force scan | Compute the 1,000-point grid directly from 0 through 99.9 rather than accumulating `z += 0.1`. This changes coordinate rounding. Grip constraints still use 100. Cosine arguments within 1e-12 of ±1 are accepted, any NaN force sample is penalized, and `problem.min`/`problem.max` are ignored because OKK has fixed variable bounds. |
| KUR and ZDT6 | Follow upstream: KUR uses `5 sin(x)^3` (the Kursawe/Van Veldhuizen form; Deb's NSGA-II paper uses `sin(x^3)`), and ZDT6 uses `g = 1 + (n - 1) (sum / (n - 1))^0.25`, which equals the literature's factor of 9 only at the default `n = 10`. |
| Statistics | Use Student-t tails and signed ranks of absolute differences, with explicit zero/tie handling and Holm correction. Wilcoxon ties use exact floating-point equality of absolute differences. `analyze` summaries report the sample standard deviation; upstream `avg.pl` used the population form. |
| Front metrics | Keep historical S/spacing separate from exact 2D/3D hypervolume with a declared reference point. Historical S reproduces upstream exactly: unique nondominated points sorted by the first objective, the reference prepended, then the sum of `(a0 - b0)(a1 - b1)` over consecutive pairs. It is not a hypervolume and can be negative. Historical spacing is the population standard deviation of adjacent gaps. |
| Randomness | Use explicit seeded ChaCha8 streams, including a valid seed zero, rather than the upstream generator and global state. |

## Default values

Some defaults differ from upstream. Upstream configurations that relied on these
defaults must set the values explicitly to get the upstream setting.

| Upstream parameter | Upstream default | Rust field | Rust default |
|---|---|---|---|
| `crossProb` | 0.5 | `operators.crossover_probability` | 0.8 |
| `crossPoints` | 2 | `operators.crossover_points` | 1 |
| `etac` | 2 | `operators.sbx_eta` | 15 |
| `etam` | 15 | `operators.mutation_eta` | 20 |
| `pruneProb` | 0.1 | `training.prune_probability` | 0 |
| `best` | population / 4 | `algorithm.best` | 0 |
| `elitism` | false | `algorithm.elitism` | true for `nsga2` and `spea2`, false for `moga` and `moes` |

Other shared parameters keep their upstream defaults, including `mutProb` (1),
`sharing` (0.1), `initRules` (15), `initGenProb` (0.75), `genProb` (0.5), `kNN` (2),
`neighbours` (10), `epsilon` (1e-10), the NGGA weights and decays, and 100 for the
population, offspring and generation counts. `algorithm.kind` defaults to `nsga2`.

## Upstream features not provided

- Separate training and testing models (upstream `model=` and `test=`, for example
  training a rule model and testing it as a mixed model). Rust uses one
  `training.model` for both.
- Per-generation output: Rust records only the final front and counts, not the
  per-generation progress lines or metric curves upstream printed and plotted.
- Final-front genomes and held-out accuracy for every front member; Rust saves and
  scores only the selected model.
- LaTeX result tables (`results.pl`) and gnuplot scripts; `analyze` writes CSV.
- The upstream configuration, log and plugin formats (see the top of this page).

## Archived upstream controls

These options default to false. The configs in
[`examples/compatibility/archived/`](../examples/compatibility/archived/) enable
selected upstream transitions. The sibling `examples/compatibility/modern/` configs retain
the modern selection and mutation defaults.

| Configuration | Behavior |
|---|---|
| `algorithm.archived_add_best=true` | For MOGA, MOES, NSGA-II and SPEA2 (requires `best > 0` for the last two), retain scalar selection ranks across generations. After survivor truncation, append one copy of each unique old nondominated objective vector, then the `best` old individuals ordered by the first objective. A front point may also appear in the best prefix. The population can exceed its configured size. Appended copies receive rank zero and fresh IDs. |
| `algorithm.archived_moes_tournament_preselection=true` | For MOES, tournament-sample an offspring-sized reproductive pool before mating. With `best > 0`, `archived_add_best` must also be enabled. |
| `operators.archived_evaluated_copy_uniform_mutation_gate=true` | For Uniform mutation under MOGA/MOES, build the offspring batch before mutation. Always invoke mutation on copied evaluated candidates; invoke it on crossed candidates with `mutation_probability`. Crossover and invoked mutation invalidate evaluation even if no locus changes. |

These flags affect selection, population size, evaluation counts and pruning-gate
schedules, not just tuning parameters. Archived tournament ties retain the first
sampled candidate. With `fitness_tournament=false`, compare Pareto dominance first
and stored rank second.

Upstream applied the evaluated-copy mutation gate more widely than this option does:
it lived in the shared mutation driver, so Normal, Polynomial and Adaptive mutation
used it too, and NSGA-II, SPEA2 and NGGA reached it as well. Rust offers it only for
Uniform mutation under MOGA/MOES, so upstream configs such as `moes.cfg` and
`ngga.cfg` (Normal mutation) are not reproduced.

They do not reproduce exact C++ trajectories. Upstream RANMAR and Rust ChaCha8
produce different streams; cut-point and normal draw order, sorting ties and
training bounds differ. The upstream real-valued Uniform mutation test uses
`<= deviation`, while Rust uses `< deviation`; integer mutation uses `<` in both.

## Regression variants

`regression_rule` evolves constant consequences. It can memorize sparse training
observations without learning useful interpolation; a good training MSE does not
establish held-out quality.

The experimental `regression_linear_rule` evolves affine consequences for exactly
one numeric/integer predictor. With saved training bounds,
`z = (x - xmin) / (xmax - xmin)` and each consequence is
`ymin + (ymax - ymin) * (a + b*z)`. Coefficient domains are `a ∈ [-1, 2]` and
`b ∈ [-4, 4]`. Both initialize uniformly; there is no least-squares initialization
or prediction clipping. The training predictor must be observed and nonconstant.
Missing prediction inputs use the training-target mean; constant targets retain
their exact value. This variant is an explicit modeling choice, not a replacement
for constant-consequence regression or a general accuracy guarantee.

## Reproducibility and versions

Initial nominal wildcards use `training.initial_generality`, as do initial full-range
numeric intervals. `training.wildcard_probability` controls the mutation sampling
band; zero disables nominal wildcards by removing their genotype sentinel.
Rule, Mixed and RegressionRule training records use learning semantics `mocs-rs-3`.
Numeric optimization and Instance training use version 1; the affine variant uses
version 2. Saved predictor semantics are 1 for the original models and 2 for the
affine variant. Model loading validates the named model's expected semantics.

Each run owns one RNG stream. After batch variation, draw one pruning gate per dirty
candidate in candidate-ID order before evaluating. Reporting consumes no draws.
NGGA keeps its dependent offspring transitions sequential. Evaluation results,
commits, pruning and post-pruning reevaluation have deterministic ordering; parallel
workers do not perform unordered floating-point reductions.

For identical inputs, configuration, executable and host/toolchain, serial and
parallel execution preserve scientific results. Build metadata and timings differ.
Different platforms or sampling-library versions may change floating-point results
or random draws; exact C++ or cross-platform trajectory equality is not promised.
