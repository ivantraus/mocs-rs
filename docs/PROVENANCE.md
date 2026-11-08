# Provenance and licensing

## Source and license

MOCS-RS is a Rust port of [MOCS](https://github.com/ivantraus/mocs), based on
[source revision e4bae37](https://github.com/ivantraus/mocs/tree/e4bae37a2e8e895c143ee24fe3fa8961cb25ea08).
The copyright holder authorized this port under Apache-2.0. The upstream C++
code retains its GPL-3.0-or-later license; this grant does not change existing
upstream distributions. See [LICENSE](../LICENSE) and [NOTICE](../NOTICE).

The evolutionary algorithms, models, problems and operators derive from upstream
behavior, with differences described in [SEMANTICS.md](SEMANTICS.md).
Random sampling uses `rand`, `rand_chacha` and `rand_distr`; the upstream random
number implementation is not included. Configuration uses serde and TOML, and
the dense ARFF parser is implemented in Rust. Statistical distributions use
`statrs`; hypervolume and paired-statistics calculations have native implementations.

## Dependencies

[Cargo.lock](../Cargo.lock) pins resolved dependencies. Their licenses and source
repositories are listed in [DEPENDENCIES.tsv](DEPENDENCIES.tsv), including optional
and development dependencies. Dependencies retain their own terms; the project's
Apache license does not replace them. Update the inventory when changing the
locked dependency graph and retain applicable third-party notices in distributions.

## Data

Bundled ARFF examples and test fixtures are synthetic. Historical datasets and
research papers are not distributed with this port. Scripts that accept paths to
upstream datasets do not grant permission to redistribute those files.

For externally obtained data, retain its source, license, attribution, checksum
and split provenance separately from the software license. Original UCI sources
used by optional diagnostics and compatibility examples are listed below. Their CC BY 4.0 terms
require attribution; verify the terms of the specific copy being distributed.

| Dataset | Source attribution |
|---|---|
| [Zoo](https://archive.ics.uci.edu/dataset/111/zoo) | Forsyth, R. (1990), DOI 10.24432/C5R59V |
| [Iris](https://archive.ics.uci.edu/dataset/53/iris) | Fisher, R. (1936), DOI 10.24432/C56C76 |
| [Primary Tumor](https://archive.ics.uci.edu/dataset/83/primary%2Btumor) | Zwitter, M. and Soklic, M. (1987), DOI 10.24432/C5WK5Q |
| [Sonar](https://archive.ics.uci.edu/dataset/151/connectionist+bench+sonar+mines+vs+rocks) | Sejnowski, T. and Gorman, R. (1988), DOI 10.24432/C5T01Q |
| [Liver Disorders](https://archive.ics.uci.edu/dataset/60/liver+disorders) | Liver Disorders (2016), DOI 10.24432/C54G67; BUPA data donated by Richard S. Forsyth |

Current source licenses do not establish how historical ARFF copies were prepared
or partitioned. In particular, Bupa's seventh field is a train/test selector, not
a disease-presence label. Classification of that field must not be interpreted as
liver-disease prediction.

`scripts/fetch_zoo.py` verifies the upstream archive checksum, removes the animal
name identifier, and writes ARFF plus a separate attribution file. The output
preserves source row order and does not reproduce historical cross-validation folds.
