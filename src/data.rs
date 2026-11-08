//! Dense ARFF parsing and immutable fitted dataset schemas.

use crate::Result;
use serde::{Deserialize, Serialize};
use std::{fs, path::Path};

/// Declared ARFF attribute type.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
pub enum AttributeKind {
    /// `numeric` or `real`.
    Numeric,
    /// `integer`.
    Integer,
    /// Nominal attribute with its declared labels, in declaration order.
    Nominal(Vec<String>),
}

/// One ARFF attribute and the statistics fitted from training rows.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Attribute {
    /// Attribute name as declared.
    pub name: String,
    /// Declared type.
    pub kind: AttributeKind,
    /// Optional `[lo,hi]` bounds declared on an `integer` attribute. Values outside
    /// them are accepted, but prediction data must declare the same bounds.
    pub declared_bounds: Option<(i64, i64)>,
    /// `(min, max)` of the observed training values, for numeric/integer attributes
    /// with at least one observed value.
    pub fitted_range: Option<(f64, f64)>,
    /// Mean of the observed training values, for numeric/integer attributes.
    pub fitted_mean: Option<f64>,
    /// Whether any training row has a non-missing value for this attribute.
    pub observed_in_training: bool,
}
impl Attribute {
    /// Whether the attribute is `numeric`/`real` or `integer`.
    pub fn is_numeric(&self) -> bool {
        matches!(self.kind, AttributeKind::Numeric | AttributeKind::Integer)
    }
    /// Declared labels of a nominal attribute.
    pub fn nominal_values(&self) -> Option<&[String]> {
        if let AttributeKind::Nominal(v) = &self.kind {
            Some(v)
        } else {
            None
        }
    }
    /// Fitted training range; same as [`Attribute::fitted_range`].
    pub fn numeric_range(&self) -> Option<(f64, f64)> {
        self.fitted_range
    }
}

/// Attributes, target choice and fitted statistics of a training dataset. Saved
/// models store it so prediction data is interpreted exactly as in training.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Schema {
    /// ARFF `@relation` name, if any.
    pub relation: Option<String>,
    /// All attributes in column order, including the target.
    pub attributes: Vec<Attribute>,
    /// Column index of the target attribute.
    pub target_index: usize,
    /// Number of rows the statistics were fitted on (also the denominator of model size).
    pub training_rows: usize,
}
impl Schema {
    /// The target attribute.
    pub fn target(&self) -> &Attribute {
        &self.attributes[self.target_index]
    }
    /// `(column index, attribute)` for every non-target attribute.
    pub fn predictors(&self) -> impl Iterator<Item = (usize, &Attribute)> {
        self.attributes
            .iter()
            .enumerate()
            .filter(move |(i, _)| *i != self.target_index)
    }
    /// Whether the target is nominal.
    pub fn is_classification(&self) -> bool {
        matches!(self.target().kind, AttributeKind::Nominal(_))
    }
    /// Number of predictor attributes.
    pub fn feature_count(&self) -> usize {
        self.attributes.len().saturating_sub(1)
    }
    /// Check internal consistency: unique nonempty names, a valid target index,
    /// positive `training_rows`, valid bounds and labels, and fitted statistics that
    /// match each attribute's kind and observed status.
    pub fn validate(&self) -> Result<()> {
        anyhow::ensure!(!self.attributes.is_empty(), "schema has no attributes");
        anyhow::ensure!(
            self.target_index < self.attributes.len(),
            "schema target index is out of range"
        );
        anyhow::ensure!(self.training_rows > 0, "schema has no training rows");
        let mut names = std::collections::BTreeSet::new();
        for (i, a) in self.attributes.iter().enumerate() {
            anyhow::ensure!(
                !a.name.is_empty() && names.insert(a.name.clone()),
                "schema attribute {i} has an empty or duplicate name"
            );
            if let Some((lo, hi)) = a.declared_bounds {
                anyhow::ensure!(lo <= hi, "schema attribute {i} has invalid integer bounds");
            }
            match a.kind {
                AttributeKind::Numeric | AttributeKind::Integer => {
                    if let Some((lo, hi)) = a.fitted_range {
                        anyhow::ensure!(
                            lo.is_finite() && hi.is_finite() && lo <= hi,
                            "schema attribute {i} has invalid fitted range"
                        );
                    }
                    if let Some(mean) = a.fitted_mean {
                        anyhow::ensure!(
                            mean.is_finite(),
                            "schema attribute {i} has nonfinite fitted mean"
                        );
                    }
                    anyhow::ensure!(
                        a.observed_in_training == a.fitted_range.is_some()
                            && a.observed_in_training == a.fitted_mean.is_some(),
                        "schema attribute {i} has inconsistent fitted numeric metadata"
                    );
                    if i == self.target_index {
                        anyhow::ensure!(
                            a.fitted_range.is_some() && a.fitted_mean.is_some(),
                            "numeric target has no fitted training values"
                        );
                    }
                }
                AttributeKind::Nominal(ref values) => {
                    anyhow::ensure!(
                        !values.is_empty(),
                        "schema nominal attribute {i} has no labels"
                    );
                    let unique = values.iter().collect::<std::collections::BTreeSet<_>>();
                    anyhow::ensure!(
                        unique.len() == values.len(),
                        "schema nominal attribute {i} has duplicate labels"
                    );
                    anyhow::ensure!(
                        a.fitted_range.is_none() && a.fitted_mean.is_none(),
                        "schema nominal attribute {i} has numeric fit metadata"
                    );
                    if i == self.target_index {
                        anyhow::ensure!(
                            a.observed_in_training,
                            "nominal target has no observed training values"
                        );
                    }
                }
            }
        }
        Ok(())
    }
}

/// A non-missing cell value.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
pub enum Value {
    /// Value of a `numeric`/`real` attribute.
    Numeric(f64),
    /// Value of an `integer` attribute.
    Integer(i64),
    /// Nominal value, stored with the declared label spelling.
    Nominal(String),
}
impl Value {
    /// Numeric value of a numeric or integer cell.
    pub fn numeric(&self) -> Option<f64> {
        match self {
            Self::Numeric(v) => Some(*v),
            Self::Integer(v) => Some(*v as f64),
            _ => None,
        }
    }
    /// Value of an integer cell.
    pub fn integer(&self) -> Option<i64> {
        if let Self::Integer(v) = self {
            Some(*v)
        } else {
            None
        }
    }
    /// Label of a nominal cell.
    pub fn nominal(&self) -> Option<&str> {
        if let Self::Nominal(v) = self {
            Some(v)
        } else {
            None
        }
    }
}

/// One data row.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Row {
    /// Zero-based position of the row in the source data section (kept in subsets).
    pub source_index: usize,
    /// One entry per schema attribute; `None` is a missing (`?`) value or an omitted target.
    pub values: Vec<Option<Value>>,
}

/// Parsed ARFF rows with their fitted [`Schema`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Dataset {
    schema: Schema,
    rows: Vec<Row>,
}

struct ParsedArff {
    relation: Option<String>,
    attributes: Vec<Attribute>,
    rows: Vec<Vec<ArffCell>>,
    target: Option<String>,
}
struct ArffCell {
    value: String,
    quoted: bool,
}
impl Dataset {
    /// Read and parse an ARFF file as training data. See [`Dataset::parse_str_with_target`].
    pub fn load(path: impl AsRef<Path>, target: Option<&str>) -> Result<Self> {
        let p = path.as_ref();
        let input =
            fs::read_to_string(p).map_err(|e| anyhow::anyhow!("read ARFF {}: {e}", p.display()))?;
        Self::parse_str_with_target(&input, target)
    }
    /// Parse ARFF training data with the default target choice.
    pub fn parse_str(input: &str) -> Result<Self> {
        Self::parse_str_with_target(input, None)
    }
    /// Parse dense ARFF training data and fit ranges and means from its rows.
    ///
    /// The target is the attribute named by `target`, else one named by an `@target`
    /// header line, else the last attribute. Errors on malformed ARFF, an empty data
    /// section, duplicate or unknown attribute names, a row with a missing target, or
    /// data with no observed nonconstant predictor.
    pub fn parse_str_with_target(input: &str, target: Option<&str>) -> Result<Self> {
        let ParsedArff {
            relation,
            mut attributes,
            rows: raw,
            target: explicit_target,
        } = parse_arff(input)?;
        anyhow::ensure!(!attributes.is_empty(), "ARFF header has no attributes");
        anyhow::ensure!(!raw.is_empty(), "ARFF data section has no rows");
        let mut names = std::collections::BTreeSet::new();
        for attr in &attributes {
            anyhow::ensure!(
                names.insert(attr.name.clone()),
                "duplicate attribute name {:?}",
                attr.name
            );
        }
        let target_index = if let Some(name) = target {
            attributes
                .iter()
                .position(|a| a.name == name)
                .ok_or_else(|| anyhow::anyhow!("target attribute {name:?} not found"))?
        } else if let Some(name) = explicit_target {
            attributes
                .iter()
                .position(|a| a.name == name)
                .ok_or_else(|| anyhow::anyhow!("target attribute {name:?} not found"))?
        } else {
            attributes.len() - 1
        };
        let rows = materialize_rows(&attributes, raw)?;
        fit_ranges(&mut attributes, &rows)?;
        let schema = Schema {
            relation,
            attributes,
            target_index,
            training_rows: rows.len(),
        };
        schema.validate()?;
        validate_rows(&schema, &rows, true)?;
        Ok(Self { schema, rows })
    }
    /// Parse scored or prediction data against the fitted training schema. The target can be omitted or missing.
    pub fn parse_str_for_prediction(input: &str, schema: &Schema) -> Result<Self> {
        schema.validate()?;
        let ParsedArff {
            relation,
            attributes: attrs,
            rows: raw_rows,
            ..
        } = parse_arff(input)?;
        anyhow::ensure!(!raw_rows.is_empty(), "ARFF data section has no rows");
        let mut mapping = vec![None; schema.attributes.len()];
        let mut source_names = std::collections::BTreeSet::new();
        for (src_i, src) in attrs.iter().enumerate() {
            anyhow::ensure!(
                source_names.insert(src.name.clone()),
                "duplicate attribute name {:?} in prediction data",
                src.name
            );
            if let Some(dst_i) = schema.attributes.iter().position(|a| a.name == src.name) {
                anyhow::ensure!(
                    same_declared_kind(src, &schema.attributes[dst_i]),
                    "attribute {:?} does not match fitted schema",
                    src.name
                );
                mapping[dst_i] = Some(src_i);
            } else {
                anyhow::bail!("unknown attribute {:?} in prediction data", src.name);
            }
        }
        for (i, attr) in schema.attributes.iter().enumerate() {
            anyhow::ensure!(
                mapping[i].is_some() || i == schema.target_index,
                "prediction data is missing predictor {:?}",
                attr.name
            );
        }
        let mut rows = Vec::with_capacity(raw_rows.len());
        for (ri, raw) in raw_rows.into_iter().enumerate() {
            anyhow::ensure!(
                raw.len() == attrs.len(),
                "row {} has {} values; expected {}",
                ri + 1,
                raw.len(),
                attrs.len()
            );
            let mut values = vec![None; schema.attributes.len()];
            for dst in 0..schema.attributes.len() {
                if let Some(src) = mapping[dst] {
                    values[dst] = parse_value(&raw[src], &schema.attributes[dst], ri + 1)?;
                }
            }
            rows.push(Row {
                source_index: ri,
                values,
            });
        }
        let result = Self {
            schema: Schema {
                relation,
                ..schema.clone()
            },
            rows,
        };
        validate_rows(&result.schema, &result.rows, false)?;
        Ok(result)
    }
    /// The fitted schema.
    pub fn schema(&self) -> &Schema {
        &self.schema
    }
    /// All rows.
    pub fn rows(&self) -> &[Row] {
        &self.rows
    }
    /// Column index of the target.
    pub fn target_index(&self) -> usize {
        self.schema.target_index
    }
    /// The target attribute.
    pub fn target_attribute(&self) -> &Attribute {
        self.schema.target()
    }
    /// Column indices of the predictors.
    pub fn predictor_indices(&self) -> Vec<usize> {
        self.schema.predictors().map(|(i, _)| i).collect()
    }
    /// Whether the target is nominal.
    pub fn is_classification(&self) -> bool {
        self.schema.is_classification()
    }
    /// Number of rows.
    pub fn len(&self) -> usize {
        self.rows.len()
    }
    /// Whether there are no rows.
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }
    /// Build a training subset and refit ranges using only the selected rows.
    pub fn subset_rows(&self, indices: &[usize]) -> Result<Self> {
        anyhow::ensure!(
            !indices.is_empty(),
            "training subset must contain at least one row"
        );
        let rows = indices
            .iter()
            .map(|&i| {
                self.rows
                    .get(i)
                    .cloned()
                    .ok_or_else(|| anyhow::anyhow!("row index {i} is out of range"))
            })
            .collect::<Result<Vec<_>>>()?;
        let mut schema = self.schema.clone();
        fit_ranges(&mut schema.attributes, &rows)?;
        schema.training_rows = rows.len();
        validate_rows(&schema, &rows, true)?;
        Ok(Self { schema, rows })
    }
    /// Select rows while retaining this dataset's fitted ranges, for held-out scoring.
    pub fn subset_rows_for_prediction(&self, indices: &[usize]) -> Result<Self> {
        anyhow::ensure!(
            !indices.is_empty(),
            "scoring subset must contain at least one row"
        );
        let rows = indices
            .iter()
            .map(|&i| {
                self.rows
                    .get(i)
                    .cloned()
                    .ok_or_else(|| anyhow::anyhow!("row index {i} is out of range"))
            })
            .collect::<Result<Vec<_>>>()?;
        let result = Self {
            schema: self.schema.clone(),
            rows,
        };
        validate_rows(&result.schema, &result.rows, false)?;
        Ok(result)
    }
    /// Clone the rows at `indices` (positions in this dataset). Errors on an
    /// out-of-range index.
    pub fn rows_with_source_indices(&self, indices: &[usize]) -> Result<Vec<Row>> {
        indices
            .iter()
            .map(|&i| {
                self.rows
                    .get(i)
                    .cloned()
                    .ok_or_else(|| anyhow::anyhow!("row index {i} is out of range"))
            })
            .collect()
    }
    /// Reuse this dataset's rows under a previously fitted schema after checking column kinds.
    pub fn with_fitted_schema(&self, schema: &Schema) -> Result<Self> {
        schema.validate()?;
        anyhow::ensure!(
            self.schema.attributes.len() == schema.attributes.len(),
            "dataset attribute count does not match fitted schema"
        );
        for (i, (src, dst)) in self
            .schema
            .attributes
            .iter()
            .zip(&schema.attributes)
            .enumerate()
        {
            anyhow::ensure!(
                src.name == dst.name && same_declared_kind(src, dst),
                "attribute {i} does not match fitted schema"
            );
        }
        let result = Self {
            schema: schema.clone(),
            rows: self.rows.clone(),
        };
        validate_rows(&result.schema, &result.rows, false)?;
        Ok(result)
    }
    pub(crate) fn schema_only(schema: Schema) -> Self {
        Self {
            schema,
            rows: Vec::new(),
        }
    }
}

fn materialize_rows(attrs: &[Attribute], raw: Vec<Vec<ArffCell>>) -> Result<Vec<Row>> {
    raw.into_iter()
        .enumerate()
        .map(|(ri, fields)| {
            anyhow::ensure!(
                fields.len() == attrs.len(),
                "row {} has {} values; expected {}",
                ri + 1,
                fields.len(),
                attrs.len()
            );
            let values = fields
                .iter()
                .zip(attrs)
                .map(|(v, a)| parse_value(v, a, ri + 1))
                .collect::<Result<Vec<_>>>()?;
            Ok(Row {
                source_index: ri,
                values,
            })
        })
        .collect()
}
fn same_declared_kind(a: &Attribute, b: &Attribute) -> bool {
    match (&a.kind, &b.kind) {
        (AttributeKind::Numeric, AttributeKind::Numeric) => true,
        (AttributeKind::Integer, AttributeKind::Integer) => a.declared_bounds == b.declared_bounds,
        (AttributeKind::Nominal(x), AttributeKind::Nominal(y)) => x == y,
        _ => false,
    }
}
fn fit_ranges(attributes: &mut [Attribute], rows: &[Row]) -> Result<()> {
    for (i, attr) in attributes.iter_mut().enumerate() {
        if !attr.is_numeric() {
            attr.fitted_range = None;
            attr.fitted_mean = None;
            attr.observed_in_training = rows
                .iter()
                .any(|row| row.values.get(i).is_some_and(Option::is_some));
            continue;
        }
        let mut low = f64::INFINITY;
        let mut high = f64::NEG_INFINITY;
        let mut mean = 0.0;
        let mut count = 0usize;
        for row in rows {
            if let Some(Some(value)) = row.values.get(i)
                && let Some(v) = value.numeric()
            {
                low = low.min(v);
                high = high.max(v);
                count += 1;
                mean = if count == 1 {
                    v
                } else {
                    mean * ((count - 1) as f64 / count as f64) + v / count as f64
                };
            }
        }
        attr.fitted_range = low.is_finite().then_some((low, high));
        if let Some((lo, hi)) = attr.fitted_range {
            anyhow::ensure!(
                (hi - lo).is_finite(),
                "attribute {:?} has a training range too wide for finite distance normalization",
                attr.name
            );
        }
        attr.fitted_mean = (count > 0).then_some(mean);
        if let Some(value) = attr.fitted_mean {
            anyhow::ensure!(
                value.is_finite(),
                "attribute {:?} has a nonfinite training mean",
                attr.name
            );
        }
        attr.observed_in_training = count > 0;
    }
    Ok(())
}
fn validate_rows(schema: &Schema, rows: &[Row], require_target: bool) -> Result<()> {
    for row in rows {
        anyhow::ensure!(
            row.values.len() == schema.attributes.len(),
            "row {} has wrong attribute count",
            row.source_index + 1
        );
        if require_target {
            anyhow::ensure!(
                row.values[schema.target_index].is_some(),
                "row {} has a missing target",
                row.source_index + 1
            );
        }
    }
    if require_target {
        // Individual training rows may lack every predictor: they can never be
        // matched, exactly as at prediction time. The data as a whole must still
        // provide at least one usable predictor.
        let usable = schema.predictors().any(|(_, a)| {
            a.observed_in_training
                && (!a.is_numeric() || a.fitted_range.is_some_and(|(lo, hi)| lo < hi))
        });
        anyhow::ensure!(
            usable,
            "training data has no observed nonconstant predictor"
        );
    }
    Ok(())
}
/// Split into physical lines, accepting `\n`, `\r\n` and lone `\r` terminators.
fn arff_lines(input: &str) -> Vec<&str> {
    let bytes = input.as_bytes();
    let mut lines = Vec::new();
    let (mut start, mut i) = (0, 0);
    while i < bytes.len() {
        match bytes[i] {
            b'\n' => {
                lines.push(&input[start..i]);
                i += 1;
                start = i;
            }
            b'\r' => {
                lines.push(&input[start..i]);
                i += if bytes.get(i + 1) == Some(&b'\n') {
                    2
                } else {
                    1
                };
                start = i;
            }
            _ => i += 1,
        }
    }
    if start < bytes.len() {
        lines.push(&input[start..]);
    }
    lines
}
#[derive(Default)]
struct Header {
    relation: Option<String>,
    target: Option<String>,
    attributes: Vec<Attribute>,
}
impl Header {
    /// Apply one complete header statement; returns true at `@data`.
    fn statement(&mut self, line: &str) -> Result<bool> {
        if starts_directive(line, "@relation") {
            self.relation = Some(unquote(line["@relation".len()..].trim())?);
        } else if starts_directive(line, "@target") {
            self.target = Some(unquote(line["@target".len()..].trim())?);
        } else if starts_directive(line, "@attribute") {
            let (name, ty) = split_name_and_rest(line["@attribute".len()..].trim())?;
            self.attributes.push(parse_attribute(name, ty)?);
        } else if starts_directive(line, "@data") {
            return Ok(true);
        } else {
            anyhow::bail!("unsupported or malformed ARFF directive");
        }
        Ok(false)
    }
}
/// Net count of unquoted `{` over `}`.
fn open_braces(s: &str) -> Result<i64> {
    let mut depth = 0i64;
    scan_unquoted(s, |_, c| {
        match c {
            '{' => depth += 1,
            '}' => depth -= 1,
            _ => {}
        }
        false
    })?;
    Ok(depth)
}
fn parse_arff(input: &str) -> Result<ParsedArff> {
    let input = input.strip_prefix('\u{feff}').unwrap_or(input);
    let mut header = Header::default();
    let mut rows = Vec::new();
    let mut in_data = false;
    // An `@attribute` statement whose nominal `{...}` list continues on later lines.
    let mut pending: Option<(usize, String)> = None;
    for (index, original) in arff_lines(input).into_iter().enumerate() {
        let line_no = index + 1;
        let at = |e: anyhow::Error| anyhow::anyhow!("line {line_no}: {e}");
        let line = strip_comment(original).map_err(at)?.trim();
        if let Some((start, mut text)) = pending.take() {
            anyhow::ensure!(
                !["@relation", "@attribute", "@data", "@target"]
                    .iter()
                    .any(|d| starts_directive(line, d)),
                "line {start}: unterminated nominal value list"
            );
            text.push(' ');
            text.push_str(line);
            if open_braces(&text).map_err(at)? > 0 {
                pending = Some((start, text));
            } else {
                header
                    .statement(&text)
                    .map_err(|e| anyhow::anyhow!("line {start}: {e}"))?;
            }
            continue;
        }
        if line.is_empty() {
            continue;
        }
        if !in_data {
            if starts_directive(line, "@attribute") && open_braces(line).map_err(at)? > 0 {
                pending = Some((line_no, line.to_string()));
                continue;
            }
            in_data = header.statement(line).map_err(at)?;
        } else {
            anyhow::ensure!(
                !line.starts_with('{'),
                "line {line_no}: sparse ARFF rows are unsupported"
            );
            rows.push(
                split_unquoted(line, ',')
                    .map_err(at)?
                    .into_iter()
                    .map(parse_cell)
                    .collect::<Result<Vec<_>>>()
                    .map_err(at)?,
            );
        }
    }
    if let Some((start, _)) = pending {
        anyhow::bail!("line {start}: unterminated nominal value list");
    }
    anyhow::ensure!(in_data, "ARFF is missing @data");
    Ok(ParsedArff {
        relation: header.relation,
        attributes: header.attributes,
        rows,
        target: header.target,
    })
}
fn starts_directive(line: &str, d: &str) -> bool {
    line.get(..d.len())
        .is_some_and(|s| s.eq_ignore_ascii_case(d))
        && (line.len() == d.len()
            || line
                .as_bytes()
                .get(d.len())
                .is_some_and(u8::is_ascii_whitespace))
}
/// Visit characters outside quoted tokens until `visit` returns true, returning
/// that offset. A `'` or `"` opens a quoted token only at the start of a token
/// (line start, or after whitespace, `,` or `{`), so an apostrophe inside a bare
/// word such as `don't` is literal. Backslash escapes the next quoted character.
fn scan_unquoted(s: &str, mut visit: impl FnMut(usize, char) -> bool) -> Result<Option<usize>> {
    let mut quote = None;
    let mut escaped = false;
    let mut prev: Option<char> = None;
    for (i, c) in s.char_indices() {
        let before = prev.replace(c);
        if let Some(q) = quote {
            if escaped {
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == q {
                quote = None;
            }
            continue;
        }
        if (c == '\'' || c == '"')
            && before.is_none_or(|p| p.is_whitespace() || p == ',' || p == '{')
        {
            quote = Some(c);
            continue;
        }
        if visit(i, c) {
            return Ok(Some(i));
        }
    }
    anyhow::ensure!(quote.is_none(), "unterminated quoted token");
    Ok(None)
}
fn strip_comment(s: &str) -> Result<&str> {
    Ok(match scan_unquoted(s, |_, c| c == '%')? {
        Some(i) => &s[..i],
        None => s,
    })
}
fn split_unquoted(s: &str, sep: char) -> Result<Vec<&str>> {
    let mut parts = Vec::new();
    let mut start = 0;
    scan_unquoted(s, |i, c| {
        if c == sep {
            parts.push(&s[start..i]);
            start = i + c.len_utf8();
        }
        false
    })?;
    parts.push(&s[start..]);
    Ok(parts)
}
fn split_name_and_rest(s: &str) -> Result<(String, &str)> {
    let s = s.trim_start();
    anyhow::ensure!(!s.is_empty(), "attribute name is missing");
    if let Some(q @ ('\'' | '"')) = s.chars().next() {
        let end = quoted_end(s, q)?;
        Ok((unquote(&s[..=end])?, s[end + 1..].trim_start()))
    } else {
        let end = s
            .find(|c: char| c.is_whitespace() || c == '{')
            .unwrap_or(s.len());
        Ok((s[..end].to_string(), s[end..].trim_start()))
    }
}
/// Strip a case-insensitive type keyword that is not followed by a word character.
fn strip_keyword<'a>(t: &'a str, keyword: &str) -> Option<&'a str> {
    let head = t.get(..keyword.len())?;
    let rest = &t[keyword.len()..];
    (head.eq_ignore_ascii_case(keyword)
        && !rest.starts_with(|c: char| c.is_alphanumeric() || c == '_'))
    .then(|| rest.trim())
}
/// Parse an integer, also accepting integral values written as reals (`1.0`, `1e2`).
fn parse_integral(s: &str) -> Option<i64> {
    if let Ok(v) = s.parse::<i64>() {
        return Some(v);
    }
    let v: f64 = s.parse().ok()?;
    const LIMIT: f64 = 9_223_372_036_854_775_808.0; // 2^63
    (v.is_finite() && v.fract() == 0.0 && (-LIMIT..LIMIT).contains(&v)).then_some(v as i64)
}
fn parse_attribute(name: String, ty: &str) -> Result<Attribute> {
    anyhow::ensure!(!name.is_empty(), "attribute name is empty");
    let t = ty.trim();
    let (kind, declared_bounds) =
        if let Some(rest) = strip_keyword(t, "numeric").or_else(|| strip_keyword(t, "real")) {
            anyhow::ensure!(rest.is_empty(), "unsupported attribute type {ty:?}");
            (AttributeKind::Numeric, None)
        } else if let Some(rest) = strip_keyword(t, "integer") {
            if rest.is_empty() {
                (AttributeKind::Integer, None)
            } else {
                let bounds = rest
                    .strip_prefix('[')
                    .and_then(|r| r.strip_suffix(']'))
                    .ok_or_else(|| anyhow::anyhow!("invalid integer bounds {rest:?}"))?;
                let parts = bounds.split(',').map(str::trim).collect::<Vec<_>>();
                let parsed = match parts[..] {
                    [lo, hi] => parse_integral(lo).zip(parse_integral(hi)),
                    _ => None,
                };
                let (lo, hi) = parsed
                    .filter(|(lo, hi)| lo <= hi)
                    .ok_or_else(|| anyhow::anyhow!("invalid integer bounds {bounds:?}"))?;
                (AttributeKind::Integer, Some((lo, hi)))
            }
        } else if let Some(inner) = t.strip_prefix('{').and_then(|r| r.strip_suffix('}')) {
            let vals = split_unquoted(inner, ',')?
                .into_iter()
                .map(|v| {
                    let v = v.trim();
                    anyhow::ensure!(!v.is_empty(), "empty nominal label");
                    unquote(v)
                })
                .collect::<Result<Vec<_>>>()?;
            let mut seen = std::collections::BTreeSet::new();
            for v in &vals {
                anyhow::ensure!(seen.insert(v.clone()), "duplicate nominal label {v:?}");
            }
            (AttributeKind::Nominal(vals), None)
        } else {
            anyhow::bail!("unsupported attribute type {ty:?}");
        };
    Ok(Attribute {
        name,
        kind,
        declared_bounds,
        fitted_range: None,
        fitted_mean: None,
        observed_in_training: false,
    })
}
/// Resolve a nominal cell to its declared spelling. Exact matches win; otherwise
/// an ASCII case-insensitive match is accepted unless the declared labels of this
/// attribute themselves differ only in case, which keeps matching exact.
fn match_nominal<'a>(labels: &'a [String], s: &str, attr: &str, row: usize) -> Result<&'a str> {
    if let Some(label) = labels.iter().find(|v| *v == s) {
        return Ok(label);
    }
    let case_distinct = labels
        .iter()
        .enumerate()
        .all(|(i, a)| labels[i + 1..].iter().all(|b| !a.eq_ignore_ascii_case(b)));
    if case_distinct && let Some(label) = labels.iter().find(|v| v.eq_ignore_ascii_case(s)) {
        return Ok(label);
    }
    let note = if case_distinct {
        ""
    } else {
        " (matching is case-sensitive because declared labels differ only in case)"
    };
    anyhow::bail!("row {row}, attribute {attr:?}: unknown nominal label {s:?}{note}")
}
fn parse_value(raw: &ArffCell, attr: &Attribute, row: usize) -> Result<Option<Value>> {
    // Keep quoted contents verbatim. Quoted question marks are ordinary
    // values, while only the unquoted ARFF token `?` means missingness.
    let s = if raw.quoted {
        raw.value.as_str()
    } else {
        raw.value.trim()
    };
    if !raw.quoted && s == "?" {
        return Ok(None);
    }
    match &attr.kind {
        AttributeKind::Numeric => {
            let v: f64 = s.parse().map_err(|_| {
                anyhow::anyhow!("row {row}, attribute {:?}: invalid number {s:?}", attr.name)
            })?;
            anyhow::ensure!(
                v.is_finite(),
                "row {row}, attribute {:?}: nonfinite number",
                attr.name
            );
            Ok(Some(Value::Numeric(v)))
        }
        AttributeKind::Integer => {
            // Declared `[lo,hi]` bounds are an initial range, not a constraint:
            // as upstream, observed values may lie outside it.
            let v = parse_integral(s).ok_or_else(|| {
                if s.parse::<f64>().is_ok_and(f64::is_finite) {
                    anyhow::anyhow!(
                        "row {row}, attribute {:?}: non-integral value {s:?} for integer attribute",
                        attr.name
                    )
                } else {
                    anyhow::anyhow!(
                        "row {row}, attribute {:?}: expected integer, got {s:?}",
                        attr.name
                    )
                }
            })?;
            Ok(Some(Value::Integer(v)))
        }
        AttributeKind::Nominal(vals) => Ok(Some(Value::Nominal(
            match_nominal(vals, s, &attr.name, row)?.to_string(),
        ))),
    }
}
fn quoted_end(s: &str, q: char) -> Result<usize> {
    let mut escaped = false;
    for (i, c) in s.char_indices().skip(1) {
        if escaped {
            escaped = false;
            continue;
        }
        if c == '\\' {
            escaped = true;
            continue;
        }
        if c == q {
            return Ok(i);
        }
    }
    anyhow::bail!("unterminated quoted token")
}
fn unquote(s: &str) -> Result<String> {
    let s = s.trim();
    if let Some(q @ ('\'' | '"')) = s.chars().next() {
        let end = quoted_end(s, q)?;
        anyhow::ensure!(
            s[end + 1..].trim().is_empty(),
            "unexpected text after quoted token"
        );
        let mut out = String::new();
        let mut escaped = false;
        for c in s[1..end].chars() {
            if escaped {
                out.push(c);
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else {
                out.push(c);
            }
        }
        if escaped {
            out.push('\\');
        }
        Ok(out)
    } else {
        Ok(s.to_string())
    }
}
fn parse_cell(s: &str) -> Result<ArffCell> {
    let trimmed = s.trim();
    Ok(ArffCell {
        value: unquote(trimmed)?,
        quoted: matches!(trimmed.chars().next(), Some('\'' | '"')),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        config::{ModelKind, OperatorsConfig, TrainingConfig},
        evaluation::Problem,
        models::LearningProblem,
        rng::Random,
    };

    fn err(input: &str) -> String {
        format!("{:#}", Dataset::parse_str(input).unwrap_err())
    }

    #[test]
    fn integer_values_outside_declared_bounds_widen_the_fitted_range() {
        let header = "@relation r\n@attribute n integer [10,90]\n@attribute class {a,b}\n@data\n";
        let data = Dataset::parse_str(&format!("{header}100,a\n5,b\n50,a\n")).unwrap();
        let n = &data.schema().attributes[0];
        assert_eq!(n.declared_bounds, Some((10, 90)));
        assert_eq!(n.fitted_range, Some((5.0, 100.0)));
        assert_eq!(data.rows()[0].values[0], Some(Value::Integer(100)));

        let scored =
            Dataset::parse_str_for_prediction(&format!("{header}200,a\n50,b\n"), data.schema())
                .unwrap();
        assert_eq!(scored.rows()[0].values[0], Some(Value::Integer(200)));
        let problem =
            LearningProblem::new(data, TrainingConfig::default(), OperatorsConfig::default())
                .unwrap();
        let genome = problem.initialize(&mut Random::new(3)).unwrap();
        let predictions = problem.predict_dataset(&genome, &scored).unwrap();
        assert!(predictions[0].out_of_range);
        assert!(!predictions[1].out_of_range);
    }

    #[test]
    fn integral_reals_are_accepted_in_integer_columns() {
        let data = Dataset::parse_str(
            "@relation r\n@attribute n INTEGER[1.0, 5]\n@attribute m integer\n@attribute class {a,b}\n@data\n1.0,-2.0,a\n1e1,+3,b\n",
        )
        .unwrap();
        assert_eq!(data.schema().attributes[0].declared_bounds, Some((1, 5)));
        assert_eq!(data.rows()[0].values[0], Some(Value::Integer(1)));
        assert_eq!(data.rows()[0].values[1], Some(Value::Integer(-2)));
        assert_eq!(data.rows()[1].values[0], Some(Value::Integer(10)));
        assert_eq!(data.rows()[1].values[1], Some(Value::Integer(3)));

        let e =
            err("@relation r\n@attribute n integer\n@attribute class {a,b}\n@data\n1.5,a\n2,b\n");
        assert!(e.contains("non-integral value \"1.5\""), "{e}");
        let e = err("@relation r\n@attribute n integer\n@attribute class {a,b}\n@data\nx,a\n2,b\n");
        assert!(e.contains("expected integer"), "{e}");
        let e =
            err("@relation r\n@attribute n integer [1.5,3]\n@attribute class {a}\n@data\n1,a\n");
        assert!(e.contains("invalid integer bounds"), "{e}");
    }

    const ALL_MISSING: &str = "@relation r\n@attribute x numeric\n@attribute c {u,v}\n@attribute y TARGET\n@data\n0,u,L0\n?,?,L1\n1,v,L0\n2,u,L1\n";

    #[test]
    fn training_rows_without_observed_predictors_are_accepted_and_never_matched() {
        let cases = [
            (ModelKind::Rule, "{a,b}", ["a", "b"]),
            (ModelKind::Instance, "{a,b}", ["a", "b"]),
            (ModelKind::Mixed, "{a,b}", ["a", "b"]),
            (ModelKind::RegressionRule, "real", ["1", "3"]),
        ];
        for (model, target, [l0, l1]) in cases {
            let input = ALL_MISSING
                .replace("TARGET", target)
                .replace("L0", l0)
                .replace("L1", l1);
            let data = Dataset::parse_str(&input).unwrap();
            assert_eq!(data.rows()[1].values[..2], [None, None]);
            let config = TrainingConfig {
                model,
                ..TrainingConfig::default()
            };
            let problem =
                LearningProblem::new(data.clone(), config, OperatorsConfig::default()).unwrap();
            let genome = problem.initialize(&mut Random::new(5)).unwrap();
            problem.evaluate(&genome).unwrap();
            let prediction = &problem.predict_dataset(&genome, &data).unwrap()[1];
            if model == ModelKind::RegressionRule {
                assert!(prediction.fallback, "{model:?}");
                assert_eq!(prediction.value, Some(Value::Numeric(2.0)));
            } else {
                assert!(prediction.abstained, "{model:?}");
            }
        }
        let data = Dataset::parse_str(
            "@relation r\n@attribute x integer\n@attribute y real\n@data\n0,1\n?,2\n1,3\n",
        )
        .unwrap();
        let config = TrainingConfig {
            model: ModelKind::RegressionLinearRule,
            ..TrainingConfig::default()
        };
        let problem =
            LearningProblem::new(data.clone(), config, OperatorsConfig::default()).unwrap();
        let genome = problem.initialize(&mut Random::new(5)).unwrap();
        problem.evaluate(&genome).unwrap();
        let prediction = &problem.predict_dataset(&genome, &data).unwrap()[1];
        assert!(prediction.fallback);
        assert_eq!(prediction.value, Some(Value::Numeric(2.0)));
    }

    #[test]
    fn training_data_needs_some_usable_predictor() {
        let e = err("@relation r\n@attribute x numeric\n@attribute class {a,b}\n@data\n?,a\n?,b\n");
        assert!(e.contains("no observed nonconstant predictor"), "{e}");
        let e = err(
            "@relation r\n@attribute x numeric\n@attribute class {a,b}\n@data\n1,a\n1,b\n?,a\n",
        );
        assert!(e.contains("no observed nonconstant predictor"), "{e}");
        let data = Dataset::parse_str(
            "@relation r\n@attribute x numeric\n@attribute class {a,b}\n@data\n?,a\n1,b\n2,a\n",
        )
        .unwrap();
        let e = data.subset_rows(&[0]).unwrap_err().to_string();
        assert!(e.contains("no observed nonconstant predictor"), "{e}");
    }

    #[test]
    fn nominal_lists_may_follow_the_name_directly_and_span_lines() {
        let data = Dataset::parse_str(
            "@relation r\n@attribute x{p,q}\n@attribute 'y'{r,s}\n@attribute class {a, % first\n  b,\n\n  'c d' % last\n}\n@data\np,r,'c d'\nq,s,a\n",
        )
        .unwrap();
        let names = data
            .schema()
            .attributes
            .iter()
            .map(|a| a.name.as_str())
            .collect::<Vec<_>>();
        assert_eq!(names, ["x", "y", "class"]);
        assert_eq!(
            data.schema().attributes[2].nominal_values().unwrap(),
            ["a", "b", "c d"]
        );
        let e = err("@relation r\n@attribute class {a,\nb\n@data\na\n");
        assert!(e.contains("line 2: unterminated nominal value list"), "{e}");
        let e = err("@relation r\n@attribute class {a,\nb,\n");
        assert!(e.contains("line 2: unterminated nominal value list"), "{e}");
        let e = err("@relation r\n@attribute class {a,,b}\n@data\na\n");
        assert!(e.contains("empty nominal label"), "{e}");
    }

    #[test]
    fn nominal_values_match_case_insensitively_unless_labels_collide() {
        let header = "@relation r\n@attribute x numeric\n@attribute class {Yes,no}\n@data\n";
        let data = Dataset::parse_str(&format!("{header}0,YES\n1,No\n2,yes\n")).unwrap();
        let labels = data
            .rows()
            .iter()
            .map(|r| r.values[1].clone().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(
            labels,
            ["Yes", "no", "Yes"].map(|s| Value::Nominal(s.into()))
        );
        let scored =
            Dataset::parse_str_for_prediction(&format!("{header}0,NO\n"), data.schema()).unwrap();
        assert_eq!(
            scored.rows()[0].values[1],
            Some(Value::Nominal("no".into()))
        );

        let header = "@relation r\n@attribute x numeric\n@attribute class {hid,hId,z}\n@data\n";
        let data = Dataset::parse_str(&format!("{header}0,hid\n1,hId\n")).unwrap();
        assert_eq!(data.rows()[1].values[1], Some(Value::Nominal("hId".into())));
        let e = err(&format!("{header}0,hid\n1,Z\n"));
        assert!(
            e.contains("unknown nominal label \"Z\"") && e.contains("case-sensitive"),
            "{e}"
        );
    }

    #[test]
    fn apostrophes_inside_tokens_are_literal() {
        // A quote at token start still opens a quoted token, which must end the token.
        let e = err("@relation r\n@attribute x {'can''t'}\n@attribute class {a}\n@data\n");
        assert!(e.contains("unexpected text after quoted token"), "{e}");
        let data = Dataset::parse_str(
            "@relation don't\n@attribute it's {don't,'a b',\"x,y\"} % it's a comment\n@attribute class {a,b}\n@data\ndon't,a % note\n\"x,y\" , b\n'a b',a\n",
        )
        .unwrap();
        assert_eq!(data.schema().relation.as_deref(), Some("don't"));
        assert_eq!(data.schema().attributes[0].name, "it's");
        assert_eq!(
            data.schema().attributes[0].nominal_values().unwrap(),
            ["don't", "a b", "x,y"]
        );
        let cells = data
            .rows()
            .iter()
            .map(|r| r.values[0].clone().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(
            cells,
            ["don't", "x,y", "a b"].map(|s| Value::Nominal(s.into()))
        );
        let e = err("@relation r\n@attribute x {a,b}\n@attribute class {a,b}\n@data\n'a,b\n");
        assert!(e.contains("line 5: unterminated quoted token"), "{e}");
    }

    #[test]
    fn bom_line_endings_and_trailing_comment_are_accepted() {
        let unix = "@relation r\n@attribute x REAL\n@attribute class {a,b}\n@data\n0,a\n1,b\n% end";
        let expected = Dataset::parse_str(unix).unwrap();
        for input in [
            format!("\u{feff}{unix}"),
            unix.replace('\n', "\r\n"),
            unix.replace('\n', "\r"),
            unix.replacen('\n', "\r", 2).replacen('\n', "\r\n", 2),
        ] {
            assert_eq!(Dataset::parse_str(&input).unwrap(), expected, "{input:?}");
        }
        let e = err("@relation r\r@attribute x real\r\r@attribute class {a,b}\r@data\r0,a\r1,c\r");
        assert!(e.contains("row 2") && e.contains("\"c\""), "{e}");
        let e = err("@relation r\r@attribute x real\r\n\r\n@attribute class bogus\r@data\r");
        assert!(e.contains("line 4:"), "{e}");
    }

    #[test]
    fn retained_strictness() {
        let e = err("@relation r\n@attribute x real\n@attribute class {a,b}\n@data\n{0 1,1 a}\n");
        assert!(e.contains("sparse ARFF rows are unsupported"), "{e}");
        let e = err("@relation r\n@attribute x real\n@attribute class {a,b}\n@data\n0,a,\n");
        assert!(e.contains("row 1 has 3 values"), "{e}");
        let e = err("@relation r\n@attribute x real [0,1]\n@attribute class {a,b}\n@data\n0,a\n");
        assert!(e.contains("unsupported attribute type"), "{e}");
        let e = err("@relation r\n@attribute x integers\n@attribute class {a,b}\n@data\n0,a\n");
        assert!(e.contains("unsupported attribute type"), "{e}");
        let train = Dataset::parse_str(
            "@relation r\n@attribute x real\n@attribute class {a,b}\n@data\n0,a\n1,b\n",
        )
        .unwrap();
        let e = Dataset::parse_str_for_prediction(
            "@relation r\n@attribute x real\n@attribute z real\n@attribute class {a,b}\n@data\n0,0,a\n",
            train.schema(),
        )
        .unwrap_err();
        assert!(e.to_string().contains("unknown attribute \"z\""), "{e}");
    }
}
