//! Traces of a compilation: the IR before the first pass and after every
//! pass, one file per compilation (`.crt`).
//!
//! A trace is a JSON header ([TraceMeta]: crate, target, the pipeline's
//! pass names, and whatever else its producer records) followed by the
//! snapshots, each starting with a `--- crabbit <label> ---` line: the
//! first is the input (`initial`), every later one is the IR after the pass
//! it is named after. The IR itself is pliron's printed form.
//!
//! A trace directory holds one subdirectory per project (the compiled
//! crate) with one trace per compilation in it.

use std::{collections::BTreeMap, path::Path};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

pub const TRACE_EXTENSION: &str = "crt";

const PART_MARKER_PREFIX: &str = "--- crabbit";
const PART_MARKER_SUFFIX: &str = "---";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceMeta {
    pub name: String,
    pub kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub entry: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub pipeline: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
    #[serde(flatten)]
    pub extra: BTreeMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trace {
    pub meta: TraceMeta,
    pub snapshots: Vec<Snapshot>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Snapshot {
    pub label: String,
    pub ir: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TraceFileInfo {
    pub filename: String,
    pub filepath: String,
    /// File modification time in milliseconds since the Unix epoch.
    /// Used to order versions of a project; 0 when unavailable.
    #[serde(default)]
    pub modified_ms: u64,
}

/// A project groups every trace produced by compiling the same crate:
/// one subdirectory of the trace directory per project, one `.crt` file
/// per compilation (a "version") inside it.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TraceProjectInfo {
    pub name: String,
    /// Versions ordered newest first.
    pub versions: Vec<TraceFileInfo>,
}

impl Trace {
    pub fn new(meta: TraceMeta) -> Self {
        Self {
            meta,
            snapshots: Vec::new(),
        }
    }

    pub fn push_snapshot(&mut self, label: impl Into<String>, ir: impl Into<String>) {
        self.snapshots.push(Snapshot {
            label: label.into(),
            ir: ir.into(),
        });
    }

    pub fn to_text(&self) -> Result<String> {
        // One line: the header may carry large machine data (profiles).
        let mut out = serde_json::to_string(&self.meta)?;
        out.push_str("\n\n");
        for dump in &self.snapshots {
            out.push_str(&format!("--- crabbit {} ---\n", dump.label));
            out.push_str(dump.ir.trim_matches('\n'));
            out.push('\n');
            out.push('\n');
        }
        Ok(out)
    }

    pub fn parse(contents: &str) -> Result<Self> {
        parse_trace(contents)
    }

    pub fn read(path: &Path) -> Result<Self> {
        let contents = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read trace file {}", path.display()))?;
        Self::parse(&contents).with_context(|| format!("failed to parse {}", path.display()))
    }

    pub fn write(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).with_context(|| {
                format!("failed to create trace directory {}", parent.display())
            })?;
        }
        std::fs::write(path, self.to_text()?)
            .with_context(|| format!("failed to write trace file {}", path.display()))
    }
}

pub fn discover_trace_files(dir: &Path) -> Result<Vec<TraceFileInfo>> {
    let mut files = Vec::new();
    if !dir.exists() {
        return Ok(files);
    }

    for entry in std::fs::read_dir(dir)
        .with_context(|| format!("failed to read trace directory {}", dir.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        if path.extension().and_then(|ext| ext.to_str()) != Some(TRACE_EXTENSION) {
            continue;
        }
        files.push(trace_file_info(&path));
    }

    files.sort_by(|a, b| a.filename.cmp(&b.filename));
    Ok(files)
}

/// Discovers traces grouped by project. Each subdirectory of `dir` is a
/// project whose `.crt` files are its versions. Loose `.crt` files in `dir`
/// itself (the pre-folder layout) are grouped under a project name derived
/// from their filename.
pub fn discover_trace_projects(dir: &Path) -> Result<Vec<TraceProjectInfo>> {
    let mut projects: BTreeMap<String, Vec<TraceFileInfo>> = BTreeMap::new();
    if !dir.exists() {
        return Ok(Vec::new());
    }

    for entry in std::fs::read_dir(dir)
        .with_context(|| format!("failed to read trace directory {}", dir.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            let name = path
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string();
            let versions = discover_trace_files(&path)?;
            if !versions.is_empty() {
                projects.entry(name).or_default().extend(versions);
            }
        } else if path.extension().and_then(|ext| ext.to_str()) == Some(TRACE_EXTENSION) {
            let stem = path
                .file_stem()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string();
            projects
                .entry(derive_project_from_stem(&stem).to_string())
                .or_default()
                .push(trace_file_info(&path));
        }
    }

    Ok(projects
        .into_iter()
        .map(|(name, mut versions)| {
            sort_versions_newest_first(&mut versions);
            TraceProjectInfo { name, versions }
        })
        .collect())
}

pub fn sort_versions_newest_first(versions: &mut [TraceFileInfo]) {
    versions.sort_by(|a, b| {
        b.modified_ms
            .cmp(&a.modified_ms)
            .then_with(|| b.filename.cmp(&a.filename))
    });
}

/// Extracts the project name from a legacy flat trace filename stem of the
/// form `{project}-{pid}-{timestamp}` by stripping up to two trailing
/// all-digit segments. Stems without such suffixes are returned unchanged.
pub fn derive_project_from_stem(stem: &str) -> &str {
    let mut out = stem;
    for _ in 0..2 {
        match out.rsplit_once('-') {
            Some((prefix, suffix))
                if !prefix.is_empty()
                    && !suffix.is_empty()
                    && suffix.chars().all(|ch| ch.is_ascii_digit()) =>
            {
                out = prefix;
            }
            _ => break,
        }
    }
    out
}

fn trace_file_info(path: &Path) -> TraceFileInfo {
    TraceFileInfo {
        filename: path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string(),
        filepath: path.to_string_lossy().to_string(),
        modified_ms: path
            .metadata()
            .and_then(|meta| meta.modified())
            .ok()
            .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|duration| duration.as_millis() as u64)
            .unwrap_or_default(),
    }
}

fn parse_trace(contents: &str) -> Result<Trace> {
    let start = contents
        .find(|c: char| !c.is_whitespace())
        .ok_or_else(|| anyhow::anyhow!("empty trace file"))?;
    let meta_end = find_json_end(&contents[start..])
        .map(|offset| start + offset)
        .ok_or_else(|| anyhow::anyhow!("trace file must start with a JSON metadata value"))?;
    let meta_src = &contents[start..meta_end];
    let meta: TraceMeta = serde_json::from_str(meta_src)?;
    let body = contents[meta_end..].trim_start();
    let snapshots = parse_parts(body);
    Ok(Trace { meta, snapshots })
}

fn find_json_end(src: &str) -> Option<usize> {
    let mut chars = src.char_indices();
    let (_, first) = chars.next()?;
    let (open, close) = match first {
        '{' => ('{', '}'),
        '[' => ('[', ']'),
        _ => return None,
    };

    let mut depth = 1usize;
    let mut in_string = false;
    let mut escaped = false;

    for (idx, ch) in chars {
        if in_string {
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                in_string = false;
            }
            continue;
        }

        if ch == '"' {
            in_string = true;
        } else if ch == open {
            depth += 1;
        } else if ch == close {
            depth -= 1;
            if depth == 0 {
                return Some(idx + ch.len_utf8());
            }
        }
    }

    None
}

fn parse_parts(body: &str) -> Vec<Snapshot> {
    let mut parts = Vec::new();
    let mut current_label: Option<String> = None;
    let mut current = String::new();

    for line in body.lines() {
        if let Some(label) = parse_part_marker(line) {
            if let Some(previous_label) = current_label.replace(label) {
                parts.push(Snapshot {
                    label: previous_label,
                    ir: current.trim_matches('\n').to_string(),
                });
                current.clear();
            }
        } else if current_label.is_some() {
            current.push_str(line);
            current.push('\n');
        } else if !line.trim().is_empty() {
            current_label = Some("part 1".to_string());
            current.push_str(line);
            current.push('\n');
        }
    }

    if let Some(label) = current_label {
        parts.push(Snapshot {
            label,
            ir: current.trim_matches('\n').to_string(),
        });
    }

    parts
}

fn parse_part_marker(line: &str) -> Option<String> {
    let trimmed = line.trim();
    if !trimmed.starts_with(PART_MARKER_PREFIX) || !trimmed.ends_with(PART_MARKER_SUFFIX) {
        return None;
    }
    let label = trimmed
        .trim_start_matches(PART_MARKER_PREFIX)
        .trim_end_matches(PART_MARKER_SUFFIX)
        .trim();
    Some(if label.is_empty() {
        "crabbit".to_string()
    } else {
        label.to_string()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_metadata_and_named_parts() {
        let trace = Trace::parse(
            r#"{"name":"demo","kind":"compiler-run","pipeline":["a","b"]}
--- crabbit input ---
builtin.module {
}
--- crabbit after a ---
arith.constant 1
"#,
        )
        .unwrap();

        assert_eq!(trace.meta.name, "demo");
        assert_eq!(trace.snapshots.len(), 2);
        assert_eq!(trace.snapshots[0].label, "input");
        assert_eq!(trace.snapshots[0].ir, "builtin.module {\n}");
        assert_eq!(trace.snapshots[1].label, "after a");
        assert_eq!(trace.snapshots[1].ir, "arith.constant 1");
    }

    #[test]
    fn derives_project_from_legacy_stem() {
        assert_eq!(derive_project_from_stem("demo-12345-1718000000000"), "demo");
        assert_eq!(
            derive_project_from_stem("my-crate-12345-1718000000000"),
            "my-crate"
        );
        assert_eq!(
            derive_project_from_stem("1718000000000-12345"),
            "1718000000000"
        );
        assert_eq!(derive_project_from_stem("demo"), "demo");
    }

    #[test]
    fn discovers_projects_from_folders_and_legacy_files() {
        let dir = std::env::temp_dir().join(format!(
            "pliron-trace-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(dir.join("demo")).unwrap();
        std::fs::write(
            dir.join("demo").join("1-1.crt"),
            "{\"name\":\"demo\",\"kind\":\"k\"}",
        )
        .unwrap();
        std::fs::write(
            dir.join("demo").join("2-1.crt"),
            "{\"name\":\"demo\",\"kind\":\"k\"}",
        )
        .unwrap();
        std::fs::write(
            dir.join("demo-99-100.crt"),
            "{\"name\":\"demo-99-100\",\"kind\":\"k\"}",
        )
        .unwrap();
        std::fs::write(
            dir.join("other-1-2.crt"),
            "{\"name\":\"other-1-2\",\"kind\":\"k\"}",
        )
        .unwrap();

        let projects = discover_trace_projects(&dir).unwrap();
        std::fs::remove_dir_all(&dir).unwrap();

        let names: Vec<_> = projects.iter().map(|p| p.name.as_str()).collect();
        assert_eq!(names, vec!["demo", "other"]);
        assert_eq!(projects[0].versions.len(), 3);
        assert_eq!(projects[1].versions.len(), 1);
    }

    #[test]
    fn round_trips_trace_file() {
        let meta = TraceMeta {
            name: "demo".to_string(),
            kind: "compiler-run".to_string(),
            entry: Some("main".to_string()),
            source: None,
            pipeline: vec!["convert".to_string()],
            target: None,
            note: None,
            extra: BTreeMap::new(),
        };
        let mut trace = Trace::new(meta);
        trace.push_snapshot("initial", "mir.module {}");
        trace.push_snapshot("convert", "llvm.module {}");

        let encoded = trace.to_text().unwrap();
        let decoded = Trace::parse(&encoded).unwrap();
        assert_eq!(decoded.meta.name, "demo");
        assert_eq!(decoded.snapshots.len(), 2);
        assert_eq!(decoded.snapshots[1].label, "convert");
    }
}
