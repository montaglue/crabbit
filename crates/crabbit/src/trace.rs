use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

pub const TRACE_EXTENSION: &str = "stx";
pub const DEFAULT_TRACE_DIR: &str = "/tmp/stair-events";

const PART_MARKER_PREFIX: &str = "--- STAIR";
const PART_MARKER_SUFFIX: &str = "---";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StairTraceMeta {
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
pub struct StairTraceFile {
    pub meta: StairTraceMeta,
    pub ir_dumps: Vec<StairTraceDump>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StairTraceDump {
    pub label: String,
    pub ir: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct StairTraceFileInfo {
    pub filename: String,
    pub filepath: String,
    /// File modification time in milliseconds since the Unix epoch.
    /// Used to order versions of a project; 0 when unavailable.
    #[serde(default)]
    pub modified_ms: u64,
}

/// A project groups every trace produced by compiling the same crate:
/// one subdirectory of the trace directory per project, one `.stx` file
/// per compilation (a "version") inside it.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct StairTraceProjectInfo {
    pub name: String,
    /// Versions ordered newest first.
    pub versions: Vec<StairTraceFileInfo>,
}

impl StairTraceFile {
    pub fn new(meta: StairTraceMeta) -> Self {
        Self {
            meta,
            ir_dumps: Vec::new(),
        }
    }

    pub fn push_dump(&mut self, label: impl Into<String>, ir: impl Into<String>) {
        self.ir_dumps.push(StairTraceDump {
            label: label.into(),
            ir: ir.into(),
        });
    }

    pub fn to_stx_string(&self) -> Result<String> {
        let mut out = serde_json::to_string_pretty(&self.meta)?;
        out.push_str("\n\n");
        for dump in &self.ir_dumps {
            out.push_str(&format!("--- STAIR {} ---\n", dump.label));
            out.push_str(dump.ir.trim_matches('\n'));
            out.push('\n');
            out.push('\n');
        }
        Ok(out)
    }

    pub fn from_stx_str(contents: &str) -> Result<Self> {
        parse_trace(contents)
    }

    pub fn read(path: &Path) -> Result<Self> {
        let contents = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read trace file {}", path.display()))?;
        Self::from_stx_str(&contents).with_context(|| format!("failed to parse {}", path.display()))
    }

    pub fn write(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).with_context(|| {
                format!("failed to create trace directory {}", parent.display())
            })?;
        }
        std::fs::write(path, self.to_stx_string()?)
            .with_context(|| format!("failed to write trace file {}", path.display()))
    }
}

pub fn discover_trace_files(dir: &Path) -> Result<Vec<StairTraceFileInfo>> {
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
/// project whose `.stx` files are its versions. Loose `.stx` files in `dir`
/// itself (the pre-folder layout) are grouped under a project name derived
/// from their filename.
pub fn discover_trace_projects(dir: &Path) -> Result<Vec<StairTraceProjectInfo>> {
    let mut projects: BTreeMap<String, Vec<StairTraceFileInfo>> = BTreeMap::new();
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
            StairTraceProjectInfo { name, versions }
        })
        .collect())
}

pub fn sort_versions_newest_first(versions: &mut [StairTraceFileInfo]) {
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

pub fn project_trace_path(project: &str, version: &str) -> PathBuf {
    Path::new(DEFAULT_TRACE_DIR)
        .join(project)
        .join(format!("{version}.{TRACE_EXTENSION}"))
}

fn trace_file_info(path: &Path) -> StairTraceFileInfo {
    StairTraceFileInfo {
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

fn parse_trace(contents: &str) -> Result<StairTraceFile> {
    let start = contents
        .find(|c: char| !c.is_whitespace())
        .ok_or_else(|| anyhow::anyhow!("empty trace file"))?;
    let meta_end = find_json_end(&contents[start..])
        .map(|offset| start + offset)
        .ok_or_else(|| anyhow::anyhow!("trace file must start with a JSON metadata value"))?;
    let meta_src = &contents[start..meta_end];
    let meta: StairTraceMeta = serde_json::from_str(meta_src)?;
    let body = contents[meta_end..].trim_start();
    let ir_dumps = parse_parts(body);
    Ok(StairTraceFile { meta, ir_dumps })
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

fn parse_parts(body: &str) -> Vec<StairTraceDump> {
    let mut parts = Vec::new();
    let mut current_label: Option<String> = None;
    let mut current = String::new();

    for line in body.lines() {
        if let Some(label) = parse_part_marker(line) {
            if let Some(previous_label) = current_label.replace(label) {
                parts.push(StairTraceDump {
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
        parts.push(StairTraceDump {
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
        "stair".to_string()
    } else {
        label.to_string()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_metadata_and_named_parts() {
        let trace = StairTraceFile::from_stx_str(
            r#"{"name":"demo","kind":"compiler-run","pipeline":["a","b"]}
--- STAIR input ---
builtin.module {
}
--- STAIR after a ---
arith.constant 1
"#,
        )
        .unwrap();

        assert_eq!(trace.meta.name, "demo");
        assert_eq!(trace.ir_dumps.len(), 2);
        assert_eq!(trace.ir_dumps[0].label, "input");
        assert_eq!(trace.ir_dumps[0].ir, "builtin.module {\n}");
        assert_eq!(trace.ir_dumps[1].label, "after a");
        assert_eq!(trace.ir_dumps[1].ir, "arith.constant 1");
    }

    #[test]
    fn derives_project_from_legacy_stem() {
        assert_eq!(derive_project_from_stem("demo-12345-1718000000000"), "demo");
        assert_eq!(
            derive_project_from_stem("my-crate-12345-1718000000000"),
            "my-crate"
        );
        assert_eq!(derive_project_from_stem("1718000000000-12345"), "1718000000000");
        assert_eq!(derive_project_from_stem("demo"), "demo");
    }

    #[test]
    fn discovers_projects_from_folders_and_legacy_files() {
        let dir = std::env::temp_dir().join(format!(
            "stair-trace-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(dir.join("demo")).unwrap();
        std::fs::write(dir.join("demo").join("1-1.stx"), "{\"name\":\"demo\",\"kind\":\"k\"}").unwrap();
        std::fs::write(dir.join("demo").join("2-1.stx"), "{\"name\":\"demo\",\"kind\":\"k\"}").unwrap();
        std::fs::write(
            dir.join("demo-99-100.stx"),
            "{\"name\":\"demo-99-100\",\"kind\":\"k\"}",
        )
        .unwrap();
        std::fs::write(dir.join("other-1-2.stx"), "{\"name\":\"other-1-2\",\"kind\":\"k\"}").unwrap();

        let projects = discover_trace_projects(&dir).unwrap();
        std::fs::remove_dir_all(&dir).unwrap();

        let names: Vec<_> = projects.iter().map(|p| p.name.as_str()).collect();
        assert_eq!(names, vec!["demo", "other"]);
        assert_eq!(projects[0].versions.len(), 3);
        assert_eq!(projects[1].versions.len(), 1);
    }

    #[test]
    fn round_trips_trace_file() {
        let meta = StairTraceMeta {
            name: "demo".to_string(),
            kind: "compiler-run".to_string(),
            entry: Some("main".to_string()),
            source: None,
            pipeline: vec!["convert".to_string()],
            target: None,
            note: None,
            extra: BTreeMap::new(),
        };
        let mut trace = StairTraceFile::new(meta);
        trace.push_dump("initial", "mir.module {}");
        trace.push_dump("convert", "llvm.module {}");

        let encoded = trace.to_stx_string().unwrap();
        let decoded = StairTraceFile::from_stx_str(&encoded).unwrap();
        assert_eq!(decoded.meta.name, "demo");
        assert_eq!(decoded.ir_dumps.len(), 2);
        assert_eq!(decoded.ir_dumps[1].label, "convert");
    }
}
