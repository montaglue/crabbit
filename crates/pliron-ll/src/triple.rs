//! Target triple parsing modeled on `llvm::Triple`.
//!
//! A triple is the `arch-vendor-os-environment` string LLVM and rustc use to
//! name a target (e.g. `aarch64-unknown-linux-gnu`, `arm64-apple-macosx11.0.0`).
//! Like LLVM, parsing normalizes aliases (`arm64` → aarch64), strips version
//! suffixes from OS components (`macosx11.0.0` → macosx), tolerates omitted
//! vendors (`x86_64-linux-gnu`), and never fails: unrecognized components are
//! preserved as [`Arch::Unknown`]-style values so callers can report them.

use std::fmt;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Arch {
    Aarch64,
    X86_64,
    Unknown(String),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Vendor {
    Apple,
    Pc,
    Unknown(String),
}

/// Operating system component. Like `llvm::Triple::OSType`, the Darwin
/// family keeps its distinct spellings (`darwin`, `macosx`, `ios`) and is
/// grouped by [`Triple::is_os_darwin`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Os {
    Darwin,
    MacOsx,
    Ios,
    Linux,
    Unknown(String),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Environment {
    Gnu,
    Musl,
    Unknown(String),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Triple {
    pub arch: Arch,
    pub vendor: Vendor,
    pub os: Os,
    pub environment: Environment,
}

impl Triple {
    /// Parses a target triple the way `llvm::Triple` does: the first
    /// component is the architecture; the remaining components are matched
    /// as vendor, OS, and environment, in order, skipping slots whose
    /// keywords don't match (so `x86_64-linux-gnu` still finds its OS).
    pub fn parse(triple: &str) -> Triple {
        let mut components = triple.split('-');
        let arch = components.next().map(parse_arch).unwrap_or_else(|| {
            Arch::Unknown(String::new())
        });

        let mut vendor = None;
        let mut os = None;
        let mut environment = None;
        for (index, component) in components.enumerate() {
            if os.is_none()
                && let Some(parsed) = try_parse_os(component)
            {
                os = Some(parsed);
                continue;
            }
            if index == 0 && vendor.is_none() && os.is_none() {
                vendor = Some(parse_vendor(component));
                continue;
            }
            if environment.is_none() && os.is_some() {
                environment = Some(parse_environment(component));
            }
        }

        Triple {
            arch,
            vendor: vendor.unwrap_or(Vendor::Unknown(String::new())),
            os: os.unwrap_or(Os::Unknown(String::new())),
            environment: environment.unwrap_or(Environment::Unknown(String::new())),
        }
    }

    /// Whether the OS is any member of the Darwin family (macOS, iOS, or a
    /// bare `darwin`), mirroring `llvm::Triple::isOSDarwin`.
    pub fn is_os_darwin(&self) -> bool {
        matches!(self.os, Os::Darwin | Os::MacOsx | Os::Ios)
    }
}

impl fmt::Display for Triple {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let arch = match &self.arch {
            Arch::Aarch64 => "aarch64",
            Arch::X86_64 => "x86_64",
            Arch::Unknown(name) => name,
        };
        let vendor = match &self.vendor {
            Vendor::Apple => "apple",
            Vendor::Pc => "pc",
            Vendor::Unknown(_) => "unknown",
        };
        let os = match &self.os {
            Os::Darwin => "darwin",
            Os::MacOsx => "macosx",
            Os::Ios => "ios",
            Os::Linux => "linux",
            Os::Unknown(_) => "unknown",
        };
        write!(f, "{arch}-{vendor}-{os}")?;
        match &self.environment {
            Environment::Gnu => write!(f, "-gnu"),
            Environment::Musl => write!(f, "-musl"),
            Environment::Unknown(_) => Ok(()),
        }
    }
}

fn parse_arch(component: &str) -> Arch {
    match component {
        "aarch64" | "arm64" | "arm64e" => Arch::Aarch64,
        "x86_64" | "amd64" | "x86_64h" => Arch::X86_64,
        other => Arch::Unknown(other.to_string()),
    }
}

fn parse_vendor(component: &str) -> Vendor {
    match component {
        "apple" => Vendor::Apple,
        "pc" => Vendor::Pc,
        other => Vendor::Unknown(other.to_string()),
    }
}

/// OS components carry version suffixes (`macosx11.0.0`), so they are
/// matched by prefix, as in `llvm::Triple`'s `parseOS`.
fn try_parse_os(component: &str) -> Option<Os> {
    if component.starts_with("darwin") {
        Some(Os::Darwin)
    } else if component.starts_with("macosx") || component.starts_with("macos") {
        Some(Os::MacOsx)
    } else if component.starts_with("ios") {
        Some(Os::Ios)
    } else if component.starts_with("linux") {
        Some(Os::Linux)
    } else {
        None
    }
}

fn parse_environment(component: &str) -> Environment {
    if component.starts_with("gnu") {
        Environment::Gnu
    } else if component.starts_with("musl") {
        Environment::Musl
    } else {
        Environment::Unknown(component.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_rustc_linux_triple() {
        let triple = Triple::parse("aarch64-unknown-linux-gnu");
        assert_eq!(triple.arch, Arch::Aarch64);
        assert_eq!(triple.os, Os::Linux);
        assert_eq!(triple.environment, Environment::Gnu);
        assert!(!triple.is_os_darwin());
        assert_eq!(triple.to_string(), "aarch64-unknown-linux-gnu");
    }

    #[test]
    fn parses_rustc_darwin_triple_with_version() {
        let triple = Triple::parse("arm64-apple-macosx11.0.0");
        assert_eq!(triple.arch, Arch::Aarch64);
        assert_eq!(triple.vendor, Vendor::Apple);
        assert_eq!(triple.os, Os::MacOsx);
        assert!(triple.is_os_darwin());
    }

    #[test]
    fn parses_bare_darwin_triple() {
        let triple = Triple::parse("x86_64-apple-darwin");
        assert_eq!(triple.arch, Arch::X86_64);
        assert!(triple.is_os_darwin());
    }

    #[test]
    fn parses_triple_with_omitted_vendor() {
        let triple = Triple::parse("x86_64-linux-gnu");
        assert_eq!(triple.arch, Arch::X86_64);
        assert_eq!(triple.os, Os::Linux);
        assert_eq!(triple.environment, Environment::Gnu);
    }

    #[test]
    fn preserves_unknown_components() {
        let triple = Triple::parse("riscv64-unknown-freebsd");
        assert_eq!(triple.arch, Arch::Unknown("riscv64".to_string()));
        assert_eq!(triple.os, Os::Unknown(String::new()));
    }
}
