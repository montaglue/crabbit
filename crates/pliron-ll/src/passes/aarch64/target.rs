//! The OS-specific knobs of the shared aarch64 backend. The instruction
//! selection, register allocation, frame lowering, and encoding passes are
//! identical across operating systems; everything that genuinely differs
//! between Mach-O/Darwin and ELF/Linux is captured here.

/// Operating system flavor of an aarch64 target. The pass pipeline is shared;
/// this selects symbol mangling, the calling-convention variant, and which
/// object container the module is written into.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TargetOs {
    /// Mach-O container, `_`-prefixed symbols, Apple's AArch64 ABI.
    Darwin,
    /// ELF container, unprefixed symbols, standard AAPCS64.
    Linux,
}

impl TargetOs {
    /// The object-file symbol for a source-level name: Mach-O prepends an
    /// underscore, ELF uses the name as-is.
    pub fn symbol_name(self, name: &str) -> String {
        match self {
            TargetOs::Darwin => format!("_{name}"),
            TargetOs::Linux => name.to_string(),
        }
    }

    /// AAPCS64 (rule C.8) starts a 16-byte-aligned integer pair at an
    /// even-numbered GPR; Apple's ABI drops that requirement.
    pub fn requires_even_gpr_pairs(self) -> bool {
        matches!(self, TargetOs::Linux)
    }
}
