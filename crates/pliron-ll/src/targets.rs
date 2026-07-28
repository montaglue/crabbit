//! Registry of the native object backends, looked up by [Triple] the way
//! LLVM resolves a registered `Target` from a triple (`TargetRegistry::
//! lookupTarget`). Each backend bundles the pass pipeline that lowers a
//! `builtin.module` to encoded machine code and the writer that turns the
//! lowered module into object-container bytes for that triple.

use pliron::{context::Context, context::Ptr, operation::Operation};

use crate::{
    conversion::pass::Passes,
    passes::{aarch64, x86_64_darwin},
    result::STAIRResult,
    triple::{Arch, Triple},
};

/// A registered object backend: everything the driver needs to compile for
/// one triple family.
pub struct TargetBackend {
    /// Registry name, e.g. `aarch64-linux` (diagnostics only).
    pub name: &'static str,
    matches: fn(&Triple) -> bool,
    pipeline: fn() -> Passes,
    write_object: fn(&mut Context, Ptr<Operation>) -> STAIRResult<Vec<u8>>,
}

impl TargetBackend {
    /// The lowering pipeline for this backend, from LLVM-dialect
    /// verification down to encoded machine code.
    pub fn pipeline(&self) -> Passes {
        (self.pipeline)()
    }

    /// Writes a module lowered by [Self::pipeline] into object-container
    /// bytes (Mach-O or ELF, per the backend's triple).
    pub fn write_object(
        &self,
        ctx: &mut Context,
        root: Ptr<Operation>,
    ) -> STAIRResult<Vec<u8>> {
        (self.write_object)(ctx, root)
    }
}

static BACKENDS: &[TargetBackend] = &[
    TargetBackend {
        name: "aarch64-darwin",
        matches: |triple| triple.arch == Arch::Aarch64 && triple.is_os_darwin(),
        pipeline: || aarch64::pipeline(aarch64::TargetOs::Darwin),
        write_object: aarch64::write_macho_object_from_ir,
    },
    TargetBackend {
        name: "aarch64-linux",
        matches: |triple| {
            triple.arch == Arch::Aarch64 && triple.os == crate::triple::Os::Linux
        },
        pipeline: || aarch64::pipeline(aarch64::TargetOs::Linux),
        write_object: aarch64::write_elf_object_from_ir,
    },
    TargetBackend {
        name: "x86_64-darwin",
        matches: |triple| triple.arch == Arch::X86_64 && triple.is_os_darwin(),
        pipeline: x86_64_darwin::pipeline,
        write_object: x86_64_darwin::write_macho_object_from_ir,
    },
];

/// The backend registered for `triple`, or `None` if no backend supports it.
pub fn lookup(triple: &Triple) -> Option<&'static TargetBackend> {
    BACKENDS.iter().find(|backend| (backend.matches)(triple))
}

/// The registry names of every registered backend, for diagnostics.
pub fn registered_names() -> impl Iterator<Item = &'static str> {
    BACKENDS.iter().map(|backend| backend.name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_backends_from_triples() {
        let cases = [
            ("aarch64-unknown-linux-gnu", Some("aarch64-linux")),
            ("arm64-apple-macosx11.0.0", Some("aarch64-darwin")),
            ("aarch64-apple-darwin", Some("aarch64-darwin")),
            ("x86_64-apple-darwin", Some("x86_64-darwin")),
            ("x86_64-unknown-linux-gnu", None),
            ("riscv64-unknown-linux-gnu", None),
        ];
        for (triple, expected) in cases {
            let backend = lookup(&Triple::parse(triple));
            assert_eq!(backend.map(|b| b.name), expected, "triple {triple}");
        }
    }
}
