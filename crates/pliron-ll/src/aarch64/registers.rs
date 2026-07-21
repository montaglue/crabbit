use core::fmt;

/// The register file an AArch64 value is permitted to occupy.
///
/// This is deliberately independent of textual attribute spellings. AArch64
/// operations must carry this information as part of their constraints instead
/// of recovering it from a string.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum RegisterClass {
    Gpr64,
    Gpr32,
    Fpr64,
    Fpr32,
    Simd128,
    Nzcv,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct VirtualRegister(pub u32);

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum PhysicalRegister {
    Gpr64(u8),
    Gpr32(u8),
    Fpr64(u8),
    Fpr32(u8),
    Simd128(u8),
    Sp,
    Nzcv,
}

impl PhysicalRegister {
    pub fn class(self) -> RegisterClass {
        match self {
            Self::Gpr64(_) | Self::Sp => RegisterClass::Gpr64,
            Self::Gpr32(_) => RegisterClass::Gpr32,
            Self::Fpr64(_) => RegisterClass::Fpr64,
            Self::Fpr32(_) => RegisterClass::Fpr32,
            Self::Simd128(_) => RegisterClass::Simd128,
            Self::Nzcv => RegisterClass::Nzcv,
        }
    }

    pub fn is_reserved(self) -> bool {
        matches!(self, Self::Sp | Self::Nzcv | Self::Gpr64(18 | 29 | 30))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum Register {
    Virtual {
        id: VirtualRegister,
        class: RegisterClass,
    },
    Physical(PhysicalRegister),
}

impl Register {
    /// Parses the compatibility spelling used by the existing attribute-based
    /// instruction dialect. Virtual GPRs use `vrN`, never `vN`: `vN` is the
    /// architectural SIMD/FP register namespace.
    pub fn parse(text: &str) -> Option<Self> {
        if let Some(number) = text.strip_prefix("vr") {
            return number.parse().ok().map(|id| Self::Virtual {
                id: VirtualRegister(id),
                class: RegisterClass::Gpr64,
            });
        }
        if text == "sp" {
            return Some(Self::Physical(PhysicalRegister::Sp));
        }
        if text == "nzcv" {
            return Some(Self::Physical(PhysicalRegister::Nzcv));
        }
        let parse_number = |prefix: char| {
            text.strip_prefix(prefix)
                .and_then(|number| number.parse::<u8>().ok())
                .filter(|number| *number <= 31)
        };
        parse_number('x')
            .map(|number| Self::Physical(PhysicalRegister::Gpr64(number)))
            .or_else(|| {
                parse_number('w').map(|number| Self::Physical(PhysicalRegister::Gpr32(number)))
            })
            .or_else(|| {
                parse_number('d').map(|number| Self::Physical(PhysicalRegister::Fpr64(number)))
            })
            .or_else(|| {
                parse_number('s').map(|number| Self::Physical(PhysicalRegister::Fpr32(number)))
            })
            .or_else(|| {
                parse_number('v').map(|number| Self::Physical(PhysicalRegister::Simd128(number)))
            })
    }

    pub const fn virtual_gpr(id: u32) -> Self {
        Self::Virtual {
            id: VirtualRegister(id),
            class: RegisterClass::Gpr64,
        }
    }

    /// The 64-bit general-purpose register `x<number>`.
    pub const fn gpr(number: u8) -> Self {
        Self::Physical(PhysicalRegister::Gpr64(number))
    }
}

/// The stack pointer.
pub const SP: Register = Register::Physical(PhysicalRegister::Sp);
/// The link register (`x30`).
pub const LR: Register = Register::gpr(30);
/// The frame pointer (`x29`).
pub const FP: Register = Register::gpr(29);
/// The indirect-result (sret) register.
pub const X8: Register = Register::gpr(8);
/// The intra-procedure-call scratch register, outside the allocatable set.
pub const X16: Register = Register::gpr(16);

impl fmt::Display for Register {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Virtual { id, .. } => write!(f, "vr{}", id.0),
            Self::Physical(PhysicalRegister::Gpr64(number)) => write!(f, "x{number}"),
            Self::Physical(PhysicalRegister::Gpr32(number)) => write!(f, "w{number}"),
            Self::Physical(PhysicalRegister::Fpr64(number)) => write!(f, "d{number}"),
            Self::Physical(PhysicalRegister::Fpr32(number)) => write!(f, "s{number}"),
            Self::Physical(PhysicalRegister::Simd128(number)) => write!(f, "v{number}"),
            Self::Physical(PhysicalRegister::Sp) => f.write_str("sp"),
            Self::Physical(PhysicalRegister::Nzcv) => f.write_str("nzcv"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{PhysicalRegister, Register, RegisterClass, VirtualRegister};

    #[test]
    fn keeps_virtual_gprs_out_of_the_simd_namespace() {
        assert_eq!(
            Register::parse("vr7"),
            Some(Register::Virtual {
                id: VirtualRegister(7),
                class: RegisterClass::Gpr64,
            })
        );
        assert_eq!(
            Register::parse("v7"),
            Some(Register::Physical(PhysicalRegister::Simd128(7)))
        );
    }
}
