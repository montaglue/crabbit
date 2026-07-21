use core::fmt;

/// The register file an x86-64 value is permitted to occupy.
///
/// This is deliberately independent of textual attribute spellings. x86-64
/// operations must carry this information as part of their constraints instead
/// of recovering it from a string.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum RegisterClass {
    Gpr64,
    Gpr32,
    Xmm,
    Rflags,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct VirtualRegister(pub u32);

/// x86-64 general-purpose registers by hardware encoding number:
/// rax=0, rcx=1, rdx=2, rbx=3, rsp=4, rbp=5, rsi=6, rdi=7, r8..r15=8..15.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum PhysicalRegister {
    Gpr64(u8),
    Gpr32(u8),
    Xmm(u8),
    Rflags,
}

pub const GPR64_NAMES: [&str; 16] = [
    "rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13",
    "r14", "r15",
];

pub const GPR32_NAMES: [&str; 16] = [
    "eax", "ecx", "edx", "ebx", "esp", "ebp", "esi", "edi", "r8d", "r9d", "r10d", "r11d", "r12d",
    "r13d", "r14d", "r15d",
];

pub const RSP: u8 = 4;

impl PhysicalRegister {
    pub fn class(self) -> RegisterClass {
        match self {
            Self::Gpr64(_) => RegisterClass::Gpr64,
            Self::Gpr32(_) => RegisterClass::Gpr32,
            Self::Xmm(_) => RegisterClass::Xmm,
            Self::Rflags => RegisterClass::Rflags,
        }
    }

    pub fn is_reserved(self) -> bool {
        // rsp and rbp frame the stack; rflags is not allocatable.
        matches!(self, Self::Gpr64(4 | 5) | Self::Rflags)
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
    /// Parses the compatibility spelling used by the attribute-based
    /// instruction dialect. Virtual GPRs use `vrN`; physical registers use
    /// their architectural names (`rax`, `r12`, `eax`, `xmm0`, ...).
    pub fn parse(text: &str) -> Option<Self> {
        if let Some(number) = text.strip_prefix("vr") {
            return number.parse().ok().map(|id| Self::Virtual {
                id: VirtualRegister(id),
                class: RegisterClass::Gpr64,
            });
        }
        if text == "rflags" {
            return Some(Self::Physical(PhysicalRegister::Rflags));
        }
        if let Some(index) = GPR64_NAMES.iter().position(|name| *name == text) {
            return Some(Self::Physical(PhysicalRegister::Gpr64(index as u8)));
        }
        if let Some(index) = GPR32_NAMES.iter().position(|name| *name == text) {
            return Some(Self::Physical(PhysicalRegister::Gpr32(index as u8)));
        }
        if let Some(number) = text.strip_prefix("xmm") {
            return number
                .parse::<u8>()
                .ok()
                .filter(|number| *number <= 15)
                .map(|number| Self::Physical(PhysicalRegister::Xmm(number)));
        }
        None
    }

    pub const fn virtual_gpr(id: u32) -> Self {
        Self::Virtual {
            id: VirtualRegister(id),
            class: RegisterClass::Gpr64,
        }
    }

    /// The 64-bit general-purpose register with this hardware number.
    pub const fn gpr(number: u8) -> Self {
        Self::Physical(PhysicalRegister::Gpr64(number))
    }
}

pub const RAX: Register = Register::gpr(0);
pub const RCX: Register = Register::gpr(1);
pub const RDX: Register = Register::gpr(2);
pub const RBX: Register = Register::gpr(3);
pub const RSP_REG: Register = Register::gpr(4);
pub const RBP: Register = Register::gpr(5);
pub const RSI: Register = Register::gpr(6);
pub const RDI: Register = Register::gpr(7);
pub const R8: Register = Register::gpr(8);
pub const R9: Register = Register::gpr(9);
pub const R10: Register = Register::gpr(10);
pub const R11: Register = Register::gpr(11);
pub const R12: Register = Register::gpr(12);
pub const R13: Register = Register::gpr(13);
pub const R14: Register = Register::gpr(14);
pub const R15: Register = Register::gpr(15);

impl fmt::Display for Register {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Virtual { id, .. } => write!(f, "vr{}", id.0),
            Self::Physical(PhysicalRegister::Gpr64(number)) => {
                f.write_str(GPR64_NAMES[*number as usize])
            }
            Self::Physical(PhysicalRegister::Gpr32(number)) => {
                f.write_str(GPR32_NAMES[*number as usize])
            }
            Self::Physical(PhysicalRegister::Xmm(number)) => write!(f, "xmm{number}"),
            Self::Physical(PhysicalRegister::Rflags) => f.write_str("rflags"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{PhysicalRegister, Register, RegisterClass, VirtualRegister};

    #[test]
    fn parses_virtual_and_physical_gprs() {
        assert_eq!(
            Register::parse("vr7"),
            Some(Register::Virtual {
                id: VirtualRegister(7),
                class: RegisterClass::Gpr64,
            })
        );
        assert_eq!(
            Register::parse("rax"),
            Some(Register::Physical(PhysicalRegister::Gpr64(0)))
        );
        assert_eq!(
            Register::parse("r12"),
            Some(Register::Physical(PhysicalRegister::Gpr64(12)))
        );
        assert_eq!(
            Register::parse("esi"),
            Some(Register::Physical(PhysicalRegister::Gpr32(6)))
        );
        assert_eq!(Register::parse("v7"), None);
    }
}
