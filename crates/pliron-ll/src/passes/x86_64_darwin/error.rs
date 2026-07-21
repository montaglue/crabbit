use thiserror::Error;

use crate::dialects::llvm::attributes::LinkageAttr;

#[derive(Debug, Error)]
pub(super) enum X86_64DarwinErr {
    #[error("expected builtin.module as x86-64 Darwin lowering root")]
    NotModule,
    #[error("unsupported LLVM operation for x86-64 Darwin lowering: {0}")]
    UnsupportedOp(String),
    #[error("unsupported LLVM type for Darwin x86-64 ABI: {0}")]
    UnsupportedType(String),
    #[error("function `{0}` uses unsupported linkage `{1:?}`")]
    UnsupportedLinkage(String, LinkageAttr),
    #[error("value `{0}` was used before instruction selection defined it")]
    UndefinedValue(String),
}
