use thiserror::Error;

use crate::dialects::llvm::attributes::LinkageAttr;

#[derive(Debug, Error)]
pub(super) enum Aarch64Err {
    #[error("expected builtin.module as AArch64 lowering root")]
    NotModule,
    #[error("unsupported LLVM operation for AArch64 lowering: {0}")]
    UnsupportedOp(String),
    #[error("unsupported LLVM type for AArch64 ABI: {0}")]
    UnsupportedType(String),
    #[error("function `{0}` uses unsupported linkage `{1:?}`")]
    UnsupportedLinkage(String, LinkageAttr),
    #[error("value `{0}` was used before instruction selection defined it")]
    UndefinedValue(String),
}
