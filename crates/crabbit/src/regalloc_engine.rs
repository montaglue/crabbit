//! Register-allocation engine selection (eregalloc/INTEGRATION-CONTRACT.md).
//!
//! | variable | values | default | effect |
//! |---|---|---|---|
//! | `CRABBIT_REGALLOC` | `linear`, `eregalloc` | `linear` | `linear` keeps the backend's own allocator; `eregalloc` swaps in `EregallocRegisterAllocatePass` |
//! | `CRABBIT_REGALLOC_ORACLE` | `c0`, `c2` | `c0` | `c0` → syntactic oracle, `c2` → saturated e-graph oracle; only read under `eregalloc` |
//! | `CRABBIT_BLOCK_FREQ` | `uniform`, `spectral` | `uniform` | under `eregalloc`: the pass's frequency source (`spectral` is the fallthrough-aware machine-CFG Perron computation) |
//!
//! Under `eregalloc`, `CRABBIT_SPILL_POLICY` / `CRABBIT_RESTORE_ESTIMATE`
//! are superseded (the oracle scoring and plan-driven remat are the engine)
//! and are ignored; `CRABBIT_BLOCK_FREQ` keeps its meaning. Under `linear`
//! all three `CodegenOpts` variables keep their exact meaning and the
//! eregalloc-only variables are ignored.
//!
//! This lives in the `crabbit` root crate, not `pliron-ll`, because
//! `eregalloc-passes` depends on `pliron-ll` (package cycle otherwise).
//!
//! `cmt_spectral_cfg::block_frequencies` is the region-level CMT analysis
//! and matches `BlockFreqSource::Provider` exactly; it is kept reachable
//! ([cmt_block_frequencies]) for explicit-edge CFGs, but the pre-placement
//! machine CFG at the RA position uses implicit fallthrough, which a
//! `block.succs` analysis cannot see — so `spectral` routes to
//! `BlockFreqSource::Spectral` (same math over the allocator's own
//! successor lists) as the contract prescribes.

use eregalloc_passes::{BlockFreqSource, EregallocRegisterAllocatePass, OracleKind};
use pliron::{context::Context, context::Ptr, region::Region};

/// CMT's region-level spectral block frequencies, in the exact shape of
/// [BlockFreqSource::Provider]. Selected by `CRABBIT_BLOCK_FREQ=cmt-provider`
/// (an explicitly experimental value: see the module docs for why it
/// under-connects the machine CFG at the RA position).
pub fn cmt_block_frequencies(ctx: &Context, region: Ptr<Region>) -> Vec<f64> {
    cmt_spectral_cfg::block_frequencies(ctx, region)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RegallocEngine {
    Linear,
    Eregalloc { oracle: OracleKind, freq: FreqChoice },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FreqChoice {
    Uniform,
    Spectral,
    CmtProvider,
}

impl RegallocEngine {
    pub fn from_env() -> Result<Self, String> {
        let engine = env("CRABBIT_REGALLOC");
        match engine.as_deref() {
            None | Some("linear") => Ok(RegallocEngine::Linear),
            Some("eregalloc") => {
                let oracle = match env("CRABBIT_REGALLOC_ORACLE").as_deref() {
                    None | Some("c0") => OracleKind::Syntactic,
                    Some("c2") => OracleKind::EGraph,
                    Some(other) => return Err(unknown("CRABBIT_REGALLOC_ORACLE", other, "c0, c2")),
                };
                let freq = match env("CRABBIT_BLOCK_FREQ").as_deref() {
                    None | Some("uniform") => FreqChoice::Uniform,
                    Some("spectral") => FreqChoice::Spectral,
                    Some("cmt-provider") => FreqChoice::CmtProvider,
                    Some(other) => {
                        return Err(unknown("CRABBIT_BLOCK_FREQ", other, "uniform, spectral, cmt-provider"));
                    }
                };
                Ok(RegallocEngine::Eregalloc { oracle, freq })
            }
            Some(other) => Err(unknown("CRABBIT_REGALLOC", other, "linear, eregalloc")),
        }
    }

    /// The allocator pass to substitute, or `None` for the backend default.
    pub fn allocator(self) -> Option<EregallocRegisterAllocatePass> {
        match self {
            RegallocEngine::Linear => None,
            RegallocEngine::Eregalloc { oracle, freq } => {
                let source = match freq {
                    FreqChoice::Uniform => BlockFreqSource::Uniform,
                    FreqChoice::Spectral => BlockFreqSource::Spectral,
                    FreqChoice::CmtProvider => BlockFreqSource::Provider(cmt_block_frequencies),
                };
                Some(EregallocRegisterAllocatePass::new(oracle, source))
            }
        }
    }
}

fn unknown(var: &str, value: &str, expected: &str) -> String {
    format!("unknown value `{value}` for {var}; expected one of: {expected}")
}

fn env(var: &str) -> Option<String> {
    std::env::var(var).ok().filter(|value| !value.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    // Env-var tests share process state; serialize them.
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn with_env(vars: &[(&str, &str)], f: impl FnOnce()) {
        let _guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());
        for var in ["CRABBIT_REGALLOC", "CRABBIT_REGALLOC_ORACLE", "CRABBIT_BLOCK_FREQ"] {
            unsafe { std::env::remove_var(var) };
        }
        for (k, v) in vars {
            unsafe { std::env::set_var(k, v) };
        }
        f();
        for (k, _) in vars {
            unsafe { std::env::remove_var(k) };
        }
    }

    #[test]
    fn defaults_to_linear() {
        with_env(&[], || {
            assert_eq!(RegallocEngine::from_env().unwrap(), RegallocEngine::Linear);
            assert!(RegallocEngine::Linear.allocator().is_none());
        });
    }

    #[test]
    fn eregalloc_variants_parse() {
        with_env(&[("CRABBIT_REGALLOC", "eregalloc")], || {
            assert_eq!(
                RegallocEngine::from_env().unwrap(),
                RegallocEngine::Eregalloc { oracle: OracleKind::Syntactic, freq: FreqChoice::Uniform }
            );
        });
        with_env(
            &[("CRABBIT_REGALLOC", "eregalloc"), ("CRABBIT_REGALLOC_ORACLE", "c2"), ("CRABBIT_BLOCK_FREQ", "spectral")],
            || {
                let engine = RegallocEngine::from_env().unwrap();
                assert_eq!(
                    engine,
                    RegallocEngine::Eregalloc { oracle: OracleKind::EGraph, freq: FreqChoice::Spectral }
                );
                assert!(engine.allocator().is_some());
            },
        );
    }

    #[test]
    fn oracle_is_ignored_under_linear_and_unknown_values_error() {
        with_env(&[("CRABBIT_REGALLOC_ORACLE", "bogus")], || {
            assert_eq!(RegallocEngine::from_env().unwrap(), RegallocEngine::Linear);
        });
        with_env(&[("CRABBIT_REGALLOC", "bogus")], || {
            let err = RegallocEngine::from_env().unwrap_err();
            assert!(err.contains("CRABBIT_REGALLOC") && err.contains("linear, eregalloc"), "{err}");
        });
        with_env(&[("CRABBIT_REGALLOC", "eregalloc"), ("CRABBIT_REGALLOC_ORACLE", "c1")], || {
            assert!(RegallocEngine::from_env().unwrap_err().contains("c0, c2"));
        });
    }
}
