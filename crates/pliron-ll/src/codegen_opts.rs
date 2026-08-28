//! Research codegen options.
//!
//! crabbit doubles as a testbed for register-allocation and cost-model
//! research (the eregalloc availability-aware spilling line and the
//! combinatorial-matrix-theory analytical models). Their published numbers
//! come from synthetic corpora, so every policy here is an *option*, chosen
//! per build via environment variables, to let the same real program be
//! compiled under each policy and compared — no crabbit rebuild required.
//!
//! | variable | values | default |
//! |---|---|---|
//! | `CRABBIT_SPILL_POLICY` | `furthest`, `weighted` | `furthest` |
//! | `CRABBIT_RESTORE_ESTIMATE` | `reload`, `remat`, `egraph` | `reload` |
//! | `CRABBIT_BLOCK_FREQ` | `uniform`, `spectral` | `uniform` |
//!
//! `furthest` is the classic Poletto–Sarkar victim choice and the exact
//! pre-options behavior. `weighted` scores candidates by
//! `restore_estimate × Σ execution frequency of uses` and evicts the
//! cheapest — the shape of eregalloc's eviction score. `remat` prices
//! trivially rematerializable values (constant defs) below a reload and
//! also rewrites their restores to re-execute the def instead of loading
//! the slot — eregalloc's "safe form": store once, recompute at uses when
//! cheaper. `egraph` (the full availability-aware e-graph oracle) is
//! reserved and errors until the vendored oracle is wired in. `spectral`
//! computes analytical block frequencies via Perron–Frobenius power
//! iteration ([crate::passes::spectral_freq], from
//! combinatorial-matrix-theory Experiment A); `uniform` weights every
//! block equally.

use thiserror::Error;

use crate::{input_error_noloc, result::STAIRResult};

#[derive(Debug, Error)]
pub enum CodegenOptsErr {
    #[error("unknown value `{value}` for {var}; expected one of: {expected}")]
    UnknownValue {
        var: &'static str,
        value: String,
        expected: &'static str,
    },
    #[error("{0} is not wired in yet")]
    NotWired(&'static str),
}

/// How the linear scan picks a spill victim under pressure.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum SpillVictimPolicy {
    /// Evict the active interval that ends furthest away (baseline).
    #[default]
    FurthestEnd,
    /// Evict the candidate with the cheapest total restore cost:
    /// `restore_estimate × Σ freq(use)`.
    WeightedCost,
}

/// How the cost of re-obtaining a spilled value is estimated.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum RestoreEstimate {
    /// Every restore is a reload from the spill slot.
    #[default]
    Reload,
    /// Trivially rematerializable values (single-def constants) are priced
    /// below a reload and restored by re-executing their def.
    TrivialRemat,
    // Reserved: the availability-aware e-graph oracle (eregalloc C1/C2).
    // Parsing `egraph` errors until it is vendored and wired.
}

/// Where block execution frequencies come from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum BlockFreqModel {
    /// Every block weighs 1.
    #[default]
    Uniform,
    /// Perron–Frobenius stationary frequencies of the CFG random walk.
    Spectral,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CodegenOpts {
    pub victim: SpillVictimPolicy,
    pub restore: RestoreEstimate,
    pub freq: BlockFreqModel,
}

impl CodegenOpts {
    /// Parse the option environment variables; unset means default.
    pub fn from_env() -> STAIRResult<Self> {
        let victim = match env("CRABBIT_SPILL_POLICY").as_deref() {
            None | Some("furthest") => SpillVictimPolicy::FurthestEnd,
            Some("weighted") => SpillVictimPolicy::WeightedCost,
            Some(other) => {
                return Err(input_error_noloc!(CodegenOptsErr::UnknownValue {
                    var: "CRABBIT_SPILL_POLICY",
                    value: other.to_string(),
                    expected: "furthest, weighted",
                }));
            }
        };
        let restore = match env("CRABBIT_RESTORE_ESTIMATE").as_deref() {
            None | Some("reload") => RestoreEstimate::Reload,
            Some("remat") => RestoreEstimate::TrivialRemat,
            Some("egraph") => {
                return Err(input_error_noloc!(CodegenOptsErr::NotWired(
                    "CRABBIT_RESTORE_ESTIMATE=egraph (availability-aware e-graph oracle)"
                )));
            }
            Some(other) => {
                return Err(input_error_noloc!(CodegenOptsErr::UnknownValue {
                    var: "CRABBIT_RESTORE_ESTIMATE",
                    value: other.to_string(),
                    expected: "reload, remat, egraph",
                }));
            }
        };
        let freq = match env("CRABBIT_BLOCK_FREQ").as_deref() {
            None | Some("uniform") => BlockFreqModel::Uniform,
            Some("spectral") => BlockFreqModel::Spectral,
            Some(other) => {
                return Err(input_error_noloc!(CodegenOptsErr::UnknownValue {
                    var: "CRABBIT_BLOCK_FREQ",
                    value: other.to_string(),
                    expected: "uniform, spectral",
                }));
            }
        };
        Ok(CodegenOpts {
            victim,
            restore,
            freq,
        })
    }
}

fn env(var: &str) -> Option<String> {
    std::env::var(var).ok().filter(|value| !value.is_empty())
}
