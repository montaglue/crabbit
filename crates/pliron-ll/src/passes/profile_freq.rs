//! Measured block execution frequencies from a profiled trace.
//!
//! The third block-frequency source next to `uniform` and
//! [spectral](crate::passes::spectral_freq): `CRABBIT_BLOCK_FREQ=profile`
//! reads `CRABBIT_PROFILE=<trace>`, a trace with the profile of a measured
//! run (written by `crabbit-bpgo` from a compilation traced with
//! `CRABBIT_TRACE=1 CRABBIT_PROFILE_MAP=1`). The frequencies are how often
//! each block of the IR the allocator received ran, per function, in
//! region order — the same order and unit the spectral model uses:
//! normalized so `freq[entry] == 1.0`, with a block nothing was measured
//! in raised to half the least measured block of its function (a sampled
//! zero is a resolution floor, not "never executes"). Determinism of the
//! pipeline makes the allocator's CFG identical between the profiled build
//! and the rebuild as long as compiler, flags, and sources match.
//!
//! A profile is advisory: a missing or unprofiled trace, an absent symbol,
//! or a block count that does not match all fall back to uniform
//! frequencies for the affected function — a stale profile must never
//! fail a build.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

/// Environment variable naming the profiled trace.
pub const PROFILE_ENV: &str = "CRABBIT_PROFILE";

type Profile = Arc<HashMap<String, Vec<f64>>>;

/// Cache of the last loaded profile, keyed by the path it was loaded from
/// so tests (and long-lived processes) that change `CRABBIT_PROFILE` get
/// the file they asked for instead of a stale first read.
fn cache() -> &'static Mutex<Option<(String, Profile)>> {
    static CACHE: OnceLock<Mutex<Option<(String, Profile)>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(None))
}

fn load(path: &str) -> HashMap<String, Vec<f64>> {
    use crate::passes::aarch64::backward::{ALLOCATOR_INPUT, block_executions, profiled_snapshot};
    match profiled_snapshot(std::path::Path::new(path), ALLOCATOR_INPUT) {
        Ok(snapshot) => block_executions(&snapshot)
            .into_iter()
            .map(|(symbol, runs)| (symbol, normalize(&runs)))
            .collect(),
        Err(error) => {
            // eprintln! rather than log::warn!: this runs inside a rustc codegen
            // dylib where no logger is installed; the warning must reach the user.
            eprintln!(
                "crabbit: warning: {PROFILE_ENV} `{path}` is unusable ({error}); \
                 all functions fall back to uniform block frequencies"
            );
            HashMap::new()
        }
    }
}

/// Runs per block → frequencies relative to the entry block, unmeasured
/// blocks raised to half the least measured one.
fn normalize(runs: &[f64]) -> Vec<f64> {
    let least = runs
        .iter()
        .copied()
        .filter(|&r| r > 0.0)
        .fold(f64::INFINITY, f64::min);
    if !least.is_finite() {
        return vec![1.0; runs.len()];
    }
    let floored: Vec<f64> = runs.iter().map(|&r| r.max(least / 2.0)).collect();
    let entry = floored.first().copied().unwrap_or(1.0);
    floored.iter().map(|r| r / entry).collect()
}

/// The profile named by `CRABBIT_PROFILE`, or `None` when the variable is
/// unset/empty. An unreadable file loads as an empty profile (with a
/// warning), so every lookup falls back to uniform.
pub fn profile() -> Option<Profile> {
    let path = std::env::var(PROFILE_ENV)
        .ok()
        .filter(|value| !value.is_empty())?;
    let mut cached = cache().lock().unwrap_or_else(|poison| poison.into_inner());
    if let Some((cached_path, profile)) = cached.as_ref()
        && *cached_path == path
    {
        return Some(profile.clone());
    }
    let profile: Profile = Arc::new(load(&path));
    *cached = Some((path, profile.clone()));
    Some(profile)
}

/// The measured per-block frequency vector for `symbol`, if the profile
/// has one that fits a function with `blocks` machine blocks. `None` means
/// "fall back to uniform": no profile configured, symbol absent, length
/// mismatch, or non-finite/negative values.
pub fn frequencies_for(symbol: &str, blocks: usize) -> Option<Vec<f64>> {
    let profile = profile()?;
    let freqs = profile.get(symbol)?;
    if freqs.len() != blocks || freqs.iter().any(|freq| !freq.is_finite() || *freq < 0.0) {
        return None;
    }
    Some(freqs.clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    // Env-var tests share process state; serialize them. Other tests that
    // read CRABBIT_PROFILE only exercise the fallback path (symbols absent
    // from any fixture written here), so this mutex only needs to cover
    // the tests below.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn with_profile_env(value: Option<&str>, f: impl FnOnce()) {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
        match value {
            // SAFETY: single-threaded here (test/serialized-by-lock env scope);
            // no other thread reads the environment concurrently.
            Some(value) => unsafe { std::env::set_var(PROFILE_ENV, value) },
            // SAFETY: single-threaded here (test/serialized-by-lock env scope);
            // no other thread reads the environment concurrently.
            None => unsafe { std::env::remove_var(PROFILE_ENV) },
        }
        f();
        // SAFETY: single-threaded here (test/serialized-by-lock env scope);
        // no other thread reads the environment concurrently.
        unsafe { std::env::remove_var(PROFILE_ENV) };
    }

    #[test]
    fn unset_env_means_no_profile() {
        with_profile_env(None, || {
            assert!(frequencies_for("anything", 3).is_none());
        });
    }

    #[test]
    fn missing_file_falls_back_to_uniform_lookups() {
        with_profile_env(Some("/nonexistent/profile.bpgo.crt"), || {
            assert!(frequencies_for("anything", 3).is_none());
        });
    }

    #[test]
    fn runs_normalize_to_the_entry_with_a_floor() {
        assert_eq!(
            normalize(&[100.0, 800.0, 0.0, 50.0]),
            vec![1.0, 8.0, 0.25, 0.5]
        );
        assert_eq!(normalize(&[0.0, 0.0]), vec![1.0, 1.0]);
    }
}
