//! Measured block execution frequencies from a perf-derived profile.
//!
//! The third block-frequency source next to `uniform` and
//! [spectral](crate::passes::spectral_freq) (docs/PROFILE-FEEDBACK-PLAN.md):
//! `CRABBIT_BLOCK_FREQ=profile` reads `CRABBIT_PROFILE=<profile.json>`, a
//! file produced by `scripts/perf-harness/profile_ingest.py` from `perf`
//! samples and the `CRABBIT_BLOCKMAP=1` sidecars. Shape:
//!
//! ```json
//! { "<function symbol>": [1.0, 12.5, 0.0, ...], ... }
//! ```
//!
//! One `f64` per machine block in RA-time region order, normalized the way
//! [spectral_frequencies](crate::passes::spectral_freq::spectral_frequencies)
//! normalizes: `freq[b] = samples[b] / samples[entry]`, entry = 1.0.
//! Determinism of the pipeline makes the RA-time CFG (and so the indices)
//! identical between the profiled build and the rebuild as long as
//! compiler, flags, and sources match.
//!
//! A profile is advisory: a missing file, unparsable JSON, an absent
//! symbol, a wrong-length vector, or non-finite/negative entries all fall
//! back to uniform frequencies for the affected function — a stale profile
//! must never fail a build.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

/// Environment variable naming the profile JSON file.
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
    let parsed = std::fs::read_to_string(path)
        .map_err(|error| error.to_string())
        .and_then(|text| {
            serde_json::from_str::<HashMap<String, Vec<f64>>>(&text)
                .map_err(|error| error.to_string())
        });
    match parsed {
        Ok(map) => map,
        Err(error) => {
            eprintln!(
                "crabbit: warning: {PROFILE_ENV} `{path}` is unusable ({error}); \
                 all functions fall back to uniform block frequencies"
            );
            HashMap::new()
        }
    }
}

/// The profile named by `CRABBIT_PROFILE`, or `None` when the variable is
/// unset/empty. An unreadable file loads as an empty profile (with a
/// warning), so every lookup falls back to uniform.
pub fn profile() -> Option<Profile> {
    let path = std::env::var(PROFILE_ENV).ok().filter(|value| !value.is_empty())?;
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
            Some(value) => unsafe { std::env::set_var(PROFILE_ENV, value) },
            None => unsafe { std::env::remove_var(PROFILE_ENV) },
        }
        f();
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
        with_profile_env(Some("/nonexistent/profile.json"), || {
            assert!(frequencies_for("anything", 3).is_none());
        });
    }

    #[test]
    fn matching_vector_is_returned_and_mismatches_fall_back() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("profile.json");
        std::fs::write(
            &path,
            r#"{"hot_fn": [1.0, 8.0, 0.5], "bad_fn": [1.0, -3.0]}"#,
        )
        .unwrap();
        with_profile_env(Some(path.to_str().unwrap()), || {
            assert_eq!(
                frequencies_for("hot_fn", 3),
                Some(vec![1.0, 8.0, 0.5]),
                "exact symbol + length match returns the measured vector"
            );
            assert!(frequencies_for("hot_fn", 4).is_none(), "wrong length");
            assert!(frequencies_for("cold_fn", 3).is_none(), "missing symbol");
            assert!(frequencies_for("bad_fn", 2).is_none(), "negative value");
        });
    }

    #[test]
    fn changing_the_env_path_reloads() {
        let dir = tempfile::tempdir().unwrap();
        let first = dir.path().join("first.json");
        let second = dir.path().join("second.json");
        std::fs::write(&first, r#"{"f": [1.0]}"#).unwrap();
        std::fs::write(&second, r#"{"f": [1.0, 2.0]}"#).unwrap();
        with_profile_env(Some(first.to_str().unwrap()), || {
            assert_eq!(frequencies_for("f", 1), Some(vec![1.0]));
        });
        with_profile_env(Some(second.to_str().unwrap()), || {
            assert_eq!(frequencies_for("f", 2), Some(vec![1.0, 2.0]));
        });
    }
}
