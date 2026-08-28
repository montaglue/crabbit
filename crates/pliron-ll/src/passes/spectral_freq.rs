//! Spectral block execution frequencies.
//!
//! Ported from combinatorial-matrix-theory Experiment A: with uniform
//! branch priors the CFG is a row-substochastic transition matrix; closing
//! the chain with exit→entry restart edges makes it irreducible, so
//! Perron–Frobenius gives a unique stationary distribution π, found by
//! power iteration on the lazy chain `(P + I) / 2` (lazy for
//! aperiodicity). The compiler-meaningful quantity is the
//! renewal-normalized `freq(b) = π(b) / π(entry)`: expected executions of
//! `b` per function invocation. Entry gets 1, the two sides of a diamond
//! get 0.5 each, a loop body with backedge probability p gets `1/(1−p)`.
//!
//! This is purely analytical — no profile, no measurement — which is what
//! lets it ride in the backend as a cost-model option (see
//! [crate::codegen_opts]). It is deliberately graph-shaped (successor
//! lists in, frequencies out) so any backend can feed it its machine CFG.

/// Expected executions of each block per function invocation.
///
/// `successors[i]` lists the successor indices of block `i`; `entry` is the
/// function entry block. Blocks unreachable from `entry` get frequency 0.
pub fn spectral_frequencies(successors: &[Vec<usize>], entry: usize) -> Vec<f64> {
    let n = successors.len();
    if n == 0 {
        return Vec::new();
    }
    assert!(entry < n, "entry block index out of range");

    // π starts with all mass at entry; one step of the closed chain moves
    // mass uniformly over successors, exits (and any dead-end) restart at
    // entry. The lazy mix keeps periodic CFGs (e.g. a two-block cycle)
    // converging.
    let mut pi = vec![0.0f64; n];
    pi[entry] = 1.0;
    let mut next = vec![0.0f64; n];
    for _ in 0..10_000 {
        next.fill(0.0);
        for (block, succs) in successors.iter().enumerate() {
            let mass = pi[block];
            if mass == 0.0 {
                continue;
            }
            if succs.is_empty() {
                next[entry] += mass;
            } else {
                let share = mass / succs.len() as f64;
                for &succ in succs {
                    next[succ] += share;
                }
            }
        }
        let mut delta = 0.0;
        for i in 0..n {
            let lazy = 0.5 * pi[i] + 0.5 * next[i];
            delta += (lazy - pi[i]).abs();
            next[i] = lazy;
        }
        std::mem::swap(&mut pi, &mut next);
        if delta < 1e-12 {
            break;
        }
    }

    let entry_mass = pi[entry];
    if entry_mass <= 0.0 {
        // Degenerate chain (should not happen with the restart closure):
        // fall back to uniform weights so consumers stay well-defined.
        return vec![1.0; n];
    }
    pi.iter().map(|&mass| mass / entry_mass).collect()
}

#[cfg(test)]
mod tests {
    use super::spectral_frequencies;

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1e-6,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn straight_line_is_all_ones() {
        let freqs = spectral_frequencies(&[vec![1], vec![2], vec![]], 0);
        for freq in freqs {
            assert_close(freq, 1.0);
        }
    }

    #[test]
    fn diamond_sides_get_half() {
        // 0 → {1, 2} → 3
        let freqs = spectral_frequencies(&[vec![1, 2], vec![3], vec![3], vec![]], 0);
        assert_close(freqs[0], 1.0);
        assert_close(freqs[1], 0.5);
        assert_close(freqs[2], 0.5);
        assert_close(freqs[3], 1.0);
    }

    #[test]
    fn loop_body_gets_expected_trip_count() {
        // 0 → 1; 1 → {1, 2}: backedge probability 0.5 → E[visits] = 2.
        let freqs = spectral_frequencies(&[vec![1], vec![1, 2], vec![]], 0);
        assert_close(freqs[0], 1.0);
        assert_close(freqs[1], 2.0);
        assert_close(freqs[2], 1.0);
    }

    #[test]
    fn unreachable_block_gets_zero() {
        let freqs = spectral_frequencies(&[vec![], vec![0]], 0);
        assert_close(freqs[0], 1.0);
        assert_close(freqs[1], 0.0);
    }
}
