//! Runtime checks for the full 128-bit integer surface — div/rem (libcalls),
//! mul-with-overflow, add/sub overflow, saturating ops, rotates, dynamic
//! shifts, ctpop/ctlz/cttz/bswap/bitreverse, int<->float casts with Rust's
//! saturating `as` semantics, and formatting (which itself exercises 128-bit
//! division) — plus the iterator/collect paths (uninhabited-enum
//! discriminants, Box unsize coercions, unsized-tail fat references) the
//! importer lowers.
//!
//! Every expected value is a constant the host rustc computed (or a literal
//! checked against normal rustc), and operands are laundered through
//! `black_box` where const-folding could otherwise hide the runtime lowering.

use std::hint::black_box;

#[inline(never)]
fn checked_mul_u128(a: u128, b: u128) -> Option<u128> {
    a.checked_mul(b)
}

#[inline(never)]
fn checked_mul_i128(a: i128, b: i128) -> Option<i128> {
    a.checked_mul(b)
}

#[inline(never)]
fn overflowing_mul_u128(a: u128, b: u128) -> (u128, bool) {
    a.overflowing_mul(b)
}

#[inline(never)]
fn overflowing_mul_i128(a: i128, b: i128) -> (i128, bool) {
    a.overflowing_mul(b)
}

#[inline(never)]
fn trailing_zeros_u128(x: u128) -> u32 {
    x.trailing_zeros()
}

#[inline(never)]
fn filter_pos(v: Vec<i64>) -> Vec<i64> {
    // Vec's in-place collect: try_fold through `Result<_, !>` and the
    // `Result<Infallible, !>` uninhabited-discriminant path.
    v.into_iter().filter(|x| *x > 0).collect()
}

fn mul_cases() {
    // u128 checked_mul edge cases.
    assert!(checked_mul_u128(u128::MAX, 1) == Some(u128::MAX), "u128 MAX*1");
    assert!(checked_mul_u128(u128::MAX, 2).is_none(), "u128 MAX*2");
    assert!(checked_mul_u128(1u128 << 64, 1u128 << 64).is_none(), "u128 2^64*2^64");
    assert!(
        checked_mul_u128((1u128 << 64) - 1, (1u128 << 64) + 1) == Some(u128::MAX),
        "u128 (2^64-1)*(2^64+1)"
    );
    assert!(checked_mul_u128(0, u128::MAX) == Some(0), "u128 0*MAX");
    assert!(checked_mul_u128(3, 5) == Some(15), "u128 3*5");
    assert!(
        checked_mul_u128(1u128 << 63, 1u128 << 63) == Some(1u128 << 126),
        "u128 2^63*2^63"
    );
    assert!(
        checked_mul_u128(1u128 << 63, 1u128 << 64) == Some(1u128 << 127),
        "u128 2^63*2^64"
    );
    assert!(
        checked_mul_u128((1u128 << 64) + 1, (1u128 << 64) + 1).is_none(),
        "u128 (2^64+1)^2"
    );
    assert!(
        checked_mul_u128((1u128 << 127) - 1, 2) == Some(u128::MAX - 1),
        "u128 (2^127-1)*2"
    );
    assert!(
        overflowing_mul_u128(u128::MAX, 2) == (u128::MAX - 1, true),
        "u128 MAX*2 wrapped"
    );
    assert!(
        overflowing_mul_u128(1u128 << 64, 1u128 << 64) == (0, true),
        "u128 2^64*2^64 wrapped"
    );

    // i128 checked_mul edge cases.
    assert!(checked_mul_i128(i128::MIN, -1).is_none(), "i128 MIN*-1");
    assert!(checked_mul_i128(i128::MAX, i128::MAX).is_none(), "i128 MAX*MAX");
    assert!(checked_mul_i128(i128::MAX, 1) == Some(i128::MAX), "i128 MAX*1");
    assert!(checked_mul_i128(i128::MIN, 1) == Some(i128::MIN), "i128 MIN*1");
    assert!(checked_mul_i128(i128::MIN, -2).is_none(), "i128 MIN*-2");
    assert!(checked_mul_i128(-3, 5) == Some(-15), "i128 -3*5");
    assert!(checked_mul_i128(-4, -5) == Some(20), "i128 -4*-5");
    assert!(checked_mul_i128(1i128 << 126, 2).is_none(), "i128 2^126*2");
    assert!(
        checked_mul_i128(i128::MIN / 2, 2) == Some(i128::MIN),
        "i128 (MIN/2)*2"
    );
    assert!(checked_mul_i128(i128::MIN / 2, -2).is_none(), "i128 (MIN/2)*-2");
    assert!(
        checked_mul_i128(-(1i128 << 64), 1i128 << 63) == Some(i128::MIN),
        "i128 -2^64*2^63"
    );
    assert!(checked_mul_i128(1i128 << 64, 1i128 << 63).is_none(), "i128 2^64*2^63");
    assert!(checked_mul_i128(7, 6) == Some(42), "i128 7*6");
    assert!(
        overflowing_mul_i128(i128::MIN, -1) == (i128::MIN, true),
        "i128 MIN*-1 wrapped"
    );
}

fn cttz_cases() {
    assert!(trailing_zeros_u128(0) == 128, "cttz 0");
    assert!(trailing_zeros_u128(1) == 0, "cttz 1");
    assert!(trailing_zeros_u128(1u128 << 64) == 64, "cttz 2^64");
    assert!(trailing_zeros_u128(1u128 << 127) == 127, "cttz 2^127");
    assert!(
        trailing_zeros_u128((0xABu128 << 70) | (1u128 << 12)) == 12,
        "cttz mixed low"
    );
    assert!(trailing_zeros_u128(0xF0u128 << 64) == 68, "cttz high 0xF0");
}

fn collect_cases() {
    // `vec![...]` exercises the `Box<[T; N]> -> Box<[T]>` unsize coercion.
    let v = vec![3i64, -1, 0, 7, -5, 42];
    assert!(v.len() == 6 && v[3] == 7, "vec! literal");
    let f = filter_pos(v);
    assert!(f == vec![3, 7, 42], "filter collect");
    let d: Vec<i64> = vec![1i64, 2, 3].into_iter().map(|x| x * 2).collect();
    assert!(d == vec![2, 4, 6], "map collect");

    // Array by-value iteration: fat references to core's PolymorphicIter
    // (an ADT with an unsized slice tail).
    let mut sum = 0i64;
    for x in [5i64, 6, 7] {
        sum += x;
    }
    assert!(sum == 18, "array for loop");
    let s: i64 = [4i64, 5, 6].into_iter().map(|x| x + 1).sum();
    assert!(s == 18, "array intoiter map");
}

#[inline(never)]
fn div_u128(a: u128, b: u128) -> u128 {
    a / b
}

#[inline(never)]
fn rem_u128(a: u128, b: u128) -> u128 {
    a % b
}

#[inline(never)]
fn div_i128(a: i128, b: i128) -> i128 {
    a / b
}

#[inline(never)]
fn rem_i128(a: i128, b: i128) -> i128 {
    a % b
}

fn div_rem_cases() {
    // Unsigned division and remainder (the `__udivti3`/`__umodti3` calls).
    assert!(div_u128(black_box(u128::MAX), black_box(3)) == const { u128::MAX / 3 }, "u128 MAX/3");
    assert!(rem_u128(black_box(u128::MAX), black_box(3)) == 0, "u128 MAX%3");
    assert!(
        div_u128(black_box(u128::MAX), black_box(u128::MAX)) == 1,
        "u128 MAX/MAX"
    );
    assert!(
        div_u128(black_box(u128::MAX - 1), black_box(u128::MAX)) == 0,
        "u128 (MAX-1)/MAX"
    );
    assert!(
        div_u128(black_box(1u128 << 127), black_box((1u128 << 64) - 1))
            == const { (1u128 << 127) / ((1u128 << 64) - 1) },
        "u128 2^127/(2^64-1)"
    );
    assert!(
        rem_u128(black_box(1u128 << 127), black_box((1u128 << 64) - 1))
            == const { (1u128 << 127) % ((1u128 << 64) - 1) },
        "u128 2^127%(2^64-1)"
    );
    assert!(div_u128(black_box(12345), black_box(7)) == 1763, "u128 12345/7");

    // Signed division and remainder (`__divti3`/`__modti3`), including the
    // C-truncation sign rules Rust shares.
    assert!(div_i128(black_box(i128::MIN), black_box(1)) == i128::MIN, "i128 MIN/1");
    assert!(div_i128(black_box(i128::MAX), black_box(3)) == const { i128::MAX / 3 }, "i128 MAX/3");
    assert!(rem_i128(black_box(i128::MAX), black_box(3)) == 1, "i128 MAX%3");
    assert!(div_i128(black_box(-7), black_box(3)) == -2, "i128 -7/3");
    assert!(rem_i128(black_box(-7), black_box(3)) == -1, "i128 -7%3");
    assert!(div_i128(black_box(7), black_box(-3)) == -2, "i128 7/-3");
    assert!(rem_i128(black_box(7), black_box(-3)) == 1, "i128 7%-3");
    assert!(div_i128(black_box(-7), black_box(-3)) == 2, "i128 -7/-3");
    assert!(rem_i128(black_box(-7), black_box(-3)) == -1, "i128 -7%-3");
    assert!(
        div_i128(black_box(i128::MIN), black_box(i128::MAX)) == -1,
        "i128 MIN/MAX"
    );
    assert!(
        rem_i128(black_box(i128::MIN), black_box(i128::MAX)) == -1,
        "i128 MIN%MAX"
    );
    assert!(
        div_i128(black_box(i128::MIN + 1), black_box(-1)) == i128::MAX,
        "i128 (MIN+1)/-1"
    );

    // MIN / -1 is the one overflowing pair; core routes it around the
    // division, so only the wrapping/checked wrappers can reach it.
    assert!(
        black_box(i128::MIN).wrapping_div(black_box(-1)) == i128::MIN,
        "i128 MIN wrapping_div -1"
    );
    assert!(
        black_box(i128::MIN).checked_div(black_box(-1)).is_none(),
        "i128 MIN checked_div -1"
    );
    assert!(
        black_box(i128::MIN).wrapping_rem(black_box(-1)) == 0,
        "i128 MIN wrapping_rem -1"
    );
    assert!(black_box(7i128).checked_div(black_box(0)).is_none(), "i128 7/0");
    assert!(black_box(7u128).checked_rem(black_box(0)).is_none(), "u128 7%0");
}

#[inline(never)]
fn rotl_u128(x: u128, n: u32) -> u128 {
    x.rotate_left(n)
}

#[inline(never)]
fn rotr_u128(x: u128, n: u32) -> u128 {
    x.rotate_right(n)
}

#[inline(never)]
fn rotl_i128(x: i128, n: u32) -> i128 {
    x.rotate_left(n)
}

const P: u128 = 0x0123_4567_89AB_CDEF_FEDC_BA98_7654_3210;
const AMOUNTS: [u32; 7] = [0, 1, 63, 64, 65, 127, 128];

fn rotate_cases() {
    // Expected values are CTFE-computed by the host rustc, so the runtime
    // lowering is checked against an independent implementation.
    const EXPECTED: [(u128, u128, i128); AMOUNTS.len()] = {
        let mut out = [(0u128, 0u128, 0i128); AMOUNTS.len()];
        let mut i = 0;
        while i < AMOUNTS.len() {
            out[i] = (
                P.rotate_left(AMOUNTS[i]),
                P.rotate_right(AMOUNTS[i]),
                (P as i128).rotate_left(AMOUNTS[i]),
            );
            i += 1;
        }
        out
    };
    for (i, &n) in AMOUNTS.iter().enumerate() {
        assert!(
            rotl_u128(black_box(P), black_box(n)) == EXPECTED[i].0,
            "u128 rotate_left"
        );
        assert!(
            rotr_u128(black_box(P), black_box(n)) == EXPECTED[i].1,
            "u128 rotate_right"
        );
        assert!(
            rotl_i128(black_box(P as i128), black_box(n)) == EXPECTED[i].2,
            "i128 rotate_left"
        );
    }
    assert!(rotl_u128(black_box(1), black_box(127)) == 1u128 << 127, "rotl 1 by 127");
    assert!(rotl_u128(black_box(1u128 << 127), black_box(1)) == 1, "rotl top by 1");
    assert!(rotr_u128(black_box(1), black_box(1)) == 1u128 << 127, "rotr 1 by 1");
}

#[inline(never)]
fn shl_u128(x: u128, n: u32) -> u128 {
    x << n
}

#[inline(never)]
fn shr_u128(x: u128, n: u32) -> u128 {
    x >> n
}

#[inline(never)]
fn shr_i128(x: i128, n: u32) -> i128 {
    x >> n
}

fn shift_cases() {
    const SHIFTS: [u32; 6] = [0, 1, 63, 64, 65, 127];
    const EXPECTED: [(u128, u128, i128); SHIFTS.len()] = {
        let mut out = [(0u128, 0u128, 0i128); SHIFTS.len()];
        let mut i = 0;
        while i < SHIFTS.len() {
            out[i] = (P << SHIFTS[i], P >> SHIFTS[i], (P as i128) >> SHIFTS[i]);
            i += 1;
        }
        out
    };
    for (i, &n) in SHIFTS.iter().enumerate() {
        assert!(shl_u128(black_box(P), black_box(n)) == EXPECTED[i].0, "u128 shl");
        assert!(shr_u128(black_box(P), black_box(n)) == EXPECTED[i].1, "u128 lshr");
        assert!(
            shr_i128(black_box(P as i128), black_box(n)) == EXPECTED[i].2,
            "i128 ashr"
        );
        assert!(
            shr_i128(black_box(-1), black_box(n)) == -1,
            "i128 ashr all-ones"
        );
    }
}

#[inline(never)]
fn popcount_u128(x: u128) -> u32 {
    x.count_ones()
}

#[inline(never)]
fn leading_zeros_u128(x: u128) -> u32 {
    x.leading_zeros()
}

#[inline(never)]
fn bswap_u128(x: u128) -> u128 {
    x.swap_bytes()
}

#[inline(never)]
fn bswap_i128(x: i128) -> i128 {
    x.swap_bytes()
}

#[inline(never)]
fn bitrev_u128(x: u128) -> u128 {
    x.reverse_bits()
}

fn bit_cases() {
    const INPUTS: [u128; 7] = [0, 1, u128::MAX, 1 << 64, 1 << 127, P, 0xF0u128 << 100];
    const EXPECTED: [(u32, u32, u32, u128, u128); INPUTS.len()] = {
        let mut out = [(0u32, 0u32, 0u32, 0u128, 0u128); INPUTS.len()];
        let mut i = 0;
        while i < INPUTS.len() {
            out[i] = (
                INPUTS[i].count_ones(),
                INPUTS[i].leading_zeros(),
                INPUTS[i].trailing_zeros(),
                INPUTS[i].swap_bytes(),
                INPUTS[i].reverse_bits(),
            );
            i += 1;
        }
        out
    };
    for (i, &x) in INPUTS.iter().enumerate() {
        assert!(popcount_u128(black_box(x)) == EXPECTED[i].0, "u128 ctpop");
        assert!(
            leading_zeros_u128(black_box(x)) == EXPECTED[i].1,
            "u128 ctlz"
        );
        assert!(
            trailing_zeros_u128(black_box(x)) == EXPECTED[i].2,
            "u128 cttz"
        );
        assert!(bswap_u128(black_box(x)) == EXPECTED[i].3, "u128 bswap");
        assert!(bitrev_u128(black_box(x)) == EXPECTED[i].4, "u128 bitreverse");
    }
    assert!(
        bswap_u128(black_box(P)) == 0x1032_5476_98BA_DCFE_EFCD_AB89_6745_2301,
        "u128 bswap pattern"
    );
    assert!(
        bswap_i128(black_box(-2i128)) == const { (-2i128).swap_bytes() },
        "i128 bswap -2"
    );
    assert!(bswap_i128(black_box(1i128)) == 1i128 << 120, "i128 bswap 1");
    assert!(bitrev_u128(black_box(1)) == 1u128 << 127, "bitrev 1");
    assert!(bitrev_u128(black_box(0b1011)) == 0b1101u128 << 124, "bitrev 0b1011");
    // Signed operands share the unsigned expansions bit-for-bit.
    assert!(black_box(-1i128).count_ones() == 128, "i128 ctpop -1");
    assert!(black_box(-1i128).trailing_zeros() == 0, "i128 cttz -1");
    assert!(black_box(i128::MIN).leading_zeros() == 0, "i128 ctlz MIN");
    assert!(black_box(i128::MIN).trailing_zeros() == 127, "i128 cttz MIN");
    assert!(black_box(-1i128).rotate_right(black_box(7)) == -1, "i128 rotr -1");
    assert!(
        black_box(1i128).reverse_bits() == i128::MIN,
        "i128 bitrev 1"
    );
}

#[inline(never)]
fn sat_add_i128(a: i128, b: i128) -> i128 {
    a.saturating_add(b)
}

#[inline(never)]
fn sat_sub_i128(a: i128, b: i128) -> i128 {
    a.saturating_sub(b)
}

#[inline(never)]
fn sat_add_u128(a: u128, b: u128) -> u128 {
    a.saturating_add(b)
}

#[inline(never)]
fn sat_sub_u128(a: u128, b: u128) -> u128 {
    a.saturating_sub(b)
}

fn saturating_cases() {
    assert!(sat_add_i128(black_box(i128::MAX), black_box(1)) == i128::MAX, "sat MAX+1");
    assert!(
        sat_add_i128(black_box(i128::MAX), black_box(i128::MAX)) == i128::MAX,
        "sat MAX+MAX"
    );
    assert!(sat_add_i128(black_box(i128::MIN), black_box(-1)) == i128::MIN, "sat MIN+-1");
    assert!(sat_add_i128(black_box(-5), black_box(3)) == -2, "sat -5+3");
    assert!(sat_sub_i128(black_box(i128::MIN), black_box(1)) == i128::MIN, "sat MIN-1");
    assert!(sat_sub_i128(black_box(i128::MAX), black_box(-1)) == i128::MAX, "sat MAX--1");
    assert!(sat_sub_i128(black_box(10), black_box(25)) == -15, "sat 10-25");
    assert!(
        sat_sub_i128(black_box(i128::MIN), black_box(i128::MAX)) == i128::MIN,
        "sat MIN-MAX"
    );
    assert!(sat_add_u128(black_box(u128::MAX), black_box(1)) == u128::MAX, "usat MAX+1");
    assert!(sat_add_u128(black_box(7), black_box(8)) == 15, "usat 7+8");
    assert!(sat_sub_u128(black_box(0), black_box(1)) == 0, "usat 0-1");
    assert!(sat_sub_u128(black_box(15), black_box(8)) == 7, "usat 15-8");
    // Signed saturating on a narrower width shares the same lowering.
    assert!(black_box(i8::MAX).saturating_add(black_box(1i8)) == i8::MAX, "sat i8");
    assert!(black_box(i8::MIN).saturating_sub(black_box(1i8)) == i8::MIN, "sat i8 sub");
}

#[inline(never)]
fn checked_add_i128(a: i128, b: i128) -> Option<i128> {
    a.checked_add(b)
}

#[inline(never)]
fn overflowing_sub_u128(a: u128, b: u128) -> (u128, bool) {
    a.overflowing_sub(b)
}

fn overflow_cases() {
    assert!(checked_add_i128(black_box(i128::MAX), black_box(1)).is_none(), "MAX+1");
    assert!(
        checked_add_i128(black_box(i128::MIN), black_box(-1)).is_none(),
        "MIN+-1"
    );
    assert!(
        checked_add_i128(black_box(i128::MAX), black_box(-1)) == Some(i128::MAX - 1),
        "MAX+-1"
    );
    assert!(
        black_box(i128::MIN).overflowing_add(black_box(i128::MIN)) == (0, true),
        "MIN+MIN wrapped"
    );
    assert!(
        black_box(i128::MIN).checked_sub(black_box(1)).is_none(),
        "MIN-1"
    );
    assert!(
        overflowing_sub_u128(black_box(0), black_box(1)) == (u128::MAX, true),
        "u128 0-1 wrapped"
    );
    assert!(
        black_box(u128::MAX).overflowing_add(black_box(1)) == (0, true),
        "u128 MAX+1 wrapped"
    );
    assert!(
        black_box(1u128 << 127).overflowing_add(black_box(1u128 << 127)) == (0, true),
        "u128 2^127+2^127 wrapped"
    );
    assert!(black_box(i128::MIN).wrapping_neg() == i128::MIN, "MIN neg wrapped");
    assert!(black_box(i128::MIN).checked_neg().is_none(), "MIN checked_neg");
    assert!(-black_box(i128::MAX) == i128::MIN + 1, "neg MAX");
    assert!(!black_box(0u128) == u128::MAX, "not 0");
    assert!(black_box(i128::MIN).checked_abs().is_none(), "MIN abs");
    assert!(black_box(-5i128).abs() == 5, "abs -5");
}

#[inline(never)]
fn i128_to_f64(x: i128) -> f64 {
    x as f64
}

#[inline(never)]
fn u128_to_f64(x: u128) -> f64 {
    x as f64
}

#[inline(never)]
fn i128_to_f32(x: i128) -> f32 {
    x as f32
}

#[inline(never)]
fn u128_to_f32(x: u128) -> f32 {
    x as f32
}

#[inline(never)]
fn f64_to_i128(x: f64) -> i128 {
    x as i128
}

#[inline(never)]
fn f64_to_u128(x: f64) -> u128 {
    x as u128
}

#[inline(never)]
fn f32_to_i128(x: f32) -> i128 {
    x as i128
}

#[inline(never)]
fn f32_to_u128(x: f32) -> u128 {
    x as u128
}

fn cast_cases() {
    // int -> float: rustc's CTFE computes every right-hand side.
    const MAX_I_F64: f64 = i128::MAX as f64;
    const MIN_I_F64: f64 = i128::MIN as f64;
    const MAX_U_F64: f64 = u128::MAX as f64;
    const P: u128 = 0x0123_4567_89AB_CDEF_FEDC_BA98_7654_3210;
    assert!(i128_to_f64(black_box(i128::MAX)) == MAX_I_F64, "i128 MAX as f64");
    assert!(i128_to_f64(black_box(i128::MIN)) == MIN_I_F64, "i128 MIN as f64");
    assert!(i128_to_f64(black_box(0)) == 0.0, "0i128 as f64");
    assert!(i128_to_f64(black_box(-7)) == -7.0, "-7i128 as f64");
    assert!(u128_to_f64(black_box(u128::MAX)) == MAX_U_F64, "u128 MAX as f64");
    assert!(
        u128_to_f64(black_box(1u128 << 64)) == const { (1u128 << 64) as f64 },
        "2^64 as f64"
    );
    assert!(u128_to_f64(black_box(P)) == const { P as f64 }, "pattern as f64");
    assert!(
        i128_to_f32(black_box(i128::MAX)) == const { i128::MAX as f32 },
        "i128 MAX as f32"
    );
    assert!(
        i128_to_f32(black_box(-(1i128 << 100))) == const { -(1i128 << 100) as f32 },
        "-2^100 as f32"
    );
    assert!(
        u128_to_f32(black_box(u128::MAX)) == const { u128::MAX as f32 },
        "u128 MAX as f32"
    );
    assert!(u128_to_f32(black_box(12345)) == 12345.0, "12345u128 as f32");

    // float -> int, Rust saturating `as` semantics. 2^127 and 2^128 are the
    // saturation pivots; NaN goes to zero.
    const TWO_127: f64 = (1u128 << 127) as f64;
    assert!(f64_to_i128(black_box(123456789.75)) == 123456789, "f64 trunc");
    assert!(f64_to_i128(black_box(-2.5)) == -2, "f64 -2.5 trunc");
    assert!(f64_to_i128(black_box(TWO_127)) == i128::MAX, "f64 2^127 sat");
    assert!(f64_to_i128(black_box(-TWO_127)) == i128::MIN, "f64 -2^127 exact");
    assert!(f64_to_i128(black_box(f64::MAX)) == i128::MAX, "f64 MAX sat");
    assert!(f64_to_i128(black_box(f64::MIN)) == i128::MIN, "f64 MIN sat");
    assert!(f64_to_i128(black_box(f64::INFINITY)) == i128::MAX, "f64 inf sat");
    assert!(
        f64_to_i128(black_box(f64::NEG_INFINITY)) == i128::MIN,
        "f64 -inf sat"
    );
    assert!(f64_to_i128(black_box(f64::NAN)) == 0, "f64 NaN");
    const JUST_BELOW: f64 = ((1u128 << 127) - (1u128 << 74)) as f64;
    assert!(
        f64_to_i128(black_box(JUST_BELOW)) == i128::MAX - ((1i128 << 74) - 1),
        "f64 just-below-2^127"
    );
    assert!(f64_to_u128(black_box(TWO_127)) == 1u128 << 127, "f64 2^127 exact u128");
    assert!(
        f64_to_u128(black_box(TWO_127 * 2.0)) == u128::MAX,
        "f64 2^128 sat u128"
    );
    assert!(f64_to_u128(black_box(f64::MAX)) == u128::MAX, "f64 MAX sat u128");
    assert!(f64_to_u128(black_box(-1.0)) == 0, "f64 -1 u128");
    assert!(f64_to_u128(black_box(-0.75)) == 0, "f64 -0.75 u128");
    assert!(f64_to_u128(black_box(0.75)) == 0, "f64 0.75 u128");
    assert!(f64_to_u128(black_box(f64::NAN)) == 0, "f64 NaN u128");
    assert!(
        f64_to_u128(black_box(f64::NEG_INFINITY)) == 0,
        "f64 -inf u128"
    );
    assert!(f32_to_i128(black_box(123.9f32)) == 123, "f32 trunc");
    assert!(f32_to_i128(black_box(f32::MAX)) == i128::MAX, "f32 MAX sat");
    assert!(f32_to_i128(black_box(f32::MIN)) == i128::MIN, "f32 MIN sat");
    assert!(f32_to_i128(black_box(f32::NAN)) == 0, "f32 NaN");
    assert!(
        f32_to_i128(black_box(f32::NEG_INFINITY)) == i128::MIN,
        "f32 -inf sat"
    );
    assert!(
        f32_to_u128(black_box(f32::MAX)) == const { f32::MAX as u128 },
        "f32 MAX u128"
    );
    assert!(f32_to_u128(black_box(f32::INFINITY)) == u128::MAX, "f32 inf u128");
    assert!(f32_to_u128(black_box(-1.5f32)) == 0, "f32 -1.5 u128");
    assert!(
        f32_to_u128(black_box(1.5e30f32)) == const { 1.5e30f32 as u128 },
        "f32 1.5e30 u128"
    );

    // Round-trips that stay exact.
    assert!(
        f64_to_i128(black_box(i128_to_f64(black_box(1i128 << 100)))) == 1i128 << 100,
        "2^100 round trip"
    );
    assert!(
        f64_to_u128(black_box(u128_to_f64(black_box(1u128 << 90)))) == 1u128 << 90,
        "2^90 round trip"
    );
}

#[inline(never)]
fn fmt_i128(x: i128) -> String {
    format!("{x}")
}

#[inline(never)]
fn fmt_u128_hex(x: u128) -> String {
    format!("{x:x}")
}

fn fmt_cases() {
    // Display for 128-bit ints runs real 128-bit division/remainder chains.
    assert!(
        fmt_i128(black_box(i128::MIN)) == "-170141183460469231731687303715884105728",
        "fmt i128 MIN"
    );
    assert!(
        fmt_i128(black_box(i128::MAX)) == "170141183460469231731687303715884105727",
        "fmt i128 MAX"
    );
    assert!(fmt_i128(black_box(0)) == "0", "fmt 0");
    assert!(fmt_i128(black_box(-42)) == "-42", "fmt -42");
    assert!(
        fmt_u128_hex(black_box(u128::MAX)) == "ffffffffffffffffffffffffffffffff",
        "fmt u128 MAX hex"
    );
    assert!(
        fmt_u128_hex(black_box(0x0123_4567_89AB_CDEF_FEDC_BA98_7654_3210))
            == "123456789abcdeffedcba9876543210",
        "fmt pattern hex"
    );
}

#[inline(never)]
fn cmp_i128(a: i128, b: i128) -> core::cmp::Ordering {
    a.cmp(&b)
}

fn cmp_cases() {
    use core::cmp::Ordering;
    assert!(cmp_i128(black_box(i128::MIN), black_box(i128::MAX)) == Ordering::Less, "cmp");
    assert!(cmp_i128(black_box(-1), black_box(1)) == Ordering::Less, "cmp -1 1");
    assert!(cmp_i128(black_box(5), black_box(5)) == Ordering::Equal, "cmp eq");
    assert!(
        cmp_i128(black_box(1i128 << 64), black_box(1)) == Ordering::Greater,
        "cmp hi"
    );
    assert!(black_box(u128::MAX) > black_box(1u128 << 127), "u128 gt");
    assert!(black_box(-1i128) < black_box(0), "i128 lt");
    assert!(black_box(3i128).max(black_box(-9)) == 3, "i128 max");
    assert!(black_box(i128::MIN).signum() == -1, "signum MIN");
}

fn main() {
    mul_cases();
    cttz_cases();
    collect_cases();
    div_rem_cases();
    rotate_cases();
    shift_cases();
    bit_cases();
    saturating_cases();
    overflow_cases();
    cast_cases();
    fmt_cases();
    cmp_cases();
    println!("int128 ok");
}
