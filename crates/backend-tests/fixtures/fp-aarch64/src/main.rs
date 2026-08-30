//! Runtime checks for scalar floating-point support (f64/f32 arithmetic,
//! compares, saturating `as` casts, spilling under register pressure) and
//! for variable-amount 128-bit shifts.
//!
//! Plain `assert!` with string messages only: formatting a float would drag
//! in the whole float-printing machinery, which this fixture is not about.

use std::hint::black_box;

#[inline(never)]
pub extern "C" fn poly(x: f64) -> f64 {
    x * x * 0.5 + x - 1.0
}

#[inline(never)]
pub extern "C" fn polyf(x: f32) -> f32 {
    x * x * 0.5 + x - 1.0
}

#[inline(never)]
fn widen(x: f32) -> f64 {
    x as f64
}

#[inline(never)]
fn narrow(x: f64) -> f32 {
    x as f32
}

#[inline(never)]
fn to_u32(x: f64) -> u32 {
    x as u32
}

#[inline(never)]
fn to_i32(x: f64) -> i32 {
    x as i32
}

#[inline(never)]
fn to_i64(x: f64) -> i64 {
    x as i64
}

#[inline(never)]
fn to_u64(x: f64) -> u64 {
    x as u64
}

#[inline(never)]
fn from_i64(x: i64) -> f64 {
    x as f64
}

#[inline(never)]
fn from_u64(x: u64) -> f64 {
    x as f64
}

fn arithmetic_cases() {
    assert!(poly(2.0) == 3.0, "poly(2)");
    assert!(poly(-1.5) == -1.375, "poly(-1.5)");
    assert!(poly(0.0) == -1.0, "poly(0)");
    assert!(polyf(2.0) == 3.0, "polyf(2)");
    assert!(polyf(-1.5) == -1.375, "polyf(-1.5)");
    assert!(widen(1.5) == 1.5, "widen 1.5");
    assert!(narrow(2.5) == 2.5, "narrow 2.5");
    assert!(narrow(f64::NAN).is_nan(), "narrow nan");
    let x = 7.0f64;
    assert!((x / 2.0) * 2.0 == 7.0, "div mul");
    assert!(-x == 0.0 - 7.0, "neg");
    assert!(x % 4.0 == 3.0, "frem f64");
    assert!((7.5f32 % 2.0f32) == 1.5f32, "frem f32");
}

fn cast_cases() {
    // Rust `as` casts saturate and map NaN to zero.
    assert!(to_u32(-1.0) == 0, "-1 as u32");
    assert!(to_u32(f64::NAN) == 0, "nan as u32");
    assert!(to_u32(4294967296.0) == u32::MAX, "2^32 as u32 saturates");
    assert!(to_u32(3.9) == 3, "3.9 as u32 truncates");
    assert!(to_i32(f64::INFINITY) == i32::MAX, "inf as i32");
    assert!(to_i32(f64::NEG_INFINITY) == i32::MIN, "-inf as i32");
    assert!(to_i32(-3.9) == -3, "-3.9 as i32");
    assert!(to_i64(1e300) == i64::MAX, "1e300 as i64");
    assert!(to_i64(-1e300) == i64::MIN, "-1e300 as i64");
    assert!(to_i64(f64::NAN) == 0, "nan as i64");
    assert!(to_u64(1e300) == u64::MAX, "1e300 as u64");
    assert!(from_i64(-5) == -5.0, "-5 as f64");
    assert!(from_i64(i64::MIN) == -9223372036854775808.0, "i64::MIN as f64");
    assert!(from_u64(u64::MAX) == 18446744073709551616.0, "u64::MAX as f64");
    assert!(from_u64(5) == 5.0, "5 as f64");
}

fn compare_cases() {
    let nan = f64::NAN;
    assert!(!(nan == nan), "nan != nan");
    assert!(nan != nan, "nan ne");
    assert!(!(nan < 1.0) && !(nan > 1.0), "nan unordered");
    assert!(1.0 < 2.0 && 2.0 > 1.0 && 1.0 <= 1.0 && 1.0 >= 1.0, "ordered");
    assert!(nan.is_nan() && !1.0f64.is_nan(), "is_nan");
    // A float-keyed max reduction with a branch on each compare.
    let values = [3.5f64, -1.25, 9.75, 0.0, 9.5];
    let mut max = values[0];
    let mut index = 0usize;
    for (position, value) in values.iter().enumerate() {
        if *value > max {
            max = *value;
            index = position;
        }
    }
    assert!(max == 9.75 && index == 2, "max reduction");
}

#[inline(never)]
fn pressure(x: f64) -> f64 {
    // Eight simultaneously-live values against a four-register FP pool.
    let a = x + 1.0;
    let b = x + 2.0;
    let c = x + 3.0;
    let d = x + 4.0;
    let e = x + 5.0;
    let f = x + 6.0;
    let g = x + 7.0;
    let h = x + 8.0;
    let doubled = poly(x); // a call: everything above is live across it
    a * 1.0 + b * 2.0 + c * 3.0 + d * 4.0 + e * 5.0 + f * 6.0 + g * 7.0 + h * 8.0 + doubled
}

fn spill_cases() {
    // sum_{i=1..8} i*(x+i) = 36x + 204; poly(0) = -1, poly(1) = 0.5.
    assert!(pressure(0.0) == 203.0, "pressure(0)");
    assert!(pressure(1.0) == 240.5, "pressure(1)");
}

fn sort_and_sum_cases() {
    let mut values = [5.5f64, -2.25, 8.0, 0.5, -7.75, 3.125];
    // Insertion sort keyed on floats.
    let mut i = 1;
    while i < values.len() {
        let mut j = i;
        while j > 0 && values[j - 1] > values[j] {
            values.swap(j - 1, j);
            j -= 1;
        }
        i += 1;
    }
    assert!(values[0] == -7.75 && values[5] == 8.0, "sorted ends");
    let mut ascending = true;
    for pair in values.windows(2) {
        if pair[0] > pair[1] {
            ascending = false;
        }
    }
    assert!(ascending, "sorted order");
    let mut sum = 0.0f64;
    for value in values.iter() {
        sum += *value;
    }
    assert!(sum == 7.125, "float sum loop");
}

#[inline(never)]
fn shl_u128(x: u128, n: u32) -> u128 {
    x << n
}

#[inline(never)]
fn lshr_u128(x: u128, n: u32) -> u128 {
    x >> n
}

#[inline(never)]
fn ashr_i128(x: i128, n: u32) -> i128 {
    x >> n
}

#[inline(never)]
fn ashr_i64(x: i64, n: u32) -> i64 {
    x >> n
}

fn dynamic_shift_cases() {
    let patterns: [u128; 4] = [
        (0x0123_4567_89ab_cdefu128 << 64) | 0xfedc_ba98_7654_3210,
        1,
        u128::MAX,
        1u128 << 127,
    ];
    // The expected side shifts by a literal, which the backend lowers
    // through its constant-shift path, so each assert cross-checks the
    // dynamic-amount lowering against the constant-amount one.
    macro_rules! check_amount {
        ($x:expr, $n:literal) => {
            assert!(shl_u128($x, $n) == $x << $n, concat!("dynamic shl ", $n));
            assert!(lshr_u128($x, $n) == $x >> $n, concat!("dynamic lshr ", $n));
            assert!(
                ashr_i128($x as i128, $n) == ($x as i128) >> $n,
                concat!("dynamic ashr ", $n)
            );
        };
    }
    for x in patterns {
        check_amount!(x, 0);
        check_amount!(x, 1);
        check_amount!(x, 63);
        check_amount!(x, 64);
        check_amount!(x, 65);
        check_amount!(x, 127);
    }
    // Anchor both paths against fully known values.
    assert!(shl_u128(1, 127) == 1u128 << 127, "shl anchor");
    assert!(lshr_u128(1u128 << 127, 127) == 1, "lshr anchor");
    assert!(ashr_i128((1u128 << 127) as i128, 127) == -1, "ashr sign anchor");
    assert!(ashr_i128(i128::MIN, 64) == (i128::MIN >> 64), "ashr -2^63 half");
    assert!(ashr_i64(-8, 1) == -4, "i64 ashr");
    assert!(ashr_i64(i64::MIN, 63) == -1, "i64 ashr sign fill");
    assert!(ashr_i64(8, 2) == 2, "i64 ashr positive");
}

#[inline(never)]
fn sqrt64(x: f64) -> f64 {
    x.sqrt()
}

#[inline(never)]
fn sqrt32(x: f32) -> f32 {
    x.sqrt()
}

#[inline(never)]
fn exp2_32(x: f32) -> f32 {
    x.exp2()
}

#[inline(never)]
fn hypot_like(a: f64, b: f64) -> f64 {
    (a * a + b * b).sqrt()
}

/// The float math intrinsics: the inline ones (`fsqrt`, `fabs`, the
/// `frint*` rounding family, `fminnm`/`fmaxnm`) and the libm-backed ones
/// (`exp2f`, `exp`, `log`, `pow`, `sin`/`cos`, `fma`, `__powidf2`).
fn math_cases() {
    let x = black_box(2.25f64);
    let y = black_box(-1.5f64);
    let xf = black_box(2.25f32);
    let yf = black_box(-1.5f32);

    assert!(sqrt64(x) == 1.5, "f64 sqrt");
    assert!(sqrt32(xf) == 1.5, "f32 sqrt");
    assert!(sqrt64(y).is_nan(), "sqrt(-1.5) is nan");
    assert!(hypot_like(black_box(3.0), black_box(4.0)) == 5.0, "hypot 3 4");
    assert!(y.abs() == 1.5 && yf.abs() == 1.5, "abs");
    assert!(x.abs() == 2.25 && xf.abs() == 2.25, "abs positive");

    assert!(x.floor() == 2.0 && y.floor() == -2.0, "f64 floor");
    assert!(xf.floor() == 2.0 && yf.floor() == -2.0, "f32 floor");
    assert!(x.ceil() == 3.0 && y.ceil() == -1.0, "f64 ceil");
    assert!(xf.ceil() == 3.0 && yf.ceil() == -1.0, "f32 ceil");
    assert!(x.trunc() == 2.0 && y.trunc() == -1.0, "f64 trunc");
    assert!(xf.trunc() == 2.0 && yf.trunc() == -1.0, "f32 trunc");
    // round: half away from zero; round_ties_even: half to even.
    assert!(black_box(2.5f64).round() == 3.0 && y.round() == -2.0, "f64 round");
    assert!(black_box(2.5f32).round() == 3.0 && yf.round() == -2.0, "f32 round");
    assert!(black_box(2.5f64).round_ties_even() == 2.0, "f64 round_ties_even");
    assert!(black_box(3.5f32).round_ties_even() == 4.0, "f32 round_ties_even");

    let nan = black_box(f64::NAN);
    assert!(x.min(y) == -1.5 && x.max(y) == 2.25, "f64 min/max");
    assert!(xf.min(yf) == -1.5 && xf.max(yf) == 2.25, "f32 min/max");
    assert!(x.min(nan) == 2.25 && nan.max(x) == 2.25, "min/max ignore nan");
    assert!(y.copysign(x) == 1.5 && x.copysign(y) == -2.25, "copysign");

    assert!(exp2_32(black_box(3.0)) == 8.0, "f32 exp2");
    assert!(black_box(10.0f64).exp2() == 1024.0, "f64 exp2");
    assert!(black_box(0.0f64).exp() == 1.0 && black_box(0.0f32).exp() == 1.0, "exp 0");
    assert!((black_box(1.0f64).exp() - core::f64::consts::E).abs() < 1e-15, "exp 1");
    assert!(black_box(1.0f64).ln() == 0.0 && black_box(8.0f32).log2() == 3.0, "ln/log2");
    assert!(black_box(1000.0f64).log10() == 3.0, "log10");
    assert!(black_box(2.0f64).powi(10) == 1024.0, "f64 powi");
    assert!(black_box(2.0f32).powi(-2) == 0.25, "f32 powi");
    assert!(black_box(2.0f64).powf(0.5) == core::f64::consts::SQRT_2, "f64 powf");
    assert!(black_box(9.0f32).powf(0.5) == 3.0, "f32 powf");
    assert!(black_box(0.0f64).sin() == 0.0 && black_box(0.0f64).cos() == 1.0, "sin/cos 0");
    assert!((black_box(core::f32::consts::FRAC_PI_2).sin() - 1.0).abs() < 1e-6, "f32 sin pi/2");
    assert!(x.mul_add(2.0, 0.5) == 5.0 && xf.mul_add(2.0, 0.5) == 5.0, "mul_add");
}

fn main() {
    arithmetic_cases();
    cast_cases();
    compare_cases();
    spill_cases();
    sort_and_sum_cases();
    dynamic_shift_cases();
    math_cases();
    println!("fp ok");
}
