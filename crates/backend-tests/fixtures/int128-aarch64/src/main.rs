//! Runtime checks for 128-bit mul-with-overflow, 128-bit cttz, and the
//! iterator/collect paths (uninhabited-enum discriminants, Box unsize
//! coercions, unsized-tail fat references) the importer lowers.
//!
//! Plain `assert!` with string messages only: formatting a 128-bit value
//! would drag in 128-bit division, which the AArch64 lowering does not
//! support yet.

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

fn main() {
    mul_cases();
    cttz_cases();
    collect_cases();
    println!("int128 ok");
}
