use std::hint::black_box;

fn add(lhs: i32, rhs: i32) -> i32 {
    lhs + rhs
}

fn double(value: i32) -> i32 {
    value + value
}

fn classify(value: i32) -> i32 {
    match value {
        0 => 10,
        1 => 11,
        2 => 12,
        _ => 13,
    }
}

fn pick() -> i32 {
    let values = [3, 5, 8, 13];
    values[2]
}

#[inline(never)]
fn div_i32(a: i32, b: i32) -> i32 {
    a / b
}

#[inline(never)]
fn rem_i32(a: i32, b: i32) -> i32 {
    a % b
}

#[inline(never)]
fn div_i16(a: i16, b: i16) -> i16 {
    a / b
}

#[inline(never)]
fn rem_i8(a: i8, b: i8) -> i8 {
    a % b
}

#[inline(never)]
fn div_by_3(a: i32) -> i32 {
    a / 3
}

#[inline(never)]
fn div_u32(a: u32, b: u32) -> u32 {
    a / b
}

/// Signed division of sub-64-bit operands: the backend runs the 64-bit
/// `sdiv`, so a negative dividend or divisor must be sign-extended into it
/// (zero-extending `-7` gave `1431655763`).
fn narrow_signed_division_cases() {
    let (seven, three) = (black_box(7i32), black_box(3i32));
    assert!(div_i32(-seven, three) == -2, "-7 / 3");
    assert!(div_i32(seven, -three) == -2, "7 / -3");
    assert!(div_i32(-seven, -three) == 2, "-7 / -3");
    assert!(div_i32(seven, three) == 2, "7 / 3");
    assert!(rem_i32(-seven, three) == -1, "-7 % 3");
    assert!(rem_i32(seven, -three) == 1, "7 % -3");
    assert!(rem_i32(-seven, -three) == -1, "-7 % -3");
    assert!(div_by_3(black_box(-7)) == -2, "-7 / const 3");
    assert!(div_by_3(black_box(-1)) == 0, "-1 / const 3");
    assert!(div_i32(black_box(i32::MIN), black_box(2)) == i32::MIN / 2, "MIN / 2");
    assert!(div_i32(black_box(i32::MIN), black_box(-1)) == i32::MIN, "MIN / -1 wraps");
    assert!(div_i16(black_box(-300i16), black_box(7)) == -42, "i16 -300 / 7");
    assert!(div_i16(black_box(300i16), black_box(-7)) == -42, "i16 300 / -7");
    assert!(rem_i8(black_box(-100i8), black_box(7)) == -2, "i8 -100 % 7");
    assert!(rem_i8(black_box(100i8), black_box(-7)) == 2, "i8 100 % -7");
    assert!(div_u32(black_box(u32::MAX), black_box(3)) == 1431655765, "u32 MAX / 3");
}

#[inline(never)]
fn ten_args(
    a: &[i32],
    b: &[i32],
    c: &mut [i32],
    t: &mut [i32; 4],
    u: &mut [i32; 4],
    v: &mut [i32; 4],
    n: usize,
) -> i32 {
    // Ten integer-class arguments (three fat slices count double): the last
    // two arrive on the stack.
    t[0] = a[0];
    u[0] = b[0];
    v[0] = c[0];
    c[1] = t[0] + u[0] + v[0] + n as i32;
    c[1]
}

#[inline(never)]
fn twelve_scalars(
    a: u64,
    b: u64,
    c: u64,
    d: u64,
    e: u64,
    f: u64,
    g: u64,
    h: u64,
    i: u64,
    j: u64,
    k: u64,
    l: u64,
) -> u64 {
    // Keep the stack-passed arguments live across a call so the callee's
    // frame and spills interact with the incoming stack-argument loads.
    let head = black_box(a + b + c + d + e + f + g + h);
    head + i * 3 + j * 5 + k * 7 + l * 11
}

/// Arguments beyond x0-x7 go through the AAPCS64 stack-argument area on
/// both sides of a non-inlined call.
fn stack_argument_cases() {
    let a = [5i32; 4];
    let b = [7i32; 4];
    let mut c = [11i32; 4];
    let (mut t, mut u, mut v) = ([0i32; 4], [0i32; 4], [0i32; 4]);
    let result = ten_args(&a, &b, &mut c, &mut t, &mut u, &mut v, black_box(100));
    assert!(result == 123, "ten_args result");
    assert!(c[1] == 123 && t[0] == 5 && u[0] == 7 && v[0] == 11, "ten_args side effects");
    let scalars = twelve_scalars(
        black_box(1),
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        black_box(9),
        black_box(10),
        11,
        black_box(12),
    );
    assert!(scalars == 36 + 27 + 50 + 77 + 132, "twelve_scalars result");
}

fn main() {
    let _answer = double(add(pick(), classify(2)));
    narrow_signed_division_cases();
    stack_argument_cases();
}
