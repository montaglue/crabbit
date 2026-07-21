unsafe extern "C" {
    fn write(fd: i32, buf: *const u8, count: usize) -> isize;
}

fn write_stderr(bytes: &[u8]) {
    unsafe {
        write(2, bytes.as_ptr(), bytes.len());
    }
}

fn fail(label: &str, reason: &str, expected: &str, actual: &str) -> ! {
    write_stderr(b"check failed: ");
    write_stderr(label.as_bytes());
    write_stderr(b"\nreason: ");
    write_stderr(reason.as_bytes());
    write_stderr(b"\nexpected: ");
    write_stderr(expected.as_bytes());
    write_stderr(b"\nactual: ");
    write_stderr(actual.as_bytes());
    write_stderr(b"\n");
    std::process::exit(1);
}

fn check_integer<T: itoa::Integer>(label: &str, value: T, expected: &str) {
    let mut buffer = itoa::Buffer::new();
    let actual = buffer.format(value);

    if actual.as_bytes().len() != expected.as_bytes().len() {
        fail(label, "length mismatch", expected, actual);
    }

    let mut index = 0;
    while index < expected.as_bytes().len() {
        if actual.as_bytes()[index] != expected.as_bytes()[index] {
            fail(label, "byte mismatch", expected, actual);
        }
        index += 1;
    }
}

fn check_u128_words(label: &str, value: u128, expected_lo: u64, expected_hi: u64) {
    if value as u64 != expected_lo {
        fail(label, "low word mismatch", "expected low word", "actual low word differed");
    }

    if (value >> 64) as u64 != expected_hi {
        fail(
            label,
            "high word mismatch",
            "expected high word",
            "actual high word differed",
        );
    }
}

fn main() {
    check_u128_words("u128::from(u64::MAX)", u128::from(u64::MAX), u64::MAX, 0);
    check_u128_words(
        "u128::MAX >> 64",
        u128::MAX >> 64,
        u64::MAX,
        0,
    );
    check_u128_words(
        "u128::from(u64::MAX) * u128::from(u64::MAX)",
        u128::from(u64::MAX) * u128::from(u64::MAX),
        1,
        u64::MAX - 1,
    );

    check_integer("0u8", 0u8, "0");
    check_integer("7u8", 7u8, "7");
    check_integer("u8::MAX", u8::MAX, "255");

    check_integer("10u16", 10u16, "10");
    check_integer("u16::MAX", u16::MAX, "65535");

    check_integer("128u32", 128u32, "128");
    check_integer("10_000u32", 10_000u32, "10000");
    check_integer("u32::MAX", u32::MAX, "4294967295");

    check_integer("7u64", 7u64, "7");
    check_integer("10u64", 10u64, "10");
    check_integer("128u64", 128u64, "128");
    check_integer("u64::MAX", u64::MAX, "18446744073709551615");

    check_integer("0usize", 0usize, "0");
    check_integer("usize::MAX", usize::MAX, "18446744073709551615");

    check_integer("0i8", 0i8, "0");
    check_integer("-1i8", -1i8, "-1");
    check_integer("i8::MIN", i8::MIN, "-128");
    check_integer("i8::MAX", i8::MAX, "127");

    check_integer("-10i16", -10i16, "-10");
    check_integer("i16::MIN", i16::MIN, "-32768");
    check_integer("i16::MAX", i16::MAX, "32767");

    check_integer("-128i32", -128i32, "-128");
    check_integer("i32::MIN", i32::MIN, "-2147483648");
    check_integer("i32::MAX", i32::MAX, "2147483647");

    check_integer("-10i64", -10i64, "-10");
    check_integer("i64::MIN", i64::MIN, "-9223372036854775808");
    check_integer("i64::MAX", i64::MAX, "9223372036854775807");

    check_integer("0isize", 0isize, "0");
    check_integer("isize::MIN", isize::MIN, "-9223372036854775808");
    check_integer("isize::MAX", isize::MAX, "9223372036854775807");

    check_integer("10u128", 10u128, "10");
    check_integer(
        "u128::MAX",
        u128::MAX,
        "340282366920938463463374607431768211455",
    );

    check_integer("-10i128", -10i128, "-10");
    check_integer(
        "i128::MIN",
        i128::MIN,
        "-170141183460469231731687303715884105728",
    );
    check_integer(
        "i128::MAX",
        i128::MAX,
        "170141183460469231731687303715884105727",
    );
}
