//! Exercises the ways Rust programs read standard input.
//!
//! Each mode leans on a different part of std: `read_to_string` and `read_line`
//! go through `Stdin`'s prebuilt entry points, while `lock().lines()` pulls the
//! generic `Lines`/`BufRead` machinery (and the `Mutex` behind `Stdin`, hence
//! the atomic intrinsics) into the importer. `trim`/`split_whitespace` exercise
//! the pattern searchers, which depend on real enum layouts and on `*const str`
//! keeping its length metadata.

use std::io::{self, BufRead, Read};

fn sum_words(line: &str) -> u64 {
    let mut total = 0;
    for word in line.split_whitespace() {
        let mut value = 0u64;
        for byte in word.bytes() {
            if byte.is_ascii_digit() {
                value = value * 10 + (byte - b'0') as u64;
            }
        }
        total += value;
    }
    total
}

fn main() {
    let mode = std::env::args().nth(1).unwrap_or_else(|| "all".to_owned());

    match mode.as_str() {
        "read_to_string" => {
            let mut input = String::new();
            io::stdin()
                .read_to_string(&mut input)
                .expect("failed to read stdin");
            println!("bytes={}", input.len());
            println!("trimmed=[{}]", input.trim());
            println!("sum={}", sum_words(&input));
        }
        "read_line" => {
            let mut first = String::new();
            io::stdin()
                .read_line(&mut first)
                .expect("failed to read a line");
            println!("first=[{}]", first.trim());
        }
        "lines" => {
            let mut count = 0;
            let mut total = 0;
            for line in io::stdin().lock().lines() {
                let line = line.expect("failed to read a line");
                if line.trim().is_empty() {
                    continue;
                }
                count += 1;
                total += sum_words(&line);
            }
            println!("lines={}", count);
            println!("total={}", total);
        }
        other => {
            eprintln!("unknown mode: {other}");
            std::process::exit(1);
        }
    }
}
