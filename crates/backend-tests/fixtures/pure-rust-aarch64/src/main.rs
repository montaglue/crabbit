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

fn main() {
    let _answer = double(add(pick(), classify(2)));
}
