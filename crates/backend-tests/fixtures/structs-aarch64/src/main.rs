struct Pair {
    lhs: i32,
    rhs: i32,
}

fn sum(pair: Pair) -> i32 {
    pair.lhs + pair.rhs
}

fn bump_rhs(mut pair: Pair) -> i32 {
    pair.rhs = pair.rhs + 1;
    sum(pair)
}

fn main() {
    let pair = Pair { lhs: 20, rhs: 22 };
    let _answer = bump_rhs(pair);
}
