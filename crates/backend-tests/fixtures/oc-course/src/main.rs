fn sumeven(x: &[isize]) -> isize {
    let mut result = 0;
    for i in 0..x.len() {
        if x[i] % 2 == 0 {
            result += x[i]
        }
    }
    return result
}

fn main() {
    println!("Hello, world!");
}
