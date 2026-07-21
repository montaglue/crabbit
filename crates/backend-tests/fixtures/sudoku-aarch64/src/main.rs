const PUZZLE: [[u8; 9]; 9] = [
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9],
];

fn find_empty(board: &[[u8; 9]; 9]) -> Option<(usize, usize)> {
    for row in 0..9 {
        for col in 0..9 {
            if board[row][col] == 0 {
                return Some((row, col));
            }
        }
    }
    None
}

fn is_valid(board: &[[u8; 9]; 9], row: usize, col: usize, digit: u8) -> bool {
    for idx in 0..9 {
        if board[row][idx] == digit || board[idx][col] == digit {
            return false;
        }
    }

    let box_row = row / 3 * 3;
    let box_col = col / 3 * 3;
    for r in box_row..box_row + 3 {
        for c in box_col..box_col + 3 {
            if board[r][c] == digit {
                return false;
            }
        }
    }

    true
}

fn solve(board: &mut [[u8; 9]; 9]) -> bool {
    let Some((row, col)) = find_empty(board) else {
        return true;
    };

    for digit in 1..=9 {
        if is_valid(board, row, col, digit) {
            board[row][col] = digit;
            if solve(board) {
                return true;
            }
            board[row][col] = 0;
        }
    }

    false
}

fn main() {
    let mut board = PUZZLE;
    if !solve(&mut board) {
        println!("sudoku puzzle has no solution");
        return;
    }

    for row in &board {
        let line: String = row.iter().map(|&digit| char::from(b'0' + digit)).collect();
        println!("{line}");
    }
}
