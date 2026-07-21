use std::env;
use std::io::{self, Read};

const SIZE: usize = 9;
const ALL_DIGITS: u16 = 0b11_1111_1110;

#[derive(Clone)]
struct Sudoku {
    cells: [u8; SIZE * SIZE],
    rows: [u16; SIZE],
    columns: [u16; SIZE],
    boxes: [u16; SIZE],
}

impl Sudoku {
    fn parse(input: &str) -> Result<Self, String> {
        let values: Vec<u8> = input
            .chars()
            .filter(|c| !c.is_whitespace())
            .map(|c| match c {
                '.' | '0' => Ok(0),
                '1'..='9' => Ok(c.to_digit(10).unwrap() as u8),
                _ => Err(format!("unexpected character '{c}'")),
            })
            .collect::<Result<_, _>>()?;

        if values.len() != SIZE * SIZE {
            return Err(format!(
                "expected 81 cells, but found {} (use 0 or . for blanks)",
                values.len()
            ));
        }

        let mut sudoku = Self {
            cells: [0; SIZE * SIZE],
            rows: [0; SIZE],
            columns: [0; SIZE],
            boxes: [0; SIZE],
        };

        for (index, value) in values.into_iter().enumerate() {
            if value == 0 {
                continue;
            }
            let row = index / SIZE;
            let column = index % SIZE;
            let box_index = Self::box_index(row, column);
            let bit = 1 << value;
            if sudoku.rows[row] & bit != 0
                || sudoku.columns[column] & bit != 0
                || sudoku.boxes[box_index] & bit != 0
            {
                return Err(format!(
                    "duplicate {value} at row {}, column {}",
                    row + 1,
                    column + 1
                ));
            }
            sudoku.place(index, value);
        }

        Ok(sudoku)
    }

    fn solve(&mut self) -> bool {
        let mut best: Option<(usize, u16)> = None;

        for index in 0..self.cells.len() {
            if self.cells[index] != 0 {
                continue;
            }
            let candidates = self.candidates(index);
            let count = candidates.count_ones();
            if count == 0 {
                return false;
            }
            if best.is_none_or(|(_, current)| count < current.count_ones()) {
                best = Some((index, candidates));
                if count == 1 {
                    break;
                }
            }
        }

        let Some((index, mut candidates)) = best else {
            return true;
        };

        while candidates != 0 {
            let bit = candidates & candidates.wrapping_neg();
            let value = bit.trailing_zeros() as u8;
            self.place(index, value);
            if self.solve() {
                return true;
            }
            self.remove(index, value);
            candidates &= !bit;
        }
        false
    }

    fn candidates(&self, index: usize) -> u16 {
        let row = index / SIZE;
        let column = index % SIZE;
        let used = self.rows[row] | self.columns[column] | self.boxes[Self::box_index(row, column)];
        ALL_DIGITS & !used
    }

    fn place(&mut self, index: usize, value: u8) {
        let row = index / SIZE;
        let column = index % SIZE;
        let box_index = Self::box_index(row, column);
        let bit = 1 << value;
        self.cells[index] = value;
        self.rows[row] |= bit;
        self.columns[column] |= bit;
        self.boxes[box_index] |= bit;
    }

    fn remove(&mut self, index: usize, value: u8) {
        let row = index / SIZE;
        let column = index % SIZE;
        let box_index = Self::box_index(row, column);
        let bit = !(1 << value);
        self.cells[index] = 0;
        self.rows[row] &= bit;
        self.columns[column] &= bit;
        self.boxes[box_index] &= bit;
    }

    const fn box_index(row: usize, column: usize) -> usize {
        (row / 3) * 3 + column / 3
    }

    fn display(&self) -> String {
        self.cells
            .chunks_exact(SIZE)
            .map(|row| row.iter().map(u8::to_string).collect::<Vec<_>>().join(" "))
            .collect::<Vec<_>>()
            .join("\n")
    }
}

fn read_input() -> Result<String, String> {
    let arguments: Vec<String> = env::args().skip(1).collect();
    if !arguments.is_empty() {
        return Ok(arguments.join(""));
    }

    let mut input = String::new();
    io::stdin()
        .read_to_string(&mut input)
        .map_err(|error| format!("could not read stdin: {error}"))?;
    if input.trim().is_empty() {
        return Err("provide a puzzle as an argument or through stdin".to_owned());
    }
    Ok(input)
}

fn run() -> Result<(), String> {
    let input = read_input()?;
    let mut sudoku = Sudoku::parse(&input)?;
    if !sudoku.solve() {
        return Err("the puzzle has no solution".to_owned());
    }
    println!("{}", sudoku.display());
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        eprintln!("usage: sudoku-solver <81 cells using digits, 0, or .>");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PUZZLE: &str = "530070000\
                          600195000\
                          098000060\
                          800060003\
                          400803001\
                          700020006\
                          060000280\
                          000419005\
                          000080079";

    #[test]
    fn solves_a_puzzle() {
        let mut sudoku = Sudoku::parse(PUZZLE).unwrap();
        assert!(sudoku.solve());
        let expected = Sudoku::parse(
            "534678912672195348198342567859761423426853791713924856961537284287419635345286179",
        )
        .unwrap();
        assert_eq!(sudoku.cells, expected.cells);
    }

    #[test]
    fn rejects_duplicate_clues() {
        let puzzle = format!("11{}", "0".repeat(79));
        assert!(Sudoku::parse(&puzzle).is_err());
    }

    #[test]
    fn reports_an_unsolvable_valid_board() {
        let puzzle = "123456780000000009".to_owned() + &"0".repeat(63);
        let mut sudoku = Sudoku::parse(&puzzle).unwrap();
        assert!(!sudoku.solve());
    }
}
