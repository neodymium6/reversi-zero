use std::collections::HashMap;
use std::env;
use std::fs::File;
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, anyhow, bail};
use rust_reversi_core::board::{Board, Turn};
use rust_reversi_core::search::{AlphaBetaSearch, BitMatrixEvaluator, Search};
use serde::Deserialize;

const DEFAULT_DEPTH: usize = 3;
const WIN_SCORE: i32 = 1 << 20;
const WEIGHTS: [i32; 6] = [-2, 1, -1, -10, 3, 38];
const MASKS: [u64; 6] = [
    0x5abd66c3c366bd5a,
    0x0000001818000000,
    0x00245a24245a2400,
    0x0042000000004200,
    0x2400810000810024,
    0x8100000000000081,
];

#[derive(Debug)]
struct Config {
    depth: usize,
    openings_file: Option<PathBuf>,
    turn: Turn,
}

fn parse_args(args: impl IntoIterator<Item = String>) -> Result<Config> {
    let mut depth = DEFAULT_DEPTH;
    let mut openings_file = None;
    let mut turn = None;
    let mut args = args.into_iter();

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--depth" => {
                let value = args.next().context("--depth requires a value")?;
                depth = value
                    .parse::<usize>()
                    .with_context(|| format!("invalid depth: {value}"))?;
            }
            "--openings-file" => {
                openings_file = Some(PathBuf::from(
                    args.next().context("--openings-file requires a path")?,
                ));
            }
            color if color.eq_ignore_ascii_case("BLACK") => turn = Some(Turn::Black),
            color if color.eq_ignore_ascii_case("WHITE") => turn = Some(Turn::White),
            _ => bail!("unknown argument: {arg}"),
        }
    }

    if depth == 0 {
        bail!("depth must be positive");
    }
    Ok(Config {
        depth,
        openings_file,
        turn: turn.context("missing color argument (BLACK or WHITE)")?,
    })
}

fn create_search(depth: usize) -> AlphaBetaSearch {
    let evaluator = BitMatrixEvaluator::<6>::new(WEIGHTS.to_vec(), MASKS.to_vec());
    AlphaBetaSearch::new(depth, Arc::new(evaluator), WIN_SCORE)
}

#[derive(Debug, Deserialize)]
struct Opening {
    moves: Vec<usize>,
}

#[derive(Debug, Deserialize)]
struct OpeningSuite {
    openings: Vec<Opening>,
}

#[derive(Debug)]
struct CompiledOpening {
    forced_moves: HashMap<(String, &'static str), usize>,
    final_piece_count: i32,
}

impl CompiledOpening {
    fn compile(opening: Opening) -> Result<Self> {
        let mut board = Board::new();
        let mut forced_moves = HashMap::new();

        for (ply, move_index) in opening.moves.into_iter().enumerate() {
            while board.is_pass() && !board.is_game_over() {
                board.do_pass().map_err(|error| {
                    anyhow!("failed to pass while compiling opening: {error:?}")
                })?;
            }
            if board.is_game_over() {
                bail!("opening continues after the game ended at ply {ply}");
            }
            if !board.is_legal_move(move_index) {
                bail!("illegal opening move {move_index} at ply {ply}");
            }

            forced_moves.insert(
                (
                    board
                        .get_board_line()
                        .map_err(|error| anyhow!("failed to serialize board: {error:?}"))?,
                    turn_name(board.get_turn()),
                ),
                move_index,
            );
            board
                .do_move(move_index)
                .map_err(|error| anyhow!("failed to apply opening move: {error:?}"))?;
        }

        Ok(Self {
            forced_moves,
            final_piece_count: board.piece_sum(),
        })
    }
}

#[derive(Debug)]
struct OpeningController {
    openings: Vec<CompiledOpening>,
    game_index: usize,
    last_piece_count: Option<i32>,
}

impl OpeningController {
    fn load(path: &Path) -> Result<Self> {
        let file = File::open(path)
            .with_context(|| format!("failed to open opening suite: {}", path.display()))?;
        let suite: OpeningSuite = serde_json::from_reader(file)
            .with_context(|| format!("failed to parse opening suite: {}", path.display()))?;
        Self::new(suite.openings)
    }

    fn new(openings: Vec<Opening>) -> Result<Self> {
        if openings.is_empty() {
            bail!("opening suite must contain at least one opening");
        }
        Ok(Self {
            openings: openings
                .into_iter()
                .map(CompiledOpening::compile)
                .collect::<Result<Vec<_>>>()?,
            game_index: 0,
            last_piece_count: None,
        })
    }

    fn select_forced_move(
        &mut self,
        board: &Board,
        board_line: &str,
        turn: Turn,
    ) -> Result<Option<usize>> {
        let piece_count = board.piece_sum();
        if self
            .last_piece_count
            .is_some_and(|last_piece_count| piece_count < last_piece_count)
        {
            self.game_index += 1;
        }
        self.last_piece_count = Some(piece_count);

        let opening = self.openings.get(self.game_index).with_context(|| {
            format!(
                "Arena requested game {}, but the opening suite contains only {} games per color assignment",
                self.game_index + 1,
                self.openings.len()
            )
        })?;
        if let Some(&move_index) = opening
            .forced_moves
            .get(&(board_line.to_owned(), turn_name(turn)))
        {
            return Ok(Some(move_index));
        }
        if piece_count < opening.final_piece_count {
            bail!(
                "game {} diverged from its forced opening",
                self.game_index + 1
            );
        }
        Ok(None)
    }
}

fn turn_name(turn: Turn) -> &'static str {
    match turn {
        Turn::Black => "BLACK",
        Turn::White => "WHITE",
    }
}

fn run(config: Config, input: impl BufRead, output: impl Write) -> Result<()> {
    let search = create_search(config.depth);
    let mut openings = config
        .openings_file
        .as_deref()
        .map(OpeningController::load)
        .transpose()?;
    let mut output = BufWriter::new(output);

    for line in input.lines() {
        let board_line = line.context("failed to read Arena input")?;
        let board_line = board_line.trim();
        if board_line.eq_ignore_ascii_case("ping") {
            writeln!(output, "pong")?;
            output.flush()?;
            continue;
        }

        let mut board = Board::new();
        board
            .set_board_str(board_line, config.turn)
            .map_err(|error| anyhow!("invalid board string from Arena: {error:?}"))?;
        let forced_move = openings
            .as_mut()
            .map(|controller| controller.select_forced_move(&board, board_line, config.turn))
            .transpose()?
            .flatten();
        let move_index = forced_move
            .or_else(|| search.get_move(&mut board))
            .context("no legal move available")?;

        writeln!(output, "{move_index}")?;
        output.flush()?;
    }
    Ok(())
}

fn main() -> Result<()> {
    let config = parse_args(env::args().skip(1))?;
    run(config, BufReader::new(io::stdin()), io::stdout())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_arena_color_after_options() {
        let config = parse_args([
            "--depth".to_owned(),
            "5".to_owned(),
            "--openings-file".to_owned(),
            "openings.json".to_owned(),
            "WHITE".to_owned(),
        ])
        .unwrap();

        assert_eq!(config.depth, 5);
        assert_eq!(config.openings_file, Some(PathBuf::from("openings.json")));
        assert_eq!(config.turn, Turn::White);
    }

    #[test]
    fn rejects_zero_depth() {
        let error =
            parse_args(["--depth".to_owned(), "0".to_owned(), "BLACK".to_owned()]).unwrap_err();

        assert!(error.to_string().contains("depth must be positive"));
    }

    #[test]
    fn bitmatrix_search_selects_a_legal_initial_move() {
        let mut board = Board::new();
        let move_index = create_search(1).get_move(&mut board).unwrap();

        assert!(board.is_legal_move(move_index));
    }

    #[test]
    fn opening_controller_forces_each_games_opening() {
        let mut controller = OpeningController::new(vec![
            Opening { moves: vec![19] },
            Opening { moves: vec![26] },
        ])
        .unwrap();
        let first_board = Board::new();
        let first_line = first_board.get_board_line().unwrap();

        assert_eq!(
            controller
                .select_forced_move(&first_board, &first_line, Turn::Black)
                .unwrap(),
            Some(19)
        );

        let mut progressed_board = first_board.clone();
        progressed_board.do_move(19).unwrap();
        let progressed_line = progressed_board.get_board_line().unwrap();
        assert_eq!(
            controller
                .select_forced_move(&progressed_board, &progressed_line, Turn::White)
                .unwrap(),
            None
        );

        let second_board = Board::new();
        let second_line = second_board.get_board_line().unwrap();
        assert_eq!(
            controller
                .select_forced_move(&second_board, &second_line, Turn::Black)
                .unwrap(),
            Some(26)
        );
    }
}
