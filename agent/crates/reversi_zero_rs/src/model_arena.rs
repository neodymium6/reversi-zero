use anyhow::{Result, anyhow, bail};
use rayon::ThreadPool;
use rayon::prelude::*;
use reversi_core::{Board, Turn};
use reversi_mcts::{Mcts, MctsConfig, PolicyValueModel};

#[derive(Debug, Default, PartialEq, Eq)]
pub(crate) struct ModelArenaSummary {
    pub wins: u32,
    pub losses: u32,
    pub draws: u32,
    pub challenger_pieces: u64,
    pub reference_pieces: u64,
}

#[derive(Debug)]
struct GameOutcome {
    challenger_won: bool,
    reference_won: bool,
    challenger_pieces: u32,
    reference_pieces: u32,
}

pub(crate) fn evaluate_models<M: PolicyValueModel + Sync>(
    challenger: &M,
    reference: &M,
    config: &MctsConfig,
    openings: &[Vec<usize>],
    pool: &ThreadPool,
) -> Result<ModelArenaSummary> {
    if openings.is_empty() {
        bail!("At least one opening is required");
    }

    let jobs: Vec<_> = openings
        .iter()
        .flat_map(|opening| [(opening, true), (opening, false)])
        .collect();
    let outcomes: Vec<Result<GameOutcome>> = pool.install(|| {
        jobs.into_par_iter()
            .with_max_len(1)
            .with_min_len(1)
            .map(|(opening, challenger_is_black)| {
                play_game(challenger, reference, config, opening, challenger_is_black)
            })
            .collect()
    });

    let mut summary = ModelArenaSummary::default();
    for outcome in outcomes {
        let outcome = outcome?;
        if outcome.challenger_won {
            summary.wins += 1;
        } else if outcome.reference_won {
            summary.losses += 1;
        } else {
            summary.draws += 1;
        }
        summary.challenger_pieces += u64::from(outcome.challenger_pieces);
        summary.reference_pieces += u64::from(outcome.reference_pieces);
    }
    Ok(summary)
}

fn play_game<M: PolicyValueModel>(
    challenger: &M,
    reference: &M,
    config: &MctsConfig,
    opening: &[usize],
    challenger_is_black: bool,
) -> Result<GameOutcome> {
    let mut board = Board::new();
    for (ply, &move_index) in opening.iter().enumerate() {
        if board.is_game_over() {
            bail!("Opening continues after the game ended at ply {ply}");
        }
        if board.is_pass() {
            bail!("Opening omits a required pass at ply {ply}");
        }
        board.do_move(move_index).map_err(|error| {
            anyhow!("Illegal opening move {move_index} at ply {ply}: {error:?}")
        })?;
    }

    while !board.is_game_over() {
        let challenger_to_move = match board.get_turn() {
            Turn::Black => challenger_is_black,
            Turn::White => !challenger_is_black,
        };
        let model = if challenger_to_move {
            challenger
        } else {
            reference
        };
        let mut mcts = Mcts::new();
        let result = mcts.search(&board, model, config)?;
        match result.best_move {
            Some(move_index) => board
                .do_move(move_index)
                .map_err(|error| anyhow!("MCTS move {move_index} failed: {error:?}"))?,
            None => board
                .do_pass()
                .map_err(|error| anyhow!("MCTS pass failed: {error:?}"))?,
        }
    }

    let winner = board
        .get_winner()
        .map_err(|error| anyhow!("Failed to determine winner: {error:?}"))?;
    let challenger_turn = if challenger_is_black {
        Turn::Black
    } else {
        Turn::White
    };
    let (black_pieces, white_pieces) = board.piece_counts();
    let (challenger_pieces, reference_pieces) = if challenger_is_black {
        (black_pieces, white_pieces)
    } else {
        (white_pieces, black_pieces)
    };

    Ok(GameOutcome {
        challenger_won: winner == Some(challenger_turn),
        reference_won: winner.is_some() && winner != Some(challenger_turn),
        challenger_pieces,
        reference_pieces,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::{Device, Kind, Tensor};

    struct UniformModel;

    impl PolicyValueModel for UniformModel {
        fn forward(&self, input: &Tensor) -> tch::Result<(Tensor, Tensor)> {
            let batch_size = input.size()[0];
            Ok((
                Tensor::zeros([batch_size, 64], (Kind::Float, Device::Cpu)),
                Tensor::zeros([batch_size, 1], (Kind::Float, Device::Cpu)),
            ))
        }

        fn device(&self) -> Device {
            Device::Cpu
        }
    }

    #[test]
    fn paired_identical_models_have_symmetric_results() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();
        let config = MctsConfig::default()
            .with_simulations(8)
            .with_temperature(0.0);

        let summary =
            evaluate_models(&UniformModel, &UniformModel, &config, &[Vec::new()], &pool).unwrap();

        assert_eq!(summary.wins, summary.losses);
        assert_eq!(summary.challenger_pieces, summary.reference_pieces);
        assert_eq!(summary.wins + summary.losses + summary.draws, 2);
    }
}
