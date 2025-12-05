#![allow(unsafe_op_in_unsafe_fn)]
use pyo3::prelude::*;
use reversi_mcts::MctsConfig;
use reversi_nn::NnModel;
use reversi_selfplay::{GameResult, play_game};
use tch::Device;

/// Python-facing stats for self-play progress.
#[pyclass]
pub struct SelfPlayStats {
    #[pyo3(get)]
    pub games: u32,
    #[pyo3(get)]
    pub black_win_rate: f32,
}

/// Stream self-play games in batches, yielding stats after each batch.
#[pyclass]
pub struct SelfPlayStream {
    total_games: u32,
    batch_size: u32,
    games_done: u32,
    black_wins: u32,
    draws: u32,
    model: NnModel,
    config: MctsConfig,
}

#[pymethods]
impl SelfPlayStream {
    #[new]
    #[pyo3(signature = (total_games, batch_size, model_path=None, num_simulations=None, batch_timeout_ms=None, device=None))]
    pub fn new(
        total_games: u32,
        batch_size: u32,
        model_path: Option<String>,
        num_simulations: Option<u32>,
        batch_timeout_ms: Option<u64>,
        device: Option<String>,
    ) -> PyResult<Self> {
        let model_path = model_path.unwrap_or_else(|| "../models/ts/latest.pt".to_string());
        let device = match device.as_deref() {
            Some("cpu") => Device::Cpu,
            Some("cuda") | Some("gpu") => Device::cuda_if_available(),
            _ => Device::cuda_if_available(),
        };

        let base_model = NnModel::load(&model_path, device).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to load model: {e}"))
        })?;

        let mut config = MctsConfig::default();
        if let Some(sims) = num_simulations {
            config = config.with_simulations(sims);
        }
        if let Some(to_ms) = batch_timeout_ms {
            config = config.with_batch_timeout_ms(to_ms);
        }
        config = config.with_batch_size(batch_size);

        Ok(SelfPlayStream {
            total_games,
            batch_size: batch_size.max(1),
            games_done: 0,
            black_wins: 0,
            draws: 0,
            model: base_model,
            config,
        })
    }

    fn __iter__(slf: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {
        slf
    }

    fn __next__(&mut self) -> Option<SelfPlayStats> {
        if self.games_done >= self.total_games {
            return None;
        }

        let remaining = (self.total_games - self.games_done) as usize;
        let batch = remaining.min(self.batch_size as usize);

        for _ in 0..batch {
            // Play one game
            let record = play_game(&self.model, &self.config).ok()?;
            self.games_done += 1;
            match record.winner {
                GameResult::BlackWin => self.black_wins += 1,
                GameResult::WhiteWin => (),
                GameResult::Draw => self.draws += 1,
            }
        }

        let black_win_rate = if self.games_done == 0 {
            0.0
        } else {
            self.black_wins as f32 / self.games_done as f32
        };

        Some(SelfPlayStats {
            games: self.games_done,
            black_win_rate,
        })
    }
}

#[pymodule]
fn reversi_zero_rs(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<SelfPlayStream>()?;
    m.add_class::<SelfPlayStats>()?;
    Ok(())
}
