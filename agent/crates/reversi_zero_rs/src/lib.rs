#![allow(unsafe_op_in_unsafe_fn)]
use std::sync::Arc;
use std::time::Duration;

use pyo3::prelude::*;
use rayon::ThreadPool;
use rayon::prelude::*;
use reversi_core::{Board, Turn};
use reversi_mcts::{BatchingModel, Mcts, MctsConfig};
use reversi_nn::NnModel;
use reversi_selfplay::{
    GameRecord, GameResult, game_to_training_examples, play_game,
    storage::append_training_data_to_dir,
};
use tch::Device;

#[pyclass]
#[derive(Debug, Clone, Default)]
pub struct MctsConfigArgs {
    #[pyo3(get, set)]
    pub num_simulations: Option<u32>,
    #[pyo3(get, set)]
    pub c_puct: Option<f32>,
    #[pyo3(get, set)]
    pub temperature: Option<f32>,
    #[pyo3(get, set)]
    pub dirichlet_alpha: Option<f32>,
    #[pyo3(get, set)]
    pub dirichlet_epsilon: Option<f32>,
    #[pyo3(get, set)]
    pub expansion_batch_size: Option<u32>,
}

#[pymethods]
impl MctsConfigArgs {
    #[new]
    #[pyo3(signature = (num_simulations=None, c_puct=None, temperature=None, dirichlet_alpha=None, dirichlet_epsilon=None, expansion_batch_size=None))]
    fn new(
        num_simulations: Option<u32>,
        c_puct: Option<f32>,
        temperature: Option<f32>,
        dirichlet_alpha: Option<f32>,
        dirichlet_epsilon: Option<f32>,
        expansion_batch_size: Option<u32>,
    ) -> Self {
        Self {
            num_simulations,
            c_puct,
            temperature,
            dirichlet_alpha,
            dirichlet_epsilon,
            expansion_batch_size,
        }
    }
}

#[pyclass]
#[derive(Debug, Clone, Default)]
pub struct BatchConfigArgs {
    #[pyo3(get, set)]
    pub batch_size: Option<u32>,
    #[pyo3(get, set)]
    pub game_concurrency: Option<u32>,
    #[pyo3(get, set)]
    pub batch_timeout_ms: Option<u64>,
}

#[pymethods]
impl BatchConfigArgs {
    #[new]
    #[pyo3(signature = (batch_size=None, game_concurrency=None, batch_timeout_ms=None))]
    fn new(
        batch_size: Option<u32>,
        game_concurrency: Option<u32>,
        batch_timeout_ms: Option<u64>,
    ) -> Self {
        Self {
            batch_size,
            game_concurrency,
            batch_timeout_ms,
        }
    }
}

/// Python-facing stats for self-play progress (incremental per step).
#[pyclass]
pub struct SelfPlayStats {
    // Incremental counts (this step only)
    #[pyo3(get)]
    pub games: u32,
    #[pyo3(get)]
    pub black_wins: u32,
    #[pyo3(get)]
    pub white_wins: u32,
    #[pyo3(get)]
    pub draws: u32,

    // Win rates (this step only)
    #[pyo3(get)]
    pub black_win_rate: f32,
    #[pyo3(get)]
    pub white_win_rate: f32,
    #[pyo3(get)]
    pub draw_rate: f32,

    // Time metrics
    #[pyo3(get)]
    pub step_duration_sec: f64,
    #[pyo3(get)]
    pub games_per_sec: f32,
    #[pyo3(get)]
    pub elapsed_time_sec: f64,

    // Game quality
    #[pyo3(get)]
    pub avg_game_length: f32,
    #[pyo3(get)]
    pub positions_generated: u64,
}

#[pyclass]
pub struct MctsPlayer {
    model: NnModel,
    config: MctsConfig,
}

#[pymethods]
impl MctsPlayer {
    #[new]
    #[pyo3(signature = (model_path, device=None, mcts=None))]
    pub fn new(
        model_path: String,
        device: Option<String>,
        mcts: Option<MctsConfigArgs>,
    ) -> PyResult<Self> {
        let device = match device.as_deref().map(|s| s.to_ascii_lowercase()) {
            Some(ref d) if d == "cpu" => Device::Cpu,
            Some(ref d) if d == "cuda" || d == "gpu" => Device::cuda_if_available(),
            Some(other) => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Unknown device '{other}', expected 'cpu' or 'cuda'"
                )));
            }
            None => Device::cuda_if_available(),
        };

        let config = build_eval_mcts_config(mcts)?;

        let model = NnModel::load(&model_path, device).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Failed to load model from {model_path}: {e}"
            ))
        })?;

        Ok(Self { model, config })
    }

    /// Select a move for the given board string and turn.
    ///
    /// Args:
    ///     board_str: Board representation compatible with rust_reversi_core.
    ///     turn: "BLACK" or "WHITE".
    ///
    /// Returns:
    ///     int: best move index (0-63). Returns -1 if no move is found.
    pub fn select_move(&self, board_str: String, turn: String) -> PyResult<i32> {
        let turn = parse_turn_str(&turn)?;

        let mut board = Board::new();
        if let Err(e) = board.set_board_str(&board_str, turn) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!("{e:?}")));
        }

        // Initialize a fresh MCTS tree for each call to avoid sync issues.
        let mut mcts = Mcts::new();
        let result = match mcts.search(&board, &self.model, &self.config) {
            Ok(res) => res,
            Err(e) => {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")));
            }
        };

        Ok(result.best_move.map_or(-1, |m| m as i32))
    }
}

/// Stream self-play games in batches, yielding stats after each batch.
#[pyclass]
pub struct SelfPlayStream {
    total_games: u32,
    report_interval: u32,
    games_done: u32,
    black_wins_total: u32,
    white_wins_total: u32,
    draws_total: u32,
    total_positions: u64,
    model: BatchingModel<NnModel>,
    config: MctsConfig,
    pool: Arc<ThreadPool>,
    start_time: std::time::Instant,
    last_report_time: std::time::Instant,
    save_dir: Option<String>,
    current_batch_records: Vec<GameRecord>,
}

#[pymethods]
impl SelfPlayStream {
    #[new]
    #[pyo3(signature = (
        total_games,
        report_interval,
        model_path=None,
        device=None,
        batch=None,
        mcts=None,
        save_dir=None,
    ))]
    pub fn new(
        total_games: u32,
        report_interval: u32,
        model_path: Option<String>,
        device: Option<String>,
        batch: Option<BatchConfigArgs>,
        mcts: Option<MctsConfigArgs>,
        save_dir: Option<String>,
    ) -> PyResult<Self> {
        if total_games == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "total_games must be > 0",
            ));
        }

        let model_path = model_path.unwrap_or_else(|| "../models/ts/latest.pt".to_string());
        let device = match device.as_deref() {
            Some("cpu") => Device::Cpu,
            Some("cuda") | Some("gpu") => Device::cuda_if_available(),
            _ => Device::cuda_if_available(),
        };

        let report_interval = report_interval.max(1);
        let batch = batch.unwrap_or_default();
        let mcts = mcts.unwrap_or_default();

        let game_concurrency = match batch.game_concurrency {
            Some(0) => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "game_concurrency must be > 0",
                ));
            }
            Some(gc) => gc,
            None => default_game_concurrency(report_interval),
        };

        let batch_size = match batch.batch_size {
            Some(0) => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "batch_size must be > 0",
                ));
            }
            Some(bs) => bs,
            None => default_batch_size(game_concurrency),
        };
        let batch_timeout_ms = match batch.batch_timeout_ms {
            Some(0) => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "batch_timeout_ms must be > 0",
                ));
            }
            Some(ms) => ms,
            None => clamp_timeout(2),
        };

        let base_model = NnModel::load(&model_path, device).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to load model: {e}"))
        })?;

        let model = BatchingModel::new(
            base_model,
            batch_size as usize,
            Duration::from_millis(batch_timeout_ms),
        );

        // MCTS expansion batch size - should be smaller than NN batch_size to avoid determinism
        let expansion_batch_size = mcts.expansion_batch_size.unwrap_or(4);

        let mut config = MctsConfig::default()
            .with_batch_size(expansion_batch_size)
            .with_batch_timeout_ms(batch_timeout_ms);

        if let Some(sims) = mcts.num_simulations {
            config = config.with_simulations(sims);
        }
        if let Some(c) = mcts.c_puct {
            config = config.with_c_puct(c);
        }
        if let Some(t) = mcts.temperature {
            config = config.with_temperature(t);
        }

        // Dirichlet noise is enabled by default for self-play; allow tuning alpha/epsilon.
        let alpha = mcts.dirichlet_alpha.unwrap_or(0.3);
        let epsilon = mcts.dirichlet_epsilon.unwrap_or(0.25);
        config = config.with_dirichlet_noise(alpha, epsilon);

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(game_concurrency as usize)
            .build()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{e}")))?;

        let now = std::time::Instant::now();

        // Warn if save_dir is not set
        if save_dir.is_none() {
            eprintln!("Warning: save_dir is not set. Game data will not be saved to disk.");
        }

        Ok(SelfPlayStream {
            total_games,
            report_interval,
            games_done: 0,
            black_wins_total: 0,
            white_wins_total: 0,
            draws_total: 0,
            total_positions: 0,
            model,
            config,
            pool: Arc::new(pool),
            start_time: now,
            last_report_time: now,
            save_dir,
            current_batch_records: Vec::new(),
        })
    }

    fn __iter__(slf: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {
        slf
    }

    fn __next__(&mut self) -> PyResult<Option<SelfPlayStats>> {
        if self.games_done >= self.total_games {
            return Ok(None);
        }

        let remaining = self.total_games - self.games_done;
        let chunk = remaining.min(self.report_interval);

        let model = self.model.clone();
        let config = self.config.clone();

        // Track incremental stats for this step
        let mut step_black_wins = 0u32;
        let mut step_white_wins = 0u32;
        let mut step_draws = 0u32;
        let mut step_total_moves = 0u64;
        let mut step_positions = 0u64;

        let results: Vec<anyhow::Result<_>> = self.pool.install(|| {
            (0..chunk)
                .into_par_iter()
                .with_max_len(1)
                .with_min_len(1)
                .map(|_| play_game(&model, &config))
                .collect()
        });

        for res in results {
            let record = res.map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Self-play failed: {e}"))
            })?;

            self.games_done += 1;

            // Track step-level incremental stats
            let game_length = record.states.len() as u64;
            step_total_moves += game_length;
            step_positions += game_length;

            match record.winner {
                GameResult::BlackWin => {
                    self.black_wins_total += 1;
                    step_black_wins += 1;
                }
                GameResult::WhiteWin => {
                    self.white_wins_total += 1;
                    step_white_wins += 1;
                }
                GameResult::Draw => {
                    self.draws_total += 1;
                    step_draws += 1;
                }
            }

            // Store record temporarily for potential saving
            if self.save_dir.is_some() {
                self.current_batch_records.push(record);
            }
        }

        // Auto-save if directory is specified (appends to disk, doesn't keep in memory)
        if let Some(ref dir) = self.save_dir {
            // Convert records to training examples
            let examples: Vec<_> = self
                .current_batch_records
                .iter()
                .flat_map(game_to_training_examples)
                .collect();

            // Append to directory (creates states.npy, policies.npy, values.npy)
            append_training_data_to_dir(&examples, dir).map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Auto-save failed: {e}"))
            })?;

            // Clear from memory after saving
            self.current_batch_records.clear();
        }

        self.total_positions += step_positions;

        // Calculate time metrics
        let now = std::time::Instant::now();
        let step_duration = now.duration_since(self.last_report_time);
        let elapsed = now.duration_since(self.start_time);
        self.last_report_time = now;

        let step_duration_sec = step_duration.as_secs_f64();
        let elapsed_sec = elapsed.as_secs_f64();
        let games_per_sec = if step_duration_sec > 0.0 {
            chunk as f32 / step_duration_sec as f32
        } else {
            0.0
        };

        // Calculate win rates for this step
        let step_games = chunk;
        let black_win_rate = if step_games > 0 {
            step_black_wins as f32 / step_games as f32
        } else {
            0.0
        };
        let white_win_rate = if step_games > 0 {
            step_white_wins as f32 / step_games as f32
        } else {
            0.0
        };
        let draw_rate = if step_games > 0 {
            step_draws as f32 / step_games as f32
        } else {
            0.0
        };

        // Calculate average game length for this step
        let avg_game_length = if step_games > 0 {
            step_total_moves as f32 / step_games as f32
        } else {
            0.0
        };

        Ok(Some(SelfPlayStats {
            games: step_games,
            black_wins: step_black_wins,
            white_wins: step_white_wins,
            draws: step_draws,
            black_win_rate,
            white_win_rate,
            draw_rate,
            step_duration_sec,
            games_per_sec,
            elapsed_time_sec: elapsed_sec,
            avg_game_length,
            positions_generated: step_positions,
        }))
    }
}

#[pymodule]
fn reversi_zero_rs(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<BatchConfigArgs>()?;
    m.add_class::<MctsConfigArgs>()?;
    m.add_class::<SelfPlayStream>()?;
    m.add_class::<SelfPlayStats>()?;
    m.add_class::<MctsPlayer>()?;
    Ok(())
}

fn default_game_concurrency(report_interval: u32) -> u32 {
    let cores = num_cpus::get().max(1) as u32;
    report_interval.max(1).min(cores)
}

fn clamp_batch_size(bs: u32) -> u32 {
    bs.clamp(1, 512)
}

fn default_batch_size(game_concurrency: u32) -> u32 {
    let upper = clamp_batch_size(game_concurrency.saturating_mul(8));
    let mut candidate = 1u32;
    while candidate <= game_concurrency {
        candidate = candidate.saturating_mul(2).max(1);
        if candidate == 0 {
            break;
        }
    }

    if candidate == 0 || candidate > upper {
        // fallback to largest power-of-two <= upper
        let mut fallback = 1u32;
        while fallback.saturating_mul(2) <= upper && fallback < u32::MAX / 2 {
            fallback *= 2;
        }
        fallback.max(1)
    } else {
        candidate
    }
}

fn clamp_timeout(ms: u64) -> u64 {
    ms.clamp(1, 500)
}

fn parse_turn_str(s: &str) -> PyResult<Turn> {
    match s.to_ascii_uppercase().as_str() {
        "BLACK" | "B" => Ok(Turn::Black),
        "WHITE" | "W" => Ok(Turn::White),
        other => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Unknown turn '{other}', expected BLACK or WHITE"
        ))),
    }
}

fn build_eval_mcts_config(args: Option<MctsConfigArgs>) -> PyResult<MctsConfig> {
    let mut config = MctsConfig::default();
    if let Some(a) = args {
        if let Some(s) = a.num_simulations {
            config = config.with_simulations(s);
        }
        if let Some(c) = a.c_puct {
            config = config.with_c_puct(c);
        }
        if let Some(t) = a.temperature {
            config = config.with_temperature(t);
        }
        if let Some(alpha) = a.dirichlet_alpha {
            let epsilon = a.dirichlet_epsilon.unwrap_or(0.25);
            config = config.with_dirichlet_noise(alpha, epsilon);
        } else if let Some(epsilon) = a.dirichlet_epsilon {
            let alpha = a.dirichlet_alpha.unwrap_or(0.3);
            config = config.with_dirichlet_noise(alpha, epsilon);
        }
    }

    Ok(config)
}
