"""Type stubs for reversi_zero_rs Rust module."""

from typing import Iterator, Optional

class MctsConfigArgs:
    """MCTS configuration arguments."""

    num_simulations: Optional[int]
    c_puct: Optional[float]
    temperature: Optional[float]
    dirichlet_alpha: Optional[float]
    dirichlet_epsilon: Optional[float]

    def __init__(
        self,
        num_simulations: Optional[int] = None,
        c_puct: Optional[float] = None,
        temperature: Optional[float] = None,
        dirichlet_alpha: Optional[float] = None,
        dirichlet_epsilon: Optional[float] = None,
    ) -> None: ...

class BatchConfigArgs:
    """Batch configuration arguments."""

    batch_size: Optional[int]
    game_concurrency: Optional[int]
    batch_timeout_ms: Optional[int]

    def __init__(
        self,
        batch_size: Optional[int] = None,
        game_concurrency: Optional[int] = None,
        batch_timeout_ms: Optional[int] = None,
    ) -> None: ...

class SelfPlayStats:
    """Statistics from a self-play batch (incremental per step)."""

    # Incremental counts (this step only)
    games: int
    black_wins: int
    white_wins: int
    draws: int

    # Win rates (this step only)
    black_win_rate: float
    white_win_rate: float
    draw_rate: float

    # Time metrics
    step_duration_sec: float
    games_per_sec: float
    elapsed_time_sec: float

    # Game quality
    avg_game_length: float
    positions_generated: int

class SelfPlayStream:
    """Stream self-play games in batches, yielding stats after each batch."""

    def __init__(
        self,
        total_games: int,
        report_interval: int,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        batch: Optional[BatchConfigArgs] = None,
        mcts: Optional[MctsConfigArgs] = None,
        save_dir: Optional[str] = None,
    ) -> None:
        """
        Initialize self-play stream.

        Args:
            total_games: Total number of games to play
            report_interval: Report stats every N games
            model_path: Path to TorchScript model (default: "../models/ts/latest.pt")
            device: Device to use ("cpu", "cuda", or "gpu", default: cuda if available)
            batch: Batch configuration
            mcts: MCTS configuration
            save_dir: Directory to auto-save training data (creates states.npy, policies.npy, values.npy)
                     If None, data will not be saved to disk (warning will be shown)

        Raises:
            ValueError: If total_games is 0, or if batch_size/game_concurrency/batch_timeout_ms is 0
            RuntimeError: If model fails to load or other runtime errors occur
        """
        ...

    def __iter__(self) -> Iterator[SelfPlayStats]: ...
    def __next__(self) -> SelfPlayStats: ...
