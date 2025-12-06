from reversi_zero_rs import BatchConfigArgs, MctsConfigArgs, SelfPlayStream

from reversi_zero_trainer.logging import (
    ConsoleConfig,
    LoggerKind,
    LoggingConfig,
    create_logger,
)


def main() -> None:
    """
    Minimal demonstration of Rust self-play from Python with logging.

    This uses the PyO3 bindings provided by `reversi_zero_rs` to run self-play
    and stream cumulative stats, with metrics logged via the logging system.
    """

    # Configuration parameters
    total_games = 10_000
    report_interval = 32
    batch_size = 512
    game_concurrency = 32
    batch_timeout_ms = 1
    num_simulations = 800
    model_path = "../models/ts/latest_resnet_c64_b6.pt"
    device = "cuda"
    save_dir = "data/selfplay"  # Auto-save directory (creates states.npy, policies.npy, values.npy)

    # Create logging configuration with rich console output
    logging_cfg = LoggingConfig(
        backends={
            LoggerKind.CONSOLE: ConsoleConfig(
                verbose=True,
                show_params_table=True,  # Show params in a nice table
                show_timestamp=True,
            ),
        }
    )

    with create_logger(logging_cfg) as logger:
        # Log configuration parameters
        logger.log_param("total_games", total_games)
        logger.log_param("report_interval", report_interval)
        logger.log_param("batch_size", batch_size)
        logger.log_param("game_concurrency", game_concurrency)
        logger.log_param("batch_timeout_ms", batch_timeout_ms)
        logger.log_param("num_simulations", num_simulations)
        logger.log_param("model_path", model_path)
        logger.log_param("device", device)
        logger.log_param("save_dir", save_dir)

        stream = SelfPlayStream(
            total_games=total_games,
            report_interval=report_interval,
            batch=BatchConfigArgs(
                batch_size=batch_size,
                game_concurrency=game_concurrency,
                batch_timeout_ms=batch_timeout_ms,
            ),
            mcts=MctsConfigArgs(
                num_simulations=num_simulations,
                # c_puct=1.5,
                # temperature=1.0,
                # dirichlet_alpha=0.3,
                # dirichlet_epsilon=0.25,
            ),
            model_path=model_path,
            device=device,
            save_dir=save_dir,  # Enables automatic cumulative saving to directory
        )

        for step, stats in enumerate(stream):
            # Log basic counts (incremental)
            logger.log_metric("selfplay/games", float(stats.games), step=step)
            logger.log_metric("selfplay/black_wins", float(stats.black_wins), step=step)
            logger.log_metric("selfplay/white_wins", float(stats.white_wins), step=step)
            logger.log_metric("selfplay/draws", float(stats.draws), step=step)

            # Log win rates (for this step)
            logger.log_metric(
                "selfplay/black_win_rate", stats.black_win_rate, step=step
            )
            logger.log_metric(
                "selfplay/white_win_rate", stats.white_win_rate, step=step
            )
            logger.log_metric("selfplay/draw_rate", stats.draw_rate, step=step)

            # Log time metrics
            logger.log_metric(
                "selfplay/step_duration_sec", stats.step_duration_sec, step=step
            )
            logger.log_metric("selfplay/games_per_sec", stats.games_per_sec, step=step)
            logger.log_metric(
                "selfplay/elapsed_time_sec", stats.elapsed_time_sec, step=step
            )

            # Log game quality metrics
            logger.log_metric(
                "selfplay/avg_game_length", stats.avg_game_length, step=step
            )
            logger.log_metric(
                "selfplay/positions_generated",
                float(stats.positions_generated),
                step=step,
            )
