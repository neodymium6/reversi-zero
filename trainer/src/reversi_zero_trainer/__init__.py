from reversi_zero_rs import BatchConfigArgs, MctsConfigArgs, SelfPlayStream


def main() -> None:
    """
    Minimal demonstration of Rust self-play from Python.

    This uses the PyO3 bindings provided by `reversi_zero_rs` to run self-play
    and stream cumulative stats.
    """

    stream = SelfPlayStream(
        total_games=10_000,
        report_interval=256,
        batch=BatchConfigArgs(
            batch_size=256,
            game_concurrency=32,
            batch_timeout_ms=1,
        ),
        mcts=MctsConfigArgs(
            num_simulations=800,
            # c_puct=1.5,
            # temperature=1.0,
            # dirichlet_alpha=0.3,
            # dirichlet_epsilon=0.25,
        ),
        model_path="../models/ts/latest_resnet_c64_b6.pt",
        device="cuda",  # or "cpu"
        # device="cpu",
    )

    for step, stats in enumerate(stream):
        print(
            f"[selfplay] step={step} games={stats.games} "
            f"black_win_rate={stats.black_win_rate:.3f} draws={stats.draws}"
        )
