from reversi_zero_rs import SelfPlayStream


def main() -> None:
    """
    Minimal demonstration of Rust self-play from Python.

    This uses the PyO3 bindings provided by `reversi_zero_rs` to run self-play
    and stream cumulative stats.
    """

    stream = SelfPlayStream(total_games=10, batch_size=2)

    for step, stats in enumerate(stream):
        print(
            f"[selfplay] step={step} games={stats.games} "
            f"black_win_rate={stats.black_win_rate:.3f}"
        )
