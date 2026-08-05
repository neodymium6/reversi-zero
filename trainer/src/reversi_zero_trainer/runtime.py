"""Runtime configuration shared by trainer entry points."""

from __future__ import annotations

import torch


def configure_mcts_player_threads(
    device: str, requested_threads: int | None
) -> int | None:
    """Configure Torch for a sequential Arena MCTS player process.

    Arena runs multiple player processes concurrently.  Letting every CPU player
    use Torch's machine-wide default causes severe oversubscription, while CUDA
    players should retain Torch's normal threading configuration.
    """

    if requested_threads is not None and requested_threads <= 0:
        raise ValueError("Torch thread count must be positive")
    if device != "cpu":
        return None

    threads = requested_threads if requested_threads is not None else 1
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    return threads
