import pytest

from reversi_zero_trainer import runtime


def test_cpu_mcts_player_defaults_to_one_torch_thread(monkeypatch):
    calls: list[tuple[str, int]] = []
    monkeypatch.setattr(
        runtime.torch, "set_num_threads", lambda value: calls.append(("intra", value))
    )
    monkeypatch.setattr(
        runtime.torch,
        "set_num_interop_threads",
        lambda value: calls.append(("interop", value)),
    )

    configured = runtime.configure_mcts_player_threads("cpu", None)

    assert configured == 1
    assert calls == [("intra", 1), ("interop", 1)]


def test_cuda_mcts_player_keeps_torch_defaults(monkeypatch):
    monkeypatch.setattr(
        runtime.torch,
        "set_num_threads",
        lambda value: pytest.fail("CUDA must keep the Torch default"),
    )
    monkeypatch.setattr(
        runtime.torch,
        "set_num_interop_threads",
        lambda value: pytest.fail("CUDA must keep the Torch default"),
    )

    assert runtime.configure_mcts_player_threads("cuda", 4) is None
