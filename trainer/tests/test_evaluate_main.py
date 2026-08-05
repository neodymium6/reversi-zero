import pytest

import reversi_zero_trainer.evaluate_main as evaluate_main
from reversi_zero_trainer.evaluate_main import (
    _bitmatrix_command,
    _mcts_command,
    _run_batched_model_arena,
    opening_suite_sha256,
    parse_args,
    score_interval,
    write_report,
)
from reversi_zero_trainer.openings import generate_openings


def test_score_interval_contains_even_score():
    low, high = score_interval(wins=20, losses=20, draws=0)

    assert low < 0.5 < high


def test_score_interval_treats_draw_as_half_point():
    assert score_interval(0, 0, 40) == pytest.approx(score_interval(20, 20, 0))


def test_opening_suite_hash_is_reproducible():
    openings = generate_openings(count=4, plies=6, seed=1)

    assert opening_suite_sha256(openings) == opening_suite_sha256(openings)
    assert opening_suite_sha256(openings) != opening_suite_sha256(
        list(reversed(openings))
    )


def test_write_report_is_atomic_and_refuses_overwrite(tmp_path):
    output = tmp_path / "evaluation.json"
    write_report({"summary": {"score": 0.5}}, output)

    assert output.exists()
    assert not (tmp_path / ".evaluation.json.next").exists()
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        write_report({"summary": {}}, output)


def test_cpu_mcts_command_limits_torch_threads(tmp_path):
    command = _mcts_command(
        tmp_path / "model.pt",
        tmp_path / "openings.json",
        "cpu",
        simulations=25,
        c_puct=1.5,
        torch_threads=2,
        expansion_batch_size=4,
    )

    assert command[-4:] == ["--device", "cpu", "--torch-threads", "2"]
    assert command[command.index("--expansion-batch-size") + 1] == "4"


def test_cuda_mcts_command_does_not_override_torch_threads(tmp_path):
    command = _mcts_command(
        tmp_path / "model.pt",
        tmp_path / "openings.json",
        "cuda",
        simulations=25,
        c_puct=1.5,
        torch_threads=1,
        expansion_batch_size=2,
    )

    assert "--torch-threads" not in command


def test_bitmatrix_command_uses_native_player(monkeypatch, tmp_path):
    module_file = tmp_path / "trainer/src/reversi_zero_trainer/evaluate_main.py"
    player = tmp_path / "agent/target/release/reversi-bitmatrix-player"
    player.parent.mkdir(parents=True)
    player.touch()
    monkeypatch.setattr("reversi_zero_trainer.evaluate_main.__file__", str(module_file))

    command = _bitmatrix_command(tmp_path / "openings.json", depth=5)

    assert command == [
        str(player),
        "--depth",
        "5",
        "--openings-file",
        str(tmp_path / "openings.json"),
    ]


def test_evaluation_accepts_bitmatrix_reference(tmp_path):
    args = parse_args(
        [
            "--challenger",
            str(tmp_path / "model.pt"),
            "--reference-bitmatrix",
            "--bitmatrix-depth",
            "5",
            "--output",
            str(tmp_path / "report.json"),
        ]
    )

    assert args.reference_bitmatrix is True
    assert args.bitmatrix_depth == 5


def test_evaluation_rejects_non_positive_torch_threads(tmp_path):
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--challenger",
                str(tmp_path / "model.pt"),
                "--reference-random",
                "--output",
                str(tmp_path / "report.json"),
                "--torch-threads",
                "0",
            ]
        )


@pytest.mark.parametrize(
    "flag",
    ["--challenger-expansion-batch-size", "--reference-expansion-batch-size"],
)
def test_evaluation_rejects_non_positive_expansion_batch_size(tmp_path, flag):
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--challenger",
                str(tmp_path / "model.pt"),
                "--reference-model",
                str(tmp_path / "reference.pt"),
                "--output",
                str(tmp_path / "report.json"),
                flag,
                "0",
            ]
        )


def test_batched_model_arena_preserves_opening_order(monkeypatch, tmp_path):
    captured = {}

    def fake_evaluate_models(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return (3, 2, 1, 101, 99)

    monkeypatch.setattr(evaluate_main, "evaluate_models", fake_evaluate_models)
    openings = generate_openings(count=3, plies=4, seed=7)

    result = _run_batched_model_arena(
        tmp_path / "challenger.pt",
        tmp_path / "reference.pt",
        openings,
        device="cuda",
        simulations=100,
        c_puct=1.5,
        expansion_batch_size=4,
    )

    assert result == (3, 2, 1, 101, 99)
    assert captured["args"][2] == [list(opening.moves) for opening in openings]
    assert captured["kwargs"]["game_concurrency"] == 6
    mcts = captured["kwargs"]["mcts"]
    assert mcts.num_simulations == 100
    assert mcts.expansion_batch_size == 4
