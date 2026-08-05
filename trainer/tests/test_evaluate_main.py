import pytest

from reversi_zero_trainer.evaluate_main import (
    _mcts_command,
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
