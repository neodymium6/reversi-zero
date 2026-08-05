import pytest

from reversi_zero_trainer.evaluate_main import (
    opening_suite_sha256,
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
