import json

import pytest
from rust_reversi import Board, Turn

from reversi_zero_trainer.openings import (
    OpeningController,
    compile_opening,
    generate_openings,
    load_openings,
    write_opening_suite,
)


def test_generate_openings_is_reproducible_and_unique():
    first = generate_openings(count=8, plies=6, seed=123)
    second = generate_openings(count=8, plies=6, seed=123)

    assert first == second
    final_states = {
        (compile_opening(item).final_board, compile_opening(item).final_turn)
        for item in first
    }
    assert len(final_states) == 8


def test_load_openings_from_evaluation_report(tmp_path):
    openings = generate_openings(count=2, plies=4, seed=5)
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps({"summary": {}, "openings": [item.to_dict() for item in openings]})
    )

    assert load_openings(report) == openings


def test_opening_controller_tracks_games_for_each_arena_process(tmp_path):
    openings = generate_openings(count=2, plies=4, seed=9)
    suite = tmp_path / "openings.json"
    write_opening_suite(openings, suite)
    controllers = {
        "BLACK": OpeningController(suite),
        "WHITE": OpeningController(suite),
    }

    for opening in openings:
        board = Board()
        for expected_move in opening.moves:
            color = "BLACK" if board.get_turn() == Turn.BLACK else "WHITE"
            actual_move = controllers[color].select_forced_move(
                board.get_board_line(), color
            )
            assert actual_move == expected_move
            board.do_move(actual_move)
        while not board.is_game_over() and board.piece_sum() < 20:
            if board.is_pass():
                board.do_pass()
            else:
                color = "BLACK" if board.get_turn() == Turn.BLACK else "WHITE"
                assert (
                    controllers[color].select_forced_move(board.get_board_line(), color)
                    is None
                )
                board.do_move(board.get_legal_moves_vec()[0])


def test_generate_openings_rejects_multiple_zero_ply_openings():
    with pytest.raises(ValueError, match="Only one unique opening"):
        generate_openings(count=2, plies=0, seed=0)
