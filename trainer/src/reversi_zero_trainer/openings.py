"""Deterministic opening suites for paired Arena evaluations."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rust_reversi import Board, Turn


def _turn_name(turn: Turn) -> str:
    return "BLACK" if turn == Turn.BLACK else "WHITE"


@dataclass(frozen=True)
class Opening:
    """A legal move sequence from the standard initial position."""

    moves: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"moves": list(self.moves)}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Opening:
        moves = payload.get("moves")
        if not isinstance(moves, list) or not all(
            isinstance(move, int) for move in moves
        ):
            raise ValueError("Each opening must contain an integer 'moves' list")
        opening = cls(tuple(moves))
        compile_opening(opening)
        return opening


@dataclass(frozen=True)
class CompiledOpening:
    forced_moves: dict[tuple[str, str], int]
    final_piece_count: int
    final_board: str
    final_turn: str


def compile_opening(opening: Opening) -> CompiledOpening:
    """Validate an opening and map each pre-move board to its forced move."""

    board = Board()
    forced_moves: dict[tuple[str, str], int] = {}

    for move in opening.moves:
        while board.is_pass() and not board.is_game_over():
            board.do_pass()
        if board.is_game_over():
            raise ValueError("Opening continues after the game has ended")

        key = (board.get_board_line(), _turn_name(board.get_turn()))
        if not board.is_legal_move(move):
            raise ValueError(f"Illegal opening move {move} at ply {len(forced_moves)}")
        forced_moves[key] = move
        board.do_move(move)

    return CompiledOpening(
        forced_moves=forced_moves,
        final_piece_count=board.piece_sum(),
        final_board=board.get_board_line(),
        final_turn=_turn_name(board.get_turn()),
    )


def generate_openings(count: int, plies: int, seed: int) -> list[Opening]:
    """Generate unique legal openings with a reproducible RNG seed."""

    if count <= 0:
        raise ValueError("Opening count must be positive")
    if plies < 0:
        raise ValueError("Opening plies must be non-negative")
    if plies == 0 and count > 1:
        raise ValueError("Only one unique opening exists when plies is zero")

    rng = random.Random(seed)  # nosec B311 - reproducible game openings
    openings: list[Opening] = []
    seen: set[tuple[str, str]] = set()
    max_attempts = max(1000, count * 100)

    for _ in range(max_attempts):
        board = Board()
        moves: list[int] = []

        while len(moves) < plies:
            if board.is_game_over():
                break
            if board.is_pass():
                board.do_pass()
                continue
            move = rng.choice(board.get_legal_moves_vec())
            moves.append(move)
            board.do_move(move)

        if len(moves) != plies:
            continue

        key = (board.get_board_line(), _turn_name(board.get_turn()))
        if key in seen:
            continue
        seen.add(key)
        openings.append(Opening(tuple(moves)))
        if len(openings) == count:
            return openings

    raise ValueError(
        f"Could only generate {len(openings)} unique openings "
        f"for count={count}, plies={plies}"
    )


def load_openings(path: Path) -> list[Opening]:
    """Load openings from a suite file or a previous evaluation report."""

    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    raw_openings = payload.get("openings") if isinstance(payload, dict) else payload
    if not isinstance(raw_openings, list) or not raw_openings:
        raise ValueError(f"No openings found in {path}")
    if not all(isinstance(item, dict) for item in raw_openings):
        raise ValueError("Openings must be JSON objects")
    return [Opening.from_dict(item) for item in raw_openings]


def write_opening_suite(openings: list[Opening], path: Path) -> None:
    """Write a temporary suite consumed by Arena player processes."""

    payload = {"schema_version": 1, "openings": [item.to_dict() for item in openings]}
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


class OpeningController:
    """Track Arena game resets and return forced opening moves when applicable."""

    def __init__(self, openings_path: Path) -> None:
        self._openings = [
            compile_opening(item) for item in load_openings(openings_path)
        ]
        self._game_index = 0
        self._last_piece_count: int | None = None

    def select_forced_move(self, board_line: str, turn_name: str) -> int | None:
        board = Board()
        turn = Turn.BLACK if turn_name.upper() == "BLACK" else Turn.WHITE
        board.set_board_str(board_line, turn)
        piece_count = board.piece_sum()

        if self._last_piece_count is not None and piece_count < self._last_piece_count:
            self._game_index += 1
        self._last_piece_count = piece_count

        if self._game_index >= len(self._openings):
            raise RuntimeError(
                f"Arena requested game {self._game_index + 1}, but the opening suite "
                f"contains only {len(self._openings)} games per color assignment"
            )

        opening = self._openings[self._game_index]
        key = (board_line, turn_name.upper())
        forced_move = opening.forced_moves.get(key)
        if forced_move is not None:
            return forced_move

        if piece_count < opening.final_piece_count:
            raise RuntimeError(
                f"Game {self._game_index + 1} diverged from its forced opening "
                f"before ply {opening.final_piece_count - 4}"
            )
        return None
