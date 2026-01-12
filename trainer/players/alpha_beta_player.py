"""Simple alpha-beta player using rust_reversi search (baseline)."""

from __future__ import annotations

import sys

from rust_reversi import AlphaBetaSearch, Board, PieceEvaluator, Turn

# Maximum search depth
DEPTH = 3


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: alpha_beta_player.py <BLACK|WHITE>", file=sys.stderr)
        sys.exit(1)

    turn = Turn.BLACK if sys.argv[1].upper() == "BLACK" else Turn.WHITE
    board = Board()

    evaluator = PieceEvaluator()
    search = AlphaBetaSearch(evaluator, DEPTH, win_score=1 << 10)

    while True:
        try:
            board_str = input().strip()

            if board_str.lower() == "ping":
                print("pong", flush=True)
                continue

            board.set_board_str(board_str, turn)
            move = search.get_move(board)
            print(move, flush=True)
        except Exception as exc:  # pragma: no cover
            print(exc, file=sys.stderr, flush=True)
            sys.exit(1)


if __name__ == "__main__":
    main()
