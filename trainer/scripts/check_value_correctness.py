#!/usr/bin/env python3
"""Check if values match game outcomes correctly by analyzing state transitions."""

import sys
from pathlib import Path

import numpy as np


def find_game_boundaries(states: np.ndarray) -> list[int]:
    """Find game boundaries by detecting board resets (mostly empty boards)."""
    boundaries = [0]

    for i in range(1, len(states)):
        # Count total pieces on the board
        # states shape: (N, 3, 8, 8) where planes are [player, opponent, empty]
        total_pieces = states[i, 0].sum() + states[i, 1].sum()

        # If very few pieces (<=4, which is initial state), likely a new game
        if total_pieces <= 4:
            boundaries.append(i)

    boundaries.append(len(states))  # Add end boundary
    return boundaries


def check_value_correctness(base_path: str) -> None:
    """Verify values match game outcomes."""
    print(f"Checking value correctness for: {base_path}")
    print("=" * 70)

    # Load data
    states = np.load(f"{base_path}_states.npy")
    values = np.load(f"{base_path}_values.npy")

    print(f"\nTotal examples: {len(values)}")

    # Find game boundaries
    boundaries = find_game_boundaries(states)
    n_games = len(boundaries) - 1

    print(f"Detected {n_games} games:")
    for i in range(n_games):
        start, end = boundaries[i], boundaries[i + 1]
        n_moves = end - start
        print(f"  Game {i+1}: moves {start}-{end-1} ({n_moves} moves)")

    errors = 0

    # Analyze each game
    for game_idx in range(n_games):
        start, end = boundaries[game_idx], boundaries[game_idx + 1]
        game_values = values[start:end]
        n_moves = len(game_values)

        print("\n" + "=" * 70)
        print(f"Game {game_idx + 1} Analysis ({n_moves} moves):")
        print("-" * 70)

        # Determine outcome from values
        # In a consistent game, all Black moves have same sign, all White moves have opposite sign
        black_values = game_values[::2]  # Even indices
        white_values = game_values[1::2]  # Odd indices

        # Check what the outcome should be
        black_mean = black_values.mean()
        white_mean = white_values.mean()

        if abs(black_mean - 1.0) < 0.5:
            outcome = "BlackWin"
            expected_black = 1.0
            expected_white = -1.0
        elif abs(black_mean + 1.0) < 0.5:
            outcome = "WhiteWin"
            expected_black = -1.0
            expected_white = 1.0
        elif abs(black_mean) < 0.5:
            outcome = "Draw"
            expected_black = 0.0
            expected_white = 0.0
        else:
            outcome = "Unknown"
            expected_black = black_mean
            expected_white = white_mean

        print(f"Detected outcome: {outcome}")
        print(f"  Expected: Black={expected_black:+.1f}, White={expected_white:+.1f}")

        # Show first 10 moves
        print("\nFirst 10 moves:")
        print("Move | Turn  | Value | Expected | Correct?")
        print("-" * 70)

        for i in range(min(10, n_moves)):
            abs_idx = start + i
            is_black_turn = i % 2 == 0
            turn = "Black" if is_black_turn else "White"
            value = game_values[i]
            expected = expected_black if is_black_turn else expected_white

            correct = abs(value - expected) < 0.01
            if not correct:
                errors += 1

            status = "✓" if correct else "✗"
            print(
                f"{abs_idx:4d} | {turn:5s} | {value:+5.1f} | {expected:+8.1f} | {status}"
            )

        # Show last 10 moves
        if n_moves > 10:
            print("\nLast 10 moves:")
            print("Move | Turn  | Value | Expected | Correct?")
            print("-" * 70)

            for i in range(max(0, n_moves - 10), n_moves):
                abs_idx = start + i
                is_black_turn = i % 2 == 0
                turn = "Black" if is_black_turn else "White"
                value = game_values[i]
                expected = expected_black if is_black_turn else expected_white

                correct = abs(value - expected) < 0.01
                if not correct:
                    errors += 1

                status = "✓" if correct else "✗"
                print(
                    f"{abs_idx:4d} | {turn:5s} | {value:+5.1f} | {expected:+8.1f} | {status}"
                )

        # Summary for this game
        print(f"\nSummary for Game {game_idx + 1}:")
        print("  Black turn values:")
        print(f"    Mean: {black_mean:+.3f}, Expected: {expected_black:+.1f}")
        black_correct = np.all(np.abs(black_values - expected_black) < 0.01)
        print(f"    All correct?: {black_correct}")
        if not black_correct:
            errors += np.sum(np.abs(black_values - expected_black) >= 0.01)
            print(f"    Unique values: {np.unique(black_values)}")

        print("  White turn values:")
        print(f"    Mean: {white_mean:+.3f}, Expected: {expected_white:+.1f}")
        white_correct = np.all(np.abs(white_values - expected_white) < 0.01)
        print(f"    All correct?: {white_correct}")
        if not white_correct:
            errors += np.sum(np.abs(white_values - expected_white) >= 0.01)
            print(f"    Unique values: {np.unique(white_values)}")

    # Final summary
    print("\n" + "=" * 70)
    if errors == 0:
        print("✓ All values are correct!")
    else:
        print(f"✗ Found {errors} incorrect values")
    print("=" * 70)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)

    base_path = sys.argv[1]

    for suffix in ["_states.npy", "_values.npy"]:
        if not Path(f"{base_path}{suffix}").exists():
            print(f"Error: File not found: {base_path}{suffix}")
            sys.exit(1)

    check_value_correctness(base_path)
