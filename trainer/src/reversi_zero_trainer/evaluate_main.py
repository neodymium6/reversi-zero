"""Paired model evaluation using rust_reversi.Arena."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import torch
from rust_reversi import Arena

from reversi_zero_trainer.openings import (
    Opening,
    generate_openings,
    load_openings,
    write_opening_suite,
)


def score_interval(wins: int, losses: int, draws: int) -> tuple[float, float]:
    """Approximate a 95% Wilson interval, counting each draw as half a win."""

    games = wins + losses + draws
    if games <= 0:
        raise ValueError("At least one game is required")
    points = wins + 0.5 * draws
    proportion = points / games
    z = 1.959963984540054
    denominator = 1.0 + z * z / games
    center = (proportion + z * z / (2.0 * games)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / games + z * z / (4.0 * games * games)
        )
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def opening_suite_sha256(openings: list[Opening]) -> str:
    encoded = json.dumps(
        [opening.to_dict() for opening in openings],
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def write_report(report: dict[str, Any], path: Path, overwrite: bool = False) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    next_path = path.with_name(f".{path.name}.next")
    if next_path.exists():
        raise FileExistsError(f"Refusing ambiguous partial report: {next_path}")
    try:
        with next_path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
        next_path.replace(path)
    except Exception:
        next_path.unlink(missing_ok=True)
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a challenger with paired fixed openings"
    )
    parser.add_argument("--challenger", type=Path, required=True)
    reference = parser.add_mutually_exclusive_group(required=True)
    reference.add_argument("--reference-model", type=Path)
    reference.add_argument("--reference-alphabeta", action="store_true")
    reference.add_argument("--reference-random", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--openings-from", type=Path)
    parser.add_argument("--num-openings", type=int, default=20)
    parser.add_argument("--opening-plies", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=1,
        help="Torch CPU threads per MCTS player process (default: 1)",
    )
    parser.add_argument("--simulations", type=int, default=400)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--challenger-expansion-batch-size", type=int, default=1)
    parser.add_argument("--reference-expansion-batch-size", type=int, default=1)
    parser.add_argument("--alphabeta-depth", type=int, default=3)
    parser.add_argument("--promotion-threshold", type=float, default=0.55)
    parser.add_argument(
        "--show-progress", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args(argv)

    if args.num_openings <= 0:
        parser.error("--num-openings must be positive")
    if args.opening_plies < 0:
        parser.error("--opening-plies must be non-negative")
    if args.simulations <= 0:
        parser.error("--simulations must be positive")
    if args.torch_threads <= 0:
        parser.error("--torch-threads must be positive")
    if args.challenger_expansion_batch_size <= 0:
        parser.error("--challenger-expansion-batch-size must be positive")
    if args.reference_expansion_batch_size <= 0:
        parser.error("--reference-expansion-batch-size must be positive")
    if args.alphabeta_depth <= 0:
        parser.error("--alphabeta-depth must be positive")
    if not 0.0 <= args.promotion_threshold <= 1.0:
        parser.error("--promotion-threshold must be between 0 and 1")
    return args


def _mcts_command(
    model: Path,
    openings_path: Path,
    device: str,
    simulations: int,
    c_puct: float,
    torch_threads: int,
    expansion_batch_size: int,
) -> list[str]:
    players_dir = Path(__file__).resolve().parents[2] / "players"
    command = [
        sys.executable,
        str(players_dir / "mcts_player.py"),
        "--model",
        str(model),
        "--sims",
        str(simulations),
        "--c-puct",
        str(c_puct),
        "--temperature",
        "0.0",
        "--openings-file",
        str(openings_path),
        "--expansion-batch-size",
        str(expansion_batch_size),
    ]
    command.extend(["--device", device])
    if device == "cpu":
        command.extend(["--torch-threads", str(torch_threads)])
    return command


def _reference_command(
    args: argparse.Namespace, openings_path: Path, actual_device: str
) -> tuple[list[str], dict[str, Any]]:
    players_dir = Path(__file__).resolve().parents[2] / "players"
    if args.reference_model is not None:
        model = args.reference_model.resolve()
        if not model.is_file():
            raise FileNotFoundError(f"Reference model not found: {model}")
        return (
            _mcts_command(
                model,
                openings_path,
                actual_device,
                args.simulations,
                args.c_puct,
                args.torch_threads,
                args.reference_expansion_batch_size,
            ),
            {"type": "model", "path": str(model), "sha256": file_sha256(model)},
        )
    if args.reference_alphabeta:
        return (
            [
                sys.executable,
                str(players_dir / "alpha_beta_player.py"),
                "--depth",
                str(args.alphabeta_depth),
                "--openings-file",
                str(openings_path),
            ],
            {"type": "alphabeta", "depth": args.alphabeta_depth},
        )
    return (
        [
            sys.executable,
            str(players_dir / "random_player.py"),
            "--seed",
            str(args.seed),
            "--openings-file",
            str(openings_path),
        ],
        {"type": "random", "seed": args.seed},
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    challenger = args.challenger.resolve()
    if not challenger.is_file():
        raise FileNotFoundError(f"Challenger model not found: {challenger}")

    actual_device = (
        "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    )
    if actual_device == "auto":
        actual_device = "cpu"
    if args.openings_from is not None:
        openings = load_openings(args.openings_from)
        opening_source = str(args.openings_from.resolve())
    else:
        openings = generate_openings(args.num_openings, args.opening_plies, args.seed)
        opening_source = "generated"

    with tempfile.TemporaryDirectory(prefix="reversi-zero-eval-") as tmp_dir:
        openings_path = Path(tmp_dir) / "openings.json"
        write_opening_suite(openings, openings_path)
        challenger_command = _mcts_command(
            challenger,
            openings_path,
            actual_device,
            args.simulations,
            args.c_puct,
            args.torch_threads,
            args.challenger_expansion_batch_size,
        )
        reference_command, reference_metadata = _reference_command(
            args, openings_path, actual_device
        )

        arena = Arena(
            challenger_command, reference_command, show_progress=args.show_progress
        )
        arena.play_n(len(openings) * 2)
        wins, losses, draws = arena.get_stats()
        challenger_pieces, reference_pieces = arena.get_pieces()

    games = wins + losses + draws
    score = (wins + 0.5 * draws) / games
    interval_low, interval_high = score_interval(wins, losses, draws)
    statistically_stronger = score >= args.promotion_threshold and interval_low > 0.5
    opening_lengths = sorted({len(opening.moves) for opening in openings})

    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "challenger": {
            "type": "model",
            "path": str(challenger),
            "sha256": file_sha256(challenger),
        },
        "reference": reference_metadata,
        "config": {
            "device": actual_device,
            "torch_threads": args.torch_threads if actual_device == "cpu" else None,
            "simulations": args.simulations,
            "c_puct": args.c_puct,
            "temperature": 0.0,
            "challenger_expansion_batch_size": (args.challenger_expansion_batch_size),
            "reference_expansion_batch_size": (
                args.reference_expansion_batch_size
                if args.reference_model is not None
                else None
            ),
            "opening_source": opening_source,
            "opening_seed": args.seed if args.openings_from is None else None,
            "opening_plies": (
                opening_lengths[0] if len(opening_lengths) == 1 else opening_lengths
            ),
            "promotion_threshold": args.promotion_threshold,
            "confidence_interval": "95% Wilson score; draws count as half a win",
        },
        "opening_suite_sha256": opening_suite_sha256(openings),
        "openings": [opening.to_dict() for opening in openings],
        "summary": {
            "openings": len(openings),
            "games": games,
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "score": score,
            "score_interval_95": [interval_low, interval_high],
            "challenger_pieces": challenger_pieces,
            "reference_pieces": reference_pieces,
            "statistically_stronger": statistically_stronger,
        },
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    report = run(args)
    write_report(report, args.output, overwrite=args.overwrite)
    summary = report["summary"]
    interval = summary["score_interval_95"]
    print(
        f"Score: {summary['score']:.3f} "
        f"(95% CI {interval[0]:.3f}-{interval[1]:.3f}); "
        f"W/D/L {summary['wins']}/{summary['draws']}/{summary['losses']}"
    )
    print(f"Statistically stronger: {summary['statistically_stronger']}")
    print(f"Report: {args.output.resolve()}")


if __name__ == "__main__":
    main()
