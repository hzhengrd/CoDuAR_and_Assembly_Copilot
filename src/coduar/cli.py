"""Command-line entry points for the future cleaned CoDuAR implementation."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CoDuAR command-line interface")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train CoDuAR")
    train_parser.add_argument("--config", required=True, help="Path to a training YAML config")

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate CoDuAR")
    eval_parser.add_argument("--config", required=True, help="Path to an evaluation YAML config")

    return parser


def main() -> None:
    args = build_parser().parse_args()
    raise NotImplementedError(
        f"The '{args.command}' command is a placeholder until the cleaned code is migrated."
    )


if __name__ == "__main__":
    main()
