"""Command-line entry points for the future cleaned Assembly Copilot implementation."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Assembly Copilot command-line interface")
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo_parser = subparsers.add_parser("demo", help="Run the Assembly Copilot demo")
    demo_parser.add_argument("--config", required=True, help="Path to a demo YAML config")

    return parser


def main() -> None:
    args = build_parser().parse_args()
    raise NotImplementedError(
        f"The '{args.command}' command is a placeholder until the cleaned code is migrated."
    )


if __name__ == "__main__":
    main()
