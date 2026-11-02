"""Unified v4.1 command-line entry point.

Only `--help` is provided here; each later task registers its own
subcommand. Argument errors are reported as a fixed code so that private
paths passed on the command line never reach a log.
"""
from __future__ import annotations

import argparse
import sys


class PrivateParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        self.print_usage(sys.stderr)
        self.exit(2, f"{self.prog}: error: invalid_arguments\n")


from pathlib import Path

REGISTRARS = ("anatobind.v41.inventory", "anatobind.v41.splits", "anatobind.v41.geometry_qc", "anatobind.v41.overlays",
              "anatobind.v41.registry")


def package_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def build_parser() -> PrivateParser:
    import importlib

    parser = PrivateParser(
        prog="anatobind.v41",
        description="AnatoBind-MRI v4.1 command contract (anatobind.v41). "
        "Subcommands appear only after their task is implemented and tested.",
    )
    subparsers = parser.add_subparsers(dest="command")
    for module_name in REGISTRARS:
        importlib.import_module(module_name).register_cli(subparsers)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
