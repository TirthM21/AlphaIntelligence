"""Unified CLI entrypoint for AlphaIntelligence workflows."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _run_script(script_name: str, passthrough: list[str]) -> int:
    script_path = Path.cwd() / script_name
    if not script_path.exists():
        print(f"Script not found: {script_name}")
        return 2
    cmd = [sys.executable, str(script_path), *passthrough]
    return subprocess.call(cmd)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="alphaintel", description="AlphaIntelligence unified CLI")
    sub = parser.add_subparsers(dest="command")

    daily = sub.add_parser("scan-daily", help="Run daily momentum scan")
    daily.add_argument("args", nargs=argparse.REMAINDER)

    quarterly = sub.add_parser("scan-quarterly", help="Run quarterly compounder scan")
    quarterly.add_argument("args", nargs=argparse.REMAINDER)

    ai = sub.add_parser("report-ai", help="Generate AI deep-dive report")
    ai.add_argument("args", nargs=argparse.REMAINDER)

    backtest = sub.add_parser("backtest", help="Run strategy backtest")
    backtest.add_argument("args", nargs=argparse.REMAINDER)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "scan-daily":
        return _run_script("run_optimized_scan.py", args.args)
    if args.command == "scan-quarterly":
        return _run_script("run_quarterly_compounder_scan.py", args.args)
    if args.command == "report-ai":
        return _run_script("run_ai_report.py", args.args)
    if args.command == "backtest":
        return _run_script("run_backtest.py", args.args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
