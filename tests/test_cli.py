from src.cli import build_parser


def test_cli_parser_accepts_daily_command():
    parser = build_parser()
    args = parser.parse_args(["scan-daily", "--", "--limit", "10"])
    assert args.command == "scan-daily"


def test_cli_parser_accepts_backtest_command():
    parser = build_parser()
    args = parser.parse_args(["backtest", "--", "--ticker", "AAPL"])
    assert args.command == "backtest"
