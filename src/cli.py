import sys
import os
import argparse
import logging
sys.path.append(os.getcwd())
from src.analysis.position_manager import PositionManager
from src.data.yfinance_positions import YFinancePositionFetcher
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def scan_daily():
    print("Executing daily scan...")
    # Wrap run_optimized_scan logic here
    # For now, just a placeholder or call the script via subprocess if moving logic is too big
    import subprocess
    subprocess.run([sys.executable, "run_optimized_scan.py"], check=True)

def scan_quarterly():
    print("Executing quarterly compounder scan...")
    import subprocess
    subprocess.run([sys.executable, "run_quarterly_compounder_scan.py"], check=True)

def report_ai():
    print("Generating AI report...")
    import subprocess
    subprocess.run([sys.executable, "run_ai_report.py"], check=True)

def manage_positions_cli():
    print("Managing positions...")
    # Inline manage_positions.py logic could go here
    import subprocess
    subprocess.run([sys.executable, "manage_positions.py"], check=True)

def main():
    parser = argparse.ArgumentParser(description="AlphaIntelligence CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # alphaintel scan-daily
    subparsers.add_parser("scan-daily", help="Run the daily momentum market scan")

    # alphaintel scan-quarterly
    subparsers.add_parser("scan-quarterly", help="Run the quarterly compounder scan")

    # alphaintel report-ai
    subparsers.add_parser("report-ai", help="Generate AI deep-dive report")
    
    # alphaintel manage-positions
    subparsers.add_parser("manage-positions", help="Manage portfolio positions")

    args = parser.parse_args()

    if args.command == "scan-daily":
        scan_daily()
    elif args.command == "scan-quarterly":
        scan_quarterly()
    elif args.command == "report-ai":
        report_ai()
    elif args.command == "manage-positions":
        manage_positions_cli()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
