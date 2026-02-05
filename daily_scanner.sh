#!/bin/bash
# Daily market scanner - runs full scan of all US stocks
# Designed to run at 6:30 AM EST daily via cron

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/data/logs"
LOG_FILE="$LOG_DIR/daily_scan_$(date +%Y%m%d).log"

# Create log directory
mkdir -p "$LOG_DIR"

# Log start
echo "========================================" | tee -a "$LOG_FILE"
echo "Daily Market Scan Started" | tee -a "$LOG_FILE"
echo "Time: $(date '+%Y-%m-%d %H:%M:%S %Z')" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

# Change to script directory
cd "$SCRIPT_DIR"

# Activate virtual environment
source venv/bin/activate

# Run the scan with resume capability (in case of previous interruption)
python run_full_market_scan.py --resume 2>&1 | tee -a "$LOG_FILE"

# Log completion
echo "========================================" | tee -a "$LOG_FILE"
echo "Daily Market Scan Completed" | tee -a "$LOG_FILE"
echo "Time: $(date '+%Y-%m-%d %H:%M:%S %Z')" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

# Keep only last 30 days of logs
find "$LOG_DIR" -name "daily_scan_*.log" -mtime +30 -delete

# Deactivate virtual environment
deactivate
