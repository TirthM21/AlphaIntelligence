#!/bin/bash
# Quick throttling detection script

LOG_FILE="./data/logs/optimized_scan_$(date +%Y%m%d).log"

if [ ! -f "$LOG_FILE" ]; then
    echo "❌ No log file found for today"
    echo "   Expected: $LOG_FILE"
    exit 0
fi

echo ""
echo "🔍 THROTTLING DETECTION REPORT"
echo "================================"
echo ""

# Count different error types
ERROR_429=$(grep -c "429\|Too Many Requests" "$LOG_FILE" 2>/dev/null || echo "0")
TIMEOUTS=$(grep -c -i "timeout\|timed out" "$LOG_FILE" 2>/dev/null || echo "0")
FAILED=$(grep -c "Failed to fetch" "$LOG_FILE" 2>/dev/null || echo "0")
EMPTY_DATA=$(grep -c "Insufficient data (0 days)" "$LOG_FILE" 2>/dev/null || echo "0")

echo "Error Breakdown:"
echo "  429 'Too Many Requests': $ERROR_429"
echo "  Timeouts: $TIMEOUTS"
echo "  Failed Fetches: $FAILED"
echo "  Empty Data Responses: $EMPTY_DATA"
echo ""

TOTAL_ERRORS=$((ERROR_429 + TIMEOUTS + FAILED))

echo "Total Error Events: $TOTAL_ERRORS"
echo ""

# Determine status
if [ $ERROR_429 -gt 50 ] || [ $TOTAL_ERRORS -gt 150 ]; then
    echo "🚨 STATUS: BEING THROTTLED!"
    echo ""
    echo "   Yahoo is rate limiting your requests."
    echo ""
    echo "   WHAT TO DO:"
    echo "   1. Stop the current scan (Ctrl+C if running)"
    echo "   2. Wait 10 minutes: sleep 600"
    echo "   3. Resume with: python run_optimized_scan.py --conservative --resume"
    echo ""
elif [ $ERROR_429 -gt 20 ] || [ $TOTAL_ERRORS -gt 75 ]; then
    echo "⚠️  STATUS: APPROACHING LIMIT"
    echo ""
    echo "   You're getting close to Yahoo's rate limit."
    echo ""
    echo "   WHAT TO DO:"
    echo "   1. If scan is running, let it finish"
    echo "   2. Next run, use: python run_optimized_scan.py --conservative"
    echo "   3. Monitor error rate closely"
    echo ""
elif [ $TOTAL_ERRORS -gt 20 ]; then
    echo "ℹ️  STATUS: MINOR ERRORS (Normal)"
    echo ""
    echo "   Some errors are normal. Error rate appears acceptable."
    echo ""
    echo "   You're fine to continue."
    echo ""
else
    echo "✅ STATUS: ALL GOOD!"
    echo ""
    echo "   Very low error rate. No throttling detected."
    echo ""
    echo "   You can safely continue or increase speed if desired."
    echo ""
fi

# Check for progress file to show error rate
PROGRESS_FILE="./data/batch_results/batch_progress.pkl"
if [ -f "$PROGRESS_FILE" ]; then
    echo "Recent Scan Statistics:"
    python3 -c "
import pickle
try:
    with open('$PROGRESS_FILE', 'rb') as f:
        data = pickle.load(f)
    error_rate = data.get('error_rate', 0) * 100
    processed = len(data.get('processed', []))
    total = data.get('total_tickers', 0)

    print(f'  Processed: {processed:,}/{total:,} stocks')
    print(f'  Error Rate: {error_rate:.2f}%')

    if error_rate > 5:
        print('  ⚠️  High error rate!')
    elif error_rate > 2:
        print('  ℹ️  Moderate error rate')
    else:
        print('  ✅ Low error rate')
except Exception as e:
    print(f'  Could not read progress file: {e}')
" 2>/dev/null
fi

echo ""
echo "================================"
echo ""
