#!/bin/bash
# Monitoring script to check cron job success and failures
# Run this manually to check the health of your scheduled jobs

echo "=== Weather & Energy Pipeline Cron Job Monitor ==="
echo "Generated: $(date)"
echo ""

# Check daily pipeline success
echo "📊 DAILY PIPELINE STATUS"
echo "Recent daily runs (last 5):"
if [ -f "logs/cron_daily.log" ]; then
    tail -20 logs/cron_daily.log | grep -E "(Status:|Error:|Pipeline execution)" | tail -5
    echo ""
    echo "Last daily run result:"
    tail -5 logs/cron_daily.log | grep -E "Status:|Records Processed:|Quality Score:" | tail -3
else
    echo "❌ Daily log file not found"
fi
echo ""

# Check quality pipeline 
echo "🔍 QUALITY CHECK STATUS"
if [ -f "logs/cron_quality.log" ]; then
    echo "Last quality check:"
    tail -10 logs/cron_quality.log | grep -E "Status:|Quality Score:" | tail -2
else
    echo "❌ Quality log file not found"
fi
echo ""

# Check monthly backfill
echo "📈 MONTHLY BACKFILL STATUS"  
if [ -f "logs/cron_monthly.log" ]; then
    echo "Last monthly backfill:"
    tail -10 logs/cron_monthly.log | grep -E "Status:|Records Processed:" | tail -2
else
    echo "❌ Monthly log file not found"
fi
echo ""

# Check data freshness
echo "🕒 DATA FRESHNESS"
if [ -d "data/processed" ]; then
    latest_file=$(ls -t data/processed/processed_data_*.csv 2>/dev/null | head -1)
    if [ -n "$latest_file" ]; then
        echo "Latest processed data: $(basename "$latest_file")"
        echo "File age: $(stat -c '%y' "$latest_file" | cut -d' ' -f1-2)"
        echo "Records: $(tail -n +2 "$latest_file" | wc -l)"
    else
        echo "❌ No processed data files found"
    fi
else
    echo "❌ Processed data directory not found"
fi
echo ""

# Check for errors in recent logs
echo "⚠️  RECENT ERRORS"
echo "Checking for errors in last 100 log lines..."
for log_file in logs/cron_*.log; do
    if [ -f "$log_file" ]; then
        errors=$(tail -100 "$log_file" | grep -i -E "error|failed|exception" | wc -l)
        if [ "$errors" -gt 0 ]; then
            echo "$(basename "$log_file"): $errors error(s) found"
            tail -100 "$log_file" | grep -i -E "error|failed|exception" | tail -3
        fi
    fi
done
echo ""

echo "=== Monitor Complete ==="
echo "💡 TIP: Set up this script to run weekly and email results for proactive monitoring"