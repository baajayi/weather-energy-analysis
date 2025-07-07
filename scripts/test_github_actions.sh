#!/bin/bash
# Test script to simulate GitHub Actions workflow locally

echo "=== GitHub Actions Local Test ==="
echo "This script simulates the GitHub Actions workflow locally"
echo ""

# Simulate the workflow steps
echo "1. Clearing Python cache..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true

echo "2. Creating runtime config (simulating secret injection)..."
cp config/config_test.yaml config/config_runtime.yaml
echo "   ✅ Runtime config created"

echo "3. Running pipeline with runtime config..."
python src/pipeline.py --mode daily --config config/config_runtime.yaml

echo "4. Checking results..."
if [ -f data/processed/processed_data_*.csv ]; then
    latest_file=$(ls -t data/processed/processed_data_*.csv | head -1)
    echo "   Latest processed file: $(basename "$latest_file")"
    echo "   Record count: $(tail -n +2 "$latest_file" | wc -l)"
    echo "   Cities in data: $(tail -n +2 "$latest_file" | cut -d',' -f2 | sort | uniq | tr '\n' ', ')"
    
    # Check for Seattle specifically
    seattle_records=$(tail -n +2 "$latest_file" | grep -c "Seattle" || echo "0")
    echo "   Seattle records: $seattle_records"
    
    if [ "$seattle_records" -gt 0 ]; then
        echo "   ✅ SUCCESS: Seattle data found in processed output"
    else
        echo "   ❌ ISSUE: No Seattle data in processed output"
    fi
else
    echo "   ❌ ERROR: No processed data files found!"
    exit 1
fi

echo ""
echo "5. Cleaning up test artifacts..."
rm -f config/config_runtime.yaml

echo "=== Test Complete ==="