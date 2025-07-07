#!/usr/bin/env python3
"""
Diagnostic script to check which version of the code is running.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def main():
    print("=== CODE VERSION DIAGNOSTIC ===")
    print(f"Python executable: {sys.executable}")
    print(f"Working directory: {os.getcwd()}")
    print(f"Python path: {sys.path[:3]}...")
    print()
    
    # Check the data_fetcher source code
    try:
        data_fetcher_path = Path(__file__).parent.parent / "src" / "data_fetcher.py"
        with open(data_fetcher_path, 'r') as f:
            content = f.read()
            
        print("=== CHECKING data_fetcher.py ===")
        
        # Check for our fixed log message
        if "Fetching daily data with aligned date ranges" in content:
            print("✅ GOOD: Found aligned date ranges log message")
        else:
            print("❌ BAD: Aligned date ranges log message NOT found")
            
        # Check for old problematic patterns
        if "Weather.*Energy" in content and "Fetching daily data -" in content:
            print("❌ BAD: Found old separate date range log pattern")
        else:
            print("✅ GOOD: No old separate date range patterns found")
            
        # Check the actual date calculation
        if "energy_end_date = end_date" in content and "weather_end_date = end_date" in content:
            print("✅ GOOD: Found aligned date calculation")
        else:
            print("❌ BAD: Aligned date calculation NOT found")
            
        print()
        
    except Exception as e:
        print(f"❌ ERROR reading data_fetcher.py: {e}")
    
    # Check what happens when we import the module
    print("=== RUNTIME IMPORT TEST ===")
    try:
        from data_fetcher import DataFetcher
        
        # Check if we can create an instance
        fetcher = DataFetcher('config/config.yaml')
        print("✅ GOOD: DataFetcher import successful")
        
        # Try to get source code of the method
        import inspect
        source = inspect.getsource(fetcher.fetch_daily_data)
        
        if "aligned date ranges" in source:
            print("✅ GOOD: Runtime method has aligned date ranges")
        else:
            print("❌ BAD: Runtime method does NOT have aligned date ranges")
            print("First 200 chars of method:")
            print(source[:200] + "...")
            
    except Exception as e:
        print(f"❌ ERROR importing DataFetcher: {e}")
    
    print()
    print("=== GIT STATUS CHECK ===")
    try:
        import subprocess
        result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                              capture_output=True, text=True, cwd=Path(__file__).parent.parent)
        if result.returncode == 0:
            print(f"Current commit: {result.stdout.strip()}")
        else:
            print("❌ Not in a git repository or git not available")
            
        # Check for uncommitted changes
        result = subprocess.run(['git', 'status', '--porcelain', 'src/'], 
                              capture_output=True, text=True, cwd=Path(__file__).parent.parent)
        if result.returncode == 0:
            if result.stdout.strip():
                print("⚠️  WARNING: Uncommitted changes in src/:")
                print(result.stdout)
            else:
                print("✅ GOOD: No uncommitted changes in src/")
        
    except Exception as e:
        print(f"❌ ERROR checking git status: {e}")
    
    print()
    print("=== RECOMMENDATIONS ===")
    print("If you see ❌ BAD messages above:")
    print("1. Clear Python cache: find . -name '*.pyc' -delete && find . -name '__pycache__' -exec rm -rf {} +")
    print("2. Pull latest changes: git pull origin main")
    print("3. Check you're on the right branch: git branch")
    print("4. Restart any running processes")
    print("5. Re-run the pipeline: python src/pipeline.py --mode daily")

if __name__ == "__main__":
    main()