#!/usr/bin/env python3
"""
Debug script to test the fetch_daily_data method and see exact date calculations.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_fetcher import DataFetcher

def debug_fetch_daily_data():
    print("=== DEBUG: Date Calculation in fetch_daily_data ===")
    
    # Simulate the exact calculation from our method
    today = datetime.now()
    print(f"Today: {today}")
    
    # This should be our FIXED logic
    print("\n--- Our Fixed Logic ---")
    end_date = today - timedelta(days=4)
    start_date = end_date - timedelta(days=2)
    
    weather_end_date = end_date
    weather_start_date = start_date
    energy_end_date = end_date  # Should be SAME as weather
    energy_start_date = start_date  # Should be SAME as weather
    
    weather_start_str = weather_start_date.strftime('%Y-%m-%d')
    weather_end_str = weather_end_date.strftime('%Y-%m-%d')
    energy_start_str = energy_start_date.strftime('%Y-%m-%d')
    energy_end_str = energy_end_date.strftime('%Y-%m-%d')
    
    print(f"Weather range: {weather_start_str} to {weather_end_str}")
    print(f"Energy range:  {energy_start_str} to {energy_end_str}")
    print(f"ALIGNED: {weather_start_str == energy_start_str and weather_end_str == energy_end_str}")
    
    # Now test what the actual DataFetcher would do
    print("\n--- Actual DataFetcher Test ---")
    try:
        fetcher = DataFetcher('config/config.yaml')
        
        # We'll simulate calling fetch_daily_data but catch any errors
        print("Calling fetcher.fetch_daily_data()...")
        
        # Let's manually check what the method would calculate
        import inspect
        source = inspect.getsource(fetcher.fetch_daily_data)
        
        if "energy_end_date = end_date" in source:
            print("✅ Method source has aligned date calculation")
        else:
            print("❌ Method source does NOT have aligned date calculation")
            print("Method source snippet:")
            lines = source.split('\n')
            for i, line in enumerate(lines[10:25], 10):  # Show lines around date calculation
                print(f"{i:2d}: {line}")
        
        # Try to actually call it (but just return the structure, don't fetch data)
        result = fetcher.fetch_daily_data()
        
        print(f"\nResult weather_date_range: {result.get('weather_date_range', 'N/A')}")
        print(f"Result energy_date_range:  {result.get('energy_date_range', 'N/A')}")
        
        weather_range = result.get('weather_date_range', '')
        energy_range = result.get('energy_date_range', '')
        
        if weather_range == energy_range and weather_range != 'N/A':
            print("✅ GOOD: Date ranges are aligned in result")
        else:
            print("❌ BAD: Date ranges are NOT aligned in result")
            print("This suggests there's still a bug in the calculation!")
        
    except Exception as e:
        print(f"❌ Error testing DataFetcher: {e}")
        
    # Check if there might be multiple versions of the method
    print("\n--- Checking for Multiple Versions ---")
    try:
        import importlib
        import data_fetcher
        
        # Force reload the module
        importlib.reload(data_fetcher)
        print("✅ Module reloaded successfully")
        
        fetcher2 = data_fetcher.DataFetcher('config/config.yaml')
        source2 = inspect.getsource(fetcher2.fetch_daily_data)
        
        if source == source2:
            print("✅ Method source is consistent after reload")
        else:
            print("❌ Method source CHANGED after reload - this indicates caching issues!")
            
    except Exception as e:
        print(f"❌ Error reloading module: {e}")

if __name__ == "__main__":
    debug_fetch_daily_data()