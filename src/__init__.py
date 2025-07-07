"""
Weather and Energy Analysis Pipeline

This package contains modules for fetching, processing, and analyzing
weather and energy consumption data from NOAA and EIA APIs.
"""

__version__ = "1.0.0"
__author__ = "Weather Energy Analysis Team"

# Import main classes for convenient package-level access
from .data_fetcher import DataFetcher
from .data_processor import DataProcessor
from .analysis import DataAnalyzer
from .pipeline import WeatherEnergyPipeline

__all__ = [
    'DataFetcher',
    'DataProcessor', 
    'DataAnalyzer',
    'WeatherEnergyPipeline'
]