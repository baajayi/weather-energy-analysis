"""
Data processing module for cleaning and transforming weather and energy data.
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import yaml
from pathlib import Path


class DataProcessor:
    """Main class for processing and cleaning weather and energy data."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the data processor with configuration."""
        self.config = self._load_config(config_path)
        self.logger = logging.getLogger(__name__)
        self.quality_checks = self.config['data']['quality_checks']
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML configuration: {e}")
    
    def process_weather_data(self, weather_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process and clean weather data from NOAA API.
        
        Args:
            weather_df: Raw weather DataFrame from NOAA API
            
        Returns:
            Processed weather DataFrame
        """
        if weather_df.empty:
            self.logger.warning("Empty weather DataFrame provided")
            return pd.DataFrame()
        
        try:
            # Create a copy to avoid modifying original
            df = weather_df.copy()
            
            # Convert date column to datetime
            df['date'] = pd.to_datetime(df['date'])
            
            # Convert temperature from tenths of Celsius to Fahrenheit
            df['temp_fahrenheit'] = df['value'].apply(self._celsius_tenths_to_fahrenheit)
            
            # Pivot to get separate columns for TMAX and TMIN
            df_pivot = df.pivot_table(
                index=['date', 'city', 'state', 'lat', 'lon'],
                columns='datatype',
                values='temp_fahrenheit',
                aggfunc='mean'
            ).reset_index()
            
            # Flatten column names
            df_pivot.columns.name = None
            
            # Rename temperature columns
            if 'TMAX' in df_pivot.columns:
                df_pivot['temp_max_f'] = df_pivot['TMAX']
            if 'TMIN' in df_pivot.columns:
                df_pivot['temp_min_f'] = df_pivot['TMIN']
            
            # Calculate average temperature
            if 'TMAX' in df_pivot.columns and 'TMIN' in df_pivot.columns:
                df_pivot['temp_avg_f'] = (df_pivot['TMAX'] + df_pivot['TMIN']) / 2
            
            # Drop original TMAX/TMIN columns
            cols_to_drop = ['TMAX', 'TMIN'] if 'TMAX' in df_pivot.columns else []
            if cols_to_drop:
                df_pivot = df_pivot.drop(columns=cols_to_drop)
            
            # Add day of week and weekend flag
            df_pivot['day_of_week'] = df_pivot['date'].dt.day_name()
            df_pivot['is_weekend'] = df_pivot['date'].dt.weekday >= 5
            
            # Add temperature range categories
            df_pivot['temp_range'] = df_pivot['temp_avg_f'].apply(self._categorize_temperature)
            
            self.logger.info(f"Processed weather data: {len(df_pivot)} records")
            return df_pivot
            
        except Exception as e:
            self.logger.error(f"Error processing weather data: {e}")
            return pd.DataFrame()
    
    def process_energy_data(self, energy_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process and clean energy data from EIA API.
        
        Args:
            energy_df: Raw energy DataFrame from EIA API
            
        Returns:
            Processed energy DataFrame
        """
        if energy_df.empty:
            self.logger.warning("Empty energy DataFrame provided")
            return pd.DataFrame()
        
        try:
            # Create a copy to avoid modifying original
            df = energy_df.copy()
            
            # Convert period to datetime
            df['date'] = pd.to_datetime(df['period'])
            
            # Rename value column to energy_consumption
            df['energy_consumption_mwh'] = df['value'].astype(float)
            
            # Add day of week and weekend flag
            df['day_of_week'] = df['date'].dt.day_name()
            df['is_weekend'] = df['date'].dt.weekday >= 5
            
            # Select relevant columns
            columns_to_keep = [
                'date', 'city', 'state', 'eia_region', 'energy_consumption_mwh',
                'day_of_week', 'is_weekend'
            ]
            
            df = df[columns_to_keep]
            
            self.logger.info(f"Processed energy data: {len(df)} records")
            return df
            
        except Exception as e:
            self.logger.error(f"Error processing energy data: {e}")
            return pd.DataFrame()
    
    def merge_weather_energy_data(self, weather_df: pd.DataFrame, 
                                 energy_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge weather and energy data on city and date.
        
        Args:
            weather_df: Processed weather DataFrame
            energy_df: Processed energy DataFrame
            
        Returns:
            Merged DataFrame
        """
        if weather_df.empty or energy_df.empty:
            self.logger.warning("One or both DataFrames are empty for merging")
            return pd.DataFrame()
        
        try:
            # Merge on city and date
            merged_df = pd.merge(
                weather_df, 
                energy_df, 
                on=['city', 'date'], 
                how='inner',
                suffixes=('', '_energy')
            )
            
            # Remove duplicate columns
            duplicate_cols = ['state_energy', 'day_of_week_energy', 'is_weekend_energy']
            existing_duplicate_cols = [col for col in duplicate_cols if col in merged_df.columns]
            if existing_duplicate_cols:
                merged_df = merged_df.drop(columns=existing_duplicate_cols)
            
            # Sort by date and city
            merged_df = merged_df.sort_values(['date', 'city']).reset_index(drop=True)
            
            self.logger.info(f"Merged data: {len(merged_df)} records")
            return merged_df
            
        except Exception as e:
            self.logger.error(f"Error merging weather and energy data: {e}")
            return pd.DataFrame()
    
    def clean_and_validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate the merged dataset.
        
        Args:
            df: Merged DataFrame to clean
            
        Returns:
            Cleaned DataFrame
        """
        if df.empty:
            self.logger.warning("Empty DataFrame provided for cleaning")
            return df
        
        try:
            # Create a copy
            cleaned_df = df.copy()
            original_count = len(cleaned_df)
            
            # Remove records with missing essential data
            essential_cols = ['date', 'city', 'energy_consumption_mwh']
            cleaned_df = cleaned_df.dropna(subset=essential_cols)
            
            # Validate temperature ranges
            if 'temp_avg_f' in cleaned_df.columns:
                temp_min = self.quality_checks['temperature']['min_fahrenheit']
                temp_max = self.quality_checks['temperature']['max_fahrenheit']
                
                temp_outliers = (
                    (cleaned_df['temp_avg_f'] < temp_min) | 
                    (cleaned_df['temp_avg_f'] > temp_max)
                )
                
                if temp_outliers.any():
                    self.logger.warning(f"Found {temp_outliers.sum()} temperature outliers")
                    cleaned_df = cleaned_df[~temp_outliers]
            
            # Validate energy consumption
            energy_min = self.quality_checks['energy']['min_consumption']
            energy_outliers = cleaned_df['energy_consumption_mwh'] < energy_min
            
            if energy_outliers.any():
                self.logger.warning(f"Found {energy_outliers.sum()} energy outliers")
                cleaned_df = cleaned_df[~energy_outliers]
            
            # Remove duplicates
            cleaned_df = cleaned_df.drop_duplicates(subset=['date', 'city'])
            
            final_count = len(cleaned_df)
            removed_count = original_count - final_count
            
            if removed_count > 0:
                self.logger.info(f"Removed {removed_count} invalid records during cleaning")
            
            self.logger.info(f"Cleaned data: {final_count} records")
            return cleaned_df
            
        except Exception as e:
            self.logger.error(f"Error cleaning data: {e}")
            return df
    
    def calculate_daily_changes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate day-over-day changes in temperature and energy consumption.
        
        Args:
            df: Cleaned DataFrame
            
        Returns:
            DataFrame with daily change columns
        """
        if df.empty:
            return df
        
        try:
            # Sort by city and date
            df_sorted = df.sort_values(['city', 'date']).copy()
            
            # Calculate daily changes by city
            for city in df_sorted['city'].unique():
                city_mask = df_sorted['city'] == city
                city_data = df_sorted[city_mask].copy()
                
                # Calculate temperature changes
                if 'temp_avg_f' in city_data.columns:
                    df_sorted.loc[city_mask, 'temp_change_f'] = city_data['temp_avg_f'].diff()
                
                # Calculate energy changes
                if 'energy_consumption_mwh' in city_data.columns:
                    df_sorted.loc[city_mask, 'energy_change_mwh'] = city_data['energy_consumption_mwh'].diff()
                    df_sorted.loc[city_mask, 'energy_change_pct'] = city_data['energy_consumption_mwh'].pct_change() * 100
            
            self.logger.info("Calculated daily changes")
            return df_sorted
            
        except Exception as e:
            self.logger.error(f"Error calculating daily changes: {e}")
            return df
    
    def save_processed_data(self, df: pd.DataFrame, filename: str = None):
        """
        Save processed data to CSV file.
        
        Args:
            df: DataFrame to save
            filename: Optional filename, defaults to timestamp-based name
        """
        if df.empty:
            self.logger.warning("No data to save")
            return
        
        try:
            processed_data_path = Path(self.config['paths']['processed_data'])
            processed_data_path.mkdir(parents=True, exist_ok=True)
            
            if filename is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"processed_data_{timestamp}.csv"
            
            filepath = processed_data_path / filename
            df.to_csv(filepath, index=False)
            self.logger.info(f"Saved processed data to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving processed data: {e}")
    
    def _celsius_tenths_to_fahrenheit(self, celsius_tenths: float) -> float:
        """
        Convert temperature from tenths of Celsius to Fahrenheit.
        
        Args:
            celsius_tenths: Temperature in tenths of Celsius
            
        Returns:
            Temperature in Fahrenheit
        """
        if pd.isna(celsius_tenths):
            return np.nan
        
        celsius = celsius_tenths / 10.0
        fahrenheit = (celsius * 9/5) + 32
        return round(fahrenheit, 1)
    
    def _categorize_temperature(self, temp_f: float) -> str:
        """
        Categorize temperature into ranges.
        
        Args:
            temp_f: Temperature in Fahrenheit
            
        Returns:
            Temperature range category
        """
        if pd.isna(temp_f):
            return 'Unknown'
        
        if temp_f < 50:
            return '<50°F'
        elif temp_f < 60:
            return '50-60°F'
        elif temp_f < 70:
            return '60-70°F'
        elif temp_f < 80:
            return '70-80°F'
        elif temp_f < 90:
            return '80-90°F'
        else:
            return '>90°F'
    
    def get_latest_data(self) -> Optional[pd.DataFrame]:
        """
        Load the most recent processed data file.
        
        Returns:
            Latest processed DataFrame or None if not found
        """
        try:
            processed_data_path = Path(self.config['paths']['processed_data'])
            
            if not processed_data_path.exists():
                self.logger.warning("Processed data directory does not exist")
                return None
            
            # Find the most recent processed data file
            csv_files = list(processed_data_path.glob("processed_data_*.csv"))
            
            if not csv_files:
                self.logger.warning("No processed data files found")
                return None
            
            latest_file = max(csv_files, key=lambda x: x.stat().st_mtime)
            self.logger.info(f"Loading latest processed data from {latest_file}")
            
            return pd.read_csv(latest_file)
            
        except Exception as e:
            self.logger.error(f"Error loading latest data: {e}")
            return None
    
    def process_all_data(self, weather_df: pd.DataFrame, 
                        energy_df: pd.DataFrame) -> pd.DataFrame:
        """
        Complete data processing pipeline.
        
        Args:
            weather_df: Raw weather DataFrame
            energy_df: Raw energy DataFrame
            
        Returns:
            Fully processed DataFrame
        """
        self.logger.info("Starting complete data processing pipeline")
        
        # Process individual datasets
        processed_weather = self.process_weather_data(weather_df)
        processed_energy = self.process_energy_data(energy_df)
        
        # Merge datasets
        merged_data = self.merge_weather_energy_data(processed_weather, processed_energy)
        
        # Clean and validate
        cleaned_data = self.clean_and_validate_data(merged_data)
        
        # Calculate daily changes
        final_data = self.calculate_daily_changes(cleaned_data)
        
        # Save processed data
        self.save_processed_data(final_data)
        
        self.logger.info("Completed data processing pipeline")
        return final_data