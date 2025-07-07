"""
Data processing module for cleaning and transforming weather and energy data.
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Dict, Optional
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
    
    def _diagnose_merge_compatibility(self, weather_df: pd.DataFrame, 
                                     energy_df: pd.DataFrame) -> Dict[str, any]:
        """
        Diagnose merge compatibility between weather and energy data.
        
        Args:
            weather_df: Processed weather DataFrame
            energy_df: Processed energy DataFrame
            
        Returns:
            Dictionary with diagnostic information
        """
        diagnosis = {
            'compatible': False,
            'common_cities': [],
            'date_overlaps': {},
            'total_possible_matches': 0,
            'issues': []
        }
        
        try:
            # Check for key overlaps
            weather_cities = set(weather_df['city'].unique())
            energy_cities = set(energy_df['city'].unique())
            common_cities = weather_cities & energy_cities
            diagnosis['common_cities'] = sorted(common_cities)
            
            if not common_cities:
                diagnosis['issues'].append("No common cities found between datasets")
                return diagnosis
            
            # Check date overlaps for each common city
            total_overlaps = 0
            for city in common_cities:
                weather_city_data = weather_df[weather_df['city'] == city]
                energy_city_data = energy_df[energy_df['city'] == city]
                
                weather_dates = set(pd.to_datetime(weather_city_data['date']).dt.date)
                energy_dates = set(pd.to_datetime(energy_city_data['date']).dt.date)
                
                common_dates = weather_dates & energy_dates
                overlap_count = len(common_dates)
                total_overlaps += overlap_count
                
                diagnosis['date_overlaps'][city] = {
                    'weather_dates': len(weather_dates),
                    'energy_dates': len(energy_dates),
                    'common_dates': overlap_count,
                    'weather_range': f"{min(weather_dates)} to {max(weather_dates)}" if weather_dates else "No dates",
                    'energy_range': f"{min(energy_dates)} to {max(energy_dates)}" if energy_dates else "No dates",
                    'sample_common_dates': sorted(list(common_dates))[:5] if common_dates else []
                }
                
                if overlap_count == 0:
                    diagnosis['issues'].append(f"No date overlap for {city}")
            
            diagnosis['total_possible_matches'] = total_overlaps
            diagnosis['compatible'] = total_overlaps > 0
            
            if total_overlaps == 0:
                diagnosis['issues'].append("No date overlaps found for any city")
            
            return diagnosis
            
        except Exception as e:
            diagnosis['issues'].append(f"Error during compatibility diagnosis: {e}")
            return diagnosis

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
            if weather_df.empty:
                self.logger.warning("Weather DataFrame is empty")
            if energy_df.empty:
                self.logger.warning("Energy DataFrame is empty")
            return pd.DataFrame()
        
        try:
            # Run comprehensive merge diagnostics
            self.logger.info("Running merge compatibility diagnostics...")
            diagnosis = self._diagnose_merge_compatibility(weather_df, energy_df)
            
            # Log diagnostic results
            self.logger.info(f"Pre-merge analysis:")
            self.logger.info(f"  Weather data: {len(weather_df)} records, {weather_df['city'].nunique()} cities")
            self.logger.info(f"  Energy data: {len(energy_df)} records, {energy_df['city'].nunique()} cities")
            self.logger.info(f"  Common cities: {diagnosis['common_cities']}")
            self.logger.info(f"  Expected merge matches: {diagnosis['total_possible_matches']}")
            
            # Log detailed date overlap information
            for city, overlap_info in diagnosis['date_overlaps'].items():
                self.logger.info(f"  {city}:")
                self.logger.info(f"    Weather: {overlap_info['weather_dates']} records ({overlap_info['weather_range']})")
                self.logger.info(f"    Energy: {overlap_info['energy_dates']} records ({overlap_info['energy_range']})")
                self.logger.info(f"    Common dates: {overlap_info['common_dates']}")
                if overlap_info['sample_common_dates']:
                    self.logger.info(f"    Sample overlap: {overlap_info['sample_common_dates']}")
            
            # Report any issues found
            if diagnosis['issues']:
                for issue in diagnosis['issues']:
                    self.logger.warning(f"  Issue: {issue}")
            
            if not diagnosis['compatible']:
                self.logger.error("Merge not compatible - no date overlaps found")
                return pd.DataFrame()
            
            # Merge on city and date
            merged_df = pd.merge(
                weather_df, 
                energy_df, 
                on=['city', 'date'], 
                how='inner',
                suffixes=('', '_energy')
            )
            
            # Log merge results
            self.logger.info(f"Post-merge analysis:")
            self.logger.info(f"  Merged records: {len(merged_df)}")
            self.logger.info(f"  Merged cities: {merged_df['city'].nunique() if len(merged_df) > 0 else 0}")
            
            if len(merged_df) == 0:
                self.logger.error("Merge resulted in zero records despite compatibility check - this should not happen!")
                # The diagnostic function should have caught this earlier
            
            # Remove duplicate columns
            duplicate_cols = ['state_energy', 'day_of_week_energy', 'is_weekend_energy']
            existing_duplicate_cols = [col for col in duplicate_cols if col in merged_df.columns]
            if existing_duplicate_cols:
                merged_df = merged_df.drop(columns=existing_duplicate_cols)
            
            # Sort by date and city
            merged_df = merged_df.sort_values(['date', 'city']).reset_index(drop=True)
            
            self.logger.info(f"Final merged data: {len(merged_df)} records")
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
            
            # Validate energy consumption - only remove negative values
            energy_min = self.quality_checks['energy']['min_consumption']
            negative_energy = cleaned_df['energy_consumption_mwh'] < energy_min
            
            if negative_energy.any():
                self.logger.warning(f"Found {negative_energy.sum()} negative energy values, removing them")
                cleaned_df = cleaned_df[~negative_energy]
            
            # Use more conservative outlier detection for extremely high values
            if 'energy_consumption_mwh' in cleaned_df.columns and len(cleaned_df) > 10:
                # Use 3 standard deviations instead of IQR for less aggressive filtering
                mean_energy = cleaned_df['energy_consumption_mwh'].mean()
                std_energy = cleaned_df['energy_consumption_mwh'].std()
                
                # Only remove values more than 3 standard deviations from mean
                extreme_outliers = cleaned_df['energy_consumption_mwh'] > (mean_energy + 3 * std_energy)
                
                if extreme_outliers.any():
                    self.logger.warning(f"Found {extreme_outliers.sum()} extreme energy outliers (>3 std dev), removing them")
                    self.logger.info(f"Energy stats - Mean: {mean_energy:.0f}, Std: {std_energy:.0f}, Threshold: {mean_energy + 3 * std_energy:.0f}")
                    cleaned_df = cleaned_df[~extreme_outliers]
                else:
                    self.logger.info(f"No extreme energy outliers found using 3-sigma rule")
            
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
    
    def process_energy_only(self, energy_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process energy data only when weather data is unavailable.
        
        Args:
            energy_df: Raw energy DataFrame
            
        Returns:
            Processed energy DataFrame
        """
        self.logger.info("Processing energy data only")
        
        # Process energy data
        processed_energy = self.process_energy_data(energy_df)
        
        # Clean and validate (with modified validation for energy-only data)
        cleaned_data = self.clean_and_validate_energy_only(processed_energy)
        
        # Save processed data
        self.save_processed_data(cleaned_data)
        
        self.logger.info("Completed energy-only data processing")
        return cleaned_data
    
    def process_weather_only(self, weather_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process weather data only when energy data is unavailable.
        
        Args:
            weather_df: Raw weather DataFrame
            
        Returns:
            Processed weather DataFrame
        """
        self.logger.info("Processing weather data only")
        
        # Process weather data
        processed_weather = self.process_weather_data(weather_df)
        
        # Clean and validate (with modified validation for weather-only data)
        cleaned_data = self.clean_and_validate_weather_only(processed_weather)
        
        # Save processed data
        self.save_processed_data(cleaned_data)
        
        self.logger.info("Completed weather-only data processing")
        return cleaned_data
    
    def clean_and_validate_energy_only(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate energy-only data.
        
        Args:
            df: Energy DataFrame to clean
            
        Returns:
            Cleaned DataFrame
        """
        if df.empty:
            self.logger.warning("Empty DataFrame provided for energy-only cleaning")
            return df
        
        try:
            # Create a copy
            cleaned_df = df.copy()
            original_count = len(cleaned_df)
            
            # Remove records with missing essential data
            essential_cols = ['date', 'city', 'energy_consumption_mwh']
            cleaned_df = cleaned_df.dropna(subset=essential_cols)
            
            # Validate energy consumption - only remove negative values
            energy_min = self.quality_checks['energy']['min_consumption']
            negative_energy = cleaned_df['energy_consumption_mwh'] < energy_min
            
            if negative_energy.any():
                self.logger.warning(f"Found {negative_energy.sum()} negative energy values, removing them")
                cleaned_df = cleaned_df[~negative_energy]
            
            # Use more conservative outlier detection for extremely high values
            if 'energy_consumption_mwh' in cleaned_df.columns and len(cleaned_df) > 10:
                # Use 3 standard deviations instead of IQR for less aggressive filtering
                mean_energy = cleaned_df['energy_consumption_mwh'].mean()
                std_energy = cleaned_df['energy_consumption_mwh'].std()
                
                # Only remove values more than 3 standard deviations from mean
                extreme_outliers = cleaned_df['energy_consumption_mwh'] > (mean_energy + 3 * std_energy)
                
                if extreme_outliers.any():
                    self.logger.warning(f"Found {extreme_outliers.sum()} extreme energy outliers (>3 std dev), removing them")
                    self.logger.info(f"Energy stats - Mean: {mean_energy:.0f}, Std: {std_energy:.0f}, Threshold: {mean_energy + 3 * std_energy:.0f}")
                    cleaned_df = cleaned_df[~extreme_outliers]
                else:
                    self.logger.info(f"No extreme energy outliers found using 3-sigma rule")
            
            # Remove duplicates
            cleaned_df = cleaned_df.drop_duplicates(subset=['date', 'city'])
            
            final_count = len(cleaned_df)
            removed_count = original_count - final_count
            
            if removed_count > 0:
                self.logger.info(f"Removed {removed_count} invalid records during energy-only cleaning")
            
            self.logger.info(f"Cleaned energy-only data: {final_count} records")
            return cleaned_df
            
        except Exception as e:
            self.logger.error(f"Error in energy-only data cleaning: {e}")
            return pd.DataFrame()
    
    def clean_and_validate_weather_only(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate weather-only data.
        
        Args:
            df: Weather DataFrame to clean
            
        Returns:
            Cleaned DataFrame
        """
        if df.empty:
            self.logger.warning("Empty DataFrame provided for weather-only cleaning")
            return df
        
        try:
            # Create a copy
            cleaned_df = df.copy()
            original_count = len(cleaned_df)
            
            # Remove records with missing essential data
            essential_cols = ['date', 'city', 'temp_avg_f']
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
            
            # Remove duplicates
            cleaned_df = cleaned_df.drop_duplicates(subset=['date', 'city'])
            
            final_count = len(cleaned_df)
            removed_count = original_count - final_count
            
            if removed_count > 0:
                self.logger.info(f"Removed {removed_count} invalid records during weather-only cleaning")
            
            self.logger.info(f"Cleaned weather-only data: {final_count} records")
            return cleaned_df
            
        except Exception as e:
            self.logger.error(f"Error in weather-only data cleaning: {e}")
            return pd.DataFrame()