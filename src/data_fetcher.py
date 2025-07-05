"""
Data fetching module for NOAA weather and EIA energy data.
"""

import requests
import pandas as pd
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
import yaml
import os
from pathlib import Path


class DataFetcher:
    """Main class for fetching weather and energy data from APIs."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the data fetcher with configuration."""
        self.config = self._load_config(config_path)
        self.logger = logging.getLogger(__name__)
        
        # API configurations
        self.noaa_config = self.config['api']['noaa']
        self.eia_config = self.config['api']['eia']
        self.cities = self.config['cities']
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML configuration: {e}")
    
    def _make_request_with_retry(self, url: str, params: Dict, 
                               headers: Dict, max_retries: int = 3,
                               retry_delay: int = 1) -> requests.Response:
        """Make HTTP request with exponential backoff retry logic."""
        for attempt in range(max_retries):
            try:
                response = requests.get(
                    url, 
                    params=params, 
                    headers=headers, 
                    timeout=30
                )
                response.raise_for_status()
                return response
            
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    self.logger.error(f"Failed to fetch data after {max_retries} attempts: {e}")
                    raise
                
                wait_time = retry_delay * (2 ** attempt)
                self.logger.warning(f"Request failed (attempt {attempt + 1}), retrying in {wait_time}s: {e}")
                time.sleep(wait_time)
    
    def fetch_weather_data(self, city: Dict, start_date: str, 
                          end_date: str) -> Optional[pd.DataFrame]:
        """
        Fetch weather data from NOAA Climate Data Online API.
        
        Args:
            city: City configuration dictionary
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with weather data or None if failed
        """
        try:
            url = f"{self.noaa_config['base_url']}/data"
            headers = {'token': self.noaa_config['token']}
            
            params = {
                'datasetid': 'GHCND',
                'stationid': city['noaa_station_id'],
                'startdate': start_date,
                'enddate': end_date,
                'datatypeid': 'TMAX,TMIN',
                'limit': 1000,
                'units': 'standard'
            }
            
            self.logger.info(f"Fetching weather data for {city['name']} from {start_date} to {end_date}")
            
            response = self._make_request_with_retry(
                url, params, headers,
                max_retries=self.noaa_config['retry_attempts'],
                retry_delay=self.noaa_config['retry_delay']
            )
            
            data = response.json()
            
            if 'results' not in data or not data['results']:
                self.logger.warning(f"No weather data found for {city['name']}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(data['results'])
            df['city'] = city['name']
            df['state'] = city['state']
            df['lat'] = city['lat']
            df['lon'] = city['lon']
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching weather data for {city['name']}: {e}")
            return None
    
    def fetch_energy_data(self, city: Dict, start_date: str, 
                         end_date: str) -> Optional[pd.DataFrame]:
        """
        Fetch energy data from EIA API.
        
        Args:
            city: City configuration dictionary
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with energy data or None if failed
        """
        try:
            url = self.eia_config['base_url']
            
            params = {
                'api_key': self.eia_config['api_key'],
                'frequency': 'daily',
                'data[0]': 'value',
                'facets[respondent][]': city['eia_region_code'],
                'start': start_date,
                'end': end_date,
                'sort[0][column]': 'period',
                'sort[0][direction]': 'desc',
                'offset': 0,
                'length': 5000
            }
            
            self.logger.info(f"Fetching energy data for {city['name']} from {start_date} to {end_date}")
            
            response = self._make_request_with_retry(
                url, params, {},
                max_retries=self.eia_config['retry_attempts'],
                retry_delay=self.eia_config['retry_delay']
            )
            
            data = response.json()
            
            if 'response' not in data or 'data' not in data['response']:
                self.logger.warning(f"No energy data found for {city['name']}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(data['response']['data'])
            df['city'] = city['name']
            df['state'] = city['state']
            df['eia_region'] = city['eia_region_code']
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching energy data for {city['name']}: {e}")
            return None
    
    def fetch_all_weather_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch weather data for all configured cities.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            Combined DataFrame with all weather data
        """
        all_data = []
        
        for city in self.cities:
            try:
                weather_df = self.fetch_weather_data(city, start_date, end_date)
                if weather_df is not None:
                    all_data.append(weather_df)
                    self.logger.info(f"Successfully fetched weather data for {city['name']}")
                else:
                    self.logger.warning(f"No weather data retrieved for {city['name']}")
            except Exception as e:
                self.logger.error(f"Failed to fetch weather data for {city['name']}: {e}")
                continue
        
        if not all_data:
            self.logger.error("No weather data retrieved for any city")
            return pd.DataFrame()
        
        combined_df = pd.concat(all_data, ignore_index=True)
        self.logger.info(f"Combined weather data: {len(combined_df)} records")
        return combined_df
    
    def fetch_all_energy_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch energy data for all configured cities.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            Combined DataFrame with all energy data
        """
        all_data = []
        
        for city in self.cities:
            try:
                energy_df = self.fetch_energy_data(city, start_date, end_date)
                if energy_df is not None:
                    all_data.append(energy_df)
                    self.logger.info(f"Successfully fetched energy data for {city['name']}")
                else:
                    self.logger.warning(f"No energy data retrieved for {city['name']}")
            except Exception as e:
                self.logger.error(f"Failed to fetch energy data for {city['name']}: {e}")
                continue
        
        if not all_data:
            self.logger.error("No energy data retrieved for any city")
            return pd.DataFrame()
        
        combined_df = pd.concat(all_data, ignore_index=True)
        self.logger.info(f"Combined energy data: {len(combined_df)} records")
        return combined_df
    
    def fetch_historical_data(self, days: int = 90) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical data for the specified number of days.
        
        Args:
            days: Number of historical days to fetch
            
        Returns:
            Dictionary containing weather and energy DataFrames
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        self.logger.info(f"Fetching {days} days of historical data from {start_str} to {end_str}")
        
        # Fetch weather and energy data
        weather_data = self.fetch_all_weather_data(start_str, end_str)
        energy_data = self.fetch_all_energy_data(start_str, end_str)
        
        return {
            'weather': weather_data,
            'energy': energy_data,
            'start_date': start_str,
            'end_date': end_str
        }
    
    def fetch_daily_data(self) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for today and yesterday.
        
        Returns:
            Dictionary containing weather and energy DataFrames
        """
        today = datetime.now()
        yesterday = today - timedelta(days=1)
        
        today_str = today.strftime('%Y-%m-%d')
        yesterday_str = yesterday.strftime('%Y-%m-%d')
        
        self.logger.info(f"Fetching daily data for {yesterday_str} to {today_str}")
        
        # Fetch weather and energy data
        weather_data = self.fetch_all_weather_data(yesterday_str, today_str)
        energy_data = self.fetch_all_energy_data(yesterday_str, today_str)
        
        return {
            'weather': weather_data,
            'energy': energy_data,
            'date': today_str
        }
    
    def save_raw_data(self, data: Dict[str, pd.DataFrame], prefix: str = "raw"):
        """
        Save raw data to CSV files.
        
        Args:
            data: Dictionary containing DataFrames to save
            prefix: Prefix for filename
        """
        raw_data_path = Path(self.config['paths']['raw_data'])
        raw_data_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        for data_type, df in data.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                filename = f"{prefix}_{data_type}_{timestamp}.csv"
                filepath = raw_data_path / filename
                df.to_csv(filepath, index=False)
                self.logger.info(f"Saved {data_type} data to {filepath}")