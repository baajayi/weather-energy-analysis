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
                               retry_delay: int = 1, api_name: str = "API") -> requests.Response:
        """Make HTTP request with exponential backoff retry logic and enhanced error handling."""
        last_error = None
        
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
                last_error = e
                
                # Provide specific error guidance
                error_msg = self._get_specific_error_message(e, api_name)
                
                if attempt == max_retries - 1:
                    self.logger.error(f"Failed to fetch {api_name} data after {max_retries} attempts: {error_msg}")
                    raise
                
                wait_time = retry_delay * (2 ** attempt)
                self.logger.warning(f"{api_name} request failed (attempt {attempt + 1}), retrying in {wait_time}s: {error_msg}")
                time.sleep(wait_time)
        
        # This should never be reached due to the raise above, but just in case
        raise last_error if last_error else Exception("Unknown error in retry logic")
    
    def _get_specific_error_message(self, error: requests.exceptions.RequestException, api_name: str) -> str:
        """Generate specific error messages with troubleshooting guidance."""
        if isinstance(error, requests.exceptions.ConnectionError):
            return f"{api_name} connection failed - check internet connectivity"
        elif isinstance(error, requests.exceptions.Timeout):
            return f"{api_name} request timed out - server may be overloaded"
        elif isinstance(error, requests.exceptions.HTTPError):
            status_code = error.response.status_code if error.response else "unknown"
            
            if status_code == 400:
                return f"{api_name} bad request (400) - check date format or station IDs in config"
            elif status_code == 401:
                return f"{api_name} unauthorized (401) - check API key/token in config"
            elif status_code == 403:
                return f"{api_name} forbidden (403) - API key may be invalid or quota exceeded"
            elif status_code == 404:
                return f"{api_name} not found (404) - check endpoint URL or station/region codes"
            elif status_code == 429:
                return f"{api_name} rate limited (429) - too many requests, will retry with backoff"
            elif status_code == 500:
                return f"{api_name} server error (500) - external service issue, will retry"
            else:
                return f"{api_name} HTTP error ({status_code}): {str(error)}"
        else:
            return f"{api_name} request error: {str(error)}"
    
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
                retry_delay=self.noaa_config['retry_delay'],
                api_name="NOAA Weather"
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
                retry_delay=self.eia_config['retry_delay'],
                api_name="EIA Energy"
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
        Fetch data for recent days with lag for weather data availability.
        Weather data typically has a 2-3 day reporting delay.
        
        Returns:
            Dictionary containing weather and energy DataFrames
        """
        try:
            today = datetime.now()
            
            # Use consistent 4-day lag for both APIs to ensure overlap
            # This accounts for weather data reporting delays
            end_date = today - timedelta(days=4)
            start_date = end_date - timedelta(days=2)
            
            # Use same date ranges for both weather and energy
            weather_end_date = end_date
            weather_start_date = start_date
            energy_end_date = end_date
            energy_start_date = start_date
            
            weather_start_str = weather_start_date.strftime('%Y-%m-%d')
            weather_end_str = weather_end_date.strftime('%Y-%m-%d')
            energy_start_str = energy_start_date.strftime('%Y-%m-%d')
            energy_end_str = energy_end_date.strftime('%Y-%m-%d')
            
            self.logger.info(f"Fetching daily data with aligned date ranges - Both APIs: {weather_start_str} to {weather_end_str}")
            
            # Initialize empty dataframes in case of failures
            weather_data = pd.DataFrame()
            energy_data = pd.DataFrame()
            
            # Fetch weather data with error handling
            try:
                weather_data = self.fetch_all_weather_data(weather_start_str, weather_end_str)
                
                # If no weather data found, try with additional lag
                if weather_data.empty:
                    self.logger.warning(f"No weather data found for {weather_start_str} to {weather_end_str}, trying with additional lag")
                    fallback_weather_end = today - timedelta(days=6)
                    fallback_weather_start = fallback_weather_end - timedelta(days=3)
                    fallback_start_str = fallback_weather_start.strftime('%Y-%m-%d')
                    fallback_end_str = fallback_weather_end.strftime('%Y-%m-%d')
                    
                    self.logger.info(f"Trying fallback weather data range: {fallback_start_str} to {fallback_end_str}")
                    weather_data = self.fetch_all_weather_data(fallback_start_str, fallback_end_str)
                    
            except Exception as e:
                self.logger.error(f"Failed to fetch weather data: {e}")
                weather_data = pd.DataFrame()
            
            # Fetch energy data with error handling
            try:
                energy_data = self.fetch_all_energy_data(energy_start_str, energy_end_str)
            except Exception as e:
                self.logger.error(f"Failed to fetch energy data: {e}")
                energy_data = pd.DataFrame()
            
            # Log final data availability
            weather_available = not weather_data.empty
            energy_available = not energy_data.empty
            self.logger.info(f"Daily data fetch summary - Weather: {len(weather_data)} records, Energy: {len(energy_data)} records")
            
            if not weather_available and not energy_available:
                self.logger.warning("No data available from either weather or energy sources")
            
            return {
                'weather': weather_data,
                'energy': energy_data,
                'date': today.strftime('%Y-%m-%d'),
                'weather_date_range': f"{weather_start_str} to {weather_end_str}",
                'energy_date_range': f"{energy_start_str} to {energy_end_str}"
            }
            
        except Exception as e:
            self.logger.error(f"Critical error in fetch_daily_data: {e}")
            # Return empty data structure to prevent pipeline crash
            return {
                'weather': pd.DataFrame(),
                'energy': pd.DataFrame(),
                'date': datetime.now().strftime('%Y-%m-%d'),
                'weather_date_range': "N/A",
                'energy_date_range': "N/A"
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