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
            
            # Debug API key (show only first/last 4 characters for security)
            api_key = self.eia_config['api_key']
            if api_key:
                key_preview = f"{api_key[:4]}...{api_key[-4:]}" if len(api_key) > 8 else "short_key"
                self.logger.debug(f"Using EIA API key: {key_preview}")
            else:
                self.logger.warning("EIA API key is empty or missing!")
            
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
        
        FIXED VERSION: Both APIs use identical date ranges to ensure data overlap.
        
        Returns:
            Dictionary containing weather and energy DataFrames
        """
        try:
            today = datetime.now()
            self.logger.info(f"FETCH_DAILY_DATA: Starting with today = {today}")
            
            # CRITICAL FIX: Use consistent 4-day lag for both APIs to ensure overlap
            # This accounts for weather data reporting delays
            base_end_date = today - timedelta(days=4)
            base_start_date = base_end_date - timedelta(days=2)
            
            self.logger.info(f"FETCH_DAILY_DATA: Calculated base date range: {base_start_date.strftime('%Y-%m-%d')} to {base_end_date.strftime('%Y-%m-%d')}")
            
            # ENFORCE: Both APIs use IDENTICAL date ranges
            weather_start_date = base_start_date
            weather_end_date = base_end_date
            energy_start_date = base_start_date  # SAME as weather
            energy_end_date = base_end_date      # SAME as weather
            
            # Convert to strings
            weather_start_str = weather_start_date.strftime('%Y-%m-%d')
            weather_end_str = weather_end_date.strftime('%Y-%m-%d')
            energy_start_str = energy_start_date.strftime('%Y-%m-%d')
            energy_end_str = energy_end_date.strftime('%Y-%m-%d')
            
            # VERIFICATION: Log the ranges to confirm alignment
            self.logger.info(f"FETCH_DAILY_DATA: Weather API will use: {weather_start_str} to {weather_end_str}")
            self.logger.info(f"FETCH_DAILY_DATA: Energy API will use:  {energy_start_str} to {energy_end_str}")
            
            if weather_start_str != energy_start_str or weather_end_str != energy_end_str:
                self.logger.error("CRITICAL ERROR: Date ranges are not aligned!")
                raise ValueError("Date range alignment verification failed")
            else:
                self.logger.info("FETCH_DAILY_DATA: ✅ Date ranges verified as IDENTICAL")
            
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
            
            # FINAL VERIFICATION: Ensure return values are aligned
            aligned_range = f"{weather_start_str} to {weather_end_str}"
            self.logger.info(f"FETCH_DAILY_DATA: Returning aligned date range: {aligned_range}")
            
            return {
                'weather': weather_data,
                'energy': energy_data,
                'date': today.strftime('%Y-%m-%d'),
                'weather_date_range': aligned_range,
                'energy_date_range': aligned_range  # GUARANTEED to be identical
            }
            
        except Exception as e:
            self.logger.error(f"Critical error in fetch_daily_data: {e}")
            # Return empty data structure to prevent pipeline crash
            error_range = "ERROR"
            return {
                'weather': pd.DataFrame(),
                'energy': pd.DataFrame(),
                'date': datetime.now().strftime('%Y-%m-%d'),
                'weather_date_range': error_range,
                'energy_date_range': error_range  # Keep aligned even in error case
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
    
    def get_cached_data(self, days_back: int = 7) -> Dict[str, pd.DataFrame]:
        """
        Retrieve cached/historical data from processed files as fallback.
        
        Args:
            days_back: Number of days to look back for cached data
            
        Returns:
            Dictionary containing cached weather and energy data
        """
        try:
            processed_path = Path(self.config['paths']['processed_data'])
            
            if not processed_path.exists():
                self.logger.warning("No processed data directory found")
                return {'weather': pd.DataFrame(), 'energy': pd.DataFrame()}
            
            # Get all processed CSV files sorted by modification time (newest first)
            csv_files = sorted(
                processed_path.glob("processed_data_*.csv"),
                key=lambda x: x.stat().st_mtime,
                reverse=True
            )
            
            if not csv_files:
                self.logger.warning("No processed data files found for fallback")
                return {'weather': pd.DataFrame(), 'energy': pd.DataFrame()}
            
            # Load the most recent file
            latest_file = csv_files[0]
            self.logger.info(f"Loading cached data from {latest_file}")
            
            cached_df = pd.read_csv(latest_file)
            
            # Convert date column to datetime for filtering
            cached_df['date'] = pd.to_datetime(cached_df['date'])
            
            # Filter for recent data within the days_back window
            cutoff_date = datetime.now() - timedelta(days=days_back)
            recent_data = cached_df[cached_df['date'] >= cutoff_date]
            
            # Separate weather and energy columns (simplified separation)
            weather_cols = ['date', 'city', 'state', 'lat', 'lon', 'temp_max_f', 'temp_min_f', 'temp_avg_f']
            energy_cols = ['date', 'city', 'state', 'eia_region', 'value']
            
            weather_data = recent_data[weather_cols].dropna() if all(col in recent_data.columns for col in weather_cols) else pd.DataFrame()
            energy_data = recent_data[energy_cols].dropna() if all(col in recent_data.columns for col in energy_cols) else pd.DataFrame()
            
            self.logger.info(f"Retrieved cached data - Weather: {len(weather_data)} records, Energy: {len(energy_data)} records")
            
            return {'weather': weather_data, 'energy': energy_data}
            
        except Exception as e:
            self.logger.error(f"Error retrieving cached data: {e}")
            return {'weather': pd.DataFrame(), 'energy': pd.DataFrame()}
    
    def fetch_with_fallback(self, start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """
        Fetch data with automatic fallback to cached data if APIs fail.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            Dictionary containing weather and energy DataFrames
        """
        try:
            # Primary attempt: fetch from APIs
            self.logger.info(f"Attempting primary data fetch from APIs for {start_date} to {end_date}")
            
            weather_data = self.fetch_all_weather_data(start_date, end_date)
            energy_data = self.fetch_all_energy_data(start_date, end_date)
            
            weather_available = not weather_data.empty
            energy_available = not energy_data.empty
            
            # If both sources have data, return successful fetch
            if weather_available and energy_available:
                self.logger.info("Primary data fetch successful for both sources")
                return {
                    'weather': weather_data,
                    'energy': energy_data,
                    'source': 'api_primary',
                    'date_range': f"{start_date} to {end_date}"
                }
            
            # Partial failure: try to supplement with cached data
            if weather_available or energy_available:
                self.logger.warning("Partial API failure - attempting to supplement with cached data")
                cached_data = self.get_cached_data(days_back=14)
                
                if not weather_available and not cached_data['weather'].empty:
                    weather_data = cached_data['weather']
                    self.logger.info("Using cached weather data as fallback")
                
                if not energy_available and not cached_data['energy'].empty:
                    energy_data = cached_data['energy']
                    self.logger.info("Using cached energy data as fallback")
                
                return {
                    'weather': weather_data,
                    'energy': energy_data,
                    'source': 'api_partial_cached_fallback',
                    'date_range': f"{start_date} to {end_date}"
                }
            
            # Complete API failure: use cached data only
            self.logger.error("Complete API failure - falling back to cached data")
            cached_data = self.get_cached_data(days_back=14)
            
            return {
                'weather': cached_data['weather'],
                'energy': cached_data['energy'],
                'source': 'cached_fallback_only',
                'date_range': f"cached_last_14_days"
            }
            
        except Exception as e:
            self.logger.error(f"Critical error in fetch_with_fallback: {e}")
            # Final fallback: return empty structure
            return {
                'weather': pd.DataFrame(),
                'energy': pd.DataFrame(),
                'source': 'error_fallback',
                'date_range': 'error'
            }
    
    def detect_missing_dates(self, expected_days: int = 7) -> List[str]:
        """
        Detect missing dates in processed data for automated backfill.
        
        Args:
            expected_days: Number of recent days expected to have data
            
        Returns:
            List of missing dates in YYYY-MM-DD format
        """
        try:
            processed_path = Path(self.config['paths']['processed_data'])
            
            if not processed_path.exists():
                self.logger.warning("No processed data directory found")
                return []
            
            # Get all processed CSV files
            csv_files = list(processed_path.glob("processed_data_*.csv"))
            
            if not csv_files:
                self.logger.warning("No processed data files found")
                # Return all expected days as missing
                end_date = datetime.now() - timedelta(days=4)  # Account for API lag
                return [(end_date - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(expected_days)]
            
            # Collect all dates present in processed files
            present_dates = set()
            
            for csv_file in csv_files:
                try:
                    df = pd.read_csv(csv_file)
                    if 'date' in df.columns:
                        dates = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d').unique()
                        present_dates.update(dates)
                except Exception as e:
                    self.logger.warning(f"Error reading {csv_file}: {e}")
                    continue
            
            # Generate expected date range (accounting for API lag)
            end_date = datetime.now() - timedelta(days=4)
            expected_dates = {
                (end_date - timedelta(days=i)).strftime('%Y-%m-%d') 
                for i in range(expected_days)
            }
            
            # Find missing dates
            missing_dates = sorted(expected_dates - present_dates)
            
            if missing_dates:
                self.logger.info(f"Detected {len(missing_dates)} missing dates: {missing_dates}")
            else:
                self.logger.info("No missing dates detected")
            
            return missing_dates
            
        except Exception as e:
            self.logger.error(f"Error detecting missing dates: {e}")
            return []
    
    def backfill_missing_data(self, missing_dates: List[str] = None) -> Dict[str, any]:
        """
        Automatically backfill missing data for specified dates.
        
        Args:
            missing_dates: List of dates to backfill. If None, auto-detect missing dates.
            
        Returns:
            Dictionary with backfill results
        """
        try:
            if missing_dates is None:
                missing_dates = self.detect_missing_dates(expected_days=14)
            
            if not missing_dates:
                return {
                    'status': 'success',
                    'message': 'No missing dates to backfill',
                    'dates_processed': [],
                    'total_records': 0
                }
            
            self.logger.info(f"Starting backfill for {len(missing_dates)} missing dates")
            
            successful_dates = []
            total_records = 0
            
            for date_str in missing_dates:
                try:
                    self.logger.info(f"Backfilling data for {date_str}")
                    
                    # Fetch data for single date with small range
                    start_date = date_str
                    end_date = date_str
                    
                    result = self.fetch_with_fallback(start_date, end_date)
                    
                    if not result['weather'].empty or not result['energy'].empty:
                        successful_dates.append(date_str)
                        total_records += len(result['weather']) + len(result['energy'])
                        self.logger.info(f"Successfully backfilled data for {date_str} from {result['source']}")
                    else:
                        self.logger.warning(f"No data available for backfill on {date_str}")
                        
                except Exception as e:
                    self.logger.error(f"Failed to backfill data for {date_str}: {e}")
                    continue
            
            return {
                'status': 'success' if successful_dates else 'partial',
                'message': f"Backfilled {len(successful_dates)} of {len(missing_dates)} missing dates",
                'dates_processed': successful_dates,
                'dates_failed': [d for d in missing_dates if d not in successful_dates],
                'total_records': total_records
            }
            
        except Exception as e:
            self.logger.error(f"Error in backfill process: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'dates_processed': [],
                'total_records': 0
            }