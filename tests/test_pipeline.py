"""
Unit tests for the weather and energy analysis pipeline.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import yaml
import sys
from pathlib import Path

# Add the project root and src directory to the path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# Import from src package  
from src.data_fetcher import DataFetcher
from src.data_processor import DataProcessor
from src.analysis import DataAnalyzer
from src.pipeline import WeatherEnergyPipeline


class TestDataFetcher:
    """Test cases for DataFetcher class."""
    
    @pytest.fixture
    def mock_config(self):
        """Mock configuration for testing."""
        return {
            'api': {
                'noaa': {
                    'base_url': 'https://api.noaa.gov',
                    'token': 'test_token',
                    'timeout': 30,
                    'retry_attempts': 3,
                    'retry_delay': 1
                },
                'eia': {
                    'base_url': 'https://api.eia.gov',
                    'api_key': 'test_key',
                    'timeout': 30,
                    'retry_attempts': 3,
                    'retry_delay': 1
                }
            },
            'cities': [
                {
                    'name': 'Test City',
                    'state': 'Test State',
                    'noaa_station_id': 'GHCND:TEST123',
                    'eia_region_code': 'TEST',
                    'lat': 40.0,
                    'lon': -74.0
                }
            ],
            'paths': {
                'raw_data': 'data/raw',
                'processed_data': 'data/processed',
                'logs': 'logs'
            }
        }
    
    @pytest.fixture
    def data_fetcher(self, mock_config):
        """Create DataFetcher instance with mocked config."""
        with patch.object(DataFetcher, '_load_config', return_value=mock_config):
            return DataFetcher()
    
    def test_init(self, data_fetcher, mock_config):
        """Test DataFetcher initialization."""
        assert data_fetcher.config == mock_config
        assert data_fetcher.cities == mock_config['cities']
        assert data_fetcher.noaa_config == mock_config['api']['noaa']
        assert data_fetcher.eia_config == mock_config['api']['eia']
    
    @patch('requests.get')
    def test_fetch_weather_data_success(self, mock_get, data_fetcher):
        """Test successful weather data fetching."""
        # Mock successful API response
        mock_response = Mock()
        mock_response.json.return_value = {
            'results': [
                {
                    'date': '2024-01-01',
                    'datatype': 'TMAX',
                    'value': 250,  # 25.0°C in tenths
                    'station': 'GHCND:TEST123'
                }
            ]
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        city = data_fetcher.cities[0]
        result = data_fetcher.fetch_weather_data(city, '2024-01-01', '2024-01-01')
        
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert 'city' in result.columns
        assert result['city'].iloc[0] == 'Test City'
    
    @patch('requests.get')
    def test_fetch_weather_data_failure(self, mock_get, data_fetcher):
        """Test weather data fetching with API failure."""
        mock_get.side_effect = Exception("API Error")
        
        city = data_fetcher.cities[0]
        result = data_fetcher.fetch_weather_data(city, '2024-01-01', '2024-01-01')
        
        assert result is None
    
    @patch('requests.get')
    def test_fetch_energy_data_success(self, mock_get, data_fetcher):
        """Test successful energy data fetching."""
        # Mock successful API response
        mock_response = Mock()
        mock_response.json.return_value = {
            'response': {
                'data': [
                    {
                        'period': '2024-01-01',
                        'value': 1000.0,
                        'respondent': 'TEST'
                    }
                ]
            }
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        city = data_fetcher.cities[0]
        result = data_fetcher.fetch_energy_data(city, '2024-01-01', '2024-01-01')
        
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert 'city' in result.columns
        assert result['city'].iloc[0] == 'Test City'


class TestDataProcessor:
    """Test cases for DataProcessor class."""
    
    @pytest.fixture
    def mock_config(self):
        """Mock configuration for testing."""
        return {
            'data': {
                'quality_checks': {
                    'temperature': {
                        'min_fahrenheit': -50,
                        'max_fahrenheit': 130
                    },
                    'energy': {
                        'min_consumption': 0
                    }
                }
            },
            'paths': {
                'processed_data': 'data/processed'
            }
        }
    
    @pytest.fixture
    def data_processor(self, mock_config):
        """Create DataProcessor instance with mocked config."""
        with patch.object(DataProcessor, '_load_config', return_value=mock_config):
            return DataProcessor()
    
    @pytest.fixture
    def sample_weather_data(self):
        """Sample weather data for testing."""
        return pd.DataFrame({
            'date': ['2024-01-01', '2024-01-02'],
            'datatype': ['TMAX', 'TMIN'],
            'value': [250, 150],  # 25.0°C and 15.0°C in tenths
            'city': ['Test City', 'Test City'],
            'state': ['Test State', 'Test State'],
            'lat': [40.0, 40.0],
            'lon': [-74.0, -74.0]
        })
    
    def test_celsius_tenths_to_fahrenheit(self, data_processor):
        """Test temperature conversion."""
        # 25.0°C (250 tenths) should be 77.0°F
        result = data_processor._celsius_tenths_to_fahrenheit(250)
        assert result == 77.0
        
        # 0°C (0 tenths) should be 32.0°F
        result = data_processor._celsius_tenths_to_fahrenheit(0)
        assert result == 32.0
        
        # Test NaN handling
        result = data_processor._celsius_tenths_to_fahrenheit(np.nan)
        assert pd.isna(result)
    
    def test_categorize_temperature(self, data_processor):
        """Test temperature categorization."""
        assert data_processor._categorize_temperature(45) == '<50°F'
        assert data_processor._categorize_temperature(55) == '50-60°F'
        assert data_processor._categorize_temperature(65) == '60-70°F'
        assert data_processor._categorize_temperature(75) == '70-80°F'
        assert data_processor._categorize_temperature(85) == '80-90°F'
        assert data_processor._categorize_temperature(95) == '>90°F'
        assert data_processor._categorize_temperature(np.nan) == 'Unknown'
    
    def test_process_weather_data(self, data_processor, sample_weather_data):
        """Test weather data processing."""
        result = data_processor.process_weather_data(sample_weather_data)
        
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert 'temp_fahrenheit' in result.columns
        assert 'day_of_week' in result.columns
        assert 'is_weekend' in result.columns
        assert 'temp_range' in result.columns
    
    def test_process_empty_data(self, data_processor):
        """Test processing of empty DataFrames."""
        empty_df = pd.DataFrame()
        result = data_processor.process_weather_data(empty_df)
        
        assert isinstance(result, pd.DataFrame)
        assert result.empty


class TestDataAnalyzer:
    """Test cases for DataAnalyzer class."""
    
    @pytest.fixture
    def mock_config(self):
        """Mock configuration for testing."""
        return {
            'data': {
                'quality_checks': {
                    'temperature': {
                        'min_fahrenheit': -50,
                        'max_fahrenheit': 130
                    },
                    'energy': {
                        'min_consumption': 0
                    },
                    'freshness_threshold': 24
                }
            },
            'paths': {
                'logs': 'logs'
            }
        }
    
    @pytest.fixture
    def data_analyzer(self, mock_config):
        """Create DataAnalyzer instance with mocked config."""
        with patch.object(DataAnalyzer, '_load_config', return_value=mock_config):
            return DataAnalyzer()
    
    @pytest.fixture
    def sample_processed_data(self):
        """Sample processed data for testing."""
        dates = pd.date_range('2024-01-01', periods=10, freq='D')
        return pd.DataFrame({
            'date': dates,
            'city': ['Test City'] * 10,
            'temp_avg_f': [70, 72, 68, 75, 73, 71, 69, 74, 76, 72],
            'energy_consumption_mwh': [1000, 1100, 950, 1200, 1050, 1000, 980, 1150, 1250, 1100],
            'day_of_week': [date.strftime('%A') for date in dates],
            'is_weekend': [date.weekday() >= 5 for date in dates],
            'temp_range': ['70-80°F'] * 10
        })
    
    def test_check_data_quality(self, data_analyzer, sample_processed_data):
        """Test data quality checking."""
        result = data_analyzer.check_data_quality(sample_processed_data)
        
        assert isinstance(result, dict)
        assert 'missing_values' in result
        assert 'outliers' in result
        assert 'data_freshness' in result
        assert 'summary_stats' in result
        assert 'total_records' in result
        assert result['total_records'] == 10
    
    def test_calculate_correlations(self, data_analyzer, sample_processed_data):
        """Test correlation calculation."""
        result = data_analyzer.calculate_correlations(sample_processed_data)
        
        assert isinstance(result, dict)
        assert 'overall' in result
        assert 'correlation_coefficient' in result['overall']
        assert 'r_squared' in result['overall']
        assert 'slope' in result['overall']
        assert 'intercept' in result['overall']
    
    def test_analyze_usage_patterns(self, data_analyzer, sample_processed_data):
        """Test usage pattern analysis."""
        result = data_analyzer.analyze_usage_patterns(sample_processed_data)
        
        assert isinstance(result, dict)
        assert 'by_temperature_range' in result
        assert 'by_day_of_week' in result
        assert 'weekend_vs_weekday' in result
    
    def test_get_data_quality_score(self, data_analyzer, sample_processed_data):
        """Test data quality score calculation."""
        score = data_analyzer.get_data_quality_score(sample_processed_data)
        
        assert isinstance(score, float)
        assert 0 <= score <= 100
    
    def test_detect_outliers(self, data_analyzer):
        """Test outlier detection."""
        # Create data with outliers
        data_with_outliers = pd.DataFrame({
            'temp_avg_f': [70, 72, 150, 75, -60],  # Contains outliers
            'energy_consumption_mwh': [1000, 1100, 950, -100, 1050]  # Contains negative value
        })
        
        outliers = data_analyzer._detect_outliers(data_with_outliers)
        
        assert isinstance(outliers, dict)
        assert 'temperature' in outliers
        assert 'energy' in outliers
        assert outliers['temperature']['count'] > 0
        assert outliers['energy']['count'] > 0


class TestWeatherEnergyPipeline:
    """Test cases for WeatherEnergyPipeline class."""
    
    @pytest.fixture
    def mock_config(self):
        """Mock configuration for testing."""
        return {
            'data': {
                'historical_days': 90
            },
            'paths': {
                'logs': 'logs'
            },
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'file': 'logs/pipeline.log'
            }
        }
    
    @pytest.fixture
    def pipeline(self, mock_config):
        """Create WeatherEnergyPipeline instance with mocked config."""
        with patch.object(WeatherEnergyPipeline, '_load_config', return_value=mock_config):
            with patch('pipeline.setup_logging'):
                return WeatherEnergyPipeline()
    
    @patch('pipeline.DataFetcher')
    @patch('pipeline.DataProcessor')
    @patch('pipeline.DataAnalyzer')
    def test_pipeline_initialization(self, mock_analyzer, mock_processor, mock_fetcher, pipeline):
        """Test pipeline initialization."""
        assert pipeline.fetcher is not None
        assert pipeline.processor is not None
        assert pipeline.analyzer is not None
    
    def test_get_pipeline_status_no_data(self, pipeline):
        """Test pipeline status when no data exists."""
        with patch.object(pipeline.processor, 'get_latest_data', return_value=None):
            result = pipeline.get_pipeline_status()
            
            assert result['status'] == 'no_data'
            assert 'message' in result
    
    def test_get_pipeline_status_with_data(self, pipeline):
        """Test pipeline status with existing data."""
        # Mock data
        mock_data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=5),
            'city': ['Test City'] * 5,
            'temp_avg_f': [70, 72, 68, 75, 73],
            'energy_consumption_mwh': [1000, 1100, 950, 1200, 1050]
        })
        
        with patch.object(pipeline.processor, 'get_latest_data', return_value=mock_data):
            with patch.object(pipeline.analyzer, 'get_data_quality_score', return_value=85.5):
                result = pipeline.get_pipeline_status()
                
                assert result['status'] == 'active'
                assert result['total_records'] == 5
                assert result['cities_covered'] == 1
                assert result['quality_score'] == 85.5
                assert 'date_range' in result
                assert 'last_update' in result


# Integration tests
class TestIntegration:
    """Integration tests for the complete pipeline."""
    
    def test_end_to_end_data_flow(self):
        """Test complete data flow from raw to processed."""
        # Create mock raw weather data
        raw_weather = pd.DataFrame({
            'date': ['2024-01-01', '2024-01-01'],
            'datatype': ['TMAX', 'TMIN'],
            'value': [250, 150],  # 25.0°C and 15.0°C in tenths
            'city': ['Test City', 'Test City'],
            'state': ['Test State', 'Test State'],
            'lat': [40.0, 40.0],
            'lon': [-74.0, -74.0]
        })
        
        # Create mock raw energy data
        raw_energy = pd.DataFrame({
            'period': ['2024-01-01'],
            'value': [1000.0],
            'city': ['Test City'],
            'state': ['Test State'],
            'eia_region': ['TEST']
        })
        
        # Mock configuration
        mock_config = {
            'data': {
                'quality_checks': {
                    'temperature': {'min_fahrenheit': -50, 'max_fahrenheit': 130},
                    'energy': {'min_consumption': 0},
                    'freshness_threshold': 24
                }
            },
            'paths': {'processed_data': 'data/processed'}
        }
        
        # Process data
        with patch.object(DataProcessor, '_load_config', return_value=mock_config):
            processor = DataProcessor()
            result = processor.process_all_data(raw_weather, raw_energy)
            
            # Verify the complete pipeline produces valid output
            assert isinstance(result, pd.DataFrame)
            assert not result.empty
            assert 'temp_avg_f' in result.columns
            assert 'energy_consumption_mwh' in result.columns
            assert 'city' in result.columns
            assert 'date' in result.columns


# Test fixtures and helpers
@pytest.fixture
def sample_config():
    """Sample configuration for tests."""
    return {
        'api': {
            'noaa': {
                'base_url': 'https://api.noaa.gov',
                'token': 'test_token',
                'retry_attempts': 3,
                'retry_delay': 1
            },
            'eia': {
                'base_url': 'https://api.eia.gov',
                'api_key': 'test_key',
                'retry_attempts': 3,
                'retry_delay': 1
            }
        },
        'cities': [
            {
                'name': 'Test City',
                'state': 'Test State',
                'noaa_station_id': 'GHCND:TEST123',
                'eia_region_code': 'TEST',
                'lat': 40.0,
                'lon': -74.0
            }
        ],
        'data': {
            'quality_checks': {
                'temperature': {
                    'min_fahrenheit': -50,
                    'max_fahrenheit': 130
                },
                'energy': {
                    'min_consumption': 0
                },
                'freshness_threshold': 24
            }
        },
        'paths': {
            'raw_data': 'data/raw',
            'processed_data': 'data/processed',
            'logs': 'logs'
        }
    }


if __name__ == "__main__":
    pytest.main([__file__, "-v"])