"""
Analysis module for data quality checks and statistical analysis.
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import yaml
from pathlib import Path
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


class DataAnalyzer:
    """Main class for data quality analysis and statistical analysis."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the data analyzer with configuration."""
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
    
    def check_data_quality(self, df: pd.DataFrame) -> Dict:
        """
        Perform comprehensive data quality checks.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary containing quality metrics
        """
        if df.empty:
            return {"error": "Empty DataFrame provided"}
        
        try:
            quality_report = {
                "timestamp": datetime.now().isoformat(),
                "total_records": len(df),
                "missing_values": {},
                "outliers": {},
                "data_freshness": {},
                "summary_stats": {}
            }
            
            # Check missing values
            missing_data = df.isnull().sum()
            quality_report["missing_values"] = {
                "by_column": missing_data.to_dict(),
                "total_missing": missing_data.sum(),
                "percentage_missing": (missing_data.sum() / len(df)) * 100
            }
            
            # Check for outliers
            outliers = self._detect_outliers(df)
            quality_report["outliers"] = outliers
            
            # Check data freshness
            freshness = self._check_data_freshness(df)
            quality_report["data_freshness"] = freshness
            
            # Generate summary statistics
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) > 0:
                quality_report["summary_stats"] = df[numeric_columns].describe().to_dict()
            
            self.logger.info("Data quality check completed")
            return quality_report
            
        except Exception as e:
            self.logger.error(f"Error in data quality check: {e}")
            return {"error": str(e)}
    
    def _detect_outliers(self, df: pd.DataFrame) -> Dict:
        """
        Detect outliers in temperature and energy data.
        
        Args:
            df: DataFrame to check for outliers
            
        Returns:
            Dictionary with outlier information
        """
        outliers = {}
        
        # Temperature outliers
        if 'temp_avg_f' in df.columns:
            temp_min = self.quality_checks['temperature']['min_fahrenheit']
            temp_max = self.quality_checks['temperature']['max_fahrenheit']
            
            temp_outliers = (
                (df['temp_avg_f'] < temp_min) | 
                (df['temp_avg_f'] > temp_max)
            )
            
            outliers["temperature"] = {
                "count": temp_outliers.sum(),
                "percentage": (temp_outliers.sum() / len(df)) * 100,
                "extreme_values": df[temp_outliers]['temp_avg_f'].tolist() if temp_outliers.any() else []
            }
        
        # Energy outliers
        if 'energy_consumption_mwh' in df.columns:
            energy_min = self.quality_checks['energy']['min_consumption']
            energy_outliers = df['energy_consumption_mwh'] < energy_min
            
            # Also check for extremely high values (using IQR method)
            Q1 = df['energy_consumption_mwh'].quantile(0.25)
            Q3 = df['energy_consumption_mwh'].quantile(0.75)
            IQR = Q3 - Q1
            energy_high_outliers = df['energy_consumption_mwh'] > (Q3 + 1.5 * IQR)
            
            total_energy_outliers = energy_outliers | energy_high_outliers
            
            outliers["energy"] = {
                "count": total_energy_outliers.sum(),
                "percentage": (total_energy_outliers.sum() / len(df)) * 100,
                "negative_values": energy_outliers.sum(),
                "extreme_high_values": energy_high_outliers.sum()
            }
        
        return outliers
    
    def _check_data_freshness(self, df: pd.DataFrame) -> Dict:
        """
        Check if data is fresh (recent).
        
        Args:
            df: DataFrame to check
            
        Returns:
            Dictionary with freshness information
        """
        if 'date' not in df.columns:
            return {"error": "No date column found"}
        
        try:
            # Convert date column to datetime if not already
            if not pd.api.types.is_datetime64_any_dtype(df['date']):
                df['date'] = pd.to_datetime(df['date'])
            
            latest_date = df['date'].max()
            oldest_date = df['date'].min()
            now = datetime.now()
            
            # Calculate hours since latest data
            hours_since_latest = (now - latest_date.to_pydatetime()).total_seconds() / 3600
            
            # Check if data is considered stale
            freshness_threshold = self.quality_checks['freshness_threshold']
            is_stale = hours_since_latest > freshness_threshold
            
            return {
                "latest_date": latest_date.isoformat(),
                "oldest_date": oldest_date.isoformat(),
                "hours_since_latest": round(hours_since_latest, 2),
                "is_stale": is_stale,
                "freshness_threshold_hours": freshness_threshold,
                "date_range_days": (latest_date - oldest_date).days
            }
            
        except Exception as e:
            self.logger.error(f"Error checking data freshness: {e}")
            return {"error": str(e)}
    
    def calculate_correlations(self, df: pd.DataFrame) -> Dict:
        """
        Calculate correlations between temperature and energy consumption.
        
        Args:
            df: DataFrame with temperature and energy data
            
        Returns:
            Dictionary with correlation analysis
        """
        if df.empty:
            return {"error": "Empty DataFrame provided"}
        
        try:
            correlations = {}
            
            # Overall correlation
            if 'temp_avg_f' in df.columns and 'energy_consumption_mwh' in df.columns:
                # Remove NaN values for correlation calculation
                clean_data = df[['temp_avg_f', 'energy_consumption_mwh']].dropna()
                
                if len(clean_data) > 1:
                    correlation_coef = clean_data['temp_avg_f'].corr(clean_data['energy_consumption_mwh'])
                    
                    # Calculate R-squared using linear regression
                    X = clean_data[['temp_avg_f']]
                    y = clean_data['energy_consumption_mwh']
                    
                    model = LinearRegression()
                    model.fit(X, y)
                    y_pred = model.predict(X)
                    r_squared = r2_score(y, y_pred)
                    
                    correlations["overall"] = {
                        "correlation_coefficient": round(correlation_coef, 4),
                        "r_squared": round(r_squared, 4),
                        "slope": round(model.coef_[0], 4),
                        "intercept": round(model.intercept_, 4),
                        "sample_size": len(clean_data)
                    }
            
            # Correlation by city
            if 'city' in df.columns:
                city_correlations = {}
                for city in df['city'].unique():
                    city_data = df[df['city'] == city]
                    if len(city_data) > 1:
                        city_clean = city_data[['temp_avg_f', 'energy_consumption_mwh']].dropna()
                        
                        if len(city_clean) > 1:
                            city_corr = city_clean['temp_avg_f'].corr(city_clean['energy_consumption_mwh'])
                            
                            # Linear regression for city
                            X_city = city_clean[['temp_avg_f']]
                            y_city = city_clean['energy_consumption_mwh']
                            
                            model_city = LinearRegression()
                            model_city.fit(X_city, y_city)
                            y_pred_city = model_city.predict(X_city)
                            r_squared_city = r2_score(y_city, y_pred_city)
                            
                            city_correlations[city] = {
                                "correlation_coefficient": round(city_corr, 4),
                                "r_squared": round(r_squared_city, 4),
                                "slope": round(model_city.coef_[0], 4),
                                "intercept": round(model_city.intercept_, 4),
                                "sample_size": len(city_clean)
                            }
                
                correlations["by_city"] = city_correlations
            
            self.logger.info("Correlation analysis completed")
            return correlations
            
        except Exception as e:
            self.logger.error(f"Error calculating correlations: {e}")
            return {"error": str(e)}
    
    def analyze_usage_patterns(self, df: pd.DataFrame) -> Dict:
        """
        Analyze energy usage patterns by temperature range and day of week.
        
        Args:
            df: DataFrame with processed data
            
        Returns:
            Dictionary with usage pattern analysis
        """
        if df.empty:
            return {"error": "Empty DataFrame provided"}
        
        try:
            patterns = {}
            
            # Usage by temperature range
            if 'temp_range' in df.columns and 'energy_consumption_mwh' in df.columns:
                temp_usage = df.groupby('temp_range')['energy_consumption_mwh'].agg([
                    'mean', 'median', 'std', 'count'
                ]).round(2)
                patterns["by_temperature_range"] = temp_usage.to_dict()
            
            # Usage by day of week
            if 'day_of_week' in df.columns:
                day_usage = df.groupby('day_of_week')['energy_consumption_mwh'].agg([
                    'mean', 'median', 'std', 'count'
                ]).round(2)
                patterns["by_day_of_week"] = day_usage.to_dict()
            
            # Weekend vs weekday comparison
            if 'is_weekend' in df.columns:
                weekend_usage = df.groupby('is_weekend')['energy_consumption_mwh'].agg([
                    'mean', 'median', 'std', 'count'
                ]).round(2)
                patterns["weekend_vs_weekday"] = weekend_usage.to_dict()
            
            # Heatmap data for temperature range vs day of week
            if all(col in df.columns for col in ['temp_range', 'day_of_week', 'energy_consumption_mwh']):
                heatmap_data = df.pivot_table(
                    index='temp_range',
                    columns='day_of_week',
                    values='energy_consumption_mwh',
                    aggfunc='mean'
                ).round(2)
                patterns["heatmap_data"] = heatmap_data.to_dict()
            
            self.logger.info("Usage pattern analysis completed")
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error analyzing usage patterns: {e}")
            return {"error": str(e)}
    
    def generate_quality_dashboard_data(self, df: pd.DataFrame) -> Dict:
        """
        Generate data for quality dashboard visualization.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary with dashboard data
        """
        try:
            dashboard_data = {
                "quality_metrics": self.check_data_quality(df),
                "correlations": self.calculate_correlations(df),
                "usage_patterns": self.analyze_usage_patterns(df),
                "time_series_summary": self._generate_time_series_summary(df)
            }
            
            return dashboard_data
            
        except Exception as e:
            self.logger.error(f"Error generating dashboard data: {e}")
            return {"error": str(e)}
    
    def _generate_time_series_summary(self, df: pd.DataFrame) -> Dict:
        """
        Generate summary statistics for time series data.
        
        Args:
            df: DataFrame with time series data
            
        Returns:
            Dictionary with time series summary
        """
        try:
            summary = {}
            
            if 'date' in df.columns:
                # Convert date to datetime if needed
                if not pd.api.types.is_datetime64_any_dtype(df['date']):
                    df['date'] = pd.to_datetime(df['date'])
                
                # Date range
                summary["date_range"] = {
                    "start": df['date'].min().isoformat(),
                    "end": df['date'].max().isoformat(),
                    "total_days": (df['date'].max() - df['date'].min()).days
                }
                
                # Monthly trends
                if len(df) > 0:
                    df['month'] = df['date'].dt.to_period('M')
                    monthly_summary = df.groupby('month').agg({
                        'temp_avg_f': 'mean',
                        'energy_consumption_mwh': 'mean'
                    }).round(2)
                    
                    # Convert Period index to string for JSON serialization
                    monthly_summary.index = monthly_summary.index.astype(str)
                    summary["monthly_trends"] = monthly_summary.to_dict()
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating time series summary: {e}")
            return {"error": str(e)}
    
    def save_quality_report(self, quality_data: Dict, filename: str = None):
        """
        Save quality report to file.
        
        Args:
            quality_data: Quality analysis data
            filename: Optional filename
        """
        try:
            logs_path = Path(self.config['paths']['logs'])
            logs_path.mkdir(parents=True, exist_ok=True)
            
            if filename is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"quality_report_{timestamp}.json"
            
            filepath = logs_path / filename
            
            import json
            import pandas as pd
            
            def json_serializer(obj):
                """Custom JSON serializer for pandas objects."""
                if isinstance(obj, pd.Period):
                    return str(obj)
                elif isinstance(obj, pd.Timestamp):
                    return obj.isoformat()
                elif isinstance(obj, pd.Timedelta):
                    return str(obj)
                elif isinstance(obj, (pd.Series, pd.DataFrame)):
                    return obj.to_dict()
                elif hasattr(obj, 'item'):  # numpy types
                    return obj.item()
                elif isinstance(obj, (np.int64, np.int32, np.float64, np.float32)):
                    return obj.item()
                elif pd.isna(obj):
                    return None
                else:
                    return str(obj)
            
            with open(filepath, 'w') as f:
                json.dump(quality_data, f, indent=2, default=json_serializer)
            
            self.logger.info(f"Quality report saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving quality report: {e}")
    
    def get_data_quality_score(self, df: pd.DataFrame) -> float:
        """
        Calculate overall data quality score (0-100).
        
        Args:
            df: DataFrame to score
            
        Returns:
            Quality score between 0 and 100
        """
        try:
            if df.empty:
                return 0.0
            
            quality_checks = self.check_data_quality(df)
            
            score = 100.0
            
            # Deduct points for missing values
            missing_percentage = quality_checks.get("missing_values", {}).get("percentage_missing", 0)
            score -= missing_percentage * 0.5  # 0.5 points per % missing
            
            # Deduct points for outliers
            temp_outliers = quality_checks.get("outliers", {}).get("temperature", {}).get("percentage", 0)
            energy_outliers = quality_checks.get("outliers", {}).get("energy", {}).get("percentage", 0)
            score -= (temp_outliers + energy_outliers) * 0.3  # 0.3 points per % outliers
            
            # Deduct points for stale data
            freshness = quality_checks.get("data_freshness", {})
            if freshness.get("is_stale", False):
                score -= 10.0  # 10 points for stale data
            
            return max(0.0, min(100.0, score))
            
        except Exception as e:
            self.logger.error(f"Error calculating quality score: {e}")
            return 0.0