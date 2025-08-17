"""
Main pipeline orchestration module for weather and energy data collection.
"""

import logging
import sys
import argparse
from datetime import datetime, timedelta
from pathlib import Path
import yaml
import pandas as pd


# Add the src directory to the path
sys.path.append(str(Path(__file__).parent))

from data_fetcher import DataFetcher
from data_processor import DataProcessor
from analysis import DataAnalyzer
from notification_system import NotificationSystem


def setup_logging(config: dict):
    """Set up logging configuration."""
    logs_path = Path(config['paths']['logs'])
    logs_path.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, config['logging']['level']),
        format=config['logging']['format'],
        handlers=[
            logging.FileHandler(config['logging']['file']),
            logging.StreamHandler(sys.stdout)
        ]
    )


class WeatherEnergyPipeline:
    """Main pipeline class for orchestrating data collection and processing."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the pipeline."""
        self.config_path = config_path
        self.config = self._load_config()
        setup_logging(self.config)
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing Weather Energy Pipeline")
        
        # Initialize components
        self.fetcher = DataFetcher(config_path)
        self.processor = DataProcessor(config_path)
        self.analyzer = DataAnalyzer(config_path)
        self.notifier = NotificationSystem(config_path)
    
    def _load_config(self) -> dict:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML configuration: {e}")
    
    def run_daily_pipeline(self):
        """Run the daily data collection and processing pipeline with enhanced robustness."""
        try:
            self.logger.info("Starting enhanced daily pipeline execution")
            
            # Step 1: Check for missing data and attempt backfill
            missing_dates = self.fetcher.detect_missing_dates(expected_days=7)
            if missing_dates:
                self.logger.info(f"Found {len(missing_dates)} missing dates, attempting backfill")
                backfill_results = self.fetcher.backfill_missing_data(missing_dates)
                
                # Notify about missing data
                self.notifier.notify_missing_data(missing_dates, backfill_results)
                
                if backfill_results['status'] == 'success' and backfill_results['dates_processed']:
                    self.logger.info(f"Successfully backfilled {len(backfill_results['dates_processed'])} dates")
            
            # Step 2: Attempt to fetch today's data with fallback mechanisms
            self.logger.info("Fetching daily data with enhanced fallback support")
            
            # Use the new fetch_with_fallback method
            today = datetime.now()
            base_end_date = today - timedelta(days=4)
            base_start_date = base_end_date - timedelta(days=2)
            
            start_str = base_start_date.strftime('%Y-%m-%d')
            end_str = base_end_date.strftime('%Y-%m-%d')
            
            daily_data = self.fetcher.fetch_with_fallback(start_str, end_str)
            daily_data['date'] = today.strftime('%Y-%m-%d')  # Add current date for tracking
            
            # Step 3: Process data based on availability
            weather_available = not daily_data['weather'].empty
            energy_available = not daily_data['energy'].empty
            data_source = daily_data.get('source', 'unknown')
            
            self.logger.info(f"Data availability - Weather: {weather_available}, Energy: {energy_available}, Source: {data_source}")
            
            if weather_available and energy_available:
                # Process the data with both weather and energy
                processed_data = self.processor.process_all_data(
                    daily_data['weather'], 
                    daily_data['energy']
                )
                data_type = "complete"
            elif energy_available:
                # Process with energy data only
                self.logger.warning("Weather data unavailable, processing energy data only")
                processed_data = self.processor.process_energy_only(daily_data['energy'])
                data_type = "energy_only"
            elif weather_available:
                # Process with weather data only
                self.logger.warning("Energy data unavailable, processing weather data only")
                processed_data = self.processor.process_weather_only(daily_data['weather'])
                data_type = "weather_only"
            else:
                self.logger.warning("No new data available from either source for daily update")
                
                # Notify about complete data failure
                error_details = {
                    "status": "error",
                    "message": "No data available from any source",
                    "data_sources": data_source,
                    "processed_records": 0
                }
                self.notifier.notify_pipeline_failure(error_details, "daily")
                
                # Try to return status based on existing data instead of failing
                try:
                    latest_data = self.processor.get_latest_data()
                    if latest_data is not None and not latest_data.empty:
                        self.logger.info("No new data available, but existing processed data found")
                        return {
                            "status": "warning",
                            "message": "No new data available for daily update",
                            "existing_records": len(latest_data),
                            "last_data_date": str(latest_data['date'].max()) if 'date' in latest_data.columns else "Unknown",
                            "data_type": "existing_only"
                        }
                    else:
                        self.logger.warning("No new data and no existing processed data found")
                        return {
                            "status": "error", 
                            "message": "No data available from either source and no existing data found",
                            "data_type": "none"
                        }
                except Exception as e:
                    self.logger.error(f"Error checking existing data: {e}")
                    return {
                        "status": "error",
                        "message": "No new data available for daily update",
                        "data_type": "none"
                    }
                
            if not processed_data.empty:
                # Step 4: Enhanced data validation and quality analysis
                self.logger.info("Running enhanced data integrity validation")
                
                # Run comprehensive data validation
                validation_results = self.processor.validate_data_integrity(processed_data)
                quality_report = self.analyzer.check_data_quality(processed_data)
                quality_score = self.analyzer.get_data_quality_score(processed_data)
                
                # Check if data quality issues require notification
                if validation_results['integrity_score'] < 70 or quality_score < 70:
                    self.logger.warning(f"Data quality issues detected. Integrity: {validation_results['integrity_score']:.1f}, Quality: {quality_score:.1f}")
                    self.notifier.notify_data_quality_issue(validation_results, threshold=70.0)
                
                # Step 5: File cleanup (maintain only last 30 days)
                try:
                    cleanup_results = self.processor.cleanup_old_files(retention_days=30)
                    if cleanup_results['files_removed']:
                        self.logger.info(f"Cleaned up {len(cleanup_results['files_removed'])} old files, freed {cleanup_results['space_freed_mb']} MB")
                except Exception as e:
                    self.logger.warning(f"File cleanup failed: {e}")
                
                # Save quality report with error handling
                try:
                    self.analyzer.save_quality_report(quality_report, 
                                                    f"daily_quality_report_{daily_data['date']}.json")
                except Exception as e:
                    self.logger.warning(f"Failed to save quality report: {e}")
                
                self.logger.info(f"Enhanced daily pipeline completed successfully. Quality score: {quality_score:.1f}/100, Data type: {data_type}, Source: {data_source}")
                
                # Notify about successful recovery if using fallback data
                if data_source != 'api_primary':
                    recovery_details = {
                        'type': 'automatic',
                        'records_processed': len(processed_data),
                        'data_sources': data_source,
                        'quality_score': quality_score,
                        'source': data_source,
                        'message': f"Pipeline successfully recovered using {data_source} data source"
                    }
                    self.notifier.notify_successful_recovery(recovery_details)
                
                return {
                    "status": "success",
                    "processed_records": len(processed_data),
                    "quality_score": quality_score,
                    "integrity_score": validation_results['integrity_score'],
                    "date": daily_data['date'],
                    "data_type": data_type,
                    "data_source": data_source,
                    "weather_available": weather_available,
                    "energy_available": energy_available,
                    "validation_warnings": len(validation_results['warnings']),
                    "validation_errors": len(validation_results['errors'])
                }
            else:
                self.logger.error("No data remained after processing")
                error_details = {
                    "status": "error",
                    "message": "No data remained after processing",
                    "data_sources": data_source,
                    "processed_records": 0
                }
                self.notifier.notify_pipeline_failure(error_details, "daily")
                return {"status": "error", "message": "No data after processing"}
                
        except Exception as e:
            self.logger.error(f"Critical error in enhanced daily pipeline: {e}")
            
            # Notify about pipeline failure
            error_details = {
                "status": "error",
                "message": str(e),
                "data_sources": "unknown",
                "processed_records": 0
            }
            self.notifier.notify_pipeline_failure(error_details, "daily")
            
            return {"status": "error", "message": str(e)}
    
    def run_historical_backfill(self, days: int = None):
        """Run historical data backfill."""
        try:
            if days is None:
                days = self.config['data']['historical_days']
            
            self.logger.info(f"Starting historical backfill for {days} days")
            
            # Fetch historical data
            historical_data = self.fetcher.fetch_historical_data(days)
            
            if not historical_data['weather'].empty and not historical_data['energy'].empty:
                # Process the data
                processed_data = self.processor.process_all_data(
                    historical_data['weather'], 
                    historical_data['energy']
                )
                
                if not processed_data.empty:
                    # Run comprehensive analysis
                    quality_report = self.analyzer.generate_quality_dashboard_data(processed_data)
                    quality_score = self.analyzer.get_data_quality_score(processed_data)
                    
                    # Save quality report
                    self.analyzer.save_quality_report(quality_report, 
                                                    f"historical_analysis_{days}days.json")
                    
                    self.logger.info(f"Historical backfill completed successfully. Quality score: {quality_score:.1f}/100")
                    
                    return {
                        "status": "success",
                        "processed_records": len(processed_data),
                        "quality_score": quality_score,
                        "date_range": f"{historical_data['start_date']} to {historical_data['end_date']}",
                        "days": days
                    }
                else:
                    self.logger.error("No data remained after processing")
                    return {"status": "error", "message": "No data after processing"}
            else:
                self.logger.error("Failed to fetch historical data")
                return {"status": "error", "message": "Failed to fetch historical data"}
                
        except Exception as e:
            self.logger.error(f"Error in historical backfill: {e}")
            return {"status": "error", "message": str(e)}
    
    def run_quality_check(self):
        """Run quality checks on existing processed data."""
        try:
            self.logger.info("Running quality check on existing data")
            
            # Load latest processed data
            latest_data = self.processor.get_latest_data()
            
            if latest_data is None or latest_data.empty:
                self.logger.warning("No processed data found for quality check")
                return {"status": "warning", "message": "No processed data found"}
            
            # Run quality analysis
            quality_report = self.analyzer.generate_quality_dashboard_data(latest_data)
            quality_score = self.analyzer.get_data_quality_score(latest_data)
            
            # Save quality report
            self.analyzer.save_quality_report(quality_report, 
                                            f"quality_check_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            
            self.logger.info(f"Quality check completed. Score: {quality_score:.1f}/100")
            
            return {
                "status": "success",
                "quality_score": quality_score,
                "total_records": len(latest_data),
                "quality_report": quality_report
            }
            
        except Exception as e:
            self.logger.error(f"Error in quality check: {e}")
            return {"status": "error", "message": str(e)}
    
    def get_pipeline_status(self):
        """Get current pipeline status and data summary."""
        try:
            # Load latest processed data
            latest_data = self.processor.get_latest_data()
            
            if latest_data is None or latest_data.empty:
                return {
                    "status": "no_data",
                    "message": "No processed data found"
                }
            
            # Basic statistics
            total_records = len(latest_data)
            cities = latest_data['city'].nunique() if 'city' in latest_data.columns else 0
            # Handle date range safely
            if 'date' in latest_data.columns:
                try:
                    # Convert to datetime if it's not already
                    date_col = pd.to_datetime(latest_data['date'])
                    date_range = {
                        "start": date_col.min().isoformat(),
                        "end": date_col.max().isoformat()
                    }
                except Exception:
                    # Fallback to string representation
                    date_range = {
                        "start": str(latest_data['date'].min()),
                        "end": str(latest_data['date'].max())
                    }
            else:
                date_range = {"start": None, "end": None}
            
            # Quality score
            quality_score = self.analyzer.get_data_quality_score(latest_data)
            
            return {
                "status": "active",
                "total_records": total_records,
                "cities_covered": cities,
                "date_range": date_range,
                "quality_score": quality_score,
                "last_update": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting pipeline status: {e}")
            return {"status": "error", "message": str(e)}


def main():
    """Main entry point for the pipeline."""
    parser = argparse.ArgumentParser(description="Weather and Energy Data Pipeline")
    parser.add_argument('--mode', choices=['daily', 'historical', 'quality-check', 'status', 'backfill', 'cleanup'], 
                       default='daily', help='Pipeline execution mode')
    parser.add_argument('--days', type=int, default=None, 
                       help='Number of days for historical backfill (default: from config)')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    
    args = parser.parse_args()
    
    try:
        # Initialize pipeline
        pipeline = WeatherEnergyPipeline(args.config)
        
        # Execute based on mode
        if args.mode == 'daily':
            result = pipeline.run_daily_pipeline()
        elif args.mode == 'historical':
            result = pipeline.run_historical_backfill(args.days)
        elif args.mode == 'quality-check':
            result = pipeline.run_quality_check()
        elif args.mode == 'status':
            result = pipeline.get_pipeline_status()
        elif args.mode == 'backfill':
            # Run missing data backfill
            missing_dates = pipeline.fetcher.detect_missing_dates(expected_days=args.days or 14)
            result = pipeline.fetcher.backfill_missing_data(missing_dates)
        elif args.mode == 'cleanup':
            # Run file cleanup
            result = pipeline.processor.cleanup_old_files(retention_days=args.days or 30)
        
        # Print result
        print(f"\nPipeline execution completed:")
        print(f"Status: {result['status']}")
        
        if result['status'] == 'success':
            if 'quality_score' in result:
                print(f"Quality Score: {result['quality_score']:.1f}/100")
            if 'processed_records' in result:
                print(f"Records Processed: {result['processed_records']}")
            if 'date_range' in result:
                print(f"Date Range: {result['date_range']}")
        elif result['status'] == 'error':
            print(f"Error: {result['message']}")
            sys.exit(1)
        elif result['status'] == 'warning':
            print(f"Warning: {result['message']}")
        
    except Exception as e:
        print(f"Pipeline execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()