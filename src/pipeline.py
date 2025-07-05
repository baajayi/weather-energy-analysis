"""
Main pipeline orchestration module for weather and energy data collection.
"""

import logging
import sys
import argparse
from datetime import datetime
from pathlib import Path
import yaml

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent))

from data_fetcher import DataFetcher
from data_processor import DataProcessor
from analysis import DataAnalyzer


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
        """Run the daily data collection and processing pipeline."""
        try:
            self.logger.info("Starting daily pipeline execution")
            
            # Fetch daily data
            daily_data = self.fetcher.fetch_daily_data()
            
            if not daily_data['weather'].empty and not daily_data['energy'].empty:
                # Process the data
                processed_data = self.processor.process_all_data(
                    daily_data['weather'], 
                    daily_data['energy']
                )
                
                if not processed_data.empty:
                    # Run quality analysis
                    quality_report = self.analyzer.check_data_quality(processed_data)
                    quality_score = self.analyzer.get_data_quality_score(processed_data)
                    
                    # Save quality report
                    self.analyzer.save_quality_report(quality_report, 
                                                    f"daily_quality_report_{daily_data['date']}.json")
                    
                    self.logger.info(f"Daily pipeline completed successfully. Quality score: {quality_score:.1f}/100")
                    
                    # Log any quality issues
                    if quality_score < 80:
                        self.logger.warning(f"Data quality score is below 80: {quality_score:.1f}")
                    
                    return {
                        "status": "success",
                        "processed_records": len(processed_data),
                        "quality_score": quality_score,
                        "date": daily_data['date']
                    }
                else:
                    self.logger.error("No data remained after processing")
                    return {"status": "error", "message": "No data after processing"}
            else:
                self.logger.error("Failed to fetch daily data")
                return {"status": "error", "message": "Failed to fetch daily data"}
                
        except Exception as e:
            self.logger.error(f"Error in daily pipeline: {e}")
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
            date_range = {
                "start": latest_data['date'].min().isoformat() if 'date' in latest_data.columns else None,
                "end": latest_data['date'].max().isoformat() if 'date' in latest_data.columns else None
            }
            
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
    parser.add_argument('--mode', choices=['daily', 'historical', 'quality-check', 'status'], 
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