# US Weather + Energy Analysis Pipeline

A production-ready data pipeline that combines weather data with energy consumption patterns to help utilities optimize power generation, reduce waste, and lower costs.

## 🎯 Business Impact

Energy companies lose millions of dollars due to inaccurate demand forecasting. This pipeline demonstrates how combining weather data with energy consumption patterns can deliver real business value through:

- **Demand Forecasting**: Predict energy needs based on temperature patterns
- **Cost Optimization**: Reduce waste by optimizing power generation
- **Regional Analysis**: Understand weather-energy relationships across different markets
- **Quality Monitoring**: Ensure data reliability for critical business decisions

## 📊 Key Features

### 1. Automated Data Pipeline
- **Daily execution** for fresh data collection
- **90-day historical backfill** for comprehensive analysis
- **Robust error handling** with retry logic and rate limiting
- **Multi-source integration** (NOAA weather + EIA energy APIs)

### 2. Data Quality Monitoring
- **Missing value detection** and reporting
- **Outlier identification** (temperature extremes, negative energy values)
- **Data freshness checks** to flag stale information
- **Quality scoring** (0-100 scale) for data reliability

### 3. Interactive Dashboard
- **Geographic overview** with current conditions and trends
- **Time series analysis** with dual-axis temperature/energy charts
- **Correlation analysis** with regression lines and R-squared values
- **Usage pattern heatmaps** by temperature range and day of week

### 4. Production-Ready Architecture
- **Modular design** with separate concerns
- **Configurable settings** via YAML files
- **Comprehensive logging** for debugging and monitoring
- **Automated testing** for code reliability

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- NOAA Climate Data API token ([Register here](https://www.ncdc.noaa.gov/cdo-web/token))
- EIA Energy API key ([Register here](https://www.eia.gov/opendata/register.php))

### Installation

1. **Clone and navigate to the project**:
   ```bash
   git clone <repository-url>
   cd data_science_project
   ```

2. **Install dependencies**:
   ```bash
   pip install -e .
   ```

3. **Configure API credentials**:
   ```bash
   cp config/config.yaml config/config.yaml.local
   # Edit config.yaml.local with your API tokens
   ```

4. **Run historical data backfill**:
   ```bash
   python src/pipeline.py --mode historical --days 90
   ```

5. **Launch the dashboard**:
   ```bash
   streamlit run dashboards/app.py
   ```

## 📋 Detailed Setup Instructions

### 1. API Configuration

Edit `config/config.yaml` with your API credentials:

```yaml
api:
  noaa:
    token: "YOUR_NOAA_TOKEN_HERE"
  eia:
    api_key: "YOUR_EIA_API_KEY_HERE"
```

### 2. Data Collection

**Historical Backfill (First Run)**:
```bash
python src/pipeline.py --mode historical --days 90
```

**Daily Updates**:
```bash
python src/pipeline.py --mode daily
```

**Quality Check**:
```bash
python src/pipeline.py --mode quality-check
```

### 3. Dashboard Usage

**Start the dashboard**:
```bash
streamlit run dashboards/app.py
```

**Access in browser**: http://localhost:8501

### 4. Automation Setup

**Daily cron job example**:
```bash
# Add to crontab: crontab -e
0 6 * * * cd /path/to/project && python src/pipeline.py --mode daily
```

## 🏗️ Project Structure

```
data_science_project/
├── README.md                 # This file
├── AI_USAGE.md              # AI assistance documentation
├── pyproject.toml           # Dependencies and build config
├── config/
│   └── config.yaml          # API keys and configuration
├── src/
│   ├── data_fetcher.py      # API clients for NOAA and EIA
│   ├── data_processor.py    # Data cleaning and transformation
│   ├── analysis.py          # Quality checks and statistics
│   └── pipeline.py          # Main orchestration script
├── dashboards/
│   └── app.py               # Streamlit dashboard
├── data/
│   ├── raw/                 # Original API responses
│   └── processed/           # Clean, analysis-ready data
├── logs/                    # Execution logs and quality reports
├── tests/                   # Unit tests
└── notebooks/               # Jupyter notebooks (optional)
```

## 📈 Dashboard Visualizations

### 1. Geographic Overview
- **Interactive US map** showing all 5 cities
- **Real-time status**: current temperature, today's energy usage
- **Performance indicators**: % change from yesterday
- **Color coding**: red for high usage, green for low

### 2. Time Series Analysis
- **Dual-axis charts**: temperature (left) vs energy (right)
- **City filtering**: view individual cities or combined data
- **Weekend highlighting**: shaded regions for weekend periods
- **90-day trend analysis**: identify seasonal patterns

### 3. Correlation Analysis
- **Scatter plot**: temperature vs energy consumption
- **City-specific coloring**: compare regional differences
- **Regression analysis**: trendlines with R-squared values
- **Statistical significance**: correlation coefficients

### 4. Usage Patterns Heatmap
- **Temperature ranges**: <50°F to >90°F in 10-degree bands
- **Day-of-week patterns**: Monday through Sunday
- **City filtering**: analyze specific regions
- **Weekend vs weekday**: usage pattern comparison

## 🔧 Configuration Options

### City Configuration
Modify `config/config.yaml` to add/remove cities:

```yaml
cities:
  - name: "Your City"
    state: "Your State"
    noaa_station_id: "GHCND:STATION_ID"
    eia_region_code: "REGION_CODE"
    lat: 40.7128
    lon: -74.0060
```

### Data Quality Thresholds
```yaml
data:
  quality_checks:
    temperature:
      min_fahrenheit: -50
      max_fahrenheit: 130
    energy:
      min_consumption: 0
    freshness_threshold: 24  # hours
```

## 🧪 Testing

Run the test suite:
```bash
pytest tests/
```

Run specific test modules:
```bash
pytest tests/test_data_fetcher.py -v
```

## 📊 Expected Results

Based on the analysis, you should expect to find:

- **Strong correlation** (r > 0.7) between temperature extremes and energy usage
- **Seasonal patterns** in energy consumption
- **Weekend vs weekday** differences in usage patterns
- **Regional variations** in weather-energy relationships

## 🔍 Data Quality Monitoring

The pipeline includes comprehensive quality checks:

- **Missing Value Detection**: Identifies gaps in temperature or energy data
- **Outlier Detection**: Flags temperatures <-50°F or >130°F, negative energy values
- **Data Freshness**: Alerts when data is more than 24 hours old
- **Quality Scoring**: 0-100 scale based on completeness, accuracy, and freshness

## 📋 Troubleshooting

### Common Issues

**API Rate Limits**:
- The pipeline includes automatic retry logic with exponential backoff
- Check logs for rate limit warnings

**Missing Data**:
- Some historical data may not be available for all cities
- The pipeline continues processing even if some API calls fail

**Dashboard Loading Issues**:
- Ensure you've run the pipeline to generate processed data
- Check that all dependencies are installed correctly

### Log Files

- **Pipeline execution**: `logs/pipeline.log`
- **Quality reports**: `logs/quality_report_*.json`
- **Error details**: Check console output for real-time issues

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Run the test suite
5. Submit a pull request

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **NOAA Climate Data Online**: Weather data source
- **EIA Energy Information Administration**: Energy consumption data
- **Streamlit**: Dashboard framework
- **Plotly**: Interactive visualizations

## 📞 Support

For issues and questions:
1. Check the troubleshooting section above
2. Review the log files for error details
3. Open an issue on the project repository

---

*This pipeline was designed to demonstrate production-ready data engineering practices while solving real business problems in the energy sector.*