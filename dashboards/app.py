"""
Streamlit dashboard for Weather and Energy Analysis.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import sys
from pathlib import Path
import yaml
from datetime import timedelta

# Add the project root and src directory to the path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# Import from src package
from src.data_processor import DataProcessor
from src.analysis import DataAnalyzer


def load_config(config_path: str = "config/config.yaml"):
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        st.error(f"Configuration file not found: {config_path}")
        return None


def load_data():
    """Load processed data for the dashboard."""
    try:
        processor = DataProcessor()
        data = processor.get_latest_data()
        
        if data is None or data.empty:
            st.error("No processed data found. Please run the data pipeline first.")
            return None
        
        # Convert date column to datetime
        data['date'] = pd.to_datetime(data['date'])
        
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None


def create_geographic_overview(data: pd.DataFrame, config: dict):
    """Create Visualization 1 - Geographic Overview."""
    st.header("🗺️ Geographic Overview")
    
    # Get latest data for each city
    latest_data = data.groupby('city').apply(lambda x: x.loc[x['date'].idxmax()]).reset_index(drop=True)
    
    # Calculate percentage change from yesterday
    yesterday_data = data[data['date'] == data['date'].max() - timedelta(days=1)]
    
    for idx, row in latest_data.iterrows():
        city = row['city']
        yesterday_row = yesterday_data[yesterday_data['city'] == city]
        
        if not yesterday_row.empty:
            yesterday_energy = yesterday_row['energy_consumption_mwh'].iloc[0]
            today_energy = row['energy_consumption_mwh']
            pct_change = ((today_energy - yesterday_energy) / yesterday_energy) * 100
            latest_data.loc[idx, 'energy_change_pct'] = pct_change
        else:
            latest_data.loc[idx, 'energy_change_pct'] = 0
    
    # Create map
    fig_map = px.scatter_mapbox(
        latest_data,
        lat='lat',
        lon='lon',
        size='energy_consumption_mwh',
        color='energy_change_pct',
        color_continuous_scale='RdYlGn_r',
        hover_name='city',
        hover_data={
            'temp_avg_f': ':.1f',
            'energy_consumption_mwh': ':.0f',
            'energy_change_pct': ':.1f'
        },
        mapbox_style="open-street-map",
        zoom=3,
        center={"lat": config['dashboard']['map']['center_lat'], 
               "lon": config['dashboard']['map']['center_lon']},
        title="Current Temperature and Energy Usage by City"
    )
    
    fig_map.update_layout(
        height=500,
        coloraxis_colorbar=dict(title="% Change from Yesterday")
    )
    
    st.plotly_chart(fig_map, use_container_width=True)
    
    # Summary table
    st.subheader("Current Status by City")
    summary_data = latest_data[['city', 'temp_avg_f', 'energy_consumption_mwh', 'energy_change_pct']].copy()
    summary_data.columns = ['City', 'Temperature (°F)', 'Energy Usage (MWh)', 'Change from Yesterday (%)']
    st.dataframe(summary_data, use_container_width=True)
    
    # Last updated timestamp
    st.caption(f"Last updated: {data['date'].max().strftime('%Y-%m-%d %H:%M:%S')}")


def create_time_series_analysis(data: pd.DataFrame):
    """Create Visualization 2 - Time Series Analysis."""
    st.header("📈 Time Series Analysis")
    
    # City selector
    cities = ['All Cities'] + sorted(data['city'].unique().tolist())
    selected_city = st.selectbox("Select City", cities)
    
    # Filter data
    if selected_city == 'All Cities':
        plot_data = data.groupby('date').agg({
            'temp_avg_f': 'mean',
            'energy_consumption_mwh': 'sum'
        }).reset_index()
        plot_data['city'] = 'All Cities'
    else:
        plot_data = data[data['city'] == selected_city].copy()
    
    # Create dual-axis plot
    fig_time = make_subplots(
        specs=[[{"secondary_y": True}]],
        subplot_titles=[f"Temperature and Energy Consumption - {selected_city}"]
    )
    
    # Add temperature trace
    fig_time.add_trace(
        go.Scatter(
            x=plot_data['date'],
            y=plot_data['temp_avg_f'],
            name="Temperature (°F)",
            line=dict(color='red', width=2),
            yaxis='y'
        ),
        secondary_y=False
    )
    
    # Add energy trace
    fig_time.add_trace(
        go.Scatter(
            x=plot_data['date'],
            y=plot_data['energy_consumption_mwh'],
            name="Energy (MWh)",
            line=dict(color='blue', width=2, dash='dot'),
            yaxis='y2'
        ),
        secondary_y=True
    )
    
    # Add weekend shading
    if not plot_data.empty:
        plot_data['is_weekend'] = plot_data['date'].dt.weekday >= 5
        weekend_dates = plot_data[plot_data['is_weekend']]['date']
        
        for weekend_date in weekend_dates:
            fig_time.add_vrect(
                x0=weekend_date,
                x1=weekend_date + timedelta(days=1),
                fillcolor="lightgray",
                opacity=0.3,
                line_width=0
            )
    
    # Update layout
    fig_time.update_xaxes(title_text="Date")
    fig_time.update_yaxes(title_text="Temperature (°F)", secondary_y=False, title_font=dict(color="red"))
    fig_time.update_yaxes(title_text="Energy Consumption (MWh)", secondary_y=True, title_font=dict(color="blue"))
    fig_time.update_layout(
        height=500,
        hovermode='x unified',
        legend=dict(x=0.02, y=0.98)
    )
    
    st.plotly_chart(fig_time, use_container_width=True)


def create_correlation_analysis(data: pd.DataFrame):
    """Create Visualization 3 - Correlation Analysis."""
    st.header("🔗 Correlation Analysis")
    
    # Calculate correlation
    analyzer = DataAnalyzer()
    correlations = analyzer.calculate_correlations(data)
    
    # Create scatter plot
    fig_scatter = px.scatter(
        data,
        x='temp_avg_f',
        y='energy_consumption_mwh',
        color='city',
        title="Temperature vs Energy Consumption",
        labels={
            'temp_avg_f': 'Temperature (°F)',
            'energy_consumption_mwh': 'Energy Consumption (MWh)'
        },
        hover_data=['date', 'city']
    )
    
    # Add trendline
    if 'overall' in correlations:
        overall_corr = correlations['overall']
        slope = overall_corr['slope']
        intercept = overall_corr['intercept']
        
        # Create trendline
        x_range = np.linspace(data['temp_avg_f'].min(), data['temp_avg_f'].max(), 100)
        y_trend = slope * x_range + intercept
        
        fig_scatter.add_trace(
            go.Scatter(
                x=x_range,
                y=y_trend,
                mode='lines',
                name=f'Trendline (y = {slope:.2f}x + {intercept:.2f})',
                line=dict(color='black', width=2, dash='dash')
            )
        )
        
        # Add correlation info
        r_value = overall_corr['correlation_coefficient']
        r_squared = overall_corr['r_squared']
        
        fig_scatter.add_annotation(
            x=0.02,
            y=0.98,
            xref="paper",
            yref="paper",
            text=f"R² = {r_squared:.4f}<br>r = {r_value:.4f}",
            showarrow=False,
            font=dict(size=14, color="black"),
            bgcolor="white",
            bordercolor="black",
            borderwidth=1
        )
    
    fig_scatter.update_layout(height=500)
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Show correlation by city
    if 'by_city' in correlations:
        st.subheader("Correlation by City")
        city_corr_data = []
        
        for city, stats in correlations['by_city'].items():
            city_corr_data.append({
                'City': city,
                'Correlation (r)': stats['correlation_coefficient'],
                'R-squared': stats['r_squared'],
                'Sample Size': stats['sample_size']
            })
        
        city_corr_df = pd.DataFrame(city_corr_data)
        st.dataframe(city_corr_df, use_container_width=True)


def create_usage_patterns_heatmap(data: pd.DataFrame):
    """Create Visualization 4 - Usage Patterns Heatmap."""
    st.header("🔥 Usage Patterns Heatmap")
    
    # City filter
    cities = ['All Cities'] + sorted(data['city'].unique().tolist())
    selected_city = st.selectbox("Filter by City", cities, key="heatmap_city")
    
    # Filter data
    if selected_city != 'All Cities':
        heatmap_data = data[data['city'] == selected_city].copy()
    else:
        heatmap_data = data.copy()
    
    # Create heatmap data
    heatmap_pivot = heatmap_data.pivot_table(
        index='temp_range',
        columns='day_of_week',
        values='energy_consumption_mwh',
        aggfunc='mean'
    )
    
    # Reorder columns to start with Monday
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    heatmap_pivot = heatmap_pivot.reindex(columns=day_order, fill_value=0)
    
    # Reorder rows by temperature
    temp_order = ['<50°F', '50-60°F', '60-70°F', '70-80°F', '80-90°F', '>90°F']
    heatmap_pivot = heatmap_pivot.reindex(temp_order, fill_value=0)
    
    # Create heatmap
    fig_heatmap = px.imshow(
        heatmap_pivot,
        labels=dict(x="Day of Week", y="Temperature Range", color="Energy Usage (MWh)"),
        title=f"Average Energy Usage by Temperature Range and Day of Week - {selected_city}",
        color_continuous_scale='Blues',
        text_auto='.0f'
    )
    
    fig_heatmap.update_layout(
        height=500,
        xaxis_title="Day of Week",
        yaxis_title="Temperature Range"
    )
    
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Show usage patterns analysis
    analyzer = DataAnalyzer()
    patterns = analyzer.analyze_usage_patterns(heatmap_data)
    
    if 'weekend_vs_weekday' in patterns:
        st.subheader("Weekend vs Weekday Usage")
        weekend_stats = patterns['weekend_vs_weekday']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                "Weekday Average (MWh)",
                f"{weekend_stats['mean'][False]:.0f}" if False in weekend_stats['mean'] else "N/A"
            )
        
        with col2:
            st.metric(
                "Weekend Average (MWh)",
                f"{weekend_stats['mean'][True]:.0f}" if True in weekend_stats['mean'] else "N/A"
            )


def create_sidebar():
    """Create sidebar with filters and controls."""
    st.sidebar.header("Dashboard Controls")
    
    # Date range selector
    data = load_data()
    if data is not None:
        min_date = data['date'].min().date()
        max_date = data['date'].max().date()
        
        date_range = st.sidebar.date_input(
            "Select Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        
        return date_range
    
    return None


def main():
    """Main Streamlit application."""
    # Load configuration
    config = load_config()
    if config is None:
        st.stop()
    
    # Set page configuration
    st.set_page_config(
        page_title=config['dashboard']['title'],
        page_icon=config['dashboard']['page_icon'],
        layout=config['dashboard']['layout']
    )
    
    # Main title
    st.title(config['dashboard']['title'])
    st.markdown("---")
    
    # Load data
    data = load_data()
    if data is None:
        st.stop()
    
    # Create sidebar
    date_range = create_sidebar()
    
    # Filter data by date range
    if date_range and len(date_range) == 2:
        start_date, end_date = date_range
        data = data[(data['date'].dt.date >= start_date) & (data['date'].dt.date <= end_date)]
    
    # Check if data is empty after filtering
    if data.empty:
        st.warning("No data available for the selected date range.")
        st.stop()
    
    # Create visualizations
    try:
        # Visualization 1: Geographic Overview
        create_geographic_overview(data, config)
        st.markdown("---")
        
        # Visualization 2: Time Series Analysis
        create_time_series_analysis(data)
        st.markdown("---")
        
        # Visualization 3: Correlation Analysis
        create_correlation_analysis(data)
        st.markdown("---")
        
        # Visualization 4: Usage Patterns Heatmap
        create_usage_patterns_heatmap(data)
        
        # Data quality summary
        st.sidebar.markdown("---")
        st.sidebar.header("Data Quality")
        
        analyzer = DataAnalyzer()
        quality_score = analyzer.get_data_quality_score(data)
        
        st.sidebar.metric("Quality Score", f"{quality_score:.1f}/100")
        st.sidebar.metric("Total Records", len(data))
        st.sidebar.metric("Cities Covered", data['city'].nunique())
        
        # Show data quality indicator
        if quality_score >= 90:
            st.sidebar.success("Excellent data quality")
        elif quality_score >= 75:
            st.sidebar.info("Good data quality")
        elif quality_score >= 60:
            st.sidebar.warning("Fair data quality")
        else:
            st.sidebar.error("Poor data quality")
        
    except Exception as e:
        st.error(f"Error creating visualizations: {e}")
        st.exception(e)


if __name__ == "__main__":
    main()