"""
Streamlit Cloud deployment version of the Weather and Energy Analysis Dashboard.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import requests
import yaml
import logging
from datetime import datetime, timedelta
import os
from pathlib import Path

# Configure page
st.set_page_config(
    page_title="US Weather & Energy Analysis Dashboard",
    page_icon="⚡",
    layout="wide"
)

@st.cache_data(ttl=3600)  # Cache for 1 hour to allow fresh data generation
def load_fresh_production_data():
    """Load real processed data from the pipeline, generating fresh data if needed."""
    processed_data_path = Path("data/processed")
    
    # Check for existing data
    csv_files = []
    if processed_data_path.exists():
        csv_files = list(processed_data_path.glob("processed_data_*.csv"))
    
    # Determine if we need fresh data
    needs_fresh_data = True
    latest_file = None
    
    if csv_files:
        latest_file = max(csv_files, key=lambda x: x.stat().st_mtime)
        df = pd.read_csv(latest_file)
        df['date'] = pd.to_datetime(df['date'])
        
        # Check data freshness and quantity
        latest_date = df['date'].max()
        days_old = (datetime.now() - latest_date).days
        has_sufficient_data = len(df) >= 20  # Need at least 20 records
        
        if days_old <= 7 and has_sufficient_data:
            needs_fresh_data = False
            st.success(f"✅ Using fresh production data ({len(df)} records, {days_old} days old)")
            return df, "production"
        else:
            st.warning(f"🔄 Data needs refresh: {days_old} days old, {len(df)} records")
    
    # Generate fresh data if needed
    if needs_fresh_data:
        with st.spinner("🔄 Generating fresh data from APIs..."):
            try:
                # Import here to avoid circular imports
                import subprocess
                import sys
                
                # Run pipeline to generate fresh historical data
                st.info("📊 Running pipeline to fetch fresh data (this may take 1-2 minutes)...")
                
                # Run 30-day historical backfill to ensure good data coverage
                result = subprocess.run([
                    sys.executable, "src/pipeline.py", 
                    "--mode", "historical", 
                    "--days", "30",
                    "--config", "config/config.yaml"
                ], capture_output=True, text=True, cwd=".")
                
                if result.returncode == 0:
                    st.success("✅ Fresh data generated successfully!")
                    
                    # Load the newly generated data
                    csv_files = list(processed_data_path.glob("processed_data_*.csv"))
                    if csv_files:
                        latest_file = max(csv_files, key=lambda x: x.stat().st_mtime)
                        df = pd.read_csv(latest_file)
                        df['date'] = pd.to_datetime(df['date'])
                        
                        st.success(f"🎉 Loaded {len(df)} fresh records spanning {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
                        return df, "fresh_production"
                    else:
                        raise Exception("No data files generated")
                        
                else:
                    error_msg = result.stderr if result.stderr else result.stdout
                    raise Exception(f"Pipeline failed: {error_msg}")
                    
            except Exception as e:
                st.error(f"❌ Failed to generate fresh data: {e}")
                st.error("Please check API keys and configuration.")
                
                # If we have any existing data, use it as fallback
                if latest_file and latest_file.exists():
                    st.warning("🔄 Using existing data as fallback...")
                    df = pd.read_csv(latest_file)
                    df['date'] = pd.to_datetime(df['date'])
                    return df, "fallback"
                
                # Final fallback - stop the app
                st.error("❌ No data available. Please:")
                st.error("1. Check API keys in config/config.yaml")
                st.error("2. Run: `python src/pipeline.py --mode historical --days 30`")
                st.error("3. Refresh this page")
                st.stop()
    
    # This should never be reached
    st.stop()


def create_geographic_overview(data: pd.DataFrame):
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
        center={"lat": 39.8283, "lon": -98.5795},
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
    
    # Calculate overall correlation
    correlation_coef = data['temp_avg_f'].corr(data['energy_consumption_mwh'])
    
    # Add trendline
    z = np.polyfit(data['temp_avg_f'], data['energy_consumption_mwh'], 1)
    p = np.poly1d(z)
    
    x_range = np.linspace(data['temp_avg_f'].min(), data['temp_avg_f'].max(), 100)
    y_trend = p(x_range)
    
    fig_scatter.add_trace(
        go.Scatter(
            x=x_range,
            y=y_trend,
            mode='lines',
            name=f'Trendline (y = {z[0]:.2f}x + {z[1]:.2f})',
            line=dict(color='black', width=2, dash='dash')
        )
    )
    
    # Calculate R-squared
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    
    X = data[['temp_avg_f']]
    y = data['energy_consumption_mwh']
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    r_squared = r2_score(y, y_pred)
    
    # Add correlation info
    fig_scatter.add_annotation(
        x=0.02,
        y=0.98,
        xref="paper",
        yref="paper",
        text=f"R² = {r_squared:.4f}<br>r = {correlation_coef:.4f}",
        showarrow=False,
        font=dict(size=14, color="black"),
        bgcolor="white",
        bordercolor="black",
        borderwidth=1
    )
    
    fig_scatter.update_layout(height=500)
    st.plotly_chart(fig_scatter, use_container_width=True)

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

def main():
    """Main Streamlit application."""
    
    # Main title
    st.title("⚡ US Weather & Energy Analysis Dashboard")
    st.markdown("---")
    
    # Load fresh production data (no demo fallback)
    data, data_mode = load_fresh_production_data()
    
    # Sidebar
    st.sidebar.header("Dashboard Controls")
    
    # Manual refresh button
    if st.sidebar.button("🔄 Force Refresh Data", help="Generate fresh data from APIs"):
        st.cache_data.clear()  # Clear all cached data
        st.rerun()  # Rerun the app to fetch fresh data
    
    if data_mode == "production":
        st.sidebar.markdown("**🟢 Production Mode**: Using existing fresh data")
        st.sidebar.success("Live data from NOAA & EIA APIs")
    elif data_mode == "fresh_production":
        st.sidebar.markdown("**🆕 Fresh Production Mode**: Just generated new data")
        st.sidebar.success("Freshly fetched from NOAA & EIA APIs")
    elif data_mode == "fallback":
        st.sidebar.markdown("**🔄 Fallback Mode**: Using existing data")
        st.sidebar.warning("API generation failed, using last available data")
    else:
        st.sidebar.markdown("**🔧 Unknown Mode**: Data loaded successfully")
        st.sidebar.info("Data source determination unclear")
    
    # Date range selector
    min_date = data['date'].min().date()
    max_date = data['date'].max().date()
    
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
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
        create_geographic_overview(data)
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
        st.sidebar.header("Data Summary")
        st.sidebar.metric("Total Records", len(data))
        st.sidebar.metric("Cities Covered", data['city'].nunique())
        st.sidebar.metric("Date Range", f"{data['date'].min().strftime('%Y-%m-%d')} to {data['date'].max().strftime('%Y-%m-%d')}")
        
        # About section
        with st.expander("About This Dashboard"):
            if data_mode == "production":
                data_info = "**📊 Data Source**: Real data from NOAA Climate Data Online and EIA Energy APIs"
            else:
                data_info = "**🎭 Data Source**: Simulated data based on realistic weather-energy correlation patterns"
                
            st.markdown(f"""
            {data_info}
            
            This dashboard demonstrates a production-ready data pipeline for analyzing the relationship 
            between weather patterns and energy consumption across 5 major US cities:
            
            - **New York, NY** - Financial hub with dense urban energy usage
            - **Chicago, IL** - Midwest industrial center with seasonal extremes
            - **Houston, TX** - Energy capital with high cooling demands
            - **Phoenix, AZ** - Desert climate with extreme heat patterns
            - **Seattle, WA** - Pacific Northwest with mild, consistent weather
            
            **Key Features:**
            - Real-time data collection from NOAA weather and EIA energy APIs
            - Automated quality monitoring and outlier detection
            - Interactive visualizations with correlation analysis
            - Production deployment with daily automated updates
            
            **Business Value:**
            - Demand forecasting for utilities
            - Cost optimization through pattern recognition
            - Regional analysis for energy planning
            - Quality monitoring for reliable insights
            """)
        
    except Exception as e:
        st.error(f"Error creating visualizations: {e}")
        st.exception(e)

if __name__ == "__main__":
    main()