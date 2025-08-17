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

@st.cache_data
def load_production_data():
    """Load real processed data from the pipeline."""
    try:
        # Try to load the latest processed data file
        processed_data_path = Path("data/processed")
        
        if processed_data_path.exists():
            # Find the most recent processed data file
            csv_files = list(processed_data_path.glob("processed_data_*.csv"))
            
            if csv_files:
                latest_file = max(csv_files, key=lambda x: x.stat().st_mtime)
                df = pd.read_csv(latest_file)
                df['date'] = pd.to_datetime(df['date'])
                
                # Check if data is recent enough (within last 30 days)
                latest_date = df['date'].max()
                days_old = (datetime.now() - latest_date).days
                
                if days_old > 30:
                    st.warning(f"⚠️ Production data is {days_old} days old. Using demo data with recent dates.")
                    return load_demo_data(), "demo"
                
                # Ensure we have a reasonable amount of data
                if len(df) < 5:
                    st.warning("⚠️ Limited production data available. Supplementing with demo data.")
                    demo_df, _ = load_demo_data()
                    # Use production data dates but fill with more demo data if needed
                    combined_df = pd.concat([df, demo_df.tail(20)], ignore_index=True)
                    combined_df = combined_df.drop_duplicates(subset=['city', 'date'], keep='first')
                    return combined_df.sort_values(['date', 'city']), "mixed"
                
                return df, "production"
        
        # If no processed data, fall back to demo data
        st.info("📊 No production data found. Using demo data with current dates.")
        return load_demo_data(), "demo"
        
    except Exception as e:
        st.error(f"Error loading production data: {e}")
        return load_demo_data(), "demo"

@st.cache_data
def load_demo_data():
    """Load sample data for demonstration when production data unavailable."""
    np.random.seed(42)
    
    cities = [
        {"name": "New York", "state": "New York", "lat": 40.7128, "lon": -74.0060},
        {"name": "Chicago", "state": "Illinois", "lat": 41.8781, "lon": -87.6298},
        {"name": "Houston", "state": "Texas", "lat": 29.7604, "lon": -95.3698},
        {"name": "Phoenix", "state": "Arizona", "lat": 33.4484, "lon": -112.0740},
        {"name": "Seattle", "state": "Washington", "lat": 47.6062, "lon": -122.3321}
    ]
    
    # Generate 30 days of sample data starting from recent date
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=29)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    data = []
    
    for city in cities:
        base_temp = {"New York": 70, "Chicago": 65, "Houston": 85, "Phoenix": 95, "Seattle": 60}[city["name"]]
        base_energy = {"New York": 25000, "Chicago": 22000, "Houston": 28000, "Phoenix": 24000, "Seattle": 18000}[city["name"]]
        
        for i, date in enumerate(dates):
            # Simulate seasonal temperature variation
            temp_variation = 10 * np.sin(i * 0.2) + np.random.normal(0, 5)
            temp_avg = base_temp + temp_variation
            
            # Energy consumption inversely correlated with moderate temperatures
            temp_deviation = abs(temp_avg - 72)  # 72°F is comfortable temperature
            energy_base = base_energy + (temp_deviation * 50) + np.random.normal(0, 1000)
            
            # Weekend effect (lower energy consumption)
            if date.weekday() >= 5:
                energy_base *= 0.85
            
            # Temperature categorization
            if temp_avg < 50:
                temp_range = '<50°F'
            elif temp_avg < 60:
                temp_range = '50-60°F'
            elif temp_avg < 70:
                temp_range = '60-70°F'
            elif temp_avg < 80:
                temp_range = '70-80°F'
            elif temp_avg < 90:
                temp_range = '80-90°F'
            else:
                temp_range = '>90°F'
            
            data.append({
                'date': date,
                'city': city["name"],
                'state': city["state"],
                'lat': city["lat"],
                'lon': city["lon"],
                'temp_avg_f': round(temp_avg, 1),
                'energy_consumption_mwh': round(energy_base, 0),
                'day_of_week': date.strftime('%A'),
                'is_weekend': date.weekday() >= 5,
                'temp_range': temp_range
            })
    
    return pd.DataFrame(data)

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
    
    # Load data
    data, data_mode = load_production_data()
    
    # Sidebar
    st.sidebar.header("Dashboard Controls")
    
    if data_mode == "production":
        st.sidebar.markdown("**🚀 Production Mode**: Using real pipeline data")
        st.sidebar.success("Live data from NOAA & EIA APIs")
    elif data_mode == "mixed":
        st.sidebar.markdown("**🟡 Mixed Mode**: Production + Demo data")
        st.sidebar.warning("Limited production data supplemented with demo data")
    else:
        st.sidebar.markdown("**🎭 Demo Mode**: Using simulated data for demonstration")
        st.sidebar.info("Deploy with processed data for production mode")
    
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