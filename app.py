import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import lightgbm as lgb

# --- Page Configuration ---
st.set_page_config(
    page_title="Energy Forecast Scenario Planner",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Energy Consumption Forecast & Scenario Planner")
st.markdown(
    "Select a city to explore its baseline LightGBM forecast or test 'What-If' scenarios "
    "by altering future predictors (e.g., temperature or wind changes)."
)

# --- Helper Functions for Data Loading ---
@st.cache_resource
def load_lgbm_model(filepath):
    """Loads a saved LightGBM model."""
    if not os.path.exists(filepath):
        return None
    return lgb.Booster(model_file=filepath)

@st.cache_data
def load_dataset(filepath):
    """Loads the full historical and forecast dataset."""
    if not os.path.exists(filepath):
        return None
    df = pd.read_json(filepath, orient='records')
    df['Date'] = pd.to_datetime(df['Date'])
    df['City'] = df['City'].astype('category')
    return df

# --- File Paths & Data Loading ---
data_path = "full_dataset.json"

if not os.path.exists(data_path):
    st.error("Missing data file. Please ensure `full_dataset.json` is in the main directory.")
    st.stop()

# Load Data First
df_all = load_dataset(data_path)

# --- Sidebar: City Selection ---
CITIES = ["Chicago", "New York", "Seattle", "Miami", "Dallas"]
selected_city = st.sidebar.selectbox("🏙️ Select City", CITIES)

# Dynamically load the correct model for the selected city
safe_city_name = selected_city.replace(" ", "_")
model_path = f"forecast_model_{safe_city_name}.txt"

if not os.path.exists(model_path):
    st.error(f"Missing model file: `{model_path}`. Please ensure it is in the main directory.")
    st.stop()

model = load_lgbm_model(model_path)

# Filter dataset for the selected city
city_df = df_all[df_all['City'] == selected_city].copy().reset_index(drop=True)

# Define features used by the model
# NOTE: 'City' MUST be in exclude_cols since the individual models were trained without it
exclude_cols = [
    'Date', 'STATE_NAME', 'COUNTY_NAME', 'MWh_Consumed', 
    'Holiday_Not_Holiday', 'Year', 'Month', 'DayOfWeek', 'Forecasted_MWh', 'City'
]
features = [col for col in city_df.columns if col not in exclude_cols]

# --- Sidebar: What-If Controls ---
st.sidebar.markdown("---")
st.sidebar.subheader("🎛️ 'What-If' Scenario Controls")
st.sidebar.caption("Adjust values for the future forecast horizon.")

temp_delta = st.sidebar.slider(
    "Temperature Shift (°C)",
    min_value=-15.0,
    max_value=15.0,
    value=0.0,
    step=0.5,
    help="Applies a temperature change to the forecast period."
)

humidity_delta = st.sidebar.slider(
    "Humidity Shift (%)",
    min_value=-30.0,
    max_value=30.0,
    value=0.0,
    step=1.0
)

wind_delta = st.sidebar.slider(
    "Wind Speed Shift (km/h)",
    min_value=-50.0,
    max_value=50.0,
    value=0.0,
    step=1.0,
    help="Applies a wind speed change to the forecast period."
)

solar_delta = st.sidebar.slider(
    "Solar Irradiance Shift (MJ/m²)",
    min_value=-20.0,
    max_value=20.0,
    value=0.0,
    step=1.0,
    help="Applies a solar irradiance change to the forecast period."
)

# --- Scenario Simulation Engine ---
scenario_df = city_df.copy()

# Identify future forecast rows (where actual historical data is missing)
future_mask = scenario_df['MWh_Consumed'].isna()

# Baseline Predictions (No adjustments)
city_df.loc[future_mask, 'yhat_baseline'] = model.predict(city_df.loc[future_mask, features])

# Apply 'What-If' Modifications
if temp_delta != 0.0:
    for col in ['Temp_Mean_C', 'Temp_Max_C', 'Temp_Min_C']:
        if col in scenario_df.columns:
            scenario_df.loc[future_mask, col] += temp_delta

if humidity_delta != 0.0 and 'Humidity_Pct' in scenario_df.columns:
    scenario_df.loc[future_mask, 'Humidity_Pct'] = (
        scenario_df.loc[future_mask, 'Humidity_Pct'] + humidity_delta
    ).clip(lower=0, upper=100)
    
if wind_delta != 0.0 and 'Wind_Speed_kmh' in scenario_df.columns:
    scenario_df.loc[future_mask, 'Wind_Speed_kmh'] = (
        scenario_df.loc[future_mask, 'Wind_Speed_kmh'] + wind_delta
    ).clip(lower=0)

if solar_delta != 0.0 and 'Solar_Irradiance_MJm2' in scenario_df.columns:
    scenario_df.loc[future_mask, 'Solar_Irradiance_MJm2'] = (
        scenario_df.loc[future_mask, 'Solar_Irradiance_MJm2'] + solar_delta
    ).clip(lower=0)

# Scenario Predictions (With adjustments)
scenario_df.loc[future_mask, 'yhat_scenario'] = model.predict(scenario_df.loc[future_mask, features])

# Merge Baseline and Scenario predictions for plotting
plot_df = city_df[['Date', 'MWh_Consumed', 'yhat_baseline']].copy()
plot_df['yhat_scenario'] = scenario_df['yhat_scenario']

# --- KPI Metric Summary ---
st.subheader(f"📈 Scenario Impact Summary: {selected_city}")

base_future_avg = plot_df.loc[future_mask, 'yhat_baseline'].mean()
scen_future_avg = plot_df.loc[future_mask, 'yhat_scenario'].mean()
diff_mwh = scen_future_avg - base_future_avg
pct_diff = (diff_mwh / base_future_avg * 100) if base_future_avg != 0 else 0.0

col1, col2, col3 = st.columns(3)
col1.metric("Baseline Avg Demand", f"{(base_future_avg / 1000):,.1f} K")
col2.metric("Scenario Avg Demand", f"{(scen_future_avg / 1000):,.1f} K", delta=f"{(diff_mwh / 1000):+,.1f} K")
col3.metric("Projected Impact", f"{pct_diff:+.2f}%")

# --- Interactive Plotly Chart ---
st.subheader("📊 Model Comparison Chart")

timeframe = st.radio(
    "Display Range:",
    ["Last 60 Days + Future Forecast", "Full Dataset"],
    horizontal=True
)

if timeframe == "Last 60 Days + Future Forecast":
    cutoff_date = plot_df['Date'].max() - pd.Timedelta(days=60)
    plot_df = plot_df[plot_df['Date'] >= cutoff_date]

fig = go.Figure()

# Historical Actual Demand
fig.add_trace(go.Scatter(
    x=plot_df['Date'],
    y=plot_df['MWh_Consumed'],
    mode='lines',
    name='Actual Consumption',
    line=dict(color='#888888', width=1.5)
))

# Baseline Model Prediction
fig.add_trace(go.Scatter(
    x=plot_df['Date'],
    y=plot_df['yhat_baseline'],
    mode='lines',
    name='Baseline Forecast',
    line=dict(color='#1f77b4', width=2.5)
))

# What-If Scenario Prediction
fig.add_trace(go.Scatter(
    x=plot_df['Date'],
    y=plot_df['yhat_scenario'],
    mode='lines',
    name='Scenario Forecast',
    line=dict(color='#ff7f0e', width=2.5, dash='dash')
))

fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Energy Demand (MWh)",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    template="plotly_white",
    height=500
)

st.plotly_chart(fig, use_container_width=True)