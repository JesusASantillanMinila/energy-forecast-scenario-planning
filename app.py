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

st.markdown("### ⚡ Energy Consumption Forecast & Scenario Planner")
st.caption(
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
exclude_cols = [
    'Date', 'STATE_NAME', 'COUNTY_NAME', 'MWh_Consumed', 
    'Holiday_Not_Holiday', 'Year', 'Month', 'DayOfWeek', 'Forecasted_MWh', 'City'
]
features = [col for col in city_df.columns if col not in exclude_cols]

# --- Sidebar: What-If Controls ---
st.sidebar.markdown("---")
st.sidebar.subheader("🎛️ 'What-If' Scenario Controls")
st.sidebar.caption("Adjust values for the future forecast horizon.")

# Callback function to reset session state variables
def reset_scenarios():
    st.session_state.temp_delta = 0.0
    st.session_state.humidity_delta = 0.0
    st.session_state.wind_delta = 0.0
    st.session_state.solar_delta = 0.0

# Add the reset button to the sidebar
st.sidebar.button("Reset Scenarios to Zero", on_click=reset_scenarios)

temp_delta = st.sidebar.slider(
    "Temperature Shift (°C)",
    min_value=-15.0,
    max_value=15.0,
    value=0.0,
    step=0.5,
    key="temp_delta",
    help="Applies a temperature change to the forecast period."
)

humidity_delta = st.sidebar.slider(
    "Humidity Shift (%)",
    min_value=-30.0,
    max_value=30.0,
    value=0.0,
    step=1.0,
    key="humidity_delta"
)

wind_delta = st.sidebar.slider(
    "Wind Speed Shift (km/h)",
    min_value=-50.0,
    max_value=50.0,
    value=0.0,
    step=1.0,
    key="wind_delta",
    help="Applies a wind speed change to the forecast period."
)

solar_delta = st.sidebar.slider(
    "Solar Irradiance Shift (MJ/m²)",
    min_value=-20.0,
    max_value=20.0,
    value=0.0,
    step=1.0,
    key="solar_delta",
    help="Applies a solar irradiance change to the forecast period."
)

# --- Scenario Simulation Engine ---
scenario_df = city_df.copy()

# Identify future forecast rows
future_mask = scenario_df['MWh_Consumed'].isna()

# Baseline Predictions
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

# Scenario Predictions
scenario_df.loc[future_mask, 'yhat_scenario'] = model.predict(scenario_df.loc[future_mask, features])

# Merge Baseline and Scenario predictions (including weather features for plotting)
plot_df = city_df[['Date', 'MWh_Consumed', 'yhat_baseline']].copy()
plot_df['yhat_scenario'] = scenario_df['yhat_scenario']

for col in ['Temp_Mean_C', 'Humidity_Pct', 'Wind_Speed_kmh', 'Solar_Irradiance_MJm2']:
    if col in scenario_df.columns:
        plot_df[col] = scenario_df[col]

# --- KPI Metric Summary & Controls ---
st.markdown(f"**📈 Impact Summary ({selected_city})**")

base_future_avg = plot_df.loc[future_mask, 'yhat_baseline'].mean()
scen_future_avg = plot_df.loc[future_mask, 'yhat_scenario'].mean()
diff_mwh = scen_future_avg - base_future_avg
pct_diff = (diff_mwh / base_future_avg * 100) if base_future_avg != 0 else 0.0

col1, col2, col3, col4 = st.columns([1, 1, 1, 1.8])

col1.metric("Baseline Avg Demand", f"{(base_future_avg / 1000):,.1f}K MWh")
col2.metric("Scenario Avg Demand", f"{(scen_future_avg / 1000):,.1f}K MWh", delta=f"{(diff_mwh / 1000):+,.1f} K MWh")
col3.metric("Projected Impact", f"{pct_diff:+.2f}%")

# --- Interactive Plotly Chart ---
st.markdown(" #### 📊 Model Comparison Chart")

chart_col, controls_col = st.columns([5, 1])

# 1. Render the new multiselect filter
with controls_col:
    st.write("\n\n")
    weather_features = st.multiselect(
        "Additional Features:",
        options=["Temperature", "Humidity", "Wind Speed", "Solar Irradiance"],
        default=[],
        help="Overlay weather features on a secondary axis."
    )

# 2. Build the Plotly figure
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=plot_df['Date'], y=plot_df['MWh_Consumed'],
    mode='lines', name='Actual Consumption',
    line=dict(color='#888888', width=1.5)
))
fig.add_trace(go.Scatter(
    x=plot_df['Date'], y=plot_df['yhat_baseline'],
    mode='lines', name='Baseline Forecast',
    line=dict(color='#1f77b4', width=2.5)
))
fig.add_trace(go.Scatter(
    x=plot_df['Date'], y=plot_df['yhat_scenario'],
    mode='lines', name='Scenario Forecast',
    line=dict(color='#ff7f0e', width=2.5, dash='dash')
))

# 3. Add Optional Weather Features on a Secondary Y-Axis
if "Temperature" in weather_features and 'Temp_Mean_C' in plot_df:
    fig.add_trace(go.Scatter(x=plot_df['Date'], y=plot_df['Temp_Mean_C'], mode='lines', name='Temp (°C)', yaxis='y2', line=dict(dash='dot', width=1.5)))
if "Humidity" in weather_features and 'Humidity_Pct' in plot_df:
    fig.add_trace(go.Scatter(x=plot_df['Date'], y=plot_df['Humidity_Pct'], mode='lines', name='Humidity (%)', yaxis='y2', line=dict(dash='dot', width=1.5)))
if "Wind Speed" in weather_features and 'Wind_Speed_kmh' in plot_df:
    fig.add_trace(go.Scatter(x=plot_df['Date'], y=plot_df['Wind_Speed_kmh'], mode='lines', name='Wind (km/h)', yaxis='y2', line=dict(dash='dot', width=1.5)))
if "Solar Irradiance" in weather_features and 'Solar_Irradiance_MJm2' in plot_df:
    fig.add_trace(go.Scatter(x=plot_df['Date'], y=plot_df['Solar_Irradiance_MJm2'], mode='lines', name='Solar (MJ/m²)', yaxis='y2', line=dict(dash='dot', width=1.5)))

layout_updates = dict(
    xaxis_title="Date",
    yaxis_title="Energy Demand (MWh)",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    template="plotly_white",
    height=380,
    margin=dict(l=10, r=10, t=30, b=10)
)

if weather_features:
    layout_updates['yaxis2'] = dict(
        title="Weather Variables",
        overlaying="y",
        side="right",
        showgrid=False
    )

fig.update_layout(**layout_updates)

# 4. Render the chart
with chart_col:
    st.plotly_chart(fig, use_container_width=True)