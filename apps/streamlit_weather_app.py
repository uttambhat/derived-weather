import os

import streamlit as st
import xarray as xr
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Weather score animator", layout="wide")


@st.cache_data
def load_dataset(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at: {path}")
    ds = xr.open_dataset(path)
    # Ensure Celsius variables exist
    if "t2m_C" not in ds and "t2m" in ds:
        ds["t2m_C"] = ds["t2m"] - 273.15
        ds["t2m_C"].attrs["units"] = "C"
    if "d2m_C" not in ds and "d2m" in ds:
        ds["d2m_C"] = ds["d2m"] - 273.15
        ds["d2m_C"].attrs["units"] = "C"
    return ds


@st.cache_data
def compute_score(_ds, t2m_ideal, d2m_ideal, t2m_width, d2m_width, downsample):
    # leading underscore prevents Streamlit from hashing the Dataset
    ds = _ds
    # Compute Gaussian-like score per variable, then multiply
    t = ds["t2m_C"]
    d = ds["d2m_C"]
    if downsample > 1:
        t = t.isel(latitude=slice(None, None, downsample), longitude=slice(None, None, downsample))
        d = d.isel(latitude=slice(None, None, downsample), longitude=slice(None, None, downsample))
    t_score = np.exp(-0.5 * ((t - t2m_ideal) / t2m_width) ** 2)
    d_score = np.exp(-0.5 * ((d - d2m_ideal) / d2m_width) ** 2)
    score = t_score * d_score
    # Convert to long form DataFrame for plotly
    df = score.to_dataframe(name="score").reset_index()
    # Make animation_frame friendly
    if "valid_time" in df.columns:
        df["valid_time_str"] = df["valid_time"].astype(str)
    else:
        df["valid_time_str"] = df.index.get_level_values("time").astype(str)
    # drop NaNs (if any)
    return df.dropna(subset=["score"])


st.title("Interactive Weather Score Animator")

# Config / dataset selection
default_path = "/home/uttam/Dropbox/Projects/derived_weather/derived-weather/resources/weather_data_lat_30_50_lon_m120_m70_2025-11-01_to_2025-11-20_extracted.nc"
data_path = st.text_input("Path to extracted NetCDF", value=default_path)
try:
    ds = load_dataset(data_path)
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

# Sliders
col1, col2 = st.columns(2)
with col1:
    t2m_ideal = st.slider("Ideal 2m temperature (°C)", -40.0, 50.0, 10.0, 0.5)
    t2m_width = st.slider("Temperature width", 0.1, 50.0, 10.0, 0.1)
with col2:
    d2m_ideal = st.slider("Ideal dewpoint (°C)", -40.0, 40.0, 0.0, 0.5)
    d2m_width = st.slider("Dewpoint width", 0.1, 50.0, 10.0, 0.1)

downsample = st.slider("Downsample factor (for responsiveness)", 1, 8, 2)

if st.button("Compute & show animation"):
    with st.spinner("Computing score and preparing animation..."):
        df = compute_score(ds, t2m_ideal, d2m_ideal, t2m_width, d2m_width, downsample)

        # Use density_mapbox for gridded color map animation
        fig = px.density_mapbox(
            df,
            lat="latitude",
            lon="longitude",
            z="score",
            animation_frame="valid_time_str",
            range_color=[0, 1],
            radius=6,
            center={"lat": float(df["latitude"].mean()), "lon": float(df["longitude"].mean())},
            zoom=3,
            mapbox_style="carto-positron",
            height=700,
        )
        fig.update_layout(coloraxis_colorbar=dict(title="Score", tickformat=".2f"))
        st.plotly_chart(fig, use_container_width=True)

st.markdown("Tip: increase Downsample factor if the animation is slow or memory-heavy.")