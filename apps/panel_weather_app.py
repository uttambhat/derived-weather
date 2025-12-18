import os

import panel as pn
import xarray as xr
import numpy as np
import hvplot.xarray  # registers hvplot for xarray

pn.extension()


DEFAULT_PATH = "/home/uttam/Dropbox/Projects/derived_weather/derived-weather/resources/weather_data_lat_30_50_lon_m120_m70_2025-11-01_to_2025-11-20_extracted.nc"



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


def compute_score(ds, t2m_ideal, d2m_ideal, t2m_width, d2m_width, downsample=1):
    """Return an xarray.DataArray with dims (valid_time, latitude, longitude) and values in [0,1]."""
    t = ds["t2m_C"]
    d = ds["d2m_C"]
    if downsample and downsample > 1:
        t = t.isel(latitude=slice(None, None, downsample), longitude=slice(None, None, downsample))
        d = d.isel(latitude=slice(None, None, downsample), longitude=slice(None, None, downsample))

    t_score = np.exp(-0.5 * ((t - t2m_ideal) / t2m_width) ** 2)
    d_score = np.exp(-0.5 * ((d - d2m_ideal) / d2m_width) ** 2)

    score = (t_score * d_score).rename("score")
    # Clip to [0,1] for safety
    score = score.clip(0, 1)
    return score


# Widgets
path_input = pn.widgets.TextInput(name="Path to extracted NetCDF", value=DEFAULT_PATH, width=450)
load_button = pn.widgets.Button(name="Load dataset", button_type="primary")

# Parameters
t2m_ideal_w = pn.widgets.FloatSlider(name="Ideal 2m temperature (°C)", start=-40, end=50, step=0.5, value=10)
t2m_width_w = pn.widgets.FloatSlider(name="Temperature width", start=0.1, end=50.0, step=0.1, value=10.0)

d2m_ideal_w = pn.widgets.FloatSlider(name="Ideal dewpoint (°C)", start=-40, end=40, step=0.5, value=0)
d2m_width_w = pn.widgets.FloatSlider(name="Dewpoint width", start=0.1, end=50.0, step=0.1, value=10.0)

downsample_w = pn.widgets.IntSlider(name="Downsample factor", start=1, end=8, step=1, value=2)
compute_button = pn.widgets.Button(name="Compute score", button_type="primary")

# Play/Frame widgets (created after dataset load)
# Use Player widget (some Panel versions do not have 'Play')
play = pn.widgets.Player(name="Play", value=0, start=0, end=0, step=1, interval=500)
frame_slider = pn.widgets.IntSlider(name="Frame", start=0, end=0, value=0)
play.link(frame_slider, value="value")

# Placeholders
_ds = None
_score = None
_plot_pane = pn.pane.HoloViews(object=None, sizing_mode="stretch_both", height=600)


def load_action(event=None):
    global _ds
    try:
        _ds = load_dataset(path_input.value)
        status.value = f"Loaded dataset with dimensions: {dict(_ds.dims)}"
    except Exception as e:
        status.value = f"Error loading dataset: {e}"
        _ds = None


def compute_action(event=None):
    global _ds, _score
    if _ds is None:
        status.value = "No dataset loaded"
        return
    try:
        _score = compute_score(
            _ds,
            t2m_ideal_w.value,
            d2m_ideal_w.value,
            t2m_width_w.value,
            d2m_width_w.value,
            downsample_w.value,
        )
        n_frames = _score.sizes.get("valid_time", 0)
        if n_frames == 0:
            status.value = "No valid_time dimension found in the score array"
            return
        play.start = 0
        play.end = n_frames - 1
        frame_slider.start = 0
        frame_slider.end = n_frames - 1
        frame_slider.value = 0
        status.value = f"Computed score: shape={_score.shape}"
        # Render first frame
        update_plot(frame_slider.value)
    except Exception as e:
        status.value = f"Error computing score: {e}"


def update_plot(frame_index):
    """Render a single frame (index into valid_time) and put it into _plot_pane."""
    global _score
    if _score is None:
        _plot_pane.object = None
        return
    try:
        da = _score.isel(valid_time=frame_index)
        title = f"Score at {_score.valid_time.values[frame_index]}"
        # Create a responsive quadmesh (avoid fixed pixel sizes so Panel can manage layout)
        # Use a fixed width/height to enforce an approximately 3:2 aspect ratio
        hv_quad = da.hvplot.quadmesh(
            x="longitude",
            y="latitude",
            cmap="viridis",
            clim=(0, 1),
            title=title,
            width=900,
            height=600,
            framewise=True,
        )

        # Optionally overlay a base tiles layer if requested and available
        if show_tiles.value:
            try:
                import holoviews as hv

                tiles = hv.tile_sources.CartoLight().opts(alpha=0.6)
                plot_obj = tiles * hv_quad
            except Exception:
                # Fallback: use quadmesh alone (no basemap)
                plot_obj = hv_quad
        else:
            plot_obj = hv_quad

        _plot_pane.object = plot_obj
    except Exception as e:
        status.value = f"Error rendering frame: {e}"


# Hook actions
load_button.on_click(load_action)
compute_button.on_click(compute_action)
frame_slider.param.watch(lambda e: update_plot(e.new), "value")

status = pn.widgets.StaticText(name="Status", value="Ready")

# Option to toggle basemap tiles (useful if offline or tiles overlap)
show_tiles = pn.widgets.Checkbox(name="Show basemap tiles", value=True)

controls = pn.Column(
    pn.Row(path_input, load_button),
    pn.Row(t2m_ideal_w, d2m_ideal_w),
    pn.Row(t2m_width_w, d2m_width_w),
    pn.Row(downsample_w, compute_button),
    pn.Row(play, frame_slider),
    show_tiles,
    status,
    width=360,
)

# Place controls above the plot to avoid overlap; allow plot to stretch widthwise
layout = pn.Column(controls, _plot_pane, sizing_mode="stretch_width")

# Make servable
layout.servable(title="Weather Score Animator (Panel)")

if __name__.startswith("bokeh") or __name__ == "__main__":
    pn.serve(layout, title="Weather Score Animator (Panel)")
