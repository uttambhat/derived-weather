from typing import Optional

import matplotlib.pyplot as plt
import xarray as xr
from matplotlib import animation
from IPython.display import HTML

def plot_weather_data(
    weather_array: xr.Dataset,
    weather_variable_name: str = "t2m",
    time_idx: int = 0,
) -> None:
    """
    Plot a heatmap of the weather variable
    
    Args
    ----
    weather_array: xarray.Dataset
        The weather data array to plot
    weather_variable_name: str
        Variable name to plot (default is "t2m" for 2m temperature)
    time_idx: int
        Time index to plot (default is 0 for the first time step)

    Returns
    -------
    None
    """
    time = weather_array.valid_time.values[time_idx]

    # extract 2m temperature (variable name in the dataset is "t2m")
    weather_variable_array = weather_array[weather_variable_name].isel(valid_time=time_idx)

    if weather_variable_name == "t2m":
        # convert to Celsius if values look like Kelvin
        if weather_variable_array.mean().item() > 200:
            weather_variable_array = weather_variable_array - 273.15
            units = "°C"
    else:
        units = weather_array[weather_variable_name].attrs.get("units", "")

    lon = weather_array.longitude
    lat = weather_array.latitude

    aspect_ratio = len(lat) / len(lon)
    plt.figure(figsize=(20, 20 * aspect_ratio))
    pcm = plt.pcolormesh(lon, lat, weather_variable_array, shading="auto", cmap="coolwarm")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(f"{weather_variable_name} at {str(time)}")
    plt.colorbar(pcm, label=units)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.show()

def animate_weather_data(
    weather_array: xr.Dataset,
    weather_variable_name: str = "t2m",
    time_range: tuple[int, int] = (0, 10),
    interval: int = 500,
    output_filepath: Optional[str] = "../resources/weather_animation.mp4",
) -> None:
    """
    Animate a heatmap of the weather variable across time steps
    
    Args
    ----
    weather_array: xarray.Dataset
        The weather data array to animate
    weather_variable_name: str
        Variable name to plot (default is "t2m" for 2m temperature)
    time_range: tuple[int, int]
        Start and end time indices (inclusive) for animation
    interval: int
        Delay between frames in milliseconds (default is 500)
    output_filepath: Optional[str]
        Optional file path to save the animation figure as an mp4. If None, returns the animation
        object to be plotted or saved by the user as required.

    Returns
    -------
    None
    """
    
    start_idx, end_idx = time_range
    time_indices = range(start_idx, end_idx + 1)
    
    lon = weather_array.longitude
    lat = weather_array.latitude
    
    weather_variable_array = weather_array[weather_variable_name]
    # set default units and convert Kelvin->C if needed
    units = weather_array[weather_variable_name].attrs.get("units", "")
    if weather_variable_name == "t2m" and weather_variable_array.mean().item() > 200:
        weather_variable_array = weather_variable_array - 273.15
        units = "°C"
        
    aspect_ratio = len(lat) / len(lon)
    fig, ax = plt.subplots(figsize=(20, 20 * aspect_ratio))
    
    # Initialize with first frame and compute a stable color range across frames
    weather_data_values = weather_variable_array.isel(valid_time=start_idx)
    subset = weather_variable_array.isel(valid_time=slice(start_idx, end_idx + 1))
    try:
        vmin = float(subset.min())
        vmax = float(subset.max())
    except Exception:
        vmin, vmax = None, None

    pcm = ax.pcolormesh(lon, lat, weather_data_values, shading="auto", cmap="coolwarm", vmin=vmin, vmax=vmax)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    title = ax.set_title("")
    cbar = plt.colorbar(pcm, ax=ax, label=units)
    ax.set_aspect("equal", adjustable="box")
    
    def update(frame):
        # Select the frame from the DataArray (weather_variable_array is already the variable)
        new_weather_data_values = weather_variable_array.isel(valid_time=frame)
        # Update the QuadMesh color values. Flatten to match the internal storage.
        pcm.set_array(new_weather_data_values.values.ravel())
        time = weather_array.valid_time.values[frame]
        title.set_text(f"{weather_variable_name} at {str(time)}")
        return (pcm, title)
    
    animated_object = animation.FuncAnimation(fig=fig, func=update, frames=time_indices, interval=interval)
    plt.close(fig)  # Prevents duplicate static plot display in some environments
    if output_filepath:
        animated_object.save(output_filepath, writer='ffmpeg', fps=2)
    else:
        return animated_object
