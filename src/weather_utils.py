import cdsapi
import inspect
import numpy as np
import xarray as xr
from typing import Tuple, List
import zipfile
import os

def pull_era5_reanalysis(
    lat_range: Tuple[float, float],
    lon_range: Tuple[float, float],
    start_date: str,
    end_date: str,
    variables: List[str] = None,
    output_file: str = "era5_data.nc"
) -> xr.Dataset:
    """
    Pull ERA5 reanalysis weather data for specified region and time period.
    
    Args
    ----
    lat_range: Tuple[float, float]
        Tuple of (min_latitude, max_latitude)
    lon_range: Tuple[float, float]
        Tuple of (min_longitude, max_longitude)
    start_date: str
        Start date in 'YYYY-MM-DD' format
    end_date: str
        End date in 'YYYY-MM-DD' format
    variables: List[str]
        List of ERA5 variable names. Defaults to common variables.
    output_file: str
        Output NetCDF filename
    
    Returns
    -------
        xarray.Dataset containing the downloaded ERA5 data
    """
    if variables is None:
        variables = ['2m_temperature', 'total_precipitation']
    
    client = cdsapi.Client()
    
    request = {
        'product_type': 'reanalysis',
        'format': 'netcdf',
        'variable': variables,
        'date': f'{start_date}/{end_date}',
        'area': [lat_range[1], lon_range[0], lat_range[0], lon_range[1]],
    }
    
    client.retrieve('reanalysis-era5-land', request, output_file)

    # If the CDS API returned a zip archive, extract a contained .nc file
    # and write it as '<basename>_extracted.nc' in the same directory.
    if zipfile.is_zipfile(output_file):
        extract_dir = os.path.dirname(output_file) or '.'
        with zipfile.ZipFile(output_file, 'r') as zf:
            nc_members = [n for n in zf.namelist() if n.endswith('.nc')]
            if not nc_members:
                raise RuntimeError(f"No .nc files found inside zip {output_file}")
            # Prefer an 'instant' file if present, otherwise take the first .nc
            chosen = next((n for n in nc_members if 'instant' in n), nc_members[0])

            base = os.path.splitext(os.path.basename(output_file))[0]
            extracted_name = f"{base}_extracted.nc"
            extracted_path = os.path.join(extract_dir, extracted_name)

            # Write the selected member to the extracted path
            with open(extracted_path, 'wb') as out_f:
                out_f.write(zf.read(chosen))

        # Update the output_file to point to the extracted NetCDF
        output_file = extracted_path

    return xr.open_dataset(output_file)


def apply_formula_to_weather_variables(
    weather_array: xr.Dataset,
    formula: callable,
    formula_kwargs: dict = None,
    new_variable_name: str = "custom_variable"
) -> xr.Dataset:
    """
    Apply a mathematical formula to existing weather variables to create a new variable.
    
    Args
    ----
    weather_array: xr.Dataset
        xarray.Dataset containing weather data
    formula: callable,
        callable function that takes in weather variables as arguments
    formula_kwargs: dict, optional
        Additional keyword arguments to pass to the formula
    new_variable_name: str
        Name for the new variable to be created

    Returns
    -------
        xarray.Dataset with the new variable added
    """
    # Extract variable names from the formula's argument names
    arg_names = inspect.getfullargspec(formula).args
    if formula_kwargs is None:
        formula_kwargs = {}
    weather_variable_names = [arg_name for arg_name in arg_names if arg_name not in formula_kwargs]

    # Ensure all required variables are present in the dataset
    for var in weather_variable_names:
        if (var not in weather_array) and (var not in formula_kwargs):
            raise ValueError(f"Variable '{var}' required by formula is not in the dataset.")

    # Apply the formula across the dataset along with any additional kwargs
    new_variable_data = formula(**{var: weather_array[var] for var in weather_variable_names} | formula_kwargs)

    # Add the new variable to the dataset
    weather_array[new_variable_name] = new_variable_data

    return weather_array


def weather_score(
    t2m: xr.DataArray,
    d2m: xr.DataArray,
    t2m_ideal_c: float = 10.0,
    d2m_ideal_c: float = 0.0,
    t2m_width_c: float = 10.0,
    d2m_width_c: float = 10.0,
):
    """Calculate a simple weather score based on temperature and dew point temperature.

    Args:
        t2m (xr.DataArray): 2-meter temperature.
        d2m (xr.DataArray): 2-meter dew point temperature.
        t2m_ideal_c (float): Ideal temperature in Celsius.
        d2m_ideal_c (float): Ideal dew point temperature in Celsius.
        t2m_width_c (float): Width of the temperature preference in Celsius.
        d2m_width_c (float): Width of the dew point temperature preference in Celsius.

    Returns:
        xr.DataArray: Weather score.
    """
    t2m_ideal_k = t2m_ideal_c + 273.15  # Ideal temperature in Kelvin
    d2m_ideal_k = d2m_ideal_c + 273.15  # Ideal dew point temperature in Kelvin
    t2m_zscore = (t2m - t2m_ideal_k) / t2m_width_c
    d2m_zscore = (d2m - d2m_ideal_k) / d2m_width_c
    score = np.exp(-t2m_zscore**2) * np.exp(-d2m_zscore**2)
    return score

