import cdsapi
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
    
    Args:
        lat_range: Tuple of (min_latitude, max_latitude)
        lon_range: Tuple of (min_longitude, max_longitude)
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        variables: List of ERA5 variable names. Defaults to common variables.
        output_file: Output NetCDF filename
    
    Returns:
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