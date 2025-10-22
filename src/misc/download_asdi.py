# Python script to download the UK MET grid data from the ASDI
# Met Office UK Deterministic (UKV) 2km on a 2-year rolling archive
# The United Kingdom domain is a 1,096km x 1,408km (2km resolution grid). Formatted via NetCDF.

# The size of files per hour should be less than 50 MBytes.
# 24h => 1200 Mbytes = 1.2 GBytes
# 365 days => 438 GBytes, which is too big to store in my PC.

# Reduce the grid size.
# 1,096km x 1,408km = 548 x 704 grids = 385,792 grids.
# 20 x 10 = 200 grids can cover Edinburgh area.
# So, the required storage per hour will be reduced to 0.05% (219 Mbytes).

# The grid size in the actual netcdf file was different. (970, 1042) => ()
# 

import os
import os.path
import tempfile
import numpy as np
from datetime import datetime, timedelta
import boto3
from botocore import UNSIGNED
from botocore.config import Config
import xarray as xr
from pyproj import Proj
import argparse
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

file_handler = logging.FileHandler('download_asdi.log')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# ASDI S3 bucket
asdi_s3_bucket = "met-office-atmospheric-model-data"

# File prefix to indicate the forecast time (00h 00m)
file_time_prefix = "PT00%02dH00M"

# Path of the output directory
output_dir = "./asdi_out"

# Forecast parameters in pressure levels and height levels are not used.

# weather_parameters = []

weather_parameters = [
  "CAPE_most_unstable_below_500hPa",
  "CAPE_surface",
  "cloud_amount_below_1000ft_ASL",
  "cloud_amount_of_high_cloud",
  "cloud_amount_of_low_cloud",
  "cloud_amount_of_medium_cloud",
  "cloud_amount_of_total_cloud",
  "fog_fraction_at_screen_level",
  "hail_fall_rate",
  "height_AGL_at_cloud_base_where_cloud_cover_2p5_oktas",
  "height_AGL_at_freezing_level",
  "height_AGL_at_wet_bulb_freezing_level",
  "landsea_mask",
  "precipitation_rate",
  "pressure_at_mean_sea_level",
  "pressure_at_surface",
  "radiation_flux_in_longwave_downward_at_surface",
  "radiation_flux_in_shortwave_diffuse_downward_at_surface",
  "radiation_flux_in_shortwave_direct_downward_at_surface",
  "radiation_flux_in_shortwave_total_downward_at_surface",
  "radiation_flux_in_uv_downward_at_surface",
  "rainfall_rate",
  "relative_humidity_at_screen_level",
  "sensible_heat_flux_at_surface",
  "snow_depth_water_equivalent",
  "snowfall_rate",
  "temperature_at_screen_level",
  "temperature_at_surface",
  "temperature_of_dew_point_at_screen_level",
  "visibility_at_screen_level",
  "wind_direction_at_10m",
  "wind_gust_at_10m",
  "wind_speed_at_10m"
]

accumulate_weather_parameters = [
  "hail_fall_accumulation-PT01H",
  "lightning_flash_accumulation-PT01H",
  "precipitation_accumulation-PT01H",
  "rainfall_accumulation-PT01H",
  "snowfall_accumulation-PT01H",
  "temperature_at_screen_level_max-PT01H",
  "temperature_at_screen_level_min-PT01H",
  "wind_gust_at_10m_max-PT01H",
]

# AWS S3 client
s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))

# Edinburgh region

edinburgh_region = {
  'north_lat': 56.0,
  'south_lat': 55.85,
  'east_lon' : -3.0,
  'west_lon' : -3.4
}

def download_all(start_date: datetime, end_date: datetime, region: dict) -> None:
  """
  Download all the required weather parameter data for the given period, extract the required region, 
  and save the data into hourly data files.
  Parameters:
  - start_date: The start date of the period (inclusive).
  - end_date: The end date of the period (exclusive).
  - region: A dict of [north_lat, south_lat, east_lon, west_lon]
  """
  delta = timedelta(hours=1)
  current_time = start_date
  while current_time < end_date:
    try:
      download_one_hour(current_time, region)
    except Exception as e:
      logger.error("Failed to create hourly data for %s" % (datetime.strftime(current_time, "%Y%m%d%H")), exc_info=True)
    finally:
      current_time += delta

def download_one_hour(dt: datetime, region: dict):
  """
  Download all the required weather parameter data for the given date and hour.
  Parameters:
  - dt: specifying the date and hour to download the data.
  - region: See download_all
  """
  # Skip if the houly data has already been saved.
  file_path = "%s/%s.npy" % (output_dir, dt.strftime("%Y%m%d%H"))
  if os.path.isfile(file_path):
    return

  # TODO: Align the data type.
  hourly_data = np.array(
    [download_one_parameter(dt, weather_param, region) for weather_param in weather_parameters] + 
    [download_one_parameter(dt, weather_param, region, hour=1) for weather_param in accumulate_weather_parameters]
  )
  
  save_one_hour_data(hourly_data, file_path)

def save_one_hour_data(data: np.ndarray, file: str) -> None:
  """
  Save a 3d numpy array containing the grid data of all parameters for the specified hour into a file.
  """

  # Disable allow_pickle to ensure compatibility between different Python environments.
  np.save(file, data, allow_pickle=False)

def download_one_parameter(dt: datetime, parameter: str, region: dict, hour: int = 0) -> np.ndarray:
  """
  Download all the specified weather parameter data for the given date and hour.
  Parameters:
  - datetime: specifying the date and hour to download the data.
  - parameter: String for the weather parameter.
  - region: See download_all
  Return:
  - A 2d numpy array containing the grid data.
  """
  folder_name = dt.strftime("%Y%m%dT%H00Z")   #20241004T0100Z
  file_dt_str = (dt + timedelta(hours=hour)).strftime("%Y%m%dT%H00Z")   #20241004T0100Z
  object_name = "uk-deterministic-2km/%s/%s-%s-%s.nc" % (folder_name, file_dt_str, file_time_prefix % hour, parameter)
  logger.info(object_name)
  with tempfile.NamedTemporaryFile() as f:
    s3.download_file(asdi_s3_bucket, object_name, f.name)
    region_data = extract_region(f.name, region)
    return region_data

def extract_region(nc_file: str, region: dict) -> np.ndarray :
  """
  Extract the specified region from the given netcdf file.
  Parameters:
  - nc_file: path to the netcdf file.
  - data_key: 
  - region: See download_all
  Return:
  - A 2d numpy array containing the grid data.
  """
  ds = xr.open_dataset(nc_file, decode_timedelta=False)
  data_key = identify_data_key(ds)
  data = ds[data_key].values

  laea_area = ds['lambert_azimuthal_equal_area']
  origin_longitude = laea_area.longitude_of_projection_origin
  origin_latitude = laea_area.latitude_of_projection_origin

  # lambert azimuthal equal area projection
  laea_proj = Proj(proj='laea', lat_0=origin_latitude, lon_0=origin_longitude, ellps='WGS84')
  # Project the region
  x1, y1 = laea_proj(region['west_lon'], region['south_lat'])
  x2, y2 = laea_proj(region['east_lon'], region['north_lat'])
  x_coordinate = ds['projection_x_coordinate'].values
  y_coordinate = ds['projection_y_coordinate'].values
  x_indices = np.where((x_coordinate >= x1) & (x_coordinate <= x2))[0]
  y_indices = np.where((y_coordinate >= y1) & (y_coordinate <= y2))[0]
  region_data = data[y_indices,:][:,x_indices]

  return region_data

def identify_data_key(ds: xr) -> str:
  non_data_variables = [
    'lambert_azimuthal_equal_area',
    'projection_x_coordinate',
    'projection_y_coordinate',
    'projection_x_coordinate_bnds',
    'projection_y_coordinate_bnds',
    'forecast_period_bnds',
    'time_bnds',
  ]
  # Identify the data variable
  data_vars = [var_name for var_name in ds.data_vars.keys() if var_name not in non_data_variables]
  assert len(data_vars) == 1
  data_key = data_vars[0]

  return data_key

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("start", help = "Start date in %Y%m%d format (inclusive)", type=str)
  parser.add_argument("end", help = "End date in %Y%m%d format(exclusive)", type=str)
  args = parser.parse_args()
  os.makedirs(output_dir, exist_ok=True)
  download_all(datetime.strptime(args.start, "%Y%m%d"), datetime.strptime(args.end, "%Y%m%d"), edinburgh_region)

if __name__ == '__main__':
  main()
