# common_weather.py
# File containing commonly used functions to process weather data.
#
# @author Takeshi Matsumura

from sklearn.preprocessing import StandardScaler
import numpy as np
from datetime import datetime, timedelta

# The list of weather parameters in the NWP data from the UK Met Office.
weather_parameters = [
  "CAPE_most_unstable_below_500hPa",
  "CAPE_surface",
  "cloud_amount_below_1000ft_ASL",
  "cloud_amount_of_high_cloud",
  "cloud_amount_of_low_cloud",
  "cloud_amount_of_medium_cloud",
  "cloud_amount_of_total_cloud",
  "fog_fraction_at_screen_level",
  # "hail_fall_accumulation-PT01H",
  "hail_fall_rate",
  "height_AGL_at_cloud_base_where_cloud_cover_2p5_oktas",
  "height_AGL_at_freezing_level",
  "height_AGL_at_wet_bulb_freezing_level",
  "landsea_mask",
  # "lightning_flash_accumulation-PT01H",
  # "precipitation_accumulation-PT01H",
  "precipitation_rate",
  "pressure_at_mean_sea_level",
  "pressure_at_surface",
  "radiation_flux_in_longwave_downward_at_surface",
  "radiation_flux_in_shortwave_diffuse_downward_at_surface",
  "radiation_flux_in_shortwave_direct_downward_at_surface",
  "radiation_flux_in_shortwave_total_downward_at_surface",
  "radiation_flux_in_uv_downward_at_surface",
  # "rainfall_accumulation-PT01H",
  "rainfall_rate",
  "relative_humidity_at_screen_level",
  "sensible_heat_flux_at_surface",
  "snow_depth_water_equivalent",
  # "snowfall_accumulation-PT01H",
  "snowfall_rate",
  "temperature_at_screen_level",
  # "temperature_at_screen_level_max-PT01H",
  # "temperature_at_screen_level_min-PT01H",
  "temperature_at_surface",
  "temperature_of_dew_point_at_screen_level",
  "visibility_at_screen_level",
  "wind_direction_at_10m",
  "wind_gust_at_10m",
  # "wind_gust_at_10m_max-PT01H",
  "wind_speed_at_10m"
]

selected_weather_parameters = [
  "temperature_at_screen_level",
  "rainfall_rate",
  # "snowfall_rate",
  "cloud_amount_of_total_cloud",
  "fog_fraction_at_screen_level",
  "wind_direction_at_10m",
  "wind_speed_at_10m"
]

# The height and width of the downloaded NWP data.
height = 8
width = 13

def extract_weather_data(dataset: np.ndarray, weather_param: str, used_weather_parameters: list):
  weather_index = used_weather_parameters.index(weather_param)
  return dataset.reshape(-1, len(selected_weather_parameters), height, width)[:, weather_index, :, :]

def remove_weather_data(dataset: np.ndarray, weather_param: str, used_weather_parameters: list):
  weather_index = used_weather_parameters.index(weather_param)
  del used_weather_parameters[weather_index]
  return np.delete(dataset, weather_index, axis=1)

def append_weather_data(dataset: np.ndarray, weather_param: str, data, used_weather_parameters: list):
  orig_shape = dataset.shape
  used_weather_parameters.append(weather_param)
  return np.append(dataset.transpose(1, 0, 2, 3), data).reshape(orig_shape[1]+1, -1, height, width).transpose(1, 0, 2, 3)

def reshape1(data: np.ndarray):
  """
  Reshape the data from (datetime, weather_parameter, y, x) to (n, weather_parameter)
  """
  # Swap datetime and weather_parameter dimension
  transposed_data = data.transpose(1, 0, 2, 3)
  original_shape = transposed_data.shape

  # Reshape the data so that each column contains all weather parameter for one grid cell on a specific time.
  data_to_standardise = transposed_data.reshape(transposed_data.shape[0], -1).T
  return data_to_standardise, original_shape

def inv_reshape1(data: np.ndarray, original_shape):
  """
  Inversion of reshape1
  """

  # Transpose and reshape the standardised_data into the original shape
  return data.T.reshape(original_shape).transpose(1, 0, 2, 3)

def standardise_data(scaler: StandardScaler, data: np.ndarray):
  """
  Standardise the data for each weather parameter.
  Parameters:
  - data: 4D data in (datetime, weather_parameter, y, x) shape
  """

  data = np.array(data)
  data_to_standardise, original_shape = reshape1(data)
  print(f"data_to_standardise.shape = {data_to_standardise.shape}")
  standardised_data = scaler.fit_transform(data_to_standardise)
  return inv_reshape1(standardised_data, original_shape)

def feature_engineering(custom_set, used_weather_parameters: list):
  """
  Convert the wind direction and wind speed into a wind vector.
  """
  wind_direction = np.radians(extract_weather_data(custom_set, "wind_direction_at_10m", used_weather_parameters))
  wind_speed = extract_weather_data(custom_set, "wind_speed_at_10m", used_weather_parameters)
  wx = wind_speed * np.cos(wind_direction)  
  wy = wind_speed * np.sin(wind_direction)
  custom_set = remove_weather_data(custom_set, "wind_direction_at_10m", used_weather_parameters)
  custom_set = remove_weather_data(custom_set, "wind_speed_at_10m", used_weather_parameters)
  custom_set = append_weather_data(custom_set, "wind_speed_x", wx, used_weather_parameters)
  custom_set = append_weather_data(custom_set, "wind_speed_y", wy, used_weather_parameters)
  return custom_set

def display_weather(weather_df, start_dt:datetime, params: list, delta_hour):
  end_dt = start_dt + timedelta(hours=delta_hour)
  start_dt_str = start_dt.strftime("%Y%m%d%H")
  end_dt_str = end_dt.strftime("%Y%m%d%H")
  start_dt_int = int(start_dt_str)
  end_dt_int = int(end_dt_str)

  print(f"Numerical weather forecast from {start_dt_str} to {end_dt_str}")
  print(weather_df[(weather_df['date_time'] >= start_dt_int) & (weather_df['date_time'] < end_dt_int)][params].mean())

def display_1h_weather(weather_df, start_dt:datetime, params: list):
  display_weather(weather_df, start_dt, params, 1)

def display_6h_weather(weather_df, start_dt:datetime, params: list):
  display_weather(weather_df, start_dt, params, 6)

def test_standardise_data():
  """
  Test that inv_reshape1 is the inversion of reshape1
  """
  data = np.random.rand(365, 10, 8, 13)
  reshaped_data, original_shape = reshape1(data)
  inv_data = inv_reshape1(reshaped_data, original_shape)
  assert np.array_equal(data, inv_data) == True
