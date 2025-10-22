# WeatherVAE
# Variational Auto Encoder (VAE) to compress weather parameters into compact latent weather vectors.
# @author Takeshi Matsumura

from tensorflow.keras.layers import Input, Flatten, Dense, Reshape, Concatenate
from tensorflow.keras.models import Model, load_model

from sklearn.preprocessing import StandardScaler, RobustScaler
import pandas as pd
from datetime import datetime
import os
import joblib

import tensorflow as tf
import numpy as np

from common import common

selected_params = [
  "cloud_amount_of_high_cloud",
  "cloud_amount_of_low_cloud",
  "cloud_amount_of_medium_cloud",
  "cloud_amount_of_total_cloud",
  "fog_fraction_at_screen_level",
  "pressure_at_surface",
  "relative_humidity_at_screen_level",
  "temperature_at_screen_level",
  "visibility_at_screen_level",
  "wind_direction_at_10m",
  "wind_speed_at_10m",
  "rainfall_accumulation-PT01H",
  "snowfall_accumulation-PT01H",
]

width = 1
height = 1

# Very small value to be added for log calculation to avoid log(0) operation.
log0_epsilon = 1e-12

class SamplingLayer(tf.keras.layers.Layer):
  def call(self, inputs):
    z_mean, z_log_var = inputs
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim), mean=0., stddev=0.1)
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon  
    
class WeatherVAE:

  def __init__(self):
    self.axis_columns = ['date_time','y','x']
    self.model_dir = common.data_path("saved_models")
    self.std_scaler_filename = "std_scaler.save"
    self.pwr_scaler_filename = "pwr_scaler.save"
    self.encoder_filename = "WeatherVAE_Encoder.tf"
    self.decoder_filename = "WeatherVAE_Decoder.tf"
    self.nwp_data_filename = common.data_path("input_data/asdi_data.parquet")
    self.focus_grid = (6, 3)

    # Set or updated by load_model
    self.std_scaler = StandardScaler()
    self.pwr_scaler = RobustScaler()
    self.vae = None
    self.encoder = None
    self.decoder = None

  def create_model(self, num_weather_features, latent_dim, units, no_KLD=False) -> None:
    """
    Create VAE for weather feature extraction.
    """

    input_shape = (height, width, num_weather_features)

    # ============= Encoder
    weather_input = Input(shape=input_shape)
    condition_input = Input(shape=(2, )) # day of the year, hour of the day.

    x = Flatten()(weather_input)
    x = Concatenate()([x, condition_input])
    for unit in units:
      x = Dense(unit, activation='relu')(x)

    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)

    z = SamplingLayer()([z_mean, z_log_var])
    encoder = Model([weather_input, condition_input], [z], name='encoder')

    # ============== Decoder
    latent_input = Input(shape=(latent_dim,))
    x = Concatenate()([latent_input, condition_input])

    for unit in units[::-1]:
      x = Dense(unit, activation='relu')(x)
    x = Dense(num_weather_features, activation='linear')(x)
    outputs = Reshape((1, 1, num_weather_features))(x)
    decoder = Model([latent_input, condition_input], outputs, name='decoder')

    # =============== VAE model
    vae_outputs = decoder([z, condition_input])
    vae = Model([weather_input, condition_input], vae_outputs, name='cvae')

    if not no_KLD:
      kl_loss = -0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
      vae.add_loss(kl_loss)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    vae.compile(optimizer, loss=tf.keras.losses.MeanSquaredError())

    self.vae = vae
    self.encoder = encoder
    self.decoder = decoder

  def load_model(self):
    """
    Load the trained VAE model in the tensorflow format.
    """
    self.std_scaler = joblib.load(f"{self.model_dir}/{self.std_scaler_filename}")
    self.pwr_scaler = joblib.load(f"{self.model_dir}/{self.pwr_scaler_filename}")
    self.encoder = load_model(f"{self.model_dir}/{self.encoder_filename}")
    self.decoder = load_model(f"{self.model_dir}/{self.decoder_filename}")

  def save_model(self):
    """
    Save the trained VAE model in the tensorflow format.
    """
    os.makedirs(self.model_dir, exist_ok=True)

    joblib.dump(self.std_scaler, f"{self.model_dir}/{self.std_scaler_filename}")
    joblib.dump(self.pwr_scaler, f"{self.model_dir}/{self.pwr_scaler_filename}")
    self.encoder.save(f"{self.model_dir}/{self.encoder_filename}", save_format='tf')
    self.decoder.save(f"{self.model_dir}/{self.decoder_filename}", save_format='tf')

  def feature_engineering(self, df):
    # Convert (wind speed, wind direction) to (wind x, wind y) vector.
    if 'wind_speed_at_10m' in selected_params:
      wind_speed = df['wind_speed_at_10m']
      wind_direction = df['wind_direction_at_10m']
      wx = wind_speed * np.cos(np.radians(wind_direction))
      wy = wind_speed * np.sin(np.radians(wind_direction))
      df['wx'] = wx
      df['wy'] = wy
      df = df.drop(columns=['wind_speed_at_10m', 'wind_direction_at_10m'])
      df = df.reset_index(drop=True)

    return df

  def scale_data(self, df):
    df4_std_scale = df.drop(columns=self.axis_columns)

    concat_list = []

    # Exclude the parameters to which scaling is not applied.
    self.std_scaler.fit(df4_std_scale)
    df_std = pd.DataFrame(self.std_scaler.transform(df4_std_scale), columns=df4_std_scale.columns)
    concat_list.append(df_std)

    df_scaled = pd.concat(concat_list, axis = 1)
    
    return df_scaled
    
  def unscale_data(self, scaled_df, used_columns):
    concat_list = []

    df_std = pd.DataFrame(self.std_scaler.inverse_transform(scaled_df), columns=scaled_df.columns)
    concat_list.append(df_std)

    # Apply inverse scaling

    df_unscaled = pd.concat(concat_list, axis = 1)

    return df_unscaled

  def load_nwp_data(self):
    """
    Load the NWP data from the file and select the focused grid point.
    """
    df = pd.read_parquet(self.nwp_data_filename)
    df['date_time'].astype('int32')

    # Select a single grid point.
    df = df[(df['x'] == self.focus_grid[0]) & (df['y'] == self.focus_grid[1])]

    axis_columns = ['date_time','y','x']
    df = df[axis_columns + selected_params]
    return df

  def prepare_dataset(self, df, df_std):
    """
    Prepare the dataset that can be fed into the model from the given standardised data frame.
    Parameters:
    - df: The original data frame
    - df_std: The data frame containing the standardised and scaled data.
    Return:
    - std_custom_set: A numpy matrix containing the data from the df_std, with the axis column values from df.
    - label_set: A numpy array containing the label for each row in the std_custom_set.
    """
    # Copy back the date_time, x, y columns.
    assert len(df_std) == len(df)
    df_std['date_time'] = df['date_time'].reset_index(drop=True)
    df_std['date_time'].astype('int32')
    df_std['x'] = df['x'].to_numpy()
    df_std['x'].astype('int16')
    df_std['y'] = df['y'].to_numpy()
    df_std['y'].astype('int16')

    height = len(np.unique(df['y'].to_numpy()))
    width = len(np.unique(df['x'].to_numpy()))

    # Ensure that the dataframe is sorted.
    df_std = df_std.sort_values(self.axis_columns, ignore_index=True)

    # Use date_time as the label of each observation.
    label_set = df_std['date_time'].unique()

    # Create the input columns by removing irrelevant ones
    input_columns = [col for col in df_std.columns if col not in self.axis_columns]

    std_custom_set = np.array([df_std[df_std['date_time'] == dt][input_columns].to_numpy().reshape((height, width, -1)) for dt in label_set])

    return std_custom_set, label_set

  def create_time_input(self, date_hour_list):
    """
    Convert the time sequence into sine curve.
    
    Parameters:
    - date_hour_list: List of integer representing the date and hour (e.g. 2023040108 for 2023/04/01 08h)
    """
    date_train = [datetime.strptime(str(y), "%Y%m%d%H") for y in date_hour_list]
    day_of_year_train = np.array([d.timetuple().tm_yday for d in date_train])
    hour_of_day_train = np.array([d.timetuple().tm_hour for d in date_train])
    # Convert them into sine curve
    day_of_year_train = np.sin(np.radians(day_of_year_train / 365 * 360))
    hour_of_day_train = np.sin(np.radians(hour_of_day_train / 24 * 360))

    time_train = np.array(list(zip(day_of_year_train, hour_of_day_train)))
    return time_train   
