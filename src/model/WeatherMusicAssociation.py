# WeatherMusicAssociation
# Integration of WeatherEmotionFNN and MusicEmotionVAE to infer music features to raise people's emotions for weather condition.
# @author Takeshi Matsumura


import pandas as pd
from datetime import timedelta, date
from model.WeatherVAE import WeatherVAE
from model.WeatherEmotionFNN import WeatherEmotionFNN
from model.MusicEmotionVAE import MusicEmotionVAE
from common.common_music import music_key_decoding
from common import common

out_dir = common.data_path("output")

def date2int(a_date: date):
  return (a_date.year * 10000 + a_date.month * 100 + a_date.day) * 100

class WeatherMusicAssociationModel:

  def __init__(self, key_encoding, mode_threshold = 0.5):
    self.scale_vars = ['mean_note_duration', 'tempo', 'pitch_range']
    self.music_columns = ["mean_note_duration", "tempo", "pitch_range", "mode"]
    if key_encoding == 'accidentals':
      self.music_columns += ["num_accidentals"]
    else: 
      self.music_columns += ["c5_index"]
    self.mode_threshold = mode_threshold
    self.weather_vae = WeatherVAE()
    self.weather_emotion_fnn = WeatherEmotionFNN()
    self.music_emotion_vae = MusicEmotionVAE(key_encoding)

  def load_trained_models(self):
    """
    Load MusicEmotionVAE and WeatherEmotionFNN that have already been trained.
    """
    self.music_emotion_vae.load_model()
    self.weather_emotion_fnn.load_model()

    # Weather encoder
    self.weather_vae.load_model()

  def generate_music_features_of_day(self, forecast_date :date, weather_df, result_interval_hours):
    """
    Generate music features to sonifiy the weather forecast of the specified date.
    Parameters:
    - forecast_date: Integer representing the forecast date (e.g. 20230101).
    - weather_df: DataFrame containing the NWP data.
    - result_interval_hours: Weather data is grouped and averaged over the specified hours and the music features are inferred for it.
    Returns:
    - result_df: Containing the inferred music features and the used emotion vectors.
    - datetime_list: List of datetime integer for each grouped weather data. 
    """
    weather_encoder = self.weather_vae.encoder

    # Filter the data by the forecast_date
    forecast_date_int = date2int(forecast_date)
    next_date_int = date2int(forecast_date + timedelta(days=1))
    daily_weather_df = weather_df[(weather_df['date_time'] >= forecast_date_int) & (weather_df['date_time'] < next_date_int)]

    # Group the data by the result_interval_hours (e.g. 6 hours)
    daily_weather_df = daily_weather_df.copy().reset_index(drop=True)
    daily_weather_df['hour_interval_index'] = daily_weather_df.index // result_interval_hours
    grouped_daily_weather_df = daily_weather_df.groupby('hour_interval_index', as_index=False).mean()
    grouped_daily_weather_df.drop(columns=['hour_interval_index'], inplace=True)
    grouped_daily_weather_df['date_time'] = daily_weather_df.loc[::result_interval_hours,'date_time'].reset_index(drop=True)

    df_std = self.weather_vae.scale_data(grouped_daily_weather_df)
    std_custom_set, label_set = self.weather_vae.prepare_dataset(grouped_daily_weather_df, df_std)
    time_set = self.weather_vae.create_time_input(label_set)

    # NWP model data to latent weather vector
    encoded_weather = weather_encoder.predict([std_custom_set, time_set], verbose=0)

    # Project the latent weather vector into the latent emotional space.
    weather_va = self.weather_emotion_fnn.model.predict(encoded_weather, verbose=0)

    # Generate music features from the latent emotional space.
    music_features = self.music_emotion_vae.model.decoder.predict(weather_va, verbose=0)
    if self.music_emotion_vae.key_encoding == 'onehot':
      predicted_columns = self.music_columns[:-1] + [f"c5_index_{c}" for c in range(12)]
    else:
      predicted_columns = self.music_columns

    music_features_df = pd.DataFrame(music_features, columns=predicted_columns)
    result_df = self.music_emotion_vae.unscale_music_features(music_features_df)
    result_df['Valence'] = weather_va[:, 0]
    result_df['Arousal'] = weather_va[:, 1]

    return result_df, grouped_daily_weather_df['date_time'].to_numpy().tolist()

  def weather_music_association(self, weather_df, start_date, end_date, result_interval_hours):
    """
    Generate music features to sonifiy the daily weather forecast during the specified period.
    Parameters:
    - weather_df: DataFrame containing the NWP data.
    - start_date: The start date of the period.
    - end_date: The end date of the period.
    - result_interval_hours: See generate_music_features_of_day.
    Returns:
    - DataFrame containing the inferred music features and the used emotion vectors.
    """
    date_times = []
    generated_music_features = []
    weather_df = self.weather_vae.feature_engineering(weather_df)

    current_date = start_date
    one_day = timedelta(days=1)
    while( current_date <= end_date ):
      result_df, result_date_time = self.generate_music_features_of_day(
        current_date, 
        weather_df,
        result_interval_hours
      )
      generated_music_features += result_df.to_numpy().tolist()
      date_times += result_date_time
      current_date = current_date + one_day

    final_df = pd.DataFrame(generated_music_features, columns=result_df.columns)
    final_df['date_time'] = date_times

    # Convert mode to string.
    final_df['mode_str'] = ['major' if m > self.mode_threshold else 'minor' for m in final_df['mode']]

    # Decode the encoded key
    final_df = music_key_decoding(final_df, self.music_emotion_vae.key_encoding)

    return final_df
  
  def load_weather_music_features(self):
    """
    Load the saved music features from the CSV file.
    Returns:
    - DataFrame given to the save_weather_music_features.
    """
    df = pd.read_csv(f"{out_dir}/generated_weather_music_features.csv", index_col=False)
    return df
  
  def save_weather_music_features(self, df):
    """
    Save the inferred music features into the CSV file.
    Parameters:
    - df: DataFrame returned by weather_music_association method.
    """
    df.to_csv(f"{out_dir}/generated_weather_music_features.csv", index=False)
