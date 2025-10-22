# WeatherEmotionFNN
# Feed-forward Neural Network to infer the emotion vectors (valence, arousal) representing input weather parameters.
# This script uses the emotion annotated weather words from this work:
# Stewart, A.E.: Affective Normative Data for English Weather Words. Atmosphere. 11, 860 (2020).
#
# @author Takeshi Matsumura

import numpy as np
import pandas as pd
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from common import common

def load_weather_words_emotion_data():
  """
  Open the Weather words - Emotion relation file.
  """

  # Ignore the Dominance and Surprise as they have high correlation with the Valence And Arousal.
  # Also, ignore the standard deviation.
  filename = "Aﬀective Normative Data for English Weather Words supp Table S1.xlsx"  
  weather_words_emotion_df = pd.read_excel(common.data_path(f"input_data/{filename}"), header=1)

  # Rename Valence_mean and Arousal_mean to Valence and Arousal respectively.
  weather_words_emotion_df.columns = ["weather_word", "Valence_mean", "Valence_SD", "Arousal_mean", "Arousal_SD", "Dominance_mean", "Dominance_SD", "Surprise_mean", "Surprise_SD", "Cluster"]

  # Remove the last row containing some footnote
  weather_words_emotion_df.drop(index=141, inplace=True)
  # Remove the word number from the weather_word column
  weather_words = weather_words_emotion_df["weather_word"]
  weather_words = weather_words.str.replace('^\d+\.', '', regex=True)
  # Remove string (n), which indicates the the word was presented as a noun to the annotators, from the words
  weather_words = weather_words.str.replace('(n)', '')
  # Remove string (weather)
  weather_words = weather_words.str.replace('(weather)', '')
  # Remove string "1" attached to the Brizzard
  weather_words = weather_words.str.replace('Blizzard 1', 'Blizzard')
  # Trim the weather words
  weather_words = weather_words.str.replace('^\s+', '', regex=True)
  weather_words = weather_words.str.replace('\s+$', '', regex=True)
  weather_words_emotion_df["weather_word"] = weather_words

  return weather_words_emotion_df

class WeatherEmotionFNN:

  """
  Feed-forward Neural Network (FNN) to infer emotion vectors for describing the input weather embeddings.
  """

  def __init__(self):
    self.model_dir = common.data_path("saved_models")
    self.model_filename = "WeatherEmotionFNN.tf"

    self.model = None

  def create_model(self, units, num_weather_features, init_dropout_rate, dropout_rate):
    """
    Create the weather-emotion FNN. Dropout layers are included to prevent it from overfitting.
    
    Parameters:
    - units: List of the number of parameters for each dense layer.
    - num_weather_features: The dimension of the input weather features.
    - init_dropout_rate: Dropout rate of the dropout layer before the first dense layer.
    - dropout_rate: Dropout rates of the dropout layers placed after the dense layers.
    """
    inputs = Input(shape=(num_weather_features,))
    x = inputs
    if init_dropout_rate > 0:
      x = Dropout(init_dropout_rate)(x)
    for unit in units:
      x = Dense(unit, activation='relu')(x)
      if dropout_rate > 0:
        x = Dropout(dropout_rate)(x)
    outputs = Dense(2, activation='sigmoid')(x)

    self.model = Model(inputs=inputs, outputs=outputs, name='weather_emotion_fnn')
    self.model.compile(Adam(learning_rate=1e-4), loss=MeanSquaredError())    

  def load_model(self):
    """
    Load the trained WeatherEmotionFNN model in the tensorflow format.
    """
    self.model = load_model(f"{self.model_dir}/{self.model_filename}")

  def save_model(self):
    """
    Save the trained WeatherEmotionFNN model in the tensorflow format.
    """
    self.model.save(f"{self.model_dir}/{self.model_filename}", save_format='tf')

  # dict to collect the indices in the df to be annotated by each weather word.

  def create_word_indices(self, df, sample_limit):
    """
    Assign weather words to the subset of the NWP data.
    See Table 10 in the dissertation report. 
    """
    word_indices = {}
    # Temperature
    temp = df['temperature_at_screen_level']
    word_indices['Hot'] = df[temp >= 273 + 20].index
    word_indices['Warm'] = df[(temp >= 273 + 15) & (temp < 273 + 20)].index
    word_indices['Comfortable'] = df[(temp >= 273 + 10) & (temp < 273 + 15)].index
    word_indices['Cool'] = df[(temp >= 273 + 5) & (temp < 273 + 10)].index
    word_indices['Cold'] = df[(temp >= 273 + 0) & (temp < 273 + 5)].index
    word_indices['Freezing'] = df[temp < 273 + 0].index

    # Humidity
    rh = df['relative_humidity_at_screen_level']
    word_indices['Humid'] = df[rh >= 0.9].index
    word_indices['Dry'] = df[rh < 0.5].index

    # Rain
    rain = df['rainfall_accumulation-PT01H'] * 1000 # meter to mm.
    word_indices['Rainy'] = df[rain >= 0.5].index
    word_indices['Drizzle'] = df[(rain > 0.0) & (rain < 0.5)].index

    # Wind
    ws = np.sqrt(df['wx'].to_numpy()**2 + df['wy'].to_numpy()**2)
    word_indices['Breezy'] = df[(ws >= 3) & (ws < 5)].index
    word_indices['Windy'] = df[(ws >= 5) & (ws < 7)].index
    word_indices['Gusty'] = df[(ws >= 7) & (ws < 10)].index
    word_indices['Blustery'] = df[ws >= 10].index

    # Cloud: Use the mean cloud amount in low and medium layers instead of the total cloud.
    cloud = (df['cloud_amount_of_low_cloud'] + df['cloud_amount_of_medium_cloud']) / 2
    word_indices['Clear'] = df[(cloud < 0.2)].index
    word_indices['Partly Cloudy'] = df[(cloud >= 0.2) & (cloud < 0.5)].index
    word_indices['Cloudy'] = df[(cloud >= 0.5) & (cloud < 0.8)].index
    word_indices['Overcast'] = df[(cloud >= 0.8)].index

    # Set the number of samples per weather word the same.
    for w in word_indices.keys():
      word_indices[w] = word_indices[w][:sample_limit]

    return word_indices

  def get_annotated_weather_data(self, weather_words_emotion_df, word_indices, latent_weather_vectors):
    """
    Annotate emotion vectors (valence, arousal) to the latent weather vectors.
    """
    weather_words_set = []
    weather_dataset = []
    label_set = []

    for weather_word in word_indices.keys():
      # Read the valence and arousal value from the weather words emotion table.
      va = weather_words_emotion_df[weather_words_emotion_df['weather_word'] == weather_word][['Valence', 'Arousal']].to_numpy()[0]
      weather_data_indices = word_indices[weather_word]
      for index in weather_data_indices:
        # Annotate the valence and arousal values to weather data samples
        latent_weather = latent_weather_vectors[index]
        weather_words_set.append(weather_word)
        weather_dataset.append(latent_weather)
        label_set.append(va)

    weather_words_set = np.array(weather_words_set)
    weather_dataset = np.array(weather_dataset)
    label_set = np.array(label_set)

    # The dimensions of the latent space created by the Weather VAE.
    weather_dataset_dimensions = weather_dataset.shape[1]
    weather_dataset_df = pd.DataFrame(weather_dataset, columns=[f"d{i}" for i in range(weather_dataset_dimensions)])
    label_set_df = pd.DataFrame(label_set, columns=['Valence', 'Arousal'])
    label_set_df["weather_word"] = weather_words_set

    return weather_dataset_df,label_set_df
