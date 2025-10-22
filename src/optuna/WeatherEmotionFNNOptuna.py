# Add phase2 directory to the sys.path so that WeatherVAE is found.
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../phase2')))

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from model.WeatherVAE import WeatherVAE
from model.WeatherEmotionFNN import WeatherEmotionFNN, load_weather_words_emotion_data
import optuna

storage_url="sqlite:///db.sqlite3"

use_early_pruning = False

# Initialize the random seed. This line makes the result reproducible.
# Restarting Kernel is not needed.
tf.keras.utils.set_random_seed(123)

# Open the Weather words - Emotion relation data.
weather_words_emotion_df = load_weather_words_emotion_data()
print(weather_words_emotion_df.head(10))

# Standardise the Valence and Arousal
# They are in the [1, 9] range. See the original paper.
mm_scaler = MinMaxScaler()

def standardise_va(df):
  # return (df - 1) / (9 - 1)
  return mm_scaler.fit_transform(df)

weather_words_emotion_df[['Valence', 'Arousal']] = standardise_va(weather_words_emotion_df[['Valence_mean', 'Arousal_mean']])
weather_words_emotion_df.describe()

# Import weather data for training

# Directory where the generated data is going to be stored.
data_dir = "./cvae_out"

notes_per_chord = 4

# The x,y index of the grid to use the data from.
focus_grid = (6, 3)

# Load the trained Weather VAE model.
weather_vae = WeatherVAE()
weather_vae.load_model()
encoder_loaded = weather_vae.encoder

## Generate corresponding latent weather vectors
df = weather_vae.load_nwp_data()
df = weather_vae.feature_engineering(df)
df_std = weather_vae.scale_data(df)
used_columns = df_std.columns
std_custom_set, label_set = weather_vae.prepare_dataset(df, df_std)
time_set = weather_vae.create_time_input(label_set)
latent_weather_vectors = encoder_loaded.predict([std_custom_set, time_set])

## Annotate weather data by words
weather_emotion_fnn = WeatherEmotionFNN()
word_indices = weather_emotion_fnn.create_word_indices(df, 300)

## Create the training data from the weather features (8D) and the annotated weather data

num_weather_features = 8

class OptunaCallback(Callback):
  def __init__(self, trial):
    self.trial = trial
  
  def on_epoch_end(self, epoch, logs=None):
    # Report the loss to Optuna
    loss = logs["loss"]
    self.trial.report(loss, step=epoch)
    
    if np.isnan(loss):
      print(f"Epoch {epoch}: Loss is NaN. Pruning the trial.")
      raise optuna.exceptions.TrialPruned()

    if use_early_pruning:
      # Check if it should prune
      if self.trial.should_prune():
        raise optuna.exceptions.TrialPruned()


def objective(trial):
  epochs = 100

  weather_dataset_df,label_set_df = weather_emotion_fnn.get_annotated_weather_data(weather_words_emotion_df, word_indices, latent_weather_vectors)
  x_train,x_test,y_train,y_test = train_test_split(weather_dataset_df,label_set_df,test_size=0.2,random_state=123)

  # dimension of the latent space.
  num_layers = trial.suggest_int('num_layers', 2, 5)
  units = [trial.suggest_int(f'units_l{i}', 128, 512, step=4) for i in range(num_layers)]

  # Drop out rate
  init_dropout_rate = 0.2
  # dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5, step=0.1)
  dropout_rate = 0.2

  optuna_callback = OptunaCallback(trial)

  weather_emotion_fnn.create_model(units, num_weather_features, init_dropout_rate, dropout_rate)
  weather_emotion_fnn.model.compile(tf.keras.optimizers.Adam(learning_rate=1e-4), loss=tf.keras.losses.MeanSquaredError())
  weather_emotion_fnn.model.fit(x_train, y_train.drop(columns=["weather_word"]), epochs=epochs, batch_size=32, callbacks=[optuna_callback])

  loss = weather_emotion_fnn.model.evaluate(x_test, y_test.drop(columns=["weather_word"]), verbose=0)
  return loss

if __name__ == '__main__':
  trials = 100

  study = optuna.create_study(direction='minimize', storage=storage_url, study_name="weather-emotion-fnn-4", load_if_exists=True)
  study.optimize(objective, n_trials=trials)

