# Add phase2 directory to the sys.path so that common_music is found.
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../phase2')))

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from model.MusicEmotionVAE import MusicEmotionVAE
from common_music import music_key_encoding
import optuna

# Parameters for Optuna
storage_url="sqlite:///db.sqlite3"
use_early_pruning = True
key_encoding = "onehot"

# Takeshi: Initialize the random seed. This line makes the result reproducible.
# Restarting Kernel is not needed.
tf.keras.utils.set_random_seed(123)

# The dimension of the latent emotional space (Valence & Arousal).
latent_dim = 2

# Prepare the training and test data outside of the objective funciton to ensure that the same data set is used every run.
df = pd.read_csv("./output/music_feature_vae_input.csv")
music_emotion_vae = MusicEmotionVAE(key_encoding)
df = music_emotion_vae.standardise_data(df)
label_set = df[music_emotion_vae.va_cols]

df_scaled = music_emotion_vae.scale_music_features(df)
df_std = music_emotion_vae.prepare_data(df_scaled)
# Remember the number of music features before the one-hot encoding (exclude mode and c5_index)
music_features_columns = [col for col in df_std.columns.to_list() if col not in ['mode', 'c5_index', 'num_accidentals']]
num_music_features = len(music_features_columns)

df_std = music_key_encoding(df_std, key_encoding)
x_train,x_test,y_train,y_test = train_test_split(df_std,label_set,test_size=0.2,random_state=123)

def objective(trial):
  epochs = 100

  # Recreate the ML model at every trial.
  music_emotion_vae = MusicEmotionVAE(key_encoding)

  # dimension of the latent space.
  # num_layers = trial.suggest_int('num_layers', 2, 5)
  # units = [trial.suggest_int(f'units_l{i}', 128, 512, step=4) for i in range(num_layers)]

  # Fixed hyper-parameters for exploring num_layers and units.
  # va_weight = 1.0
  # mode_weight = 1.0
  # key_weight = 1.0
  # beta = 0.1

  # One of the Pareto optimal solutions in music-emotion-vae-12 to search for num_layers and units only.
  num_layers = 5
  units = [452, 308, 484, 444, 256]

  # # va_weight = 30
  va_weight = trial.suggest_float('va_weight', 0.1, 10.0, step=0.1)
  mode_weight = trial.suggest_float('mode_weight', 0.1, 1.0, step=0.1)
  key_weight = trial.suggest_float('key_weight', 0.0, 1.0, step=0.1)
  # key_weight = 0.0
  beta = trial.suggest_float('beta', 0.1, 0.5, step=0.1)
  # # beta = 0.1

  music_emotion_vae.create_model(key_encoding, latent_dim, units, num_music_features, va_weight=va_weight, mode_weight=mode_weight, key_weight=key_weight, beta=beta)
  music_emotion_vae.model.compile(tf.keras.optimizers.legacy.Adam(learning_rate=1e-4))
  music_emotion_vae.model.fit(x_train, y_train, epochs=epochs, batch_size=32, verbose=0)
  loss_dict = music_emotion_vae.model.evaluate(x_test, y_test, return_dict=True, verbose=0)

  print(loss_dict)
  return loss_dict['total_loss'], loss_dict['music_loss'], loss_dict['key_loss'], loss_dict['mode_loss'], loss_dict['reconstruction_loss'], loss_dict['va_loss']

if __name__ == '__main__':
  trials = 300

  study = optuna.create_study(directions=['minimize'] * 6, storage=storage_url, study_name="music-emotion-vae-13", load_if_exists=True)
  study.optimize(objective, n_trials=trials)

