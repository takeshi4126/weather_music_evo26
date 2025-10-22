# MusicEmotionVAE
# Variational Auto Encoder (VAE) to infer music features to raise people's emotions.
# This script uses the emotion-annotated MIDI data from the paper below:
# Wang, Y., Chen, M., Li, X.: Continuous Emotion-Based Image-to-Music Generation. 
# IEEE Trans. Multimed. 26, 5670–5679 (2024).
#
# @author Takeshi Matsumura

import sys
import os

# Add phase2 directory to the sys.path so that common_music is found.
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../phase2')))

import numpy as np
import math
import joblib
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Input, Dense, Concatenate
from tensorflow.keras.models import Model, load_model
from common import common

total_loss_tracker = keras.metrics.Mean(name="total_loss")
music_loss_tracker = keras.metrics.Mean(name="music_loss")
key_loss_tracker = keras.metrics.Mean(name="key_loss")
mode_loss_tracker = keras.metrics.Mean(name="mode_loss")
reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
supervised_loss_tracker = keras.metrics.Mean(name="va_loss")

class MusicEmotionVAE:
  """
  The VAE to construct the latent emotion space from which the music features raising that emotion are inferred.
  The latent emotion space is 2-dimension (valence, arousal).
  The music features consist of tempo, pitch range, mean note duration, major/minor scale (or mode), and key.
  """

  def __init__(self, key_encoding):
    """
    Initialise the MusicEmotionVAE
    """
    self.model = None
    self.model_dir = common.data_path("saved_models")
    self.va_cols = ['Valence', 'Arousal']
    self.scaler = StandardScaler()
    self.scale_vars = ['mean_note_duration', 'tempo', 'pitch_range']
    self.key_encoding = key_encoding

  def create_model(self, key_encoding, latent_dim, units, num_music_features, va_weight=None, mode_weight=None, key_weight=None, beta=None):
    """
    Create the MusicEmotionVAEModel with the give hyper-parameters.
    Parameters:
    - key_encoding: One of 'onehot', 'circular', 'linear' and 'accidentals'. The options other than 'onehot' was used for experiments.
    - latent_dim: The dimensions of the latent space. It is 2 (valence, arousal) at the moment.
    - va_weight: The weight of the valence-arousal loss term in the ELBO.
    - mode_weight: The weight of the mode (major/minor scale) loss term in the ELBO.
    - key_weight: The weight of the key loss term in the ELBO.
    - beta: The weight of KL-divergence term in the ELBO.
    """
    if key_encoding == 'onehot':
      self.model = MusicEmotionVAEOneHotKeyEncoding(latent_dim, units, num_music_features, va_weight=va_weight, mode_weight=mode_weight, key_weight=key_weight, beta=beta)
    elif key_encoding == 'circular':
      self.model = MusicEmotionVAECircularEncoding(latent_dim, units, num_music_features, va_weight=va_weight, mode_weight=mode_weight, key_weight=key_weight, beta=beta)
    elif key_encoding == 'linear':
      self.model = MusicEmotionVAELinearEncoding(latent_dim, units, num_music_features, va_weight=va_weight, mode_weight=mode_weight, key_weight=key_weight, beta=beta)
    elif key_encoding == 'accidentals':
      self.model = MusicEmotionVAEAccidentalsEncoding(latent_dim, units, num_music_features, va_weight=va_weight, mode_weight=mode_weight, key_weight=key_weight, beta=beta)

  def load_model(self):
    """
    Load the tensorflow file containing the already trained MusicEmotionVAE and the scaling data.
    """
    self.model = load_model(f"{self.model_dir}/MusicEmotionVAE.tf")
    self.scaler = joblib.load(f"{self.model_dir}/music_feature_scaler.save")
    
  def save_model(self):
    """
    Save the trained MusicEmotionVAE into the tensorflow file together with the scaling data.
    """
    self.model.save(f"{self.model_dir}/MusicEmotionVAE.tf", save_format='tf')
    joblib.dump(self.scaler, f"{self.model_dir}/music_feature_scaler.save")

  def standardise_va(self, df):
    """
    Scale Valence and Arousal into [0, 1] range.
    Valence and Arousal are annotated in the [1, 9] range. See the original paper.
    """
    mm_scaler = MinMaxScaler()
    return mm_scaler.fit_transform(df)

  def standardise_data(self, df):
    """
    Standardise the dataframe containing the annotated music features.
    Parameters:
    - key_encoding: one of 'onehot', 'circular' and 'linear'.
    """

    # Score_NotesMean gives the total number of notes in the entire score, averaged by two parts for 0001.mid.
    # On the other hand, it seems giving the mean number of notes per measure for 0015.mid. 
    # This parameter is not reliable so drop it.

    df = df.drop(columns=['Score_NotesMean'])
    # Drop unused columns
    df = df.drop(columns=['NumberOfBeats', 'Score_Density', 'Score_Notes', 'Score_RhythmInt', 'pc', 'mean_note_duration'])
    df = df.drop(columns=['tonic'])
    # Linear encoding uses 'num_accidentals' as the key index. The others uses 'c5_index'.
    if self.key_encoding == 'accidentals':
      df = df.drop(columns=['c5_index'])
    else:
      df = df.drop(columns=['num_accidentals'])

    # Rename the columns for better names
    df = df.rename(columns = {
      'Score_AverageDuration': 'mean_note_duration',
      'interval': 'pitch_range'
    })

    # Remove the rows containing na.
    # mean_note_duration gives empty values for some midi files. 

    df = df.dropna()
    df[self.va_cols] = self.standardise_va(df[self.va_cols])

    return df

  def scale_music_features(self, df):
    """
    Scale the specified columns of the DataFrame by the given scaler.
    Parameters:
    - df: The DataFrame object containing the music features to be scaled.
    """
    # Duplicate the original DataFrame, not create a new one, in order to preserve the indices.
    df_scaled = df.copy()
    scaled = self.scaler.fit_transform(df[self.scale_vars])
    for i, col in enumerate(self.scale_vars):
      df_scaled[col] = scaled[:, i]

    return df_scaled
    
  def unscale_music_features(self, df):
    """
    Unscale the DataFrame by using the scaler (Inverse operation of scale_data).
    Parameters:
    - df: The DataFrame object containing the music features to be descaled.
    """
    # Duplicate the original DataFrame, not create a new one, in order to preserve the indices.
    unscaled_df = df.copy()
    unscaled = self.scaler.inverse_transform(df[self.scale_vars])
    for i, col in enumerate(self.scale_vars):
      unscaled_df[col] = unscaled[:, i]

    return unscaled_df
  
  def prepare_data(self, df_scaled):
    """
    Prepare the data for the VAE training by standardising the already scaled data.
    Parameters:
    - df_scaled: The DataFrame object containing the already scaled music features.
    """
    # Create the combined dataset.
    df_std = df_scaled.drop(columns=self.va_cols)

    # Move the 'mode' and 'key_index' / 'num_accidentals' columns to the last as a different loss function is applied to it.
    mode_col = df_std.pop("mode")
    df_std["mode"] = mode_col

    key_col_name = "num_accidentals" if self.key_encoding == 'accidentals' else "c5_index"
    key_col = df_std.pop(key_col_name)
    df_std[key_col_name] = key_col

    # Here, 'mode' should be either 'major' or 'minor'. Confirm that.
    assert len(df_std['mode'].unique()) == 2

    df_std['mode'] = np.array([1 if mode == 'major' else 0 for mode in df_std['mode'].to_list()], dtype=int)

    return df_std

class MusicEmotionVAEModel(keras.Model):
  def __init__(self, latent_dim, units, num_music_features, *args, beta=None, va_weight=None, mode_weight=None, key_weight=None, **kwargs):
    """
    Parameters:
    - num_music_features: Number of music features except for the mode and key.
    - va_weight: Weight on the VA supervised parametere in the total loss.
    - mode_weight: Weight on the 'mode' parameter in the reconstruction loss.
    - key_weight: Weight on the 'key' parameter in the reconstruction loss.
    """
    super(MusicEmotionVAEModel, self).__init__(*args, **kwargs)
    if beta is None or va_weight is None or mode_weight is None:
      raise "beta, va_weight and mode_weight parameters must be given in the constructor."
    self.latent_dim = latent_dim
    self.units = units
    self.num_music_features = num_music_features

    self.beta = beta
    self.va_weight = va_weight
    self.mode_weight = mode_weight
    self.key_weight = key_weight
    self.mse_loss_func = tf.keras.losses.MeanSquaredError()
    self.bce_loss_func = tf.keras.losses.BinaryCrossentropy()
    self.cce_loss_func = tf.keras.losses.CategoricalCrossentropy()

  def init_encoder_decoder(self):
    """
    Subclass must call this method from the constructor.
    """
    mode_size = 1
    input_size = self.num_music_features + mode_size + self.key_encoding_size
    latent_dim = self.latent_dim
    units = self.units

    # ============= Encoder
    # All the music features but the mode.
    music_input = Input(shape=(input_size), name="music_input")
    x = music_input
    for unit in units:
      x = Dense(unit, activation='relu')(x)

    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)
    latent_layer = Concatenate()([z_mean, z_log_var])

    self.encoder = Model(music_input, latent_layer, name='encoder')

    # ============== Decoder
    latent_input = Input(shape=(latent_dim,))
    x = latent_input
    
    for unit in units[::-1]:
      x = Dense(unit, activation='relu')(x)

    # output_0: Music features other than the mode.
    output_0 = Dense(self.num_music_features, activation='linear')(x)
    
    # output_1: Music mode (major or minor).
    output_1 = Dense(mode_size, activation='sigmoid')(x)

    # output_2: Music key that has been input in the one-hot encoding and processed by softmax.
    output_2 = Dense(self.key_encoding_size, activation=self.key_decoding_layer_activation)(x)

    # Output latent_input together with the reconstructed output.
    outputs = Concatenate()([output_0, output_1, output_2])
    self.decoder = Model(latent_input, outputs, name='decoder')

  def split_music_tensor(self, x):
    """
    Split the given tensor x into 3 parts: music features, key and mode.
    How to split it depends on the encoding of these parameters.
    Subclass must implement this method.
    """
    return x[:,:-1], x[:,-1:]

  def call(self, inputs):
    x = self.encoder(inputs)
    z_mean, z_log_var = tf.split(x, num_or_size_splits=2, axis=1)
    epsilon = tf.keras.backend.random_normal(shape=tf.shape(z_mean), mean=0., stddev=0.1)
    z = z_mean + tf.exp(0.5 * z_log_var) * epsilon
    reconstructed = self.decoder(z)
    return reconstructed, z, z_mean, z_log_var

  def _compute_loss(self, x_true, x_pred, z_true, z_pred, z_mean, z_log_var):

    # Compute the loss value
    x_true_music, x_true_mode, x_true_key = self.split_music_tensor(x_true)
    x_pred_music, x_pred_mode, x_pred_key = self.split_music_tensor(x_pred)

    # Reconstruction loss of the decoder output from the VAE input.
    music_loss = self.mse_loss_func(x_true_music, x_pred_music)
    key_loss   = self.key_loss_func(x_true_key, x_pred_key)
    mode_loss  = self.bce_loss_func(x_true_mode, x_pred_mode)
    reconstruction_loss = music_loss + self.mode_weight * mode_loss + self.key_weight * key_loss

    # Supervised loss in the latent space z.
    supervised_loss = self.mse_loss_func(z_true, z_pred)

    # KL-divergence loss
    kl_loss = -0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)

    # Combine two losses to get the total loss.
    total_loss = reconstruction_loss + self.va_weight * supervised_loss + self.beta * kl_loss

    # Return the losses. The order of the loss variables must match the argument of _updated_metrics methods.
    return total_loss, music_loss, key_loss, mode_loss, reconstruction_loss, supervised_loss

  def train_step(self, data):
    # x_true is the input to the encoder. z_true is the VA values to be compared to the latent emotional value (z).
    x_true, z_true = data
    
    with tf.GradientTape() as tape:
      x_pred, z_pred, z_mean, z_log_var = self(x_true, training=True)  # Forward pass with training enabled.
      total_loss, *other_losses = self._compute_loss(x_true, x_pred, z_true, z_pred, z_mean, z_log_var)

    # Compute gradients
    trainable_vars = self.trainable_variables
    gradients = tape.gradient(total_loss, trainable_vars)

    # Update weights
    self.optimizer.apply_gradients(zip(gradients, trainable_vars))
    
    return self._update_metrics(total_loss, *other_losses)
  
  def test_step(self, data):
    x_true, z_true = data

    x_pred, z_pred, z_mean, z_log_var = self(x_true, training=False)  # Forward pass with training disabled.
    total_loss, *other_losses = self._compute_loss(x_true, x_pred, z_true, z_pred, z_mean, z_log_var)
    return self._update_metrics(total_loss, *other_losses)

  def _update_metrics(self, total_loss, music_loss, key_loss, mode_loss, reconstruction_loss, supervised_loss):

    total_loss_tracker.update_state(total_loss)
    music_loss_tracker.update_state(music_loss)
    key_loss_tracker.update_state(key_loss)
    mode_loss_tracker.update_state(mode_loss)
    reconstruction_loss_tracker.update_state(reconstruction_loss)
    supervised_loss_tracker.update_state(supervised_loss)

    return {
      "total_loss": total_loss_tracker.result(), 
      "music_loss": music_loss_tracker.result(), 
      "key_loss": key_loss_tracker.result(), 
      "mode_loss": mode_loss_tracker.result(), 
      "reconstruction_loss": reconstruction_loss_tracker.result(), 
      "va_loss": supervised_loss_tracker.result()
    }

  @property
  def metrics(self):
    # We list our `Metric` objects here so that `reset_states()` can be
    # called automatically at the start of each epoch
    # or at the start of `evaluate()`.
    # If you don't implement this property, you have to call
    # `reset_states()` yourself at the time of your choosing.
    return [
      total_loss_tracker, 
      music_loss_tracker, 
      mode_loss_tracker, 
      reconstruction_loss_tracker, 
      supervised_loss_tracker
    ]

class MusicEmotionVAEOneHotKeyEncoding(MusicEmotionVAEModel):
  """
  Implementation of MusicEmotionVAEModel with one-hot key encoding of the music key parameter.
  """

  def __init__(self, *args, **kwargs):
    super(MusicEmotionVAEOneHotKeyEncoding, self).__init__(*args, **kwargs)
    self.key_encoding_size = 12
    self.key_decoding_layer_activation = 'softmax'
    self.init_encoder_decoder()

  def split_music_tensor(self, x):
    """
    Split the given tensor x into 3 parts: music features, key and mode.
    How to split it depends on the encoding of these parameters.
    """
    enc_size = self.key_encoding_size
    return x[:,:-enc_size-1], x[:,-enc_size-1:-enc_size], x[:,-enc_size:]

  def key_loss_func(self, true_key, pred_key):
    """
    Calculate the loss of music key by using the Categorical Cross Entropy.
    """
    return self.cce_loss_func(true_key, pred_key)


class MusicEmotionVAECircularEncoding(MusicEmotionVAEModel):
  """
  Implementation of MusicEmotionVAEModel with circular index encoding of the music key parameter.
  """

  def __init__(self, *args, **kwargs):
    super(MusicEmotionVAECircularEncoding, self).__init__(*args, **kwargs)
    self.key_encoding_size = 1
    self.key_decoding_layer_activation = 'linear'
    self.init_encoder_decoder()

  def split_music_tensor(self, x):
    """
    Split the given tensor x into 3 parts: music features, key and mode.
    How to split it depends on the encoding of these parameters.
    """
    enc_size = self.key_encoding_size
    return x[:,:-enc_size-1], x[:,-enc_size-1:-enc_size], x[:,-enc_size:]

  def key_loss_func(self, true_key, pred_key):
    """
    Calculate the loss of music key by using the Mean Square Error.
    """
    sector = 2 * math.pi / 12
    radians_true = tf.cast(true_key * sector, tf.float32)
    radians_pred = tf.cast(pred_key * sector, tf.float32)
    angle_diff = radians_true - radians_pred
    shortest_angle = tf.math.atan2(tf.math.sin(angle_diff), tf.math.cos(angle_diff))
    # Calculate the mean square.
    return tf.reduce_mean(tf.square(shortest_angle))


class MusicEmotionVAELinearEncoding(MusicEmotionVAEModel):
  """
  Implementation of MusicEmotionVAEModel with linear index encoding of the music key parameter.
  """

  def __init__(self, *args, **kwargs):
    super(MusicEmotionVAELinearEncoding, self).__init__(*args, **kwargs)
    self.key_encoding_size = 1
    self.key_decoding_layer_activation = 'tanh'
    self.init_encoder_decoder()

  def split_music_tensor(self, x):
    """
    Split the given tensor x into 3 parts: music features, key and mode.
    How to split it depends on the encoding of these parameters.
    """
    enc_size = self.key_encoding_size
    return x[:,:-enc_size-1], x[:,-enc_size-1:-enc_size], x[:,-enc_size:]

  def key_loss_func(self, true_key, pred_key):
    """
    Calculate the loss of music key by using the Mean Square Error.
    """
    return self.mse_loss_func(true_key, pred_key)
  
class MusicEmotionVAEAccidentalsEncoding(MusicEmotionVAEModel):
  """
  Implementation of MusicEmotionVAEModel with linear index encoding of the music key parameter.
  """

  def __init__(self, *args, **kwargs):
    super(MusicEmotionVAEAccidentalsEncoding, self).__init__(*args, **kwargs)
    self.key_encoding_size = 1
    # self.key_decoding_layer_activation = 'sigmoid'
    self.key_decoding_layer_activation = 'linear'
    self.init_encoder_decoder()

  def split_music_tensor(self, x):
    """
    Split the given tensor x into 3 parts: music features, key and mode.
    How to split it depends on the encoding of these parameters.
    """
    enc_size = self.key_encoding_size
    return x[:,:-enc_size-1], x[:,-enc_size-1:-enc_size], x[:,-enc_size:]

  def key_loss_func(self, true_key, pred_key):
    """
    Calculate the loss of music key by using the Mean Square Error.
    """
    return self.mse_loss_func(true_key, pred_key)
  