
import numpy as np
import optuna

from tensorflow.keras.callbacks import Callback
from sklearn.model_selection import train_test_split
from model.WeatherVAE import WeatherVAE

storage_url="sqlite:///db.sqlite3"

use_early_pruning = False

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

  weather_vae = WeatherVAE()
  df = weather_vae.load_nwp_data()
  df = weather_vae.feature_engineering(df)
  df_std = weather_vae.scale_data(df)
  used_columns = df_std.columns
  std_custom_set, label_set = weather_vae.prepare_dataset(df, df_std)
  x_train,x_test,y_train,y_test=train_test_split(std_custom_set,label_set,test_size=0.2,random_state=123)
  time_train = weather_vae.create_time_input(y_train)
  time_test = weather_vae.create_time_input(y_test)

  # dimension of the latent space.
  # latent_dim = trial.suggest_int('latent_dim', 4, 20)
  latent_dim = 8
  num_layers = trial.suggest_int('num_layers', 1, 5)
  units = [trial.suggest_int(f'units_l{i}', 128, 512, step=4) for i in range(num_layers)]

  optuna_callback = OptunaCallback(trial)

  weather_vae.create_model(len(used_columns), latent_dim, units)
  weather_vae.vae.fit([x_train, time_train], x_train, epochs=epochs, batch_size=32, callbacks=[optuna_callback])

  loss = weather_vae.vae.evaluate([x_test, time_test], x_test, verbose=0)
  return loss

if __name__ == '__main__':
  trials = 300

  study = optuna.create_study(direction='minimize', storage=storage_url, study_name="cvae1d-reduce-params-2", load_if_exists=True)
  study.optimize(objective, n_trials=trials)

