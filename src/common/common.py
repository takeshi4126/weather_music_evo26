# common.py
# Common functions irrelevant to music and weather.
# The data_dir relative path must be changed if you move the location of this file.
# @author Takeshi Matsumura

import os
import random
import seaborn as sns
import tensorflow as tf
import matplotlib.pylab as plt

script_dir = os.path.dirname(os.path.abspath(__file__))

# Relative path to the data directory.
data_dir = f"{script_dir}/../../data"

def data_path(rel_path):
  """
  Return the path in the data dir.
  """
  return f"{data_dir}/{rel_path}"

def init_notebook():
  """
  Initialise the random seek and set the seaborn font size, which is common to all the notebooks used in this project.
  """
  # Takeshi: Initialize the random seed. This line makes the result reproducible.
  tf.keras.utils.set_random_seed(123)
  random.seed(123)
  # Scale the Seaborn font size.
  sns.set_context("notebook", font_scale=1.2)

def scatter_plots_side_by_side(true_data, pred_data, weather_parameter):
  """
  Draw the scatter plots of true values and predictions side by side.
  """
  fig, ax = plt.subplots(1, 2, figsize=(18, 6))
  fig.suptitle(weather_parameter)

  df1 = true_data[['date_time', weather_parameter]].sort_values(['date_time'], ignore_index=True).reset_index()
  df1['data'] = ["true"] * len(df1)
  df2 = pred_data[['date_time', weather_parameter]].sort_values(['date_time'], ignore_index=True).reset_index()
  df2['data'] = ["pred"] * len(df2)

  vmin = df1[weather_parameter].min()
  vmax = df1[weather_parameter].max()
  sns.scatterplot(df1, x="index", y=weather_parameter, hue="data", s=8, ax=ax[0])
  ax[0].set(ylim=(vmin, vmax))
  sns.scatterplot(df2, x="index", y=weather_parameter, hue="data", s=8, ax=ax[1])  
  ax[1].set(ylim=(vmin, vmax))
