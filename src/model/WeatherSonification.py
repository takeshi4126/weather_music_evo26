# WeatherMusicAssociation
# Convert weather data into music by using the weather-music association created in the latent emotion space.
# This script uses Google Magenta's MelodyRNN running in a separate docker container, 
# The docker container should not run in parallel.
#
# @author Takeshi Matsumura

import numpy as np
import pandas as pd
import random
from datetime import datetime, date
import subprocess
import os
import glob
import time
import subprocess
import shutil

from music21 import note, stream, tempo, key, converter, stream, note
from common import common

random.seed(123)

# Directory where the output MIDI files are stored.
OUTPUT_DIR = common.data_path("weather_sonification_output")

# The directory where the Melody RNN docker stores the generated MIDI files.
# To change this, you will have to update _generate_midi function.
MAGENTA_OUTPUT_DIR = common.data_path("../magenta-docker/output")

# The name of the Melody RNN docker container
MAGENTA_CONTAINER_NAME = "my-magenta-container"

# The docker executable
DOCKER = "/opt/homebrew/bin/docker"

# The probabilities to use the notes in C-major and A-minor scales.
# R. Hart, “Key-finding algorithm,” 19 Aug 2012.
# Available: https://rnhart.net/articles/key-finding/. [Accessed 5 Apr 2025].

major_profile = [6.35, 3.48, 4.38, 4.09, 5.19, 3.66, 2.88]
cmajor_scale = ['C', 'D', 'E', 'F', 'G', 'A', 'B']

minor_profile = [6.33, 3.52, 5.38, 3.53, 4.75, 3.98, 3.34]
aminor_scale = ['A', 'B', 'C', 'D', 'E', 'F', 'G']

def generate_music_of_day(forecast_date: date, df: pd.DataFrame):
  """
  Generate MIDI files for the given forecast_date.
  Parameters:
  - forecast_date: date object of the forecast date.
  - df: DataFrame containing the music features for the weather of the forecast date.
  """
  date_str = forecast_date.strftime("%Y%m%d")
  df_of_date = df[df['date_time'].str.startswith(date_str)]
  assert len(df_of_date) > 0, f"There is no data for {forecast_date}."

  return _generate_midi_and_combine(date_str, df_of_date)

def generate_music_of_hour(forecast_datetime: datetime, df: pd.DataFrame):
  """
  Generate MIDI files for the given forecast_date.
  Parameters:
  - forecast_date: date object of the forecast date.
  - df: DataFrame containing the music features for the weather of the forecast date.
  """
  datetime_str = forecast_datetime.strftime("%Y%m%d%H")
  df_of_date = df[df['date_time'].str.startswith(datetime_str)]
  assert len(df_of_date) > 0, f"There is no data for {forecast_datetime}."

  return _generate_midi_and_combine(datetime_str, df_of_date)

def create_trigger(length, _music_key, mode, pitch_range):
  """
  Create a trigger, or seed melody, for the time period (e.g. 0 - 6 hours)
  Parameters:
  - length: The number of notes in the trigger.
  - mode: Either 'major' or 'minor'.
  - pitch_range: The pitch range in the unit of semitone where a note pitch is sampled.
  """
  melody = stream.Part()

  ## Create the profile to sample note pitches, based on the known major/minor note probability profile 
  ## and the given pitch range.

  # pitch_range is the number of semi-tones (12 in an octave.)
  chromatic_scale = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
  c_scale_3_octaves = [f"{p}{o}" for o in range(4,7) for p in chromatic_scale]
  if mode == 'minor':
    c_scale_3_octaves = ['A3', 'A#3', 'B3'] + c_scale_3_octaves[:-3]
  adjusted_pitch_range = int(min(pitch_range / 2, len(c_scale_3_octaves)))
  limited_c_scale_3_octaves = c_scale_3_octaves[:adjusted_pitch_range]
  # Remove '#' semi-tones to convert it to the diatonic scale.
  scale = [p for p in limited_c_scale_3_octaves if '#' not in p]

  # Finally, create the profile to sample note pitches.
  profile = major_profile if mode == 'major' else minor_profile
  profile = (profile * 3)[:len(scale)]

  # Create the seed melody by sampling the notes from the created profile.
  # np.random.choice does the sampling according to the given probabilities.
  notes = np.random.choice(scale, size=length-1, p=np.array(profile)/np.sum(profile))
  for n in notes:
    melody.append(note.Note(n))

  # End the trigger with the base note (I) to clarify if it is in the major or minor scale.
  melody.append(note.Note(scale[0]))

  return melody

def _stream_to_midi_list(s):
  return [n.pitch.midi for n in s.notes if n.isNote]

def _combine_midi(midi_files, music_keys, output_path):
  """
  Combine sub-daily-weather-music MIDI files into a daily-weather-music MIDI file.
  Parameters:
  - midi_files: List of MIDI files representing sub-daily weather music pieces.
  - music_keys: List of music key for each MIDI file.
  - output_path: The path of the combined MIDI file.
  """
  merged = stream.Stream()
  offset = 0.0
  REST_DURATION = 4.0
  use_rest = False

  for i, midi_path in enumerate(midi_files):
    if use_rest and i > 0:
      # Insert a full rest to separate the parts.
      r = note.Rest()
      r.quarterLength = REST_DURATION
      r.offset = offset
      merged.append(r)
      offset += REST_DURATION

    s = converter.parse(midi_path)
    part = s.parts[0] # monophonic

    # Insert tempo mark
    tempos = part.flat.getElementsByClass(tempo.MetronomeMark)
    if tempos:
      t = tempos[0]
      t.offset = offset
      merged.insert(t)

    # Insert music key (flats and sharps)
    key_str = music_keys[i]
    if len(key_str) == 2 and key_str[1] == 'b':
      # music21 interprets '-' as flat, not 'b'.
      key_str = f"{key_str[0]}-"

    k = key.Key(key_str)
    print(f"key_str = {key_str} num of sharps = {k.sharps}")
    merged.append(k)

    for elem in part.flat.notesAndRests:
      elem.offset += offset
      merged.append(elem)

    offset = merged.highestTime

  merged.write("midi", fp=output_path)
  return merged, output_path

def _generate_midi_and_combine(date_str, df_of_date):
  """
  Generate weather music of the specified date by using the music features.
  Parameters:
  - date_str: String of the date to generate the weather music.
  - df_of_date: DataFrame containing the sub-daily music features for the specified date.
    (i.e., including 4 rows of 6-hourly music features for 2023-08-12.)
  """
  midi_files = []
  music_keys = []
  modes = []
  for index, row in df_of_date.iterrows():
    midi_file, mode, music_key = _generate_music_for_one_row(row)
    midi_files.append(midi_file)
    music_keys.append(music_key)
    modes.append(mode)

  # Combine created sub-day (e.g. 6-hourly) music into one-day music.
  filename = f"{date_str}_{'_'.join(music_keys)}.mid"
  combined_midi, midi_path = _combine_midi(midi_files, music_keys, f"{OUTPUT_DIR}/{filename}")
  return combined_midi, midi_path

def _generate_music_for_one_row(mf):
  """
  Generate music according to the specified music features expressing weather conditions of a day / hour.
  Parameters:
  - mf: Dict of music features.
  """
  # The number of notes in the seed melody.
  num_notes_per_trigger = 8
  dt = datetime.strptime(mf['date_time'], "%Y%m%d%H")
  # Create the trigger to generate music by LSTM by using mode, pitch range and music key (C-major or A-minor).
  mode = mf['mode_str']
  pitch_range = mf['pitch_range']
  mean_note_duration = mf['mean_note_duration']
  tempo = mf['tempo']
  music_key = "C" if mode == 'major' else "a"
  trigger = create_trigger(num_notes_per_trigger, music_key, mode, pitch_range)

  # Generate the MIDI file using the generated trigger, mean note duration and tempo.
  midi_file = _generate_midi(dt, trigger, mean_note_duration, tempo, music_key)
  return midi_file, mode, music_key

def _generate_midi(forecast_datetime, trigger, mean_note_duration, tempo, key):
  """
  Generate MIDI file for the given music features by using Magenta MelodyRNN (LSTM) program. 
  Note that this function relies on the docker container where the MelodyRNN runs.

  Parameters:
  - forecast_datetime: the datetime object of the forecast date & hour.
  - trigger: the seed melody used as the trigger for the MelodyRNN to generate the following music sequence.
  - mean_note_duration: 1 = quarter note, 0.5 = 1/8 note etc.
  - tempo: Music tempo in integer.
  - key: Music key signature such as 'C' (C major) and 'a' (A minor).
  Returns:
  - target_path: The file path of the generated MIDI file.
  """

  # Convert the trigger, or the seed melody, into the primer melody that MelodyRNN can accept.
  hold_duration = int(mean_note_duration * 4)
  trigger_midi_list = _stream_to_midi_list(trigger)
  primer_melody = [-2] * len(trigger_midi_list) * hold_duration
  for i, note_num in enumerate(trigger_midi_list):
    primer_melody[i * hold_duration] = note_num

  # qpm is the tempo of the music.
  qpm = int(tempo)

  # The hyperparameter to control the randomness of the generated melody. Fix it to the default value.
  temperature = 1.0

  # Run the MelodyRNN in the docker container.
  cmd = [
    DOCKER, "exec", MAGENTA_CONTAINER_NAME,
    "melody_rnn_generate",
    '--config=basic_rnn',
    '--bundle_file=/app/models/basic_rnn.mag',
    '--output_dir=/app/output',
    '--num_outputs=1',
    '--num_steps=128',
    f'--primer_melody={str(primer_melody)}',
    '--condition_on_primer=true',
    '--inject_primer_during_generation=false',
    f'--temperature={temperature}',
    f'--qpm={qpm}'
  ]

  subprocess.run(cmd)

  time.sleep(1)

  # List the MIDI files in the output directory of the docker container.
  mid_files = glob.glob(os.path.join(MAGENTA_OUTPUT_DIR, "*.mid"))
  if not mid_files:
    raise FileNotFoundError("MIDI file was not generated.")

  # Move the last generated MIDI file, which should've come from this run, to the target path.
  # The output file name cannot be specified when running MelodyRNN.
  # NOTE: This code breaks if docker container is executed in parallel.
  latest_mid = max(mid_files, key=os.path.getmtime)
  print("Latest MIDI file", latest_mid)

  new_filename = f"{forecast_datetime.strftime('%Y%m%d%H')}_{key}.mid"
  target_path = os.path.join(OUTPUT_DIR, new_filename)

  if os.path.exists(target_path):
    os.remove(target_path)

  shutil.move(latest_mid, target_path)

  # Return the path to the moved MIDI file.
  return target_path