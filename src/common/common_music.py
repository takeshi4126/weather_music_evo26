# common_music.py
# Music related common functions.
# @author Takeshi Matsumura

from music21 import note, chord
import math
import pandas as pd
import numpy as np

dur_notes = ["whole","half", "quarter", "eighth"]

notes_per_chord = 4

def join_pitches(element):
  """
  Here it is assumed that element is either Note or Chord.
  """
  assert isinstance(element, chord.Chord)
  return '.'.join(str(n) for n in element.pitches)

def _element_to_chord(element):
  """
  Convert an element in midi data to a chord. 
  If the element is a Note, it will return a Chord object containing the single Note.
  Watch out that the element can be something other than a Note or a Chord.
  """
  if isinstance(element, note.Note):
    return element.pitch
  elif isinstance(element, chord.Chord):
    return join_pitches(element)
  else:
    return None

def _element_to_duration(element):
  """
  Convert an element in midi data to a chord. 
  If the element is a Note, it will return a Chord object containing the single Note.
  Watch out that the element can be something other than a Note or a Chord.
  """
  if isinstance(element, note.Note) or isinstance(element, chord.Chord):
    return element.duration.quarterLength
  else:
    return None

def extract_key_chords_durations(midi):
  """
  Extract key, chords and durations from the given music score.
  """
  chords = [_element_to_chord(element) for element in midi]
  chords = [c for c in chords if c is not None]
  durations = [_element_to_duration(element) for element in midi]
  durations = [d for d in durations if d is not None]
  key = str(midi.analyze('key'))

  return key, chords, durations

def get_note_duration(sd):
  """
  Get the note duration for the given spectra duration
  """
  if sd > 2:
    figure = dur_notes[0]
  elif sd > 1:
    figure = dur_notes[1]
  elif sd > 0.5:
    figure = dur_notes[2]
  else:
    figure = dur_notes[3]
  return figure

# Map key to the position on the circle of 5th (clockwise)

circle5th_major_keys = ["Gb", "Db", "Ab", "Eb", "Bb", "F", "C", "G", "D", "A", "E", "B"]
circle5th_minor_keys = ["eb", "bb", "f", "c", "g", "d", "a", "e", "b", "f#", "c#", "g#"]

circle5th_major_keys_2 = ["G-", "D-", "A-", "E-", "B-", "F", "C", "G", "D", "A", "E", "B"]
circle5th_minor_keys_2 = ["E-", "B-", "F", "C", "G", "D", "A", "E", "B", "F#", "C#", "G#"]
enharmonic = {
  "Cb": "B",
  "F#": "Gb",
  "C#": "Db",
  "d#": "eb"
}

enharmonic_major_keys_2 = {
  "C-": "B",
  "F#": "G-",
  "C#": "D-"
}

enharmonic_minor_keys_2 = {
  "D#": "E-",
  "A#": "B-",
  "A-": "G#"
}

def circle5th_index(k:str):
  """
  Get the index of the given music key on the circle of 5th .
  Parameters:
  - k: music key
  Returns:
  - index: [0, ..., 11] indicating the clockwise index on the circle of 5th (C = 0, F = 11)
  """
  # Convert some keys of different names
  k = enharmonic[k] if k in enharmonic else k
  if k in circle5th_major_keys:
    return circle5th_major_keys.index(k)
  elif k in circle5th_minor_keys:
    return circle5th_minor_keys.index(k)
  else:
    raise Exception(f"Key {k} is not on the circle of 5th. This must be a bug.")

def circle5th_index_2(k:str, mode:str):
  """
  Get the index of the given music key on the circle of 5th .
  Parameters:
  - k: music key
  Returns:
  - index: [0, ..., 11] indicating the clockwise index on the circle of 5th (C = 0, F = 11)
  """
  # Convert some keys of different names
  if mode == 'major':
    k = enharmonic_major_keys_2[k] if k in enharmonic_major_keys_2 else k
    if k in circle5th_major_keys_2:
      return circle5th_major_keys_2.index(k)  
  elif mode == 'minor':
    k = enharmonic_minor_keys_2[k] if k in enharmonic_minor_keys_2 else k
    if k in circle5th_minor_keys_2:
      return circle5th_minor_keys_2.index(k)
  raise Exception(f"Key {k} is not on the circle of 5th. This must be a bug.")

def circle5th_key(index:int, mode:str):
  """
  Get the key for the given index on the circle of fifths.
  Parameters:
  - index: [0, ..., 11] indicating the clockwise index on the circle of 5th (C = 0, F = 11)
  Returns:
  - k: music key
  """
  if index < 0 or index > 11:
    raise Exception(f"The circle of fifths index {index} is out of range of [1, 11].")
  if mode == 'major':
    return circle5th_major_keys[index]
  elif mode == 'minor':
    return circle5th_minor_keys[index]
  else:
    raise Exception(f"Mode {mode} is unknown. It must be either 'major' or 'minor'.")

def circle5th_position(circle5th_index):
  """
  Get the (x, y) position for the given circle5th index [0, 11].
  Returns:
  - x: [-1, 1]
  - y: [-1, 1]
  """
  rad = math.radians(circle5th_index * 30)
  return math.cos(rad), math.sin(rad)

def encode_circle5th(k:str):
  """
  Encode the circle of fifths string into 2D cartesian coordinates.
  """
  return circle5th_position(circle5th_index(k))

def decode_circle5th(circle5th_pos:tuple, mode:str):
  """
  Decode the 2D cartesian coordinates to the circle of fifths string.
  """
  x, y = circle5th_pos
  c5index = int(math.degrees(math.atan2(y, x)) / 30 % 12)
  if mode == 'major':
    return circle5th_major_keys[c5index]
  elif mode == 'minor':
    return circle5th_minor_keys[c5index]
  else:
    raise f"Wrong mode {mode} was given to decode_circle5th."

key_index_prefix = 'c5_index'

def onehot_encoding(df):
  df = pd.get_dummies(df, columns=[key_index_prefix], dtype=float)
  return df

def onehot_decoding(df):
  # Recreate c5_index from the one-hot encoding
  df[key_index_prefix] = df.filter(like=key_index_prefix+"_").idxmax(axis=1).str.replace(key_index_prefix+"_", "").astype(int)
  df = df.drop(columns=df.filter(like=key_index_prefix+"_").columns)
  return df

def circular_encoding(df):
  # Nothing to do. c5_index is already in [0, 11] range.
  return df

def circular_decoding(df):
  # Put the c5_index in the [0, 11] range.
  df[key_index_prefix] = df[key_index_prefix] % 12
  return df

def linear_encoding(df):      
  # Scaling of c5_index from [0, 11] to [-1, 1]
  df[key_index_prefix] = (df[key_index_prefix] - 6) / 6
  return df

def linear_decoding(df):      
  # Unscaling of num_accidentals from [-1, 1] to [0, 11]
  df[key_index_prefix] = round(df[key_index_prefix] * 6 + 6, 0)
  return df

def accidentals_encoding(df):      
  # Scaling of num_accidentals from [0, 5] to [0, 1]
  if 'num_accidentals' in df:
    df['num_accidentals'] = df['num_accidentals'] / 5
  return df

def accidentals_decoding(df):      
  # Unscaling of num_accidentals from [0, 1] to [0, 5]
  if 'num_accidentals' in df:
    df['num_accidentals'] = np.floor(df['num_accidentals'] * 5)
  return df

def music_key_encoding(df, key_encoding: str):
  """
  Encode the music key column.
  """
  if key_encoding == 'onehot':
    return onehot_encoding(df)
  if key_encoding == 'circular':
    return circular_encoding(df)
  if key_encoding == 'linear':
    return linear_encoding(df)
  if key_encoding == 'accidentals':
    return accidentals_encoding(df)

def music_key_decoding(df, key_encoding: str):
  """
  Decode the music key column.
  """
  if key_encoding == 'onehot':
    return onehot_decoding(df)
  if key_encoding == 'circular':
    return circular_decoding(df)
  if key_encoding == 'linear':
    return linear_decoding(df)
  if key_encoding == 'accidentals':
    return accidentals_decoding(df)
