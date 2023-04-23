
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import tensorflow as tf

from tqdm import tqdm

#%%

clean_folder = "C:\\Users\\xochipilli\\Documents\\EPiC-2023-Challenge\\data\\preprocessed\\cleaned_and_prepro\\scenario_1\\train\\physiology\\"
raw_folder = "C:\\Users\\xochipilli\\Documents\\EPiC-2023-Challenge\\data\\raw\\scenario_1\\train\\annotations\\"

def concat_csv_folder(clean_folder, raw_folder):
    subject_list = sorted([x for x in os.listdir(clean_folder)])
    df = pd.DataFrame({})
    for file in tqdm(subject_list):
        df_physio = pd.read_csv(clean_folder+file)
        df_annot = pd.read_csv(raw_folder+file)
        df_annot["time"] = df_annot["time"] + 10000
        df_merge = df_annot.merge(df_physio, how='right', on="time")

        df = pd.concat([df, df_merge])
    #df = df.fillna(method='ffill')
    return df

df = concat_csv_folder(clean_folder, raw_folder)
df = df.dropna()

#%%
column_indices = {name: i for i, name in enumerate(df.columns)}

n = len(df)
train_df = df[0:int(n*0.7)]
val_df = df[int(n*0.7):int(n*0.9)]
test_df = df[int(n*0.9):]

num_features = df.shape[1]

train_mean = train_df.mean()
train_std = train_df.std()

train_df = (train_df - train_mean) / train_std
val_df = (val_df - train_mean) / train_std
test_df = (test_df - train_mean) / train_std

#%%

# input_width: Pasado
# label_width: Prediccion
# shift: "Futuro"

class WindowGenerator():
  def __init__(self, input_width, label_width, shift,
               train_df=train_df, val_df=val_df, test_df=test_df,
               label_columns=None):
      
    # Store the raw data.
    self.train_df = train_df
    self.val_df = val_df
    self.test_df = test_df

    # Work out the label column indices.
    self.label_columns = label_columns
    if label_columns is not None:
      self.label_columns_indices = {name: i for i, name in
                                    enumerate(label_columns)}
    self.column_indices = {name: i for i, name in
                           enumerate(train_df.columns)}

    # Work out the window parameters.
    self.input_width = input_width
    self.label_width = label_width
    self.shift = shift

    self.total_window_size = input_width + shift

    self.input_slice = slice(0, input_width)
    self.input_indices = np.arange(self.total_window_size)[self.input_slice]

    self.label_start = self.total_window_size - self.label_width
    self.labels_slice = slice(self.label_start, None)
    self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

  def __repr__(self):
    return '\n'.join([
        f'Total window size: {self.total_window_size}',
        f'Input indices: {self.input_indices}',
        f'Label indices: {self.label_indices}',
        f'Label column name(s): {self.label_columns}'])

#%%

def split_window(self, features):
  inputs = features[:, self.input_slice, :]
  labels = features[:, self.labels_slice, :]
  if self.label_columns is not None:
    inputs = tf.stack(
        [inputs[:, :, self.column_indices[name]] for name in self.column_indices
        if name not in self.label_columns], axis=-1)

    labels = tf.stack(
        [labels[:, :, self.column_indices[name]] for name in self.label_columns],
        axis=-1)

  # Slicing doesn't preserve static shape information, so set the shapes
  # manually. This way the `tf.data.Datasets` are easier to inspect.
  inputs.set_shape([None, self.input_width, None])
  labels.set_shape([None, self.label_width, None])

  return inputs, labels

WindowGenerator.split_window = split_window

#%%


# def plot(self, model=None, plot_col='T (degC)', max_subplots=3):
#   inputs, labels = self.example
#   plt.figure(figsize=(12, 8))
#   plot_col_index = self.column_indices[plot_col]
#   max_n = min(max_subplots, len(inputs))
#   for n in range(max_n):
#     plt.subplot(max_n, 1, n+1)
#     plt.ylabel(f'{plot_col} [normed]')
#     plt.plot(self.input_indices, inputs[n, :, plot_col_index],
#              label='Inputs', marker='.', zorder=-10)

#     if self.label_columns:
#       label_col_index = self.label_columns_indices.get(plot_col, None)
#     else:
#       label_col_index = plot_col_index

#     if label_col_index is None:
#       continue

#     plt.scatter(self.label_indices, labels[n, :, label_col_index],
#                 edgecolors='k', label='Labels', c='#2ca02c', s=64)
#     if model is not None:
#       predictions = model(inputs)
#       plt.scatter(self.label_indices, predictions[n, :, label_col_index],
#                   marker='X', edgecolors='k', label='Predictions',
#                   c='#ff7f0e', s=64)

#     if n == 0:
#       plt.legend()

#   plt.xlabel('Time [h]')

# WindowGenerator.plot = plot


#%%

def make_dataset(self, data):
  data = np.array(data, dtype=np.float32)
  ds = tf.keras.utils.timeseries_dataset_from_array(
      data=data,
      targets=None,
      sequence_length=self.total_window_size,
      sequence_stride=1,
      shuffle=True,
      batch_size=32,)

  ds = ds.map(self.split_window)

  return ds

WindowGenerator.make_dataset = make_dataset


#%%

@property
def train(self):
  return self.make_dataset(self.train_df)

@property
def val(self):
  return self.make_dataset(self.val_df)

@property
def test(self):
  return self.make_dataset(self.test_df)


WindowGenerator.train = train
WindowGenerator.val = val
WindowGenerator.test = test

#%%

MAX_EPOCHS = 500
val_performance = {}
performance = {}

def compile_and_fit(model, window, patience=10):
  early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=patience,
                                                    mode='min',
                                                    restore_best_weights=True)

  model.compile(loss=tf.keras.losses.MeanSquaredError(),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=[tf.keras.metrics.RootMeanSquaredError()])

  history = model.fit(window.train, epochs=MAX_EPOCHS,
                      validation_data=window.val,
                      callbacks=[early_stopping])
  return history

#%%

# input_width: Pasado
# label_width: Predicciones
# shift: Futuro

wide_window = WindowGenerator(
    input_width=50, label_width=1, shift=1,
    label_columns=["valence", "arousal"])


#%%


lstm_model = tf.keras.models.Sequential([
    # Shape [batch, time, features] => [batch, time, lstm_units]
    tf.keras.layers.LSTM(32, return_sequences=True),
    
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(units=64, activation='relu'),
    
    # Shape => [batch, time, features]
    tf.keras.layers.Dense(units=2)
])

history = compile_and_fit(lstm_model, wide_window)

val_performance['LSTM'] = lstm_model.evaluate( wide_window.val)
performance['LSTM'] = lstm_model.evaluate( wide_window.test, verbose=0)


#%%

# class ResidualWrapper(tf.keras.Model):
#   def __init__(self, model):
#     super().__init__()
#     self.model = model

#   def call(self, inputs, *args, **kwargs):
#     delta = self.model(inputs, *args, **kwargs)

#     # The prediction for each time step is the input
#     # from the previous time step plus the delta
#     # calculated by the model.
#     return inputs + delta
    
# residual_lstm = ResidualWrapper(
#     tf.keras.Sequential([
#     tf.keras.layers.LSTM(32, return_sequences=True),
#     tf.keras.layers.Dense(units=2,
#         # The predicted deltas should start small.
#         # Therefore, initialize the output layer with zeros.
#         kernel_initializer=tf.initializers.zeros())
# ]))

# history = compile_and_fit(residual_lstm, wide_window)

# val_performance['Residual LSTM'] = residual_lstm.evaluate(wide_window.val)
# performance['Residual LSTM'] = residual_lstm.evaluate(wide_window.test, verbose=0)