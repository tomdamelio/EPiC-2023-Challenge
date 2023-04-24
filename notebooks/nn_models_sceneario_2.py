#%%
import glob
import re
import json
import itertools
import pandas as pd
from pathlib import Path
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Activation, Flatten, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

from tqdm import tqdm

# all the functions from helpers.py
from helpers_scenario2 import *

#%%
### CAMBIAR ESTO A LO QUE APARECE EN |BASELINE_MODELS_SCENARIO2|,
#  UNA VEZ QUE YA HAYA HECHO TODAS LAS PRUEBAS ###

# UNCOMMENT
#annotations_folder = '../data/raw/scenario_4_test/fold_0/train/annotations/'
annotations_folder = '../data/raw/scenario_4/fold_0/train/annotations/'

physiology_folder = "../data/preprocessed/cleaned_and_prepro_improved/scenario_4/fold_0/train/physiology/" #'../data/raw/scenario_1/train/physiology/'data\preprocessed\

#UNCOMMENT
df_physiology = load_read_and_append_csvs(physiology_folder)
df_annotations = load_read_and_append_csvs(annotations_folder)

videos = df_physiology.video.unique()
subjects = df_physiology.subject.unique()
#UNCOMMENT
#splits = split_videos_train_test(videos, 3)

#%%
# Realizar un left join en las columnas 'time', 'subject' y 'video'
df= pd.merge(df_annotations, df_physiology, on=['time', 'subject', 'video'], how='left')

# Verificar el tamaño del DataFrame resultante
print("Shape of merged_df:", df.shape)
# %%
del df_annotations
del df_physiology

#%%
# Subsetear para quedarme unicamente con las filas de un mismo video
df = df.loc[df.video == 16]

# %%
# Preprocess the data
physio_signals = ['ecg_cleaned', 'rr_signal', 'bvp_cleaned', 'gsr_cleaned', 'gsr_tonic', 'gsr_phasic', 'gsr_SMNA', 'rsp_cleaned', 'resp_rate', 'emg_zygo_cleaned', 'emg_coru_cleaned', 'emg_trap_cleaned', 'skt_filtered']
df_physio = df[physio_signals]

# Standardize the data
scaler = StandardScaler()
scaled_physio = scaler.fit_transform(df_physio)

#%%
del df_physio

#%%

# Prepare the data for training and testing
# Get unique subject IDs
unique_subjects = df['subject'].unique()

# Shuffle and split subjects into train and test sets (80/20)
np.random.seed(42)  # For reproducibility
shuffled_subjects = shuffle(unique_subjects)
train_subjects = shuffled_subjects[:int(0.8 * len(shuffled_subjects))]
test_subjects = shuffled_subjects[int(0.8 * len(shuffled_subjects)):]

# Filter the data based on train and test subjects
X = scaled_physio
y_valence = df['valence'].values
y_arousal = df['arousal'].values

train_mask = df['subject'].isin(train_subjects)
test_mask = df['subject'].isin(test_subjects)

X_train, y_valence_train, y_arousal_train = X[train_mask], y_valence[train_mask], y_arousal[train_mask]
X_test, y_valence_test, y_arousal_test = X[test_mask], y_valence[test_mask], y_arousal[test_mask]

#%%
# Create the CNN model
def create_cnn_model(input_shape):
    input_signal = Input(shape=input_shape)

    x = Conv1D(24, 5, padding='same')(input_signal)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv1D(16, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv1D(8, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Flatten()(x)
    f_peripheral = Dense(128, activation='relu')(x)

    return Model(inputs=input_signal, outputs=f_peripheral)

input_shape = (X_train.shape[1], 1)
cnn_model = create_cnn_model(input_shape)

# Create separate heads for valence and arousal prediction
valence_output = Dense(1, activation='linear', name='valence_output')(cnn_model.output)
arousal_output = Dense(1, activation='linear', name='arousal_output')(cnn_model.output)

# Combine the model
final_model = Model(inputs=cnn_model.input, outputs=[valence_output, arousal_output])

# Compile the model
final_model.compile(optimizer='adam',
                    loss={'valence_output': 'mse',
                    'arousal_output': 'mse'},
                    metrics=[tf.keras.metrics.RootMeanSquaredError()])

# Train the model
history = final_model.fit(X_train, {'valence_output': y_valence_train, 'arousal_output': y_arousal_train}, validation_split=0.2, epochs=50, batch_size=32)

#%%
print(final_model.evaluate(X_test, [y_valence_test, y_arousal_test]))


#%%
import math

# Predict arousal values for the test set
y_arousal_pred = final_model.predict(X_test)[1]

# Create a new dataframe with subject IDs, stimulus IDs, actual arousal values, and predicted arousal values
df_results = pd.DataFrame({'subject': df.loc[test_mask, 'subject'].values,
                           'video': df.loc[test_mask, 'video'].values,
                           'time': df.loc[test_mask, 'time'].values,
                           'actual_arousal': y_arousal_test,
                           'predicted_arousal': y_arousal_pred.flatten()})

# Plot actual vs predicted arousal time series for each test subject and stimulus
for subject in test_subjects:
    df_subject = df_results[df_results['subject'] == subject]
    unique_videos = df_subject['video'].unique()
    
    # Calculate the number of rows and columns for subplots
    ncols = 4
    nrows = math.ceil(len(unique_videos) / ncols)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, nrows * 4), sharex=True, sharey=True)
    fig.suptitle(f'Subject {subject}: Actual vs Predicted Arousal', fontsize=16)

    if nrows == 1:
        axes = [axes]

    for i, video in enumerate(unique_videos):
        df_video = df_subject[df_subject['video'] == video]

        row, col = divmod(i, ncols)
        ax = axes[row][col]
        
        ax.plot(df_video['time'], df_video['actual_arousal'], label='Actual Arousal')
        ax.plot(df_video['time'], df_video['predicted_arousal'], label='Predicted Arousal')
        ax.set_title(f'Video {video}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Arousal')
    
    # Remove empty subplots
    for i in range(len(unique_videos), nrows * ncols):
        row, col = divmod(i, ncols)
        fig.delaxes(axes[row][col])

    # Display the legend on the first subplot
    axes[0][0].legend()
    plt.tight_layout()
    plt.show()
# %%
