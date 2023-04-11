#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import re

#%%

def load_data(file_path):
    # Load data from the file
    data = pd.read_csv(file_path)

    # Extract the physiological signals and self-reported emotion columns
    column_names = [
        'bvp', 'ecg', 'emg_coru', 'emg_trap', 'emg_zygo',
        'gsr', 'rsp', 'skt', 'valence', 'arousal'
    ]
    signals = data[column_names]

    # Create an MNE info object with the appropriate channel types
    ch_types = ['misc', 'ecg', 'emg', 'emg', 'emg', 'misc', 'misc', 'misc', 'misc', 'misc']
    ch_names = column_names
    sfreq = 1000  # Sampling frequency is 1000 Hz
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

    # Create an MNE Raw object from the extracted data
    raw_data = mne.io.RawArray(signals.T, info)

    return raw_data

#%%

def load_train_data(sub, video, scenario):
    # create file name based on parameters
    file_name = f"sub_{sub}_vid_{video}.csv"

    # load data files
    train_physiology = pd.read_csv(Path(f"../data/raw/scenario_{scenario}/train/physiology", file_name), index_col="time")
    train_annotations = pd.read_csv(Path(f"../data/raw/scenario_{scenario}/train/annotations", file_name), index_col="time")

    return train_physiology, train_annotations

sub = 1
video = 1
scenario = 1

train_physiology, train_annotations = load_train_data(sub, video, scenario)

print(train_annotations.head())

#%%
# PREPROCESSING MANTAINING CONTINUOUS DATA

# SEGUIR DESDE ACA -> HACER CORRER ESTE ESCRIPT PARA OBTENER LAS SEÑALES PERFIERICAS 
# PREPROCESADAS. LUEGO, GENERAR SCRIPT PARA OBTENER FEATURES DE LAS SEÑALES PROCESADAS
# SEGUN VENTANAS DE TIEMPO ELEGIDAS (HABLAR PRIMERO CON NICO Y CON ENZO)
"""
the input DataFrame should have columns for each physiological signal.
The output DataFrame will have the same length as the input time series and
contain continuous features extracted from the physiological signals.
You can further use these continuous features to predict arousal and valence
in your machine learning model.
"""

import pandas as pd
import numpy as np
import biosppy.signals as bio_signals
import neurokit2 as nk
from biosppy.signals import ecg as bio_ecg
from biosppy import qrs_detect

def process_ecg(ecg_signal, sampling_rate):
    out = qrs_detect(ecg_signal, sampling_rate=sampling_rate)
    processed_ecg = nk.ecg_process(ecg_signal, sampling_rate=sampling_rate, rpeaks=out['Rpeaks'])
    features = {'ECG_RR_Interval': processed_ecg['ECG_RR_Interval'],
                'ECG_QRS_Duration': processed_ecg['ECG_QRS_Duration']}
    return features

def process_bvp(bvp_signal, sampling_rate):
    out = bio_signals.bvp(bvp_signal, sampling_rate=sampling_rate)
    processed_bvp = nk.ppg_process(bvp_signal, sampling_rate=sampling_rate, peaks=out['peaks'])
    features = {'PPG_Peaks_Amplitude': processed_bvp['PPG_Peaks_Amplitude']}
    return features

def process_gsr(gsr_signal, sampling_rate):
    processed_gsr = nk.eda_process(gsr_signal, sampling_rate=sampling_rate)
    features = {'EDA_Phasic': processed_gsr['EDA_Phasic']}
    return features

def process_rsp(rsp_signal, sampling_rate):
    processed_rsp = nk.rsp_process(rsp_signal, sampling_rate=sampling_rate)
    features = {'RSP_Rate': processed_rsp['RSP_Rate']}
    return features

def process_emg(emg_signal, sampling_rate):
    processed_emg = nk.emg_process(emg_signal, sampling_rate=sampling_rate)
    features = {'EMG_Amplitude': processed_emg['EMG_Amplitude']}
    return features

def process_skt(skt_signal, sampling_rate):
    # No specific processing needed for skin temperature, return the signal as-is
    features = {'SKT': skt_signal}
    return features

def preprocess_physiological_signals(df, sampling_rate):
    features = {}
    features.update(process_ecg(df['ecg'], sampling_rate))
    features.update(process_bvp(df['bvp'], sampling_rate))
    features.update(process_gsr(df['gsr'], sampling_rate))
    features.update(process_rsp(df['rsp'], sampling_rate))
    features.update(process_emg(df['emg_coru'], sampling_rate))
    features.update(process_emg(df['emg_trap'], sampling_rate))
    features.update(process_emg(df['emg_zygo'], sampling_rate))
    features.update(process_skt(df['skt'], sampling_rate))
    return pd.DataFrame(features)

# Define the sampling rate of your data
sampling_rate = 1000  # Modify this value according to your dataset

# Preprocess the physiological signals and extract relevant features
features_df = preprocess_physiological_signals(train_physiology, sampling_rate)



# %%
# PREPROCESSING AND FEATURE EXTRACTION
import pandas as pd
import numpy as np
import biosppy.signals as bio_signals
import neurokit2 as nk

def process_ecg(ecg_signal, sampling_rate):
    out = bio_signals.qrs_detect(ecg_signal, sampling_rate=sampling_rate)
    processed_ecg = nk.ecg_process(ecg_signal, sampling_rate=sampling_rate, rpeaks=out['Rpeaks'])
    features = nk.ecg_intervalrelated(processed_ecg)
    features['ecg_heart_rate_variability'] = np.std(processed_ecg['ECG_RR_Interval'])
    features['ecg_qrs_duration'] = np.mean(processed_ecg['ECG_QRS_Duration'])
    features['ecg_tpeaks'] = len(processed_ecg['ECG_Tpeaks'])
    return features

def process_bvp(bvp_signal, sampling_rate):
    out = bio_signals.bvp(bvp_signal, sampling_rate=sampling_rate)
    processed_bvp = nk.ppg_process(bvp_signal, sampling_rate=sampling_rate, peaks=out['peaks'])
    features = nk.ppg_intervalrelated(processed_bvp)
    features['bvp_peak_count'] = len(out['peaks'])
    features['bvp_peak_amplitudes'] = np.mean(processed_bvp['PPG_Peaks_Amplitude'])
    return features

def process_gsr(gsr_signal, sampling_rate):
    processed_gsr = nk.eda_process(gsr_signal, sampling_rate=sampling_rate)
    features = nk.eda_phasic(processed_gsr, sampling_rate=sampling_rate)
    features['gsr_peak_count'] = len(processed_gsr['EDA_Peaks'])
    features['gsr_mean_amplitude'] = np.mean(processed_gsr['EDA_Peaks_Amplitude'])
    features['gsr_rise_time'] = np.mean(processed_gsr['EDA_Rise_Time'])
    return features

def process_rsp(rsp_signal, sampling_rate):
    processed_rsp = nk.rsp_process(rsp_signal, sampling_rate=sampling_rate)
    features = nk.rsp_intervalrelated(processed_rsp)
    features['rsp_inspiration_duration'] = np.mean(processed_rsp['RSP_Inspiration_Duration'])
    features['rsp_expiration_duration'] = np.mean(processed_rsp['RSP_Expiration_Duration'])
    features['rsp_respiratory_rate'] = np.mean(processed_rsp['RSP_Rate'])
    return features

def process_emg(emg_signal, sampling_rate):
    processed_emg = nk.emg_process(emg_signal, sampling_rate=sampling_rate)
    features = {}
    features['emg_mean_amplitude'] = np.mean(processed_emg['EMG_Amplitude'])
    features['emg_std_amplitude'] = np.std(processed_emg['EMG_Amplitude'])
    features['emg_mean_frequency'] = np.mean(processed_emg['EMG_Frequency'])
    features['emg_median_frequency'] = np.median(processed_emg['EMG_Frequency'])
    return features

def process_skt(skt_signal, sampling_rate):
    features = {}
    features['skt_mean'] = np.mean(skt_signal)
    features['skt_std'] = np.std(skt_signal)
    features['skt_max'] = np.max(skt_signal)
    features['skt_min'] = np.min(skt_signal)
    return features

def preprocess_physiological_signals(df, sampling_rate):
    features = {}
    features.update(process_ecg(df['ecg'], sampling_rate))
    features.update(process_bvp(df['bvp'], sampling_rate))
    features.update(process_gsr(df['gsr'], sampling_rate))
    features.update(process_rsp(df['rsp'], sampling_rate))
    features.update(process_emg(df['emg_coru'], sampling_rate))
    features.update(process_emg(df['emg_trap'], sampling_rate))
    features.update(process_emg(df['emg_zygo'], sampling_rate))
    features.update(process_skt(df['skt']))
    return features

# Define the sampling rate of your data
sampling_rate = 1000  # Modify this value according to your dataset

# Preprocess the physiological signals and extract relevant features
features = preprocess_physiological_signals(train_physiology, sampling_rate)

# Convert the extracted features into a DataFrame
features_df = pd.DataFrame(features)
