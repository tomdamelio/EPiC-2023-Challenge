#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

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

print(train_physiology.head())

#%%
import neurokit2 as nk
import pandas as pd
import scipy

def preprocess_physiology(data):
    index = data.index

    # Preprocess ECG signal
    ecg_cleaned = nk.ecg_clean(data["ecg"])
    ecg_signals, _ = nk.ecg_process(ecg_cleaned, sampling_rate=1000)

    # Preprocess BVP signal
    bvp_cleaned = nk.ppg_clean(data["bvp"])
    bvp_signals, _ = nk.ppg_process(bvp_cleaned, sampling_rate=1000)

    # Preprocess GSR signal
    gsr_cleaned = nk.eda_clean(data["gsr"])
    gsr_signals, _ = nk.eda_process(gsr_cleaned, sampling_rate=1000)

    # Preprocess RSP signal
    rsp_cleaned = nk.rsp_clean(data["rsp"])
    rsp_signals, _ = nk.rsp_process(rsp_cleaned, sampling_rate=1000)
    
    # Preprocess EMG ZYGO signal
    emg_zygo_cleaned = nk.emg_clean(data["emg_zygo"])
    emg_zygo_signals, _ = nk.emg_process(emg_zygo_cleaned, sampling_rate=1000)
    
    # Preprocess EMG CORU signal
    emg_coru_cleaned = nk.emg_clean(data["emg_coru"])
    emg_coru_signals, _ = nk.emg_process(emg_coru_cleaned, sampling_rate=1000)
    
    # Preprocess EMG TRAP signal
    emg_trap_cleaned = nk.emg_clean(data["emg_trap"])
    emg_trap_signals, _ = nk.emg_process(emg_trap_cleaned, sampling_rate=1000)
    
    # Preprocess Skin Temperature signal
    def low_pass_filter(signal, cutoff_frequency, sampling_rate, order=5):
        nyquist_frequency = 0.5 * sampling_rate
        normalized_cutoff = cutoff_frequency / nyquist_frequency
        b, a = scipy.signal.butter(order, normalized_cutoff, btype="low")
        return scipy.signal.lfilter(b, a, signal)
    
    SKT_filtered = low_pass_filter(data["skt"], cutoff_frequency=1.0, sampling_rate=1000)
    skt_signals = pd.DataFrame({"skt_filtered": SKT_filtered}, index=index)

    # Combine preprocessed signals into one DataFrame
    preprocessed_data = pd.concat([ecg_signals,
                                    bvp_signals,
                                    gsr_signals,
                                    rsp_signals,
                                    emg_zygo_signals,
                                    emg_coru_signals,
                                    emg_trap_signals,
                                    skt_signals], axis=1)

    return preprocessed_data

# Preprocess the data
preprocessed_train_physiology = preprocess_physiology(train_physiology)

# Print the preprocessed data
print(preprocessed_train_physiology.head())
