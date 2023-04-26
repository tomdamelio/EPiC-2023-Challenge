#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from biosppy.signals import ecg
import os
import pandas as pd
import neurokit2 as nk
import scipy
from scipy import interpolate


#%%

def load_train_data(sub, video, scenario):
    # create file name based on parameters
    file_name = f"sub_{sub}_vid_{video}.csv"

    # load data files
    train_physiology = pd.read_csv(Path(f"../data/raw/scenario_{scenario}/train/physiology", file_name), index_col="time")
    train_annotations = pd.read_csv(Path(f"../data/raw/scenario_{scenario}/train/annotations", file_name), index_col="time")

    return train_physiology, train_annotations

sub = 9
video = 1
scenario = 1

train_physiology, train_annotations = load_train_data(sub, video, scenario)

#%%
import biosppy
import numpy as np  
from hrvanalysis import remove_outliers
from hrvanalysis import remove_ectopic_beats
from hrvanalysis import interpolate_nan_values
from matplotlib import pyplot as plt
import neurokit2 as nk

def process_ecg(signal, fs):
    """Get NN interval from ecg signal."""
    _, _, rpeaks, _, _, _, _ = ecg.ecg(signal,
                                       sampling_rate=fs,
                                       show=False)
    rr = get_nn(rpeaks, fs)

    return rr, rpeaks

def get_nn(peaks, fs):
    """Convert beat peaks in samples to NN intervals and timestamp."""
    rr = np.diff(peaks, prepend=0) * 1000 / fs

    # This remove outliers from signal
    rr = remove_outliers(rr, low_rri=300, high_rri=2000, verbose=False)
    # This replace outliers nan values with linear interpolation
    rr = interpolate_nan_values(rr, interpolation_method="linear")

    # This remove ectopic beats from signal
    # TODO: esto puede no tener sentido en PPG, pero los metodos de features
    #  estan basados en NN y no en RR.
    rr = remove_ectopic_beats(rr, method="malik", verbose=False)
    # This replace ectopic beats nan values with linear interpolation
    rr = np.array(interpolate_nan_values(rr))

    # rr[np.where(np.isnan(rr))] = 0  # Remove this line
    
    return np.array(rr)


#%%

def preprocess_physiology(data):
    index = data.index
    
    fs = 1000 # 1khz 
    rr, peaks = process_ecg(data["ecg"], fs)
    
    interpf = interpolate.interp1d(peaks/fs, rr, bounds_error=False, fill_value=np.nan)
    timestamp = np.linspace(0, len(data["ecg"]) / fs, len(data["ecg"]))

    rr_signal = interpf(timestamp)
    rr_signals = pd.DataFrame({"rr_cleaned": rr_signal}, index=timestamp * fs)

    # Preprocess ECG signal
    ecg_cleaned = nk.ecg_clean(data["ecg"])
    ecg_signals = pd.DataFrame({"ecg_cleaned": ecg_cleaned}, index=index)

    # Preprocess BVP signal
    bvp_cleaned = nk.ppg_clean(data["bvp"])
    bvp_signals = pd.DataFrame({"bvp_cleaned": bvp_cleaned}, index=index)

    # Preprocess GSR signal
    gsr_cleaned = nk.eda_clean(data["gsr"], method='BioSPPy')
    gsr_signals = pd.DataFrame({"gsr_cleaned": gsr_cleaned}, index=index)

    # Preprocess RSP signal
    rsp_cleaned = nk.rsp_clean(data["rsp"])
    rsp_signals = pd.DataFrame({"rsp_cleaned": rsp_cleaned}, index=index)

    # Preprocess EMG ZYGO signal
    emg_zygo_cleaned = nk.emg_clean(data["emg_zygo"])        
    emg_zygo_signals = pd.DataFrame({"emg_zygo_cleaned": emg_zygo_cleaned}, index=index)

    # Preprocess EMG CORU signal
    emg_coru_cleaned = nk.emg_clean(data["emg_coru"])
    emg_coru_signals = pd.DataFrame({"emg_coru_cleaned": emg_coru_cleaned}, index=index)

    # Preprocess EMG TRAP signal
    emg_trap_cleaned = nk.emg_clean(data["emg_trap"])
    emg_trap_signals = pd.DataFrame({"emg_trap_cleaned": emg_trap_cleaned}, index=index)

    # Preprocess Skin Temperature signal
    def low_pass_filter(signal, cutoff_frequency, sampling_rate, order=5):
        nyquist_frequency = 0.5 * sampling_rate
        normalized_cutoff = cutoff_frequency / nyquist_frequency
        b, a = scipy.signal.butter(order, normalized_cutoff, btype="low")
        return scipy.signal.lfilter(b, a, signal)
    
    SKT_filtered = low_pass_filter(data["skt"], cutoff_frequency=1.0, sampling_rate=1000)
    skt_signals = pd.DataFrame({"skt_filtered": SKT_filtered}, index=index)

    # Combine preprocessed signals into one DataFrame
    preprocessed_data = data.copy()
    preprocessed_data.update(ecg_signals)
    preprocessed_data.update(bvp_signals)
    preprocessed_data.update(gsr_signals)
    preprocessed_data.update(rsp_signals)
    preprocessed_data.update(emg_zygo_signals)
    preprocessed_data.update(emg_coru_signals)
    preprocessed_data.update(emg_trap_signals)
    preprocessed_data.update(skt_signals)

    # Add the rr_cleaned column to the DataFrame
    preprocessed_data["rr_cleaned"] = rr_signals["rr_cleaned"]

    return preprocessed_data

# %%

data = preprocess_physiology(train_physiology)
data.head()
# %%
# Create a new figure
fig, ax = plt.subplots(10, 1, figsize=(15, 20), sharex=True)

# Plot all 8 physiological signals
for idx, signal in enumerate(data.columns):
    ax[idx].plot(data[signal], label=signal)
    ax[idx].set_ylabel(signal)
    ax[idx].legend()

# Plot the annotations timeseries
ax[9].plot(train_annotations, label="Annotations")
ax[9].set_ylabel("Annotations")
ax[9].legend()

# Set the xlabel for the last subplot
ax[9].set_xlabel("Time")

# Show the plot
plt.show()
# %%

