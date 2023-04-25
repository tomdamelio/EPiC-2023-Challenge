#%%
import os
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import interpolate, signal
import neurokit2 as nk
from biosppy.signals import ecg


#%%
# Copy dir and subdir structure from `../data/raw``data to `../raq/preprocessed``
#def replicate_dir_structure(src, dst):
#    for root, dirs, _ in os.walk(src):
#        for directory in dirs:
#            src_dir = os.path.join(root, directory)
#            dst_dir = os.path.join(dst, os.path.relpath(src_dir, src))
#            if not os.path.exists(dst_dir):
#                os.makedirs(dst_dir)
#
#source_dir = "../data/raw"
#destination_dir = "../data/preprocessed"
#replicate_dir_structure(source_dir, destination_dir)
#
#print(train_physiology.head())

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

#%%
import biosppy
import numpy as np  
from hrvanalysis import remove_outliers
from hrvanalysis import remove_ectopic_beats
from hrvanalysis import interpolate_nan_values
from matplotlib import pyplot as plt
from scipy import interpolate
from biosppy.signals import ecg


def preprocess_physiology(data):
    index = data.index

    # Preprocess ECG signal
    fs= 1000
    
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

        rr[np.where(np.isnan(rr))] = 0
        
        return np.array(rr)


    rr, peaks = process_ecg(data.ecg, fs)
    
    from scipy import interpolate

    # Interpolate RR
    interpf = interpolate.interp1d(peaks/fs, rr)
    timestamp = np.linspace(min(peaks/fs), max(peaks/fs),
                            int((max(peaks/fs) - min(
                                peaks/fs)) * fs))
    signal = interpf(timestamp)
    
    rr_signals =  pd.DataFrame({"rr_signal": rr_signal}, index=index)
    
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
    preprocessed_data = pd.concat([ecg_signals,
                                    rr_signals,
                                    bvp_signals,
                                    gsr_signals,
                                    rsp_signals,
                                    emg_zygo_signals,
                                    emg_coru_signals,
                                    emg_trap_signals,
                                    skt_signals], axis=1)

    return preprocessed_data

#%%
data = preprocess_physiology(train_physiology)



#%%
# Create a new figure
fig, ax = plt.subplots(9, 1, figsize=(15, 20), sharex=True)

# Plot all 8 physiological signals
for idx, signal in enumerate(data.columns):
    ax[idx].plot(data[signal], label=signal)
    ax[idx].set_ylabel(signal)
    ax[idx].legend()

# Plot the annotations timeseries
ax[8].plot(train_annotations, label="Annotations")
ax[8].set_ylabel("Annotations")
ax[8].legend()

# Set the xlabel for the last subplot
ax[8].set_xlabel("Time")

# Show the plot
plt.show()
#%%
def process_files_in_physiology(src_path, dst_path):
    for root, _, files in os.walk(src_path):
        if os.path.basename(root) == "physiology":
            processed_files = 0
            for file in files:
                if file.endswith(".csv"):
                    file_path = os.path.join(root, file)
                    data = pd.read_csv(file_path)
                    preprocessed_data = preprocess_physiology(data)
                    preprocessed_file_path = file_path.replace(src_path, dst_path)
                    #os.makedirs(os.path.dirname(preprocessed_file_path), exist_ok=True)
                    #print(preprocessed_file_path)
                    #preprocessed_data.to_csv(preprocessed_file_path, index=False)

                    # Increment the counter and break the loop if 2 files have been processed
                    #processed_files += 1
                    #if processed_files == 2:
                    #    break

#source_dir = "../data/raw"
#destination_dir = "../data/preprocessed/clean"

data [¿ process_files_in_physiology(source_dir, destination_dir)

#%%
# PRUEBA DE PREPROCESSING PARA GENERAR CLEAND DATA MEJOR

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

#%%

