import os
import pandas as pd
import neurokit2 as nk
import scipy
from pathlib import Path

def preprocess_physiology(data):
    index = data.index

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
    preprocessed_data = pd.concat([ecg_signals,
                                    bvp_signals,
                                    gsr_signals,
                                    rsp_signals,
                                    emg_zygo_signals,
                                    emg_coru_signals,
                                    emg_trap_signals,
                                    skt_signals], axis=1)

    return preprocessed_data

def process_files_in_physiology(src_path, dst_path, test_mode=False):
    processed_files = 0
    outer_loop_break = False  # Add this flag variable to break the outer loop
    for root, _, files in os.walk(src_path):
        if os.path.basename(root) == "physiology":
            for file in files:
                if file.endswith(".csv"):
                    file_path = os.path.join(root, file)
                    data = pd.read_csv(file_path, index_col="time")
                    preprocessed_data = preprocess_physiology(data)
                    preprocessed_file_path = file_path.replace(src_path, dst_path)
                    os.makedirs(os.path.dirname(preprocessed_file_path), exist_ok=True)
                    preprocessed_data.to_csv(preprocessed_file_path, index=True)

                    if test_mode:
                        processed_files += 1
                        if processed_files == 1:
                            outer_loop_break = True  # Set the flag to True when you want to break the outer loop
                            break
            if outer_loop_break:  # Check the flag in the outer loop, and break if it's True
                break

source_dir = "../data/raw"
destination_dir = "../data/preprocessed/cleaned"
test_mode = False  # Set this to False if you want to run the script for all participants


process_files_in_physiology(source_dir, destination_dir, test_mode)


