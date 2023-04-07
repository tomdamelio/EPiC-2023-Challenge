import mne
import pandas as pd

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