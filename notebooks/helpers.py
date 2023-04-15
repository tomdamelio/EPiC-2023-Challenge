import glob
import re

import numpy as np
import pandas as pd

from scipy.signal import resample
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error
from joblib import Parallel, delayed

def load_read_and_append_csvs(folder_path):
    """Read and append df from all CSV files in a folders physiology or annotations. It also adds subject and video numbers as new columns. 
    Use this function to join training and validation data using multiple files.

    Args:
        folder_path (_type_): _description_

    Returns:
        _type_: _description_
    """
    all_files = glob.glob(folder_path + '/*.csv')

    data_frames = []

    for file in all_files:
        # Extract subject and video numbers from the file path
        subject_num, video_num = map(int, file.split('/')[-1].replace('.csv', '').split('_')[1::2])

        # Read the CSV file
        df = pd.read_csv(file)

        # Add subject and video numbers as new columns
        df['subject'] = subject_num
        df['video'] = video_num

        # Append the DataFrame to the list
        data_frames.append(df)

    # Concatenate all DataFrames into a single DataFrame
    combined_df = pd.concat(data_frames, ignore_index=True)

    return combined_df

def zip_csv_files(folder_path_1, folder_path_2):
    """reads all csv files in the folder and returns a list of tuples with corresponding CSV file paths in both folders. Useful to loop over all files in two folders.

    Args:
        folder_path_1 (_type_): _description_
        folder_path_2 (_type_): _description_

    Returns:
        zipped_files: (tuple) list of tuples with corresponding CSV file paths in both folders
    """
    files_1 = glob.glob(folder_path_1 + '/*.csv')
    files_2 = glob.glob(folder_path_2 + '/*.csv')

    # Create a dictionary with keys as (subject_num, video_num) and values as the file path
    files_dict_1 = {(int(s), int(v)): f for f in files_1 for s, v in re.findall(r'sub_(\d+)_vid_(\d+)', f)}
    files_dict_2 = {(int(s), int(v)): f for f in files_2 for s, v in re.findall(r'sub_(\d+)_vid_(\d+)', f)}

    # Create a list of tuples with corresponding CSV file paths in both folders
    zipped_files = [(files_dict_1[key], files_dict_2[key]) for key in files_dict_1 if key in files_dict_2]

    return zipped_files


def sliding_window_with_step(df, window_size, step):
    arr = df.values
    nrows = ((arr.shape[0] - window_size) // step) + 1
    n = arr.strides[0]
    strided_arr = np.lib.stride_tricks.as_strided(
        arr,
        shape=(nrows, window_size, arr.shape[1]),
        strides=(step * n, n, arr.strides[1]),
    )
    return strided_arr

def preprocess(df_physiology, df_annotations, aggregate=None, window_duration=1000, step_duration=20, resample_rate=50):
    """
    Preprocesses the input data for further processing and modeling.
    
    Parameters:
    ----------
    df_physiology : pd.DataFrame
        The input physiological data with columns: time, and physiological signals.
    df_annotations : pd.DataFrame
        The annotations DataFrame with columns: time and arousal.
    aggregate : list of str, optional
        The list of aggregation functions to apply on the input data.
        Available options are: 'mean', 'std', 'enlarged', or any combination of these.
        If None, it will return the 3D matrix as is.
    window_duration : int, optional
        The duration of the sliding window in milliseconds (default is 1000).
    step_duration : int, optional
        The step duration of the sliding window in milliseconds (default is 20).
    resample_rate : int, optional
        The resampling rate in Hz (default is 50).
    
    Returns:
    -------
    X : np.ndarray
        The preprocessed input data (features) as a 2D array.
    y : np.ndarray
        The arousal values corresponding to each window.
    numeric_column_indices : list of int
        The indices of numeric columns in the input data.
    categorical_column_indices : list of int
        The indices of categorical columns in the input data.
    """
    df_physiology['time'] = pd.to_timedelta(df_physiology['time'], unit='ms')
    df_physiology.set_index('time', inplace=True)

    resample_interval = int(1000 / resample_rate)
    df_resampled = df_physiology.resample(f'{resample_interval}L').mean()

    window_size = window_duration // resample_interval
    step = step_duration // resample_interval

    aligned_numeric = sliding_window_with_step(df_resampled, window_size, step)

    X_windows = aligned_numeric[:len(df_annotations)]

def preprocess(df_physiology, df_annotations, aggregate=None, window_duration=1000, step_duration=20, resample_rate=50):
    """
    Preprocesses the input data for further processing and modeling.
    
    Parameters:
    ----------
    df_physiology : pd.DataFrame
        The input physiological data with columns: time, and physiological signals.
    df_annotations : pd.DataFrame
        The annotations DataFrame with columns: time and arousal.
    aggregate : list of str, optional
        The list of aggregation functions to apply on the input data.
        Available options are: 'mean', 'std', 'enlarged', or any combination of these.
        If None, it will return the 3D matrix as is.
    window_duration : int, optional
        The duration of the sliding window in milliseconds (default is 1000).
    step_duration : int, optional
        The step duration of the sliding window in milliseconds (default is 20).
    resample_rate : int, optional
        The resampling rate in Hz (default is 50).
    
    Returns:
    -------
    X : np.ndarray
        The preprocessed input data (features) as a 2D array.
    y : np.ndarray
        The arousal values corresponding to each window.
    numeric_column_indices : list of int
        The indices of numeric columns in the input data.
    categorical_column_indices : list of int
        The indices of categorical columns in the input data.
    """
    df_physiology['time'] = pd.to_timedelta(df_physiology['time'], unit='ms')
    df_physiology.set_index('time', inplace=True)

    resample_interval = int(1000 / resample_rate)
    df_resampled = df_physiology.resample(f'{resample_interval}L').mean()

    window_size = window_duration // resample_interval
    step = step_duration // resample_interval

    aligned_numeric = sliding_window_with_step(df_resampled, window_size, step)

    X_windows = aligned_numeric[:len(df_annotations)]


    aggregate_local = aggregate.copy() if aggregate is not None else None

    X = np.array([np.array(X_windows[:, :, i].tolist()) for i in range(X_windows.shape[2])]).T
    X_aggregated = []

    if aggregate_local is not None:
        if "enlarged" in aggregate_local:
            X_enlarged = np.array([np.array(X_windows[i].flatten().tolist()) for i in range(X_windows.shape[0])])
            X_aggregated.append(X_enlarged)
            aggregate_local.remove("enlarged")

        agg_funcs = {
            "mean": np.mean,
            "std": np.std,
            "max": np.max,
            "min": np.min,
            # Add more aggregation functions here
        }

        for agg in aggregate_local:
            X_agg = np.apply_along_axis(agg_funcs[agg], axis=1, arr=X_windows)
            X_aggregated.append(X_agg)

        if X_aggregated:
            X = np.concatenate(X_aggregated, axis=1)


    y = df_annotations['arousal'].values

    numeric_column_indices = [i for i, col_dtype in enumerate(df_resampled.dtypes) if np.issubdtype(col_dtype, np.number)]
    categorical_column_indices = [i for i, col_dtype in enumerate(df_resampled.dtypes) if not np.issubdtype(col_dtype, np.number)]

    return X, y, numeric_column_indices, categorical_column_indices

    y = df_annotations['arousal'].values

    numeric_column_indices = [i for i, col_dtype in enumerate(df_resampled.dtypes) if np.issubdtype(col_dtype, np.number)]
    categorical_column_indices = [i for i, col_dtype in enumerate(df_resampled.dtypes) if not np.issubdtype(col_dtype, np.number)]

    return X, y, numeric_column_indices, categorical_column_indices



def _fit_and_evaluate(train_index, test_index, X, y, pipeline):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    rmse = mean_squared_error(y_test, y_pred, squared=False)
    return rmse

def time_series_cross_validation_with_hyperparameters(X, y, model, hyperparameters, n_splits=5, n_jobs=-1, numeric_column_indices=None, categorical_column_indices=None):
    """
    Perform time series cross-validation with hyperparameters for a given model.

    Args:
        X (array-like): The feature matrix.
        y (array-like): The target vector.
        model (callable): The model class to be used.
        hyperparameters (dict): The hyperparameters for the model.
        n_splits (int, optional): The number of splits for time series cross-validation. Defaults to 5.
        n_jobs (int, optional): The number of parallel jobs to run. -1 means using all processors. Defaults to -1.
        numeric_column_indices (list, optional): The indices of numeric features. Defaults to None.
        categorical_column_indices (list, optional): The indices of categorical features. Defaults to None.

    Returns:
        float: The average root mean squared error (RMSE) across all splits.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_column_indices),
        ('cat', categorical_transformer, categorical_column_indices)
    ])

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model(**hyperparameters))
    ])

    rmse_values = Parallel(n_jobs=n_jobs)(
        delayed(_fit_and_evaluate)(train_index, test_index, X, y, pipeline)
        for train_index, test_index in tscv.split(X)
    )

    average_rmse = np.mean(rmse_values)
    print("Average Root Mean Squared Error:", average_rmse)
    return average_rmse
