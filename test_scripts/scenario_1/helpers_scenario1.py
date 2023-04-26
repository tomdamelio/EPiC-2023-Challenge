import glob
import re
import os

import random

import numpy as np
import pandas as pd

import contextlib
import joblib
from joblib import Parallel, delayed

from scipy.signal import resample
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.multioutput import MultiOutputRegressor

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


def create_folder_structure(scenario_folder,):
    # Convert to absolute path
    scenario_folder = os.path.abspath(scenario_folder)
    
    # Join the path
    path = os.path.join(scenario_folder,)

    # Create the preprocessed folder if it doesn't exist
    preprocessed_folder = os.path.join(path, "preprocessed")
    os.makedirs(preprocessed_folder, exist_ok=True)

    # Create the physiology and annotations folders if they don't exist
    phys_folder = os.path.join(preprocessed_folder, 'physiology')
    ann_folder = os.path.join(preprocessed_folder,"annotations")

    os.makedirs(phys_folder, exist_ok=True)
    os.makedirs(ann_folder, exist_ok=True)

    # Return the path
    return phys_folder, ann_folder

def save_files(x, y, file_path, phys_folder, ann_folder):
    subject_num, video_num = map(int, file_path.split('/')[-1].replace('.csv', '').split('_')[1::2])
    
    file_base_name = f'sub_{subject_num}_vid_{video_num}'
    
    np.save(os.path.join(phys_folder, file_base_name), x)
    np.save(os.path.join(ann_folder, file_base_name), y)
    
    return None
    
    
def preprocess(df_physiology, df_annotations, predictions_cols  = 'arousal', aggregate=None, window = [-1000, 500], partition_window = 1):
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
    window_duration : list, optional
        The duration of the sliding window in milliseconds (default is -1000, 500).
    
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
    df_annotations['time'] = pd.to_timedelta(df_annotations['time'], unit='ms')

    X_windows =  sliding_window_with_annotation(df_physiology, df_annotations, start=window[0], end=window[1])
    # print(f'X_windows dimensions: {X_windows.shape}')

    aggregate_local = aggregate.copy() if aggregate is not None else None

    X = np.array([np.array(X_windows[:, :, i].tolist()) for i in range(X_windows.shape[2])]).T
    
    # print('X shape: ', X.shape)
    
    
    def partition_and_aggregate(arr, agg_func, partition_window):
        partition_size = arr.shape[1] // partition_window
        partitions = [arr[:, i * partition_size:(i + 1) * partition_size] for i in range(partition_window)]
        partitions_aggregated = [np.apply_along_axis(agg_func, axis=1, arr=partition) for partition in partitions]
        return np.concatenate(partitions_aggregated, axis=1)

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
            X_agg = partition_and_aggregate(X_windows, agg_funcs[agg], partition_window)
            X_aggregated.append(X_agg)

        if X_aggregated:
            X = np.concatenate(X_aggregated, axis=1)
    
    # print('X shape: ', X.shape)


    y = df_annotations[predictions_cols].values

    numeric_column_indices = [i for i, col_dtype in enumerate(df_physiology.dtypes) if np.issubdtype(col_dtype, np.number)]
    # categorical_column_indices = [i for i, col_dtype in enumerate(df_physiology.dtypes) if not np.issubdtype(col_dtype, np.number)]

    return X, y, #numeric_column_indices #,categorical_column_indices

def process_annotation(arr, timestamps, annotation_time, start, end, window_size):
    window_start_time = max(0, annotation_time + start)
    window_end_time = annotation_time + end

    mask = (timestamps >= window_start_time) & (timestamps <= window_end_time)
    
    window_data = arr[mask, :]
    
    return window_data

def sliding_window_with_annotation(df, df_annotations, start=-1000, end=500):
    df_annotations.set_index('time', inplace=True)
    window_size = abs(end - start) + 1

    # Convert index to integer (milliseconds)
    df.index = (df.index / pd.to_timedelta('1ms')).astype(int)
    df_annotations.index = (df_annotations.index / pd.to_timedelta('1ms')).astype(int)

    # Convert DataFrame to NumPy array
    arr = df.values
    timestamps = df.index.values

    # Initialize the time_adjusted_arr list and max_rows variable
    time_adjusted_arr = []
    max_rows = 0

    # Iterate through the annotations DataFrame
    for _, row in df_annotations.iterrows():
        annotation_time = row.name
        result = process_annotation(arr, timestamps, annotation_time, start, end, window_size)
        max_rows = max(max_rows, result.shape[0])
        time_adjusted_arr.append(result)

    # Pre-allocate the final_time_adjusted_arr with zeros
    final_time_adjusted_arr = np.zeros((len(df_annotations), max_rows, arr.shape[1]))

    # Fill the final_time_adjusted_arr with the data from time_adjusted_arr
    for i, result in enumerate(time_adjusted_arr):
        final_time_adjusted_arr[i, :result.shape[0], :] = result

    # print(f'final_time_adjusted_arr dimensions: {final_time_adjusted_arr.shape}')
    return final_time_adjusted_arr


def _fit_and_evaluate(train_index, test_index, X, y, pipeline):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Calculate RMSE for each output separately
    rmse_per_output = mean_squared_error(y_test, y_pred, squared=False, multioutput='raw_values')
    return rmse_per_output

from sklearn.multioutput import MultiOutputRegressor

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
        # ('cat', categorical_transformer, categorical_column_indices)
    ])

    # Check if y has multiple outputs
    multi_output = y.ndim > 1 and y.shape[1] > 1

    # Wrap the model in a MultiOutputRegressor if needed
    model_instance = model(**hyperparameters)
    if multi_output:
        model_instance = MultiOutputRegressor(model_instance)

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model_instance)
    ])

    # Initialize an empty list to store RMSE values for each output
    rmse_values_per_output = []

    # Parallelize the computation using Joblib
    rmse_values_per_output = Parallel(n_jobs=n_jobs)(
        delayed(_fit_and_evaluate)(train_index, test_index, X, y, pipeline)
        for train_index, test_index in tscv.split(X)
    )

    # Calculate the average RMSE for each output separately
    average_rmse_per_output = np.mean(rmse_values_per_output, axis=0)
    print("Average Root Mean Squared Error per output:", average_rmse_per_output)
    return average_rmse_per_output


# Define the context manager
@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()