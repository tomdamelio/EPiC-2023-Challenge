import glob
import re

import random

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
from sklearn.multioutput import MultiOutputRegressor

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

def split_subjects_train_test(subjects, splits):
    """Split subjects into train and test sets.

    Args:
        subjects (_type_): _description_
        splits (_type_): _description_

    Returns:
        splits: list of dictionaries with keys 'train' and 'test' and values as lists of subject numbers.
    """
    def partition (list_in, n):
        random.shuffle(list_in)
        return [list_in[i::n] for i in range(n)]
    
    partitions  = partition(subjects, 3)
    
    splits = []

    for i in partitions:
        train = [x for x in subjects if x not in i]
        test = i
        
        splits.append({'train': train, 'test': test})
        
    return splits



def sliding_window_with_step(df, window_size, step = 0):
    arr = df.values
    nrows = ((arr.shape[0] - window_size) // step) + 1
    n = arr.strides[0]
    strided_arr = np.lib.stride_tricks.as_strided(
        arr,
        shape=(nrows, window_size, arr.shape[1]),
        strides=(step * n, n, arr.strides[1]),
    )
    return strided_arr


def preprocess(df_phys, df_annotations,split, predictions_cols  = 'arousal', aggregate=None, window_duration=1000, step_duration=20, resample_rate=50):
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
    
    df_phys['time'] = pd.to_timedelta(df_phys['time'], unit='ms')
    df_phys.set_index('time', inplace=True)

    train = split['train']
    test = split['test']
    resample_interval = int(1000 / resample_rate)
    
    df_physiology_train = df_phys.query("subject in @train")#.reset_index()
    df_physiology_train = df_physiology_train.resample(f'{resample_interval}L').mean()#
    df_annotations_train = df_annotations.query("subject in @train")
    
    df_physiology_test = df_phys.query("subject in @test")#.reset_index()
    df_physiology_test = df_physiology_test.resample(f'{resample_interval}L').mean()#.reset_index()
    df_annotations_test = df_annotations.query("subject in @test")


    window_size = window_duration // resample_interval
    step = step_duration // resample_interval

    aligned_numeric_train = sliding_window_with_step(df_physiology_train, window_size, step)
    aligned_numeric_test = sliding_window_with_step(df_physiology_test, window_size, step)
    

    X_windows_train = aligned_numeric_train[:len(df_annotations_train)]
    X_windows_test = aligned_numeric_test[:len(df_annotations_test)]
    
    aggregate_local = aggregate.copy() if aggregate is not None else None

    X_train = np.array([np.array(X_windows_train[:, :, i].tolist()) for i in range(X_windows_train.shape[2])]).T
    X_test = np.array([np.array(X_windows_test[:, :, i].tolist()) for i in range(X_windows_test.shape[2])]).T
    
    X_train_aggregated = []
    X_test_aggregated = []
    

    if aggregate_local is not None:
        if "enlarged" in aggregate_local:
            X_train_enlarged = np.array([np.array(X_windows_train[i].flatten().tolist()) for i in range(X_windows_train.shape[0])])
            X_train_aggregated.append(X_train_enlarged)
            
            X_test_enlarged = np.array([np.array(X_windows_test[i].flatten().tolist()) for i in range(X_windows_test.shape[0])])
            X_test_aggregated.append(X_test_enlarged)
            
            aggregate_local.remove("enlarged")

        agg_funcs = {
            "mean": np.mean,
            "std": np.std,
            "max": np.max,
            "min": np.min,
            # Add more aggregation functions here
        }

        for agg in aggregate_local:
            X_train_agg = np.apply_along_axis(agg_funcs[agg], axis=1, arr=X_windows_train)
            X_train_aggregated.append(X_train_agg)
            
            X_test_agg = np.apply_along_axis(agg_funcs[agg], axis=1, arr=X_windows_test)
            X_test_aggregated.append(X_test_agg)

        if X_train_aggregated:
            X_train = np.concatenate(X_train_aggregated, axis=1)
            
            X_test = np.concatenate(X_test_aggregated, axis=1)
            


    y_train = df_annotations_train[predictions_cols].values
    y_test = df_annotations_test[predictions_cols].values

    numeric_column_indices = [i for i, col_dtype in enumerate(df_physiology_train.dtypes) if np.issubdtype(col_dtype, np.number)]
    categorical_column_indices = [i for i, col_dtype in enumerate(df_physiology_train.dtypes) if not np.issubdtype(col_dtype, np.number)]

    return X_train,X_test, y_train, y_test, numeric_column_indices, categorical_column_indices


def _fit_and_evaluate(X_train, X_test, y_train, y_test, pipeline):
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Calculate RMSE for each output separately
    rmse_per_output = mean_squared_error(y_test, y_pred, squared=False, multioutput='raw_values')
    return rmse_per_output



def time_series_cross_validation_with_hyperparameters(X_train, X_test, y_train, y_test, model, hyperparameters, n_jobs=-1, numeric_column_indices=None, categorical_column_indices=None):
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

    # Check if y has multiple outputs
    multi_output = y_train.ndim > 1 and y_train.shape[1] > 1

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

     # Number of instances to run in parallel
    n_instances = 5  # Adjust this value based on the available resources and dataset size
    
    # Parallelize the computation using Joblib
    rmse_values_per_output = Parallel(n_jobs=n_jobs)(
        delayed(_fit_and_evaluate)(X_train, X_test, y_train, y_test, pipeline)
        for _ in range(n_instances)
    )

    # Calculate the average RMSE for each output separately
    average_rmse_per_output = np.mean(rmse_values_per_output, axis=0)
    
    model_name = model.__name__

    print(f"Testing model: {model_name}. Average Root Mean Squared Error per output: {average_rmse_per_output}. ")
    return average_rmse_per_output
