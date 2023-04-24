#%%
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
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error
from joblib import Parallel, delayed



#%%
def zip_csv_train_test_files(folder_phys_train, folder_ann_train, folder_phys_test, folder_ann_test):
    """reads all csv files in the folder and returns a list of tuples with corresponding CSV file paths in both folders. Useful to loop over all files in two folders.

    Args:
        folder_path_1 (_type_): _description_
        folder_path_2 (_type_): _description_

    Returns:
        zipped_files: (tuple) list of tuples with corresponding CSV file paths in both folders
    """
    files_phys_train = glob.glob(folder_phys_train + '/*.csv')
    files_ann_train = glob.glob(folder_ann_train + '/*.csv')
    files_phys_test = glob.glob(folder_phys_test + '/*.csv')
    files_ann_test = glob.glob(folder_ann_test + '/*.csv')


    # Create a dictionary with keys as (subject_num, video_num) and values as the file path
    files_dict_phys_train = {(int(s), int(v)): f for f in files_phys_train for s, v in re.findall(r'sub_(\d+)_vid_(\d+)', f)}
    files_dict_ann_train = {(int(s), int(v)): f for f in files_ann_train for s, v in re.findall(r'sub_(\d+)_vid_(\d+)', f)}
    
    files_dict_phys_test = {(int(s), int(v)): f for f in files_phys_test for s, v in re.findall(r'sub_(\d+)_vid_(\d+)', f)}
    files_dict_ann_test = {(int(s), int(v)): f for f in files_ann_test for s, v in re.findall(r'sub_(\d+)_vid_(\d+)', f)}

    # Create a list of tuples with corresponding CSV file paths in both folders
    zipped_files_train = [(files_dict_phys_train[key], files_dict_ann_train[key]) for key in files_dict_phys_train if key in files_dict_ann_train]
    zipped_files_test = [(files_dict_phys_test[key], files_dict_ann_test[key]) for key in files_dict_phys_test if key in files_dict_ann_test]
    
    zipped_dict = {'train': zipped_files_train, 'test': zipped_files_test}
    

    return zipped_dict

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


def preprocess(df_physiology, df_annotations, predictions_cols  = 'arousal', aggregate=None, window_duration=1000, step_duration=20,):
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

    window_size = window_duration 
    step = step_duration

    aligned_numeric = sliding_window_with_step(df_physiology, window_size, step)

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


    y = df_annotations[predictions_cols].values

    numeric_column_indices = [i for i, col_dtype in enumerate(df_physiology.dtypes) if np.issubdtype(col_dtype, np.number)]
    categorical_column_indices = [i for i, col_dtype in enumerate(df_physiology.dtypes) if not np.issubdtype(col_dtype, np.number)]

    return X, y, numeric_column_indices, categorical_column_indices


def time_series_cross_validation_with_hyperparameters(X_train, X_test, y_train, y_test, model, hyperparameters, numeric_column_indices=None, categorical_column_indices=None):
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
    multi_output = y.ndim > 1 and y.shape[1] > 1

    # Wrap the model in a MultiOutputRegressor if needed
    model_instance = model(**hyperparameters)
    if multi_output:
        model_instance = MultiOutputRegressor(model_instance)

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model_instance)
    ])


    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Calculate RMSE for each output separately
    rmse_per_output = mean_squared_error(y_test, y_pred, squared=False, multioutput='raw_values')

    return y_pred, rmse_per_output


#%%
path_phys_train = '../data/preprocessed/cleaned_and_prepro_improved/scenario_1/train/physiology'
path_ann_train = '../data/raw/scenario_1/train/annotations'

path_phys_test = '../data/preprocessed/cleaned_and_prepro_improved/scenario_1/test/physiology'
path_ann_test = '../data/raw/scenario_1/test/annotations'

zipped_dict = zip_csv_train_test_files(path_phys_train, path_ann_train, path_phys_test, path_ann_test)

# %%
# Define aggregate metric combinations
aggregate_combinations = [
    # ['enlarged'],
    # ['mean'],
    # ['std'],
    # ['max'],
    ['min'],
    # ['mean', 'std'],
    # ['mean', 'max'],
    # ['mean', 'min'],
    # ['std', 'max'],
    # ['std', 'min'],
    # ['max', 'min'],
    ['mean', 'std', 'max', 'min']
]

# Define models and hyperparameters
models_hyperparameters = [
    # (LinearRegression, {}),
    # (SVR, {
    #     'kernel': ['linear', 'rbf'],
    #     'C': [0.1, 1, 10],
    #     'epsilon': [0.1, 1],
    #     'gamma': ['scale', 'auto'],  # Only used for 'rbf' kernel
    # }),
    (RandomForestRegressor, {
        'n_estimators': [50, 100],
        'max_depth': [10, None],
        'min_samples_split': [2, 5],
        # 'min_samples_leaf': [1, 2],
        # 'max_features': ['auto', 'sqrt'],
    }),
    (XGBRegressor, {
        'n_estimators': [50, 100],
        'max_depth': [6, 10],
    #     'learning_rate': [0.01, 0.1],
    #     'subsample': [0.5, 0.8],
        # 'colsample_bytree': [0.5, 0.8],
        # 'reg_alpha': [0, 0.1],
        # 'reg_lambda': [0.1, 1],
    }),
]
# %%
for i in range(len(zipped_dict['train'])):
    df_phys_train = pd.read_csv(zipped_dict['train'][i][0])
    df_ann_train = pd.read_csv(zipped_dict['train'][i][1])
    
    df_phys_test = pd.read_csv(zipped_dict['test'][i][0])
    df_ann_test = pd.read_csv(zipped_dict['test'][i][1])
    
    X_train, y_train, numeric_column_indices, categorical_column_indices =  preprocess(df_phys_train, df_ann_train, predictions_cols  = ['arousal', 'valence'], aggregate=None, window_duration=1000)
    X_test, y_test, numeric_column_indices, categorical_column_indices =  preprocess(df_phys_test, df_ann_test, predictions_cols  = ['arousal', 'valence'], aggregate=None, window_duration=1000)
    
    y_pred, rmse_per_output = time_series_cross_validation_with_hyperparameters(X_train, X_test, y_train, y_test, model, hyperparameters, numeric_column_indices=None, categorical_column_indices=None)
    
    
# %%
