#%%
import glob
import re

import random

import numpy as np
import pandas as pd

from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
import joblib
from joblib import Parallel, delayed, parallel_backend
import multiprocessing
from multiprocessing import Value
from tqdm.auto import tqdm
import itertools

from scipy.signal import resample
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error
from joblib import Parallel, delayed

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from helpers_scenario1 import *

#%%
def zip_csv_train_test_files(folder_phys_train, folder_ann_train, folder_phys_test, folder_ann_test, format = '.csv'):
    """reads all csv or npy files in the folder and returns a list of tuples with corresponding CSV file paths in both folders. Useful to loop over all files in two folders.

    Args:
        folder_path_1 (_type_): _description_
        folder_path_2 (_type_): _description_

    Returns:
        zipped_files: (tuple) list of tuples with corresponding CSV file paths in both folders
    """
    if format == '.csv':
        files_phys_train = glob.glob(folder_phys_train + '/*.csv')
        files_ann_train = glob.glob(folder_ann_train + '/*.csv')
        files_phys_test = glob.glob(folder_phys_test + '/*.csv')
        files_ann_test = glob.glob(folder_ann_test + '/*.csv')

    elif format == '.npy':
        files_phys_train = glob.glob(folder_phys_train + '/*.npy')
        files_ann_train = glob.glob(folder_ann_train + '/*.npy')
        files_phys_test = glob.glob(folder_phys_test + '/*.npy')
        files_ann_test = glob.glob(folder_ann_test + '/*.npy')


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

def time_series_cross_validation_with_hyperparameters(X_train, X_test, y_train, y_test, model, numeric_column_indices=None, categorical_column_indices=None):
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
        # ('cat', categorical_transformer, categorical_column_indices)
    ])

    # Check if y has multiple outputs
    multi_output = y_train.ndim > 1 and y_train.shape[1] > 1

    # Wrap the model in a MultiOutputRegressor if needed
    model_instance = model
    if multi_output:
        model_instance = MultiOutputRegressor(model_instance)

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model_instance)
    ])
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Calculate RMSE for each output separately
    # rmse_per_output = mean_squared_error(y_test, y_pred, squared=False, multioutput='raw_values')

    return y_pred, rmse_per_output

import os

def create_folder_structure(root_physiology_folder, root_annotations_folder, save_output_folder, scenario, fold=None, test=False):
    # Create scenario path
    scenario_str = f"scenario_{scenario}"
    
    # Create fold path if fold is not None
    fold_str = "" if fold is None else f"fold_{fold}"

    # Create paths
    if test:
        phys_folder_train = os.path.join(root_physiology_folder, scenario_str, fold_str, "train", "physiology")
        ann_folder_train = os.path.join(root_annotations_folder, scenario_str, fold_str, "train", "annotations")
        phys_folder_test = os.path.join(root_physiology_folder, scenario_str, fold_str, "test", "physiology")
        ann_folder_test = os.path.join(root_annotations_folder, scenario_str, fold_str, "test", "annotations")
    else:
        phys_folder_train = os.path.join(root_physiology_folder, scenario_str, fold_str, "physiology")
        ann_folder_train = os.path.join(root_annotations_folder, scenario_str, fold_str, "annotations")
        phys_folder_test = None
        ann_folder_test = None

    output_folder = os.path.join(save_output_folder, scenario_str, fold_str)

    # Create directories if they don't exist
    for folder in [phys_folder_train, ann_folder_train, phys_folder_test, ann_folder_test, output_folder]:
        if folder is not None:
            os.makedirs(folder, exist_ok=True)
            
    if test:
        return phys_folder_train, ann_folder_train, phys_folder_test, ann_folder_test, output_folder
    else:
        return phys_folder_train, ann_folder_train, output_folder
#%%config
scenario = 1
fold = None
root_physiology_folder = "../../data/preprocessed/cleaned_and_prepro_improved/"
root_annotations_folder = "../../data/raw/"
save_output_folder = "../../test/"

phys_folder_train, ann_folder_train, phys_folder_test, ann_folder_test, output_folder = create_folder_structure(root_physiology_folder, root_annotations_folder, save_output_folder, scenario, fold, test=True)


zipped_dict = zip_csv_train_test_files(phys_folder_train, ann_folder_train, phys_folder_test, output_folder, format = '.csv')
# print(len(zipped_dict['train']))


# %%
def process_files(annotation_file, physiology_file,):
    df_annotations = pd.read_csv(annotation_file)
    df_physiology = pd.read_csv(physiology_file)
    
    print(physiology_file)
    X, y = preprocess(df_physiology, df_annotations,  predictions_cols=['arousal','valence'], aggregate=['mean','min'], window=[-10000, 5000])
    print(X.shape, y.shape)
    

    
    save_files(X, y, annotation_file, os.path.dirname(physiology_file), os.path.dirname(annotation_file))
    
    return None

# Process the files using the context manager
#%%
for key in zipped_dict.keys():
    with parallel_backend('multiprocessing', n_jobs= multiprocessing.cpu_count()//2):
        with tqdm_joblib(tqdm(total=len(zipped_dict[key]), desc=f"{key} files", leave=False)) as progress_bar:
            results = Parallel()(
                (delayed(process_files)(ann_file, phys_file) for phys_file, ann_file in zipped_dict[key])
            )

#%%
# Define aggregate metric combinations
aggregate_combination = ['mean', 'max']

# Define models and hyperparameters
model= (RandomForestRegressor, {
        'n_estimators': 50,#[50, 100],
        'max_depth': 10,#[10, None],
        'min_samples_split': 2,#[2, 5],
        'min_samples_leaf': 1,
        'max_features': 'auto',
    })


# %%
y_pred, rmse_per_output = time_series_cross_validation_with_hyperparameters(X_train, X_test, y_train, y_test, model, numeric_column_indices=numeric_column_indices, categorical_column_indices=categorical_column_indices)

# %%
