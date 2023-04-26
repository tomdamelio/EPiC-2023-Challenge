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

#%%config
scenario = 1
fold = None
root_physiology_folder = "../../data/preprocessed/cleaned_and_prepro_improved/"
root_annotations_folder = "../../data/raw/"
save_output_folder = "../../test/annotations/"

phys_folder_train, ann_folder_train, phys_folder_test, ann_folder_test, output_folder = create_folder_structure(root_physiology_folder, root_annotations_folder, save_output_folder, scenario, fold, test=True)


zipped_dict = zip_csv_train_test_files(phys_folder_train, ann_folder_train, phys_folder_test, output_folder, format = '.csv')
# print(len(zipped_dict['train']))

def process_files(annotation_file, physiology_file,):
    df_annotations = pd.read_csv(annotation_file)
    df_physiology = pd.read_csv(physiology_file)
    
    print(physiology_file)
    X, y = preprocess(df_physiology, df_annotations,  predictions_cols=['arousal','valence'], aggregate=['mean','min'], window=[-10000, 10000], partition_window = 3)
    print(X.shape, y.shape)
    
    save_files(X, y, annotation_file, os.path.dirname(physiology_file), os.path.dirname(annotation_file))
    
    return None

# Process the files using the context manager
# for key in zipped_dict.keys():
#     with parallel_backend('multiprocessing', n_jobs= multiprocessing.cpu_count()//2):
#         with tqdm_joblib(tqdm(total=len(zipped_dict[key]), desc=f"{key} files", leave=False)) as progress_bar:
#             results = Parallel()(
#                 (delayed(process_files)(ann_file, phys_file) for phys_file, ann_file in zipped_dict[key])
#             )

#%%

# Define models and hyperparameters
random_forest = RandomForestRegressor(
    n_estimators=50, 
    max_depth=10, 
    min_samples_split=2, 
    min_samples_leaf=1, 
    max_features='auto'
)
# %%
zipped_dict_npy = zip_csv_train_test_files(phys_folder_train, ann_folder_train, phys_folder_test, output_folder, format = '.npy')

for i in range(len(zipped_dict['train'])):
    X_train = np.load(zipped_dict_npy['train'][i][0])
    y_train = np.load(zipped_dict_npy['train'][i][1])
    X_test = np.load(zipped_dict_npy['test'][i][0])
    # y_test = np.load(zipped_dict_npy['test'][i][1])
    
    # y_pred, rmse_per_output = time_series_cross_validation_with_hyperparameters(
        #                        X_train, X_test, y_train, y_test,
                                # random_forest, numeric_column_indices=np.array(range(X_train.shape[1])))
# %%
