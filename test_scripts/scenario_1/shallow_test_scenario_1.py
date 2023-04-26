#%%
import os
import numpy as np
import pandas as pd

from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
import joblib
from joblib import Parallel, delayed, parallel_backend
import multiprocessing
from multiprocessing import Value
from tqdm.auto import tqdm


from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from helpers_scenario1 import *

#%%config
scenario = 1
fold = None
root_physiology_folder = "../../data/preprocessed/cleaned_and_prepro_improved/"
root_annotations_folder = "../../data/raw/"
# save_output_folder = "../../test/annotations/"
save_output_folder = "../../results/test/scenario_1/annotations/"


phys_folder_train, ann_folder_train, phys_folder_test, ann_folder_test, output_folder, = create_folder_structure(
    root_physiology_folder, root_annotations_folder, save_output_folder, scenario, fold, test=True)


zipped_dict = zip_csv_train_test_files(phys_folder_train, ann_folder_train, phys_folder_test, ann_folder_test, format = '.csv')
# print(len(zipped_dict['train']))

def process_files(annotation_file, physiology_file,):
    df_annotations = pd.read_csv(annotation_file)
    df_physiology = pd.read_csv(physiology_file)
    
    # print(physiology_file)
    X, y = preprocess(df_physiology, df_annotations,  predictions_cols=['arousal','valence'], aggregate=['mean','min'], window=[-10000, 10000], partition_window = 3)
    # print(X.shape, y.shape)
    
    save_files(X, y, annotation_file, os.path.dirname(physiology_file), os.path.dirname(annotation_file))
    
    return None

#%%
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
    # max_features='auto'
)

zipped_dict_npy = zip_csv_train_test_files(phys_folder_train, ann_folder_train, phys_folder_test, ann_folder_test, format = '.npy')


def test_function(i):
    # print(zipped_dict_npy['train'][i][0])
    result_list = []
    X = np.load(zipped_dict_npy['train'][i][0])
    y = np.load(zipped_dict_npy['train'][i][1])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle = False)
    
    y_pred, rmse_per_output = time_series_cross_validation_with_hyperparameters(
                               X_train, X_test, y_train, y_test,
                                random_forest, numeric_column_indices=np.array(range(X_train.shape[1])), test= False)
    
    # print(rmse_per_output)
    result_list.append(rmse_per_output)
    save_test_data(y_pred, output_folder, zipped_dict_npy['train'][i][1], test = False, y_test = y_test)    
    return results

# def test_function(i):
#     X_train = np.load(zipped_dict_npy['train'][i][0])
#     y_train = np.load(zipped_dict_npy['train'][i][1])
#     X_test = np.load(zipped_dict_npy['test'][i][0])
#     y_test = np.load(zipped_dict_npy['test'][i][1])
    
#     y_pred = time_series_cross_validation_with_hyperparameters(
#                                X_train, X_test, y_train, y_test,
#                                 random_forest, numeric_column_indices=np.array(range(X_train.shape[1])))
    
    
    save_test_data(y_pred, output_folder, zipped_dict_npy['test'][i][1])
    
num_cpu_cores = multiprocessing.cpu_count()
with parallel_backend('multiprocessing', n_jobs=  num_cpu_cores - 10):
    with tqdm_joblib(tqdm(total=len(zipped_dict['train']), desc="Files", leave=False)) as progress_bar:
        rmse_results = Parallel()(
            (delayed(test_function)(i) for i in range(len(zipped_dict['train'])))
        )

# %%
# for i in range(len(zipped_dict['train'])):
#     test_function(i)

# # %%
