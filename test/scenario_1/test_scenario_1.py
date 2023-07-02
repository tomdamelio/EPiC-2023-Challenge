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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsRegressor
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from helpers_scenario1 import *

scenario = 1
fold = None
root_physiology_folder = "../../data/preprocessed/"
root_annotations_folder = "../../data/raw/"
save_output_folder = "../../results/"

phys_folder_train, ann_folder_train, phys_folder_test, ann_folder_test, output_folder, = create_folder_structure(
    root_physiology_folder, root_annotations_folder, save_output_folder, scenario, fold, test=True)

zipped_dict = zip_csv_train_test_files(phys_folder_train, ann_folder_train, phys_folder_test, ann_folder_test, format='.csv')

def process_files(annotation_file, physiology_file,):
    df_annotations = pd.read_csv(annotation_file)
    df_physiology = pd.read_csv(physiology_file)

    X, y = preprocess(df_physiology, df_annotations,  predictions_cols=['arousal','valence'], aggregate=['mean','min'], window=[-3000, 3000], partition_window=3)
    save_files(X, y, annotation_file, os.path.dirname(physiology_file), os.path.dirname(annotation_file))

# Process the files using the context manager
for key in zipped_dict.keys():
    with parallel_backend('loky', n_jobs= multiprocessing.cpu_count()//2):
        with tqdm_joblib(tqdm(total=len(zipped_dict[key]), desc=f"{key} files", leave=False)) as progress_bar:
            results = Parallel()(
                (delayed(process_files)(ann_file, phys_file) for phys_file, ann_file in zipped_dict[key])
            )
            

column_names_all = ['ecg_cleaned_mean_p1', 'rr_signal_mean_p1', 'bvp_cleaned_mean_p1', 'gsr_cleaned_mean_p1', 'gsr_tonic_mean_p1', 'gsr_phasic_mean_p1', 'gsr_SMNA_mean_p1', 'rsp_cleaned_mean_p1', 'resp_rate_mean_p1', 'emg_zygo_cleaned_mean_p1', 'emg_coru_cleaned_mean_p1', 'emg_trap_cleaned_mean_p1', 'skt_filtered_mean_p1',
'ecg_cleaned_min_p1', 'rr_signal_min_p1', 'bvp_cleaned_min_p1', 'gsr_cleaned_min_p1', 'gsr_tonic_min_p1', 'gsr_phasic_min_p1', 'gsr_SMNA_min_p1', 'rsp_cleaned_min_p1', 'resp_rate_min_p1', 'emg_zygo_cleaned_min_p1', 'emg_coru_cleaned_min_p1', 'emg_trap_cleaned_min_p1', 'skt_filtered_min_p1',

'ecg_cleaned_mean_p2', 'rr_signal_mean_p2', 'bvp_cleaned_mean_p2', 'gsr_cleaned_mean_p2', 'gsr_tonic_mean_p2', 'gsr_phasic_mean_p2', 'gsr_SMNA_mean_p2', 'rsp_cleaned_mean_p2', 'resp_rate_mean_p2', 'emg_zygo_cleaned_mean_p2', 'emg_coru_cleaned_mean_p2', 'emg_trap_cleaned_mean_p2', 'skt_filtered_mean_p2',
'ecg_cleaned_min_p2', 'rr_signal_min_p2', 'bvp_cleaned_min_p2', 'gs`gsr_cleaned_min_p2', 'gsr_tonic_min_p2', 'gsr_phasic_min_p2', 'gsr_SMNA_min_p2', 'rsp_cleaned_min_p2', 'resp_rate_min_p2', 'emg_zygo_cleaned_min_p2', 'emg_coru_cleaned_min_p2', 'emg_trap_cleaned_min_p2', 'skt_filtered_min_p2',

'ecg_cleaned_mean_p3', 'rr_signal_mean_p3', 'bvp_cleaned_mean_p3', 'gsr_cleaned_mean_p3', 'gsr_tonic_mean_p3', 'gsr_phasic_mean_p3', 'gsr_SMNA_mean_p3', 'rsp_cleaned_mean_p3', 'resp_rate_mean_p3', 'emg_zygo_cleaned_mean_p3', 'emg_coru_cleaned_mean_p3', 'emg_trap_cleaned_mean_p3', 'skt_filtered_mean_p3',
'ecg_cleaned_min_p3', 'rr_signal_min_p3', 'bvp_cleaned_min_p3', 'gsr_cleaned_min_p3', 'gsr_tonic_min_p3', 'gsr_phasic_min_p3', 'gsr_SMNA_min_p3', 'rsp_cleaned_min_p3', 'resp_rate_min_p3', 'emg_zygo_cleaned_min_p3', 'emg_coru_cleaned_min_p3', 'emg_trap_cleaned_min_p3', 'skt_filtered_min_p3']

def run_experiment(top_features=None):
    random_forest = RandomForestRegressor(
    n_estimators=50,
    max_depth=10,
    min_samples_split=2,
    min_samples_leaf=1,
    )
    
    zipped_dict_npy = zip_csv_train_test_files(phys_folder_train, ann_folder_train, phys_folder_test, ann_folder_test, format='.npy')

    def test_function(i, top_features=None):
        # X = np.load(zipped_dict_npy['train'][i][0])
        # y = np.load(zipped_dict_npy['train'][i][1])

        
        X_train = np.load(zipped_dict_npy['train'][i][0])
        y_train = np.load(zipped_dict_npy['train'][i][1])
        X_test = np.load(zipped_dict_npy['test'][i][0])
        y_test = np.load(zipped_dict_npy['test'][i][1])
        

        
        X_train_df = pd.DataFrame(X_train, columns=column_names_all)
        X_test_df = pd.DataFrame(X_test, columns=column_names_all)
        

        if top_features is not None:
            X_train_filtered = X_train_df[top_features.index].values
            X_test_filtered = X_test_df[top_features.index].values

        else:
            X_train_filtered = X_train_df.values
            X_test_filtered = X_test_df.values

            # X_train, X_test, y_train, y_test = train_test_split(X_filtered, y, test_size=0.2, shuffle=False)
        

        y_pred, importances = time_series_cross_validation_with_hyperparameters(
            X_train_filtered, X_test_filtered, y_train, y_test,
            random_forest, numeric_column_indices=np.array(range(X_train_filtered.shape[1])), test=True)

        if top_features is not None:
            save_test_data(y_pred, output_folder, zipped_dict_npy['test'][i][1], test=True)
        return (0,0), importances

    num_cpu_cores = multiprocessing.cpu_count()
    all_results = []
    all_importances_list = []

    with parallel_backend('loky', n_jobs=num_cpu_cores - 5):
        with tqdm_joblib(tqdm(total=len(zipped_dict['train']), desc="Files", leave=False)) as progress_bar:
            results = Parallel()(
                (delayed(test_function)(i, top_features) for i in range(len(zipped_dict['train'])))
            )
        for i in range(len(zipped_dict['train'])):
            all_results.append(results[i][0])
            all_importances_list.append(results[i][1])

    all_importances = pd.concat(all_importances_list, ignore_index=True)

    if top_features is not None:
        column_names = top_features.index
    else:
        column_names = column_names_all
    
    all_importances.columns = column_names

    df_results = pd.DataFrame(all_results, columns=['arousal', 'valence'])
    
    # create directory if it doesn't exist
    results_dir = '../../results/scenario_1'
    os.makedirs(results_dir, exist_ok=True)
    
    df_results.to_csv(os.path.join('../../results/scenario_1', 'results_rf.csv'), index=False)

    return all_importances, df_results

all_importances, df_results = run_experiment(top_features=None) # Pass in a DataFrame with top features or None

def plot_and_save_top_features(all_importances, n_features, plot=False):
    mean_importances = all_importances.mean(axis=0)
    std_importances = all_importances.std(axis=0)
    top_features = mean_importances.nlargest(n_features)
    top_std = std_importances[top_features.index]
    
    return top_features

def rmse_improved(n_features):
    top_features_n = plot_and_save_top_features(all_importances, n_features=n_features, plot=False)
    all_importances_n, df_results_n = run_experiment(top_features=top_features_n)
    df_results_melted = df_results.melt(value_name='RMSE', var_name='Dimension', ignore_index=False)
    df_results_melted['Dataset'] = 'All Features'

    df_results_n_melted = df_results_n.melt(value_name='RMSE', var_name='Dimension', ignore_index=False)
    df_results_n_melted['Dataset'] = f'Top {n_features} Features'

    combined_results = pd.concat([df_results_melted, df_results_n_melted], ignore_index=True)

    mean_rmse_all = df_results.mean(axis=0)
    mean_rmse_top_n = df_results_n.mean(axis=0)

    comparison_df = pd.DataFrame({'All Features': mean_rmse_all, f'Top {n_features} Features': mean_rmse_top_n})

    return all_importances_n, df_results_n, comparison_df

n_features = 5
all_importances_n, df_results_n, comparison_df = rmse_improved(n_features)
print(f"Comparison of Mean RMSE for All Features vs Top {n_features} Features:")
print(comparison_df)

# %%
