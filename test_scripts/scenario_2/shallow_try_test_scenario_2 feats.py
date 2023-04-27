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

from helpers_scenario2 import *

#%%config

scenario = 2
folds = [0,1,2,3,4]
for fold in folds[3:]:
    print(fold)
    root_physiology_folder = "../../data/preprocessed/cleaned_and_prepro_improved/"
    root_annotations_folder = "../../data/raw/"
    # save_output_folder = "../../test/annotations/"
    save_output_folder = "../../results/test/"


    phys_folder_train, ann_folder_train, phys_folder_test, ann_folder_test, output_folder, = create_folder_structure(
        root_physiology_folder, root_annotations_folder, save_output_folder, scenario, fold, test=True)


    zipped_dict = zip_csv_train_test_files(phys_folder_train, ann_folder_train, phys_folder_test, ann_folder_test, format = '.csv')
    # print(len(zipped_dict['train']))

    subjects, videos = get_subs_vids(phys_folder_train)

    splits = splits = split_subjects_train_test(subjects, 3)

    def process_files(annotation_file, physiology_file,):
        df_annotations = pd.read_csv(annotation_file)
        df_physiology = pd.read_csv(physiology_file)
        
        # print(physiology_file)
        X, y = preprocess(df_physiology, df_annotations,  predictions_cols=['arousal','valence'], aggregate=['mean','min'], window=[-5000, 5000], partition_window = 3)
        # print(X.shape, y.shape)
        
        save_files(X, y, annotation_file, os.path.dirname(physiology_file), os.path.dirname(annotation_file))
        
        return None


    # # Process the files using the context manager
    # for key in zipped_dict.keys():
    #     with parallel_backend('multiprocessing', n_jobs= multiprocessing.cpu_count()//2):
    #         with tqdm_joblib(tqdm(total=len(zipped_dict[key]), desc=f"{key} files", leave=False)) as progress_bar:
    #             results = Parallel()(
    #                 (delayed(process_files)(ann_file, phys_file) for phys_file, ann_file in zipped_dict[key])
    #             )


#%%
# Define models and hyperparameters
xgb = XGBRegressor(
    n_estimators=50, 
    max_depth=6, 
    learning_rate = 0.1, 
)

xgb_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=10)),
    ('xgb', MultiOutputRegressor(xgb))
])



num_cpu_cores = multiprocessing.cpu_count()
def evaluate_features(vid):
    X_train  = load_and_concatenate_train(phys_folder_train, vid =vid, split=splits[1])
    y_train = load_and_concatenate_train(ann_folder_train, vid =vid, split=splits[1])
    
    xgb_pipeline.fit(X_train, y_train)
    importances = xgb_pipeline.named_steps['xgb'].estimators_[0].feature_importances_
    
    return importances

all_importances = []
with parallel_backend('multiprocessing', n_jobs=  num_cpu_cores - 5):
    with tqdm_joblib(tqdm(total=len(videos), desc="Files", leave=False)) as progress_bar:
        results = Parallel()(
            (delayed(test_function)(i) for i in videos))
        
    # Combine results for all subjects
    for i in range(len(videos)):
        all_importances.append(results[i][1])

df_importances = pd.DataFrame(all_importances)

### Promediar importancias
### Top 10 o lo que sea
### Obtener numero de mejores columnas

### Trimear en test function por esas columnas X_train y X_test

#%%




def test_function(vid):
    X_train  = load_and_concatenate_train(phys_folder_train, vid =vid, split=splits[1])
    y_train = load_and_concatenate_train(ann_folder_train, vid =vid, split=splits[1])
    
    xgb_pipeline.fit(X_train, y_train)
    importances = xgb_pipeline.named_steps['xgb'].estimators_[0].feature_importances_
    

    ##### INSERT FEAT SELECTION HERE #####
    
    rmse_subject = []
    importances_subject = []
    for sub in splits[1]['test']:
        X_test = np.load(os.path.join(phys_folder_train, f"sub_{sub}_vid_{vid}.npy"))
        y_test = np.load(os.path.join(ann_folder_train, f"sub_{sub}_vid_{vid}.npy"))
        
        y_pred = xgb_pipeline.predict(X_test)
        y_pred_filtered = gaussian_filter_multi_output(y_pred, 80)
        y_pred_filtered = low_pass_filter(y_pred_filtered, 2, 20,6)
        
        
        rmse_per_output = mean_squared_error(y_test, y_pred_filtered, squared=False, multioutput='raw_values')
        
        path_csv_test =  os.path.join(ann_folder_train, f"sub_{sub}_vid_{vid}.csv")

        save_test_data(y_pred_filtered, output_folder, path_csv_test, test = False, y_test = y_test)
        
        rmse_subject.append(rmse_per_output)
        importances_subject.append(importances)
    
    return np.mean(rmse_subject, axis = 0), np.mean(importances_subject, axis=0)
        
        
    
    
    # knn_pipeline.fit(X_train, y_train)
    # knn_output_train = knn_pipeline.predict(X_train)
    # knn_output_test = knn_pipeline.predict(X_test)

    # X_train_knn = np.column_stack((knn_output_train, X_train))
    # X_test_knn = np.column_stack(( knn_output_test, X_test))
    
    # print(rmse_per_output)
    save_test_data(y_pred, output_folder, subject, test = False, y_test = y_test)    
    return rmse_per_output, importances

    save_test_data(y_pred, output_folder, zipped_dict_npy['test'][i][1])


#%%
all_results = []
all_importances = []
with parallel_backend('multiprocessing', n_jobs=  num_cpu_cores - 5):
    with tqdm_joblib(tqdm(total=len(videos), desc="Files", leave=False)) as progress_bar:
        results = Parallel()(
            (delayed(test_function)(i) for i in videos))
        
    # Combine results for all subjects
    for i in range(len(videos)):
        all_results.append(results[i][0])
        all_importances.append(results[i][1])

df_results = pd.DataFrame(all_results, columns=['arousal', 'valence'])
df_results.to_csv(os.path.join('../../results/scenario_2', 'results_xgb_pca_filter.csv'), index=False)

display(df_results.describe())
pd.DataFrame(all_importances).describe()
# %%
for i in videos:
    test_function(i, xgb_pipeline)
    

# %%
    
    