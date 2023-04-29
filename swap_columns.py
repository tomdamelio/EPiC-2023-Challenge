#%%
import pandas as pd
import os
import glob

# set where are the results with the columns to swap
original_output_folder = "./results/"
# where you want them to be saved. If same folder will be overwritten
swaped_output_folder = "./results_swaped_columns/"


def swap_columns_in_csv(file_name, col_name1, col_name2, save_directory):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_name)

    # Check if the provided column names exist in the DataFrame
    if col_name1 not in df.columns or col_name2 not in df.columns:
        print(f"Error: Column names '{col_name1}' or '{col_name2}' not found in the CSV file.")
        return

    # Exchange the data in the two columns without changing column names
    df[col_name1], df[col_name2] = df[col_name2].copy(), df[col_name1].copy()

    file_to_save = os.path.join(save_directory, os.path.basename(file_name))
    # Save the modified DataFrame to a new CSV file
    df.to_csv(file_to_save, index=False)
    # print(f"Data in columns '{col_name1}' and '{col_name2}' has been exchanged and saved to 'swapped_columns.csv'.")

    
def create_folder_path(original_output_folder, swaped_output_folder, scenario, fold = None):
    scenario_str = f"scenario_{scenario}"

    # Create fold path if fold is not None
    fold_str = "" if fold is None else f"fold_{fold}"


    original_folder  = os.path.join(original_output_folder, scenario_str, fold_str, 'test','annotations')
    swaped_folder = os.path.join(swaped_output_folder, scenario_str, fold_str, 'test','annotations')
    
    # Create directories if they don't exist
    for folder in [original_folder, swaped_folder]:
        if folder is not None:
            os.makedirs(folder, exist_ok=True)

    return original_folder, swaped_folder

scenarios = [1,2,3,4]
folds = {1: [None], 2: [0,1,2,3, 4], 3: [0,1,2,3], 4: [0,1,]}


for scenario in scenarios:
    for fold in folds[scenario]:
        original_folder, swaped_folder = create_folder_path(original_output_folder, swaped_output_folder, scenario, fold = fold)
        for file in glob.glob(original_folder + '/*.csv'):
            swap_columns_in_csv(file, 'valence', 'arousal', swaped_folder)




# %%
