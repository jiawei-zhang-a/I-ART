import numpy as np
import os

def read_npz_files(directory, small_size=False, multiple=False, type="original"):
    # Initialize variables to sum the results across files
    total_rejections = {
        'median': 0,
        'LR': 0,
        'lightgbm': 0,
        'xgboost': 0,
        'oracle': 0
    }

    counts = {
        'median': 0,
        'LR': 0,
        'lightgbm': 0,
        'xgboost': 0,
        'oracle': 0
    }

    for filename in os.listdir(directory):
        if filename.endswith(".npy"):
            filepath = os.path.join(directory, filename)
            # Load the dictionary from the .npy file
            data = np.load(filepath, allow_pickle=True).item()
            
            # Extract the p-values from the dictionary
            reject = data.get('reject', False)

            # Tally the rejections for each model type
            if "results_median" in filename:
                counts['median'] += 1
                total_rejections['median'] += reject
            elif "results_LR" in filename:
                counts['LR'] += 1
                total_rejections['LR'] += reject
            elif "results_lightgbm" in filename:
                counts['lightgbm'] += 1
                total_rejections['lightgbm'] += reject
            elif "results_xgboost" in filename:
                counts['xgboost'] += 1
                total_rejections['xgboost'] += reject
            elif "results_oracle" in filename:
                counts['oracle'] += 1
                total_rejections['oracle'] += reject

    # Calculate rejection rates (power) for each model
    results = {}
    for key in total_rejections:
        if counts[key] > 0:
            results[f"{key}_power"] = total_rejections[key] / counts[key]
        else:
            results[f"{key}_power"] = -1  # If no results were processed for this model

    return results


import numpy as np
import os

def read_npz_files_time(directory, small_size=False, multiple=False, type="original"):
    # Initialize variables to sum the results across files
    total_rejections = {
        'median': 0,
        'LR': 0,
        'lightgbm': 0,
        'xgboost': 0,
        'oracle': 0
    }

    counts = {
        'median': 0,
        'LR': 0,
        'lightgbm': 0,
        'xgboost': 0,
        'oracle': 0
    }

    for filename in os.listdir(directory):
        if filename.endswith(".npy"):
            filepath = os.path.join(directory, filename)
            # Load the dictionary from the .npy file
            data = np.load(filepath, allow_pickle=True).item()
            
            # Extract the p-values from the dictionary
            elapsed_time = data.get('elapsed_time', False)

            # Tally the rejections for each model type
            if "results_median" in filename:
                counts['median'] += 1
                total_rejections['median'] += elapsed_time
            elif "results_LR" in filename:
                counts['LR'] += 1
                total_rejections['LR'] += elapsed_time
            elif "results_lightgbm" in filename:
                counts['lightgbm'] += 1
                total_rejections['lightgbm'] += elapsed_time
            elif "results_xgboost" in filename:
                counts['xgboost'] += 1
                total_rejections['xgboost'] += elapsed_time
            elif "results_oracle" in filename:
                counts['oracle'] += 1
                total_rejections['oracle'] += elapsed_time

    # Calculate rejection rates (power) for each model
    results = {}
    for key in total_rejections:
        if counts[key] > 0:
            results[f"{key}_power"] = total_rejections[key] / counts[key]
        else:
            results[f"{key}_power"] = -1  # If no results were processed for this model

    return results

def read_npz_files_L(directory, small_size=False, multiple=False, type="original"):
    results = {}
    for filename in os.listdir(directory):
        if filename.endswith("1005.npy"):
            filepath = os.path.join(directory, filename)
            # Load the dictionary from the .npy file
            data = np.load(filepath, allow_pickle=True).item()
            
            t_obs = data.get('t_obs', False)
            t_sim = data.get('t_sim', False)

            # for each model type
            if "results_median" in filename:
                results['median_obs'] = t_obs
                results['median_sim'] = t_sim
            elif "results_LR" in filename:
                results['LR_obs'] = t_obs
                results['LR_sim'] = t_sim
            elif "results_lightgbm" in filename:
                results['lightgbm_obs'] = t_obs
                results['lightgbm_sim'] = t_sim
            elif "results_xgboost" in filename:
                results['xgboost_obs'] = t_obs
                results['xgboost_sim'] = t_sim
            elif "results_oracle" in filename:
                results['oracle_obs'] = t_obs
                results['oracle_sim'] = t_sim

    return results
