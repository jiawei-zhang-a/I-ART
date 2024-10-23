import numpy as np
import os
import matplotlib.pyplot as plt

def find_least_L(t_obs, t_sim, p_values, error_threshold=0.01, stable=False):
    """
    Find the least L for convergence to p-values within the given error threshold.
    
    Parameters:
    - t_obs: Observed test statistic
    - t_sim: Simulated test statistics
    - p_values: Current p_values from the file
    - error_threshold: Threshold for p-value convergence
    
    Returns:
    - L: Least L where p-values converge within the error threshold
    """
    if stable == False:
        # Initialize L to the maximum number of simulations
        L = len(t_sim)
        p_values = np.zeros(L)
        final_p_value = np.mean(t_sim >= t_obs)
        
        # Compute running p-values for the model
        for i in range(1, L + 1):
            p_values[i-1] = np.mean(t_sim[:i] >= t_obs)
            
            # Check for convergence
            if np.abs(p_values[i-1] - final_p_value) < error_threshold:
                L = i
                break

        return L
    else:
        L = len(t_sim)
        cumulative_count = np.cumsum(t_sim >= t_obs)
        running_p_values = cumulative_count / np.arange(1, L + 1)
        
        final_p_value = running_p_values[-1]
        
        # Iterate over possible L values and check if all future p-values are stable
        for i in range(L):
            # Check if all p-values from current i to the end are within the threshold
            if np.all(np.abs(running_p_values[i:] - final_p_value) < error_threshold):
                return i + 1  # Return the least L (1-based index)
        
        return L  # If no stabilization, return max L

def read_npz_files_L(directory, error_threshold=0.01, stable=False):
    """
    Process .npz files in the directory to calculate the least L for each model type.
    
    Parameters:
    - directory: Directory containing the .npy files
    - error_threshold: Threshold for p-value convergence
    
    Returns:
    - mean_L: The mean of the least L values where p-values converge, or None if no valid L found
    """
    least_Ls = {
        'median': [],
        'LR': [],
        'lightgbm': [],
        'xgboost': [],
        'oracle': []
    }
    
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        data = np.load(filepath, allow_pickle=True).item()
        
        t_obs = data.get('t_obs', False)
        t_sim = data.get('t_sim', False)
        p_values = data.get('p_values', False)

        if t_obs is not False and t_sim is not False and p_values is not False:
            # Determine the model type based on the file name
            if "results_LR" in filename:
                least_Ls['LR'].append(find_least_L(t_obs, t_sim, p_values, error_threshold, stable))
            elif "results_lightgbm" in filename:
                least_Ls['lightgbm'].append(find_least_L(t_obs, t_sim, p_values, error_threshold, stable))
            elif "results_xgboost" in filename:
                least_Ls['xgboost'].append(find_least_L(t_obs, t_sim, p_values, error_threshold, stable))

    # Compute the average of least Ls for each model
    average_least_Ls = {model: np.mean(Ls) for model, Ls in least_Ls.items() if Ls}

    # Return both the dictionary of least Ls and the dictionary of their averages
    return average_least_Ls#least_Ls, average_least_Ls


def plot(range, range_small, path, path_small, thereshold=0.01, stable=False):
    # Convergence plots for larger size
    for coef in range:
        if coef != 0.0:
            continue
        for directory in [f'{path}/{coef}']:
            results = read_npz_files_L(directory, error_threshold=thereshold, stable=stable)
            print(results)

    # Convergence plots for smaller size
    for coef in range_small:
        if coef != 0.0:
            continue
        for directory in [f'{path_small}/{coef}']:
            results = read_npz_files_L(directory, error_threshold=thereshold, stable=stable)
            print(results)
    

            

def main_pic_generator():
    print("Unstable")
    print("threshold=0.01")
    plot(np.arange(0.0,0.42,0.07), np.arange(0,1.5,0.25), "../Power/timeL/HPC_power_1000_model1", "../Power/timeL/HPC_power_50_model1")
    plot(np.arange(0.0,0.96,0.16), np.arange(0.0,4.8,0.8), "../Power/timeL/HPC_power_1000_model2", "../Power/timeL/HPC_power_50_model2")
    plot(np.arange(0.0,0.36,0.06), np.arange(0.0,1.5,0.25), "../Power/timeL/HPC_power_1000_model3", "../Power/timeL/HPC_power_50_model3")
    plot(np.arange(0.0,0.36,0.06), np.arange(0.0,1.5,0.25), "../Power/timeL/HPC_power_1000_model4", "../Power/timeL/HPC_power_50_model4")

    print("threshold=0.005")
    plot(np.arange(0.0,0.42,0.07), np.arange(0,1.5,0.25), "../Power/timeL/HPC_power_1000_model1", "../Power/timeL/HPC_power_50_model1", thereshold=0.005)
    plot(np.arange(0.0,0.96,0.16), np.arange(0.0,4.8,0.8), "../Power/timeL/HPC_power_1000_model2", "../Power/timeL/HPC_power_50_model2", thereshold=0.005)
    plot(np.arange(0.0,0.36,0.06), np.arange(0.0,1.5,0.25), "../Power/timeL/HPC_power_1000_model3", "../Power/timeL/HPC_power_50_model3", thereshold=0.005)
    plot(np.arange(0.0,0.36,0.06), np.arange(0.0,1.5,0.25), "../Power/timeL/HPC_power_1000_model4", "../Power/timeL/HPC_power_50_model4", thereshold=0.005)

    print("Stable")
    print("threshold=0.01")
    plot(np.arange(0.0,0.42,0.07), np.arange(0,1.5,0.25), "../Power/timeL/HPC_power_1000_model1", "../Power/timeL/HPC_power_50_model1", stable=True)
    plot(np.arange(0.0,0.96,0.16), np.arange(0.0,4.8,0.8), "../Power/timeL/HPC_power_1000_model2", "../Power/timeL/HPC_power_50_model2", stable=True)
    plot(np.arange(0.0,0.36,0.06), np.arange(0.0,1.5,0.25), "../Power/timeL/HPC_power_1000_model3", "../Power/timeL/HPC_power_50_model3", stable=True)
    plot(np.arange(0.0,0.36,0.06), np.arange(0.0,1.5,0.25), "../Power/timeL/HPC_power_1000_model4", "../Power/timeL/HPC_power_50_model4", stable=True)

    print("threshold=0.005")
    plot(np.arange(0.0,0.42,0.07), np.arange(0,1.5,0.25), "../Power/timeL/HPC_power_1000_model1", "../Power/timeL/HPC_power_50_model1", thereshold=0.005, stable=True)
    plot(np.arange(0.0,0.96,0.16), np.arange(0.0,4.8,0.8), "../Power/timeL/HPC_power_1000_model2", "../Power/timeL/HPC_power_50_model2", thereshold=0.005, stable=True)
    plot(np.arange(0.0,0.36,0.06), np.arange(0.0,1.5,0.25), "../Power/timeL/HPC_power_1000_model3", "../Power/timeL/HPC_power_50_model3", thereshold=0.005, stable=True)
    plot(np.arange(0.0,0.36,0.06), np.arange(0.0,1.5,0.25), "../Power/timeL/HPC_power_1000_model4", "../Power/timeL/HPC_power_50_model4", thereshold=0.005, stable=True)

main_pic_generator()
