import numpy as np
import os

def read_npz_files(directory,small_size=False,type="original"):
    summed_p_values_median = None
    summed_p_values_LR = None
    summed_p_values_lightGBM = None
    summed_p_values_xgboost = None
    summed_p_values_oracle = None
    if type == "original":
        N = int(len(os.listdir(directory)) / 2)
    else:
        N = int(len(os.listdir(directory)))

    for filename in os.listdir(directory):
        if filename.endswith(".npy"):
            filepath = os.path.join(directory, filename)
            p_values = np.load(filepath)

            if "p_values_median" in filename:
                if summed_p_values_median is None:
                    summed_p_values_median = (p_values<= 0.05).astype(int)
                else:
                    summed_p_values_median += (p_values<= 0.05).astype(int)
            elif "p_values_LR" in filename:
                if summed_p_values_LR is None:
                    summed_p_values_LR = (p_values<= 0.05).astype(int)
                else:
                    summed_p_values_LR += (p_values<= 0.05).astype(int)
            elif "p_values_lightGBM" in filename:
                if summed_p_values_lightGBM is None:
                    summed_p_values_lightGBM = (p_values<= 0.05).astype(int)
                else:
                    summed_p_values_lightGBM += (p_values<= 0.05).astype(int)
            elif "p_values_xgboost" in filename:
                if summed_p_values_xgboost is None:
                    summed_p_values_xgboost = (p_values<= 0.05).astype(int)
                else:
                    summed_p_values_xgboost += (p_values<= 0.05).astype(int)
            elif "p_values_oracle" in filename:
                if summed_p_values_oracle is None:
                    summed_p_values_oracle = (p_values<= 0.05).astype(int)
                else:
                    summed_p_values_oracle += (p_values<= 0.05).astype(int)

    if summed_p_values_median is None:
        summed_p_values_median = np.zeros(1)
    if summed_p_values_LR is None:
        summed_p_values_LR = np.zeros(1)
    if summed_p_values_lightGBM is None:
        summed_p_values_lightGBM = np.zeros(1)
    if summed_p_values_xgboost is None:
        summed_p_values_xgboost = np.zeros(1)
    if summed_p_values_oracle is None:
        summed_p_values_oracle = np.zeros(1)

    if small_size:
        results = {
            'median_power': summed_p_values_median[0] / N,
            'lr_power': summed_p_values_LR[0] / N,
            'xgboost_power': summed_p_values_xgboost[0] / N,
            'oracle_power': summed_p_values_oracle[0] / N,
        }
    else:
        results = {
            'median_power': summed_p_values_median[0] / N,
            'lr_power': summed_p_values_LR[0] / N,
            'lightGBM_power': summed_p_values_lightGBM[0] / N,
            'oracle_power': summed_p_values_oracle[0] / N,
        }

    return results

def read_npz_files_main(directory,small_size=False, multiple=False):
    summed_p_values_median = None
    summed_p_values_LR = None
    summed_p_values_lightGBM = None
    summed_p_values_xgboost = None
    summed_p_values_oracle = None

    N = int(len(os.listdir(directory)) / 4)

    if multiple:
        for filename in os.listdir(directory):
            if filename.endswith(".npy"):
                filepath = os.path.join(directory, filename)
                p_values = np.load(filepath)

                if "p_values_median" in filename:
                    if summed_p_values_median is None:
                        summed_p_values_median = p_values
                    else:
                        summed_p_values_median += p_values
                elif "p_values_LR" in filename:
                    if summed_p_values_LR is None:
                        summed_p_values_LR = p_values
                    else:
                        summed_p_values_LR += p_values
                elif "p_values_lightGBM" in filename:
                    if summed_p_values_lightGBM is None:
                        summed_p_values_lightGBM = p_values
                    else:
                        summed_p_values_lightGBM += p_values
                elif "p_values_xgboost" in filename:
                    if summed_p_values_xgboost is None:
                        summed_p_values_xgboost = p_values
                    else:
                        summed_p_values_xgboost += p_values
                elif "p_values_oracle" in filename:
                    if summed_p_values_oracle is None:
                        summed_p_values_oracle = p_values
                    else:
                        summed_p_values_oracle += p_values

        column = 3
        if small_size:
            results = {
                'median_power': summed_p_values_median[column] / N,
                'lr_power': summed_p_values_LR[column] / N,
                'xgboost_power': summed_p_values_xgboost[column] / N,
                'oracle_power': summed_p_values_oracle[column] / N,
            }
        else:
            results = {
                'median_power': summed_p_values_median[column] / N,
                'lr_power': summed_p_values_LR[column] / N,
                'lightGBM_power': summed_p_values_lightGBM[column] / N,
                'oracle_power': summed_p_values_oracle[column] / N,
            }

        return results    
    else:
        for filename in os.listdir(directory):
            if filename.endswith(".npy"):
                filepath = os.path.join(directory, filename)
                p_values = np.load(filepath)

                if "p_values_median" in filename:
                    if summed_p_values_median is None:
                        summed_p_values_median = (p_values<= 0.05).astype(int)
                    else:
                        summed_p_values_median += (p_values<= 0.05).astype(int)
                elif "p_values_LR" in filename:
                    if summed_p_values_LR is None:
                        summed_p_values_LR = (p_values<= 0.05).astype(int)
                    else:
                        summed_p_values_LR += (p_values<= 0.05).astype(int)
                elif "p_values_lightGBM" in filename:
                    if summed_p_values_lightGBM is None:
                        summed_p_values_lightGBM = (p_values<= 0.05).astype(int)
                    else:
                        summed_p_values_lightGBM += (p_values<= 0.05).astype(int)
                elif "p_values_xgboost" in filename:
                    if summed_p_values_xgboost is None:
                        summed_p_values_xgboost = (p_values<= 0.05).astype(int)
                    else:
                        summed_p_values_xgboost += (p_values<= 0.05).astype(int)
                elif "p_values_oracle" in filename:
                    if summed_p_values_oracle is None:
                        summed_p_values_oracle = (p_values<= 0.05).astype(int)
                    else:
                        summed_p_values_oracle += (p_values<= 0.05).astype(int)

        if small_size:
            results = {
                'median_power': summed_p_values_median[0] / N,
                'lr_power': summed_p_values_LR[0] / N,
                'xgboost_power': summed_p_values_xgboost[0] / N,
                'oracle_power': summed_p_values_oracle[0] / N,
            }
        else:
            results = {
                'median_power': summed_p_values_median[0] / N,
                'lr_power': summed_p_values_LR[0] / N,
                'lightGBM_power': summed_p_values_lightGBM[0] / N,
                'oracle_power': summed_p_values_oracle[0] / N,
            }

        return results