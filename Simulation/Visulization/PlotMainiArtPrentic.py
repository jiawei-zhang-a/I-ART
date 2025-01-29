import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def read_npz_files(directory,small_size=False, multiple=False, type="original"):
    summed_p_values_median = None
    summed_p_values_LR = None
    summed_p_values_lightgbm = None
    summed_p_values_xgboost = None
    summed_p_values_oracle = None

    N_p_values_median = 0
    N_p_values_LR = 0
    N_p_values_lightgbm = 0
    N_p_values_xgboost = 0
    N_p_values_oracle = 0

    summed_p_values_median = 0
    summed_p_values_LR = 0
    summed_p_values_lightgbm = 0
    summed_p_values_xgboost = 0
    summed_p_values_oracle = 0

    for filename in os.listdir(directory):
        if filename.endswith(".npy"):
            filepath = os.path.join(directory, filename)
            p_value = np.load(filepath)

            reject = (p_value < 0.05).astype(int)
            if "p_values_median" in filename and 'p_values_medianadjusted' not in filename:
                N_p_values_median += 1
                summed_p_values_median += reject
            elif "p_values_LR" in filename:
                N_p_values_LR += 1
                summed_p_values_LR += reject
            elif ("p_values_lightGBM"  in filename) or ("p_values_lightgbm" in filename):
                N_p_values_lightgbm += 1
                summed_p_values_lightgbm += reject
            elif "p_values_xgboost" in filename:
                N_p_values_xgboost += 1
                summed_p_values_xgboost += reject
            elif "p_values_oracle" in filename:
                N_p_values_oracle += 1
                summed_p_values_oracle += reject

    if N_p_values_median != 0:
        rejection_rate_median = summed_p_values_median / N_p_values_median
    else:
        rejection_rate_median = -1
    if N_p_values_LR != 0:
        rejection_rate_LR = summed_p_values_LR / N_p_values_LR
    else:
        rejection_rate_LR = -1
    if N_p_values_lightgbm != 0:
        rejection_rate_lightgbm = summed_p_values_lightgbm / N_p_values_lightgbm
    else:
        rejection_rate_lightgbm = -1
    if N_p_values_xgboost != 0:
        rejection_rate_xgboost = summed_p_values_xgboost / N_p_values_xgboost
    else:
        rejection_rate_xgboost = -1
    if N_p_values_oracle != 0:
        rejection_rate_oracle = summed_p_values_oracle / N_p_values_oracle
    else:
        rejection_rate_oracle = -1

    results = {
        'median_power': rejection_rate_median,
        'lr_power': rejection_rate_LR,
        'xgboost_power':    rejection_rate_xgboost,
        'oracle_power': rejection_rate_oracle,
        'lightgbm_power': rejection_rate_lightgbm,
    }

    return results     

def plot_results(data, title, xsticks):
    columns = ['beta', 'Imputer_Median', 'Imputer_PREP-RidgeReg',  'Imputer_PREP-GBM', 'Imputer_Oracle']

    df = pd.DataFrame(data, columns=columns)

    plt.figure(figsize=(10, 6))

    colors = {'Median': 'blue', 'PREP-RidgeReg': 'red', 'PREP-GBM': 'green', 'Oracle':'purple'}
    linestyles = {'Imputer': '-'}

    for col in columns[1:]:
        method = col.split('_')[1]
        dataset = col.split('_')[0]
        linestyle = linestyles[dataset]
        plt.plot(df['beta'], df[col], marker='o', label=method, color=colors[method], linestyle=linestyle, linewidth=2.0)
        
    plt.xlabel(r'$\beta$',fontsize=30)
    plt.ylabel('Rejection Rate',fontsize=30)
    plt.grid()
    # Setting y-axis ticks with custom intervals
    y_ticks = [i/100.0 for i in range(25, 105, 25)]  # Starts from 0, ends at 1.05, with an interval of 0.05
    y_ticks.append(0.05)
    plt.yticks(y_ticks)
    X_ticks = xsticks
    plt.xticks(X_ticks)
    plt.tick_params(axis='both', which='major', labelsize=25)

    #plt.show()
    if not os.path.exists("pic"):
        os.makedirs("pic")

    plt.savefig("pic/" + title + ".pdf", bbox_inches='tight')


def plot(range,range_small, path,path_small, title, title_small, multiple = False):
    Power_data = []
    Power_data_small = []

    for coef in range:
        row_power = [coef]
        for directory in [path + "/%f" % (coef)]:
            results = read_npz_files(directory,small_size=False, multiple = multiple)
            row_power.extend([results['median_power'], results['lr_power'], results['lightgbm_power'],results['oracle_power']])
        Power_data.append(row_power)
    print(Power_data)
    plot_results(Power_data, title, range)

    for coef in range_small:
        row_power_small = [coef]
        for directory in [path_small + "/%f" % (coef)]:
            results = read_npz_files(directory,small_size=True, multiple = multiple)
            row_power_small.extend([results['median_power'], results['lr_power'], results['xgboost_power'],results['oracle_power']])
        Power_data_small.append(row_power_small)
    print(Power_data_small)
    plot_results(Power_data_small, title_small, range_small)

def main_pic_generator():

    plot(np.arange(0.0,0.42,0.07), np.arange(0,1.5,0.25), "../Power/Result/HPC_power_1000_survival", "../Power/Result/HPC_power_50_survival", "Size1000_survival", "Size50_survival")

main_pic_generator()