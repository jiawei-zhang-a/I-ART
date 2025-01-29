import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from ReadData import read_npz_files

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

    plot(np.arange(0.0,0.42,0.07), np.arange(0,1.5,0.25), "../Data/Simulation/HPC_power_1000_survival", "../Data/Simulation/HPC_power_50_survival", "Size1000_survival", "Size50_survival")

main_pic_generator()