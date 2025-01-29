import sys
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer
from sklearn.impute import IterativeImputer
from sklearn import linear_model
import SingleOutcomeModelGenerator as Generator
import os
import lightgbm as lgb
import xgboost as xgb
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import iArt as iArt


# Do not change this parameter
beta_coef = None
task_id = 1

# Set the default values
max_iter = 3

def report_delta():
    DataGen = Generator.DataGenerator(N = 1000, beta = beta_coef)
    X, Z, T, C, M_delta,S = DataGen.GenerateData()

    # Filter only non-missing cases (M_delta = 0)
    non_missing_mask = (M_delta == 0)
    T_non_missing = T[non_missing_mask]
    C_non_missing = C[non_missing_mask]

    # Calculate Delta for non-missing cases
    Delta_non_missing = (T_non_missing <= C_non_missing).astype(int)

    # Compute proportions
    proportion_diagnosed = np.mean(Delta_non_missing == 1)  # Proportion Delta_ij = 1
    proportion_censored = np.mean(Delta_non_missing == 0)   # Proportion Delta_ij = 0

    # Report results
    print("Proportions among 50% non-missing cases (M_delta = 0):")
    print(f"Proportion Delta_ij = 1 (Diagnosed): {proportion_diagnosed:.2f}")
    print(f"Proportion Delta_ij = 0 (Censored): {proportion_censored:.2f}")

def run(Nsize, filepath, verbose=1, small_size = True):

    if beta_coef == 0.0:
        Iter = 10000
    else:
        Iter = 1000

    report_delta()

    # Simulate data
    DataGen = Generator.DataGenerator(N = Nsize, beta = beta_coef,verbose=verbose)
    X, Z, T, C, M_delta,S = DataGen.GenerateData()

    values_oracle= iArt.imputation_reimputation_survival(Z=Z,X_star=X,T_star = T,C_star=C, S=S, L=Iter, G = None,verbose=verbose)
    # Append p-values to corresponding lists

    # Mask the T and C values based on M_delta
    T_masked = np.where(M_delta == 1, np.nan, T)
    C_masked = np.where(M_delta == 1, np.nan, C)

    median_imputer = SimpleImputer(missing_values=np.nan, strategy='median')
    values_median = iArt.imputation_reimputation_survival(Z=Z, X_star=X, T_star=T_masked, C_star=C_masked, S=S, G=median_imputer, L=Iter, verbose=verbose)

    BayesianRidge = IterativeImputer(estimator = linear_model.BayesianRidge(),max_iter=max_iter)
    values_LR = iArt.imputation_reimputation_survival(Z=Z, X_star=X, T_star=T_masked, C_star=C_masked, S=S, G=BayesianRidge, L=Iter, verbose=verbose)

    if small_size == True:
        xgboost_imputer = IterativeImputer(estimator = xgb.XGBRegressor(),max_iter=max_iter)
        values_xgboost = iArt.imputation_reimputation_survival(Z=Z, X_star=X, T_star=T_masked, C_star=C_masked, S=S, G=xgboost_imputer, L=Iter, verbose=verbose)
    else:
        lightgbm_imputer = IterativeImputer(estimator = lgb.LGBMRegressor(verbose=-1),max_iter=max_iter)
        values_lightgbm = iArt.imputation_reimputation_survival(Z=Z, X_star=X, T_star=T_masked, C_star=C_masked, S=S, G=lightgbm_imputer, L=Iter, verbose=verbose)


    os.makedirs("%s/%f"%(filepath,beta_coef), exist_ok=True)
    np.save('%s/%f/p_values_median_%d.npy' % (filepath, beta_coef, task_id), values_median)
    np.save('%s/%f/p_values_oracle_%d.npy' % (filepath, beta_coef, task_id), values_oracle)
    np.save('%s/%f/p_values_LR_%d.npy' % (filepath, beta_coef, task_id), values_LR)
    if small_size == True:
        np.save('%s/%f/p_values_xgboost_%d.npy' % (filepath, beta_coef, task_id), values_xgboost)
    else:
        np.save('%s/%f/p_values_lightgbm_%d.npy' % (filepath, beta_coef, task_id), values_lightgbm)

task_id_origin = 0
if __name__ == '__main__':
    if len(sys.argv) == 2:
        task_id_origin = int(sys.argv[1])
    else:
        print("Please add the job number like this\nEx.python Power.py 1")
        exit()

    task_id = task_id_origin
    # Model 1
    beta_to_lambda = {0.0: 2.159275141001102, 0.07: 2.165387531267955, 0.14: 2.285935405246937, 0.21: 2.258923945496463, 0.28: 2.2980720651301794, 0.35: 2.3679216299985613}
    for coef in np.arange(0.0,0.42,0.07):
        beta_coef = coef
        # Round to two decimal places to match dictionary keys
        beta_coef_rounded = round(beta_coef, 2)
        if beta_coef_rounded in beta_to_lambda:
            lambda_value = beta_to_lambda[beta_coef_rounded]
            run(1000, filepath = "Result/HPC_power_1000_survival", small_size=False)
        else:
            print(f"No lambda value found for beta_coef: {beta_coef_rounded}")

    beta_to_lambda = {0.0: 2.1577587265653126, 0.25: 2.2946233479956843, 0.5: 2.42339283727788, 0.75: 2.544154767644711, 1.0: 2.669166349074493, 1.25: 2.792645016605368}
    for coef in np.arange(0,1.5,0.25):
        beta_coef = coef
        # Round to two decimal places to match dictionary keys
        beta_coef_rounded = round(beta_coef, 2)
        if beta_coef_rounded in beta_to_lambda:
            lambda_value = beta_to_lambda[beta_coef_rounded]
            run(50, filepath = "Result/HPC_power_50_survival",small_size=True)
        else:
            print(f"No lambda value found for beta_coef: {beta_coef_rounded}")
    
