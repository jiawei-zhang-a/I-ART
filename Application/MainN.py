import numpy as np
import lightgbm as lgb
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer
import xgboost as xgb
from sklearn import linear_model
from sklearn.base import BaseEstimator, TransformerMixin
import iArt

# Load the arrays from the .npz file
arrays = np.load('Data/arrays.npz')

# Accessing each array using its key
Z = arrays['Z']
X = arrays['X']
Y = arrays['Y']
S = arrays['S']

# Read the job number from the arguments
import sys,os
job_number = int(sys.argv[1])

os.makedirs('Result', exist_ok=True)

# Run the iArt test
file_path = "Result/p_values_" + str(job_number) + ".txt"
L = 10000
verbose = 0
random_state = job_number - 1
threshholdForX = 0.0

"""with open(file_path, 'a') as file:
    file.write("One-sided test\n")
median_imputer = SimpleImputer(missing_values=np.nan, strategy='median')
result = iArt.test(G=median_imputer,Z=Z, X=X, Y=Y, S=S, L=L, verbose=verbose, randomization_design='cluster', threshold_covariate_median_imputation=0.0, random_state=random_state)
with open(file_path, 'a') as file:
    file.write("median " + str(result) + '\n')

median_imputer = SimpleImputer(missing_values=np.nan, strategy='median')
result = iArt.test(G=median_imputer,Z=Z, X=X, Y=Y, S=S, L=L, verbose=verbose, randomization_design='cluster', threshold_covariate_median_imputation=0.0, random_state=random_state, covariate_adjustment='linear')
with open(file_path, 'a') as file:
    file.write("median LR adjusted: " + str(result) + '\n')

median_imputer = SimpleImputer(missing_values=np.nan, strategy='median')
result = iArt.test(G=median_imputer,Z=Z, X=X, Y=Y, S=S, L=L, verbose=verbose, randomization_design='cluster', threshold_covariate_median_imputation=0.0, random_state=random_state, covariate_adjustment='xgboost')
with open(file_path, 'a') as file:
    file.write("median XGBOOST adjusted: " + str(result) + '\n')

median_imputer = SimpleImputer(missing_values=np.nan, strategy='median')
result = iArt.test(G=median_imputer,Z=Z, X=X, Y=Y, S=S, L=L, verbose=verbose, randomization_design='cluster', threshold_covariate_median_imputation=0.0, random_state=random_state, covariate_adjustment='lightgbm')
with open(file_path, 'a') as file:
    file.write("median Lightgbm adjusted: " + str(result) + '\n')

XGBoost = IterativeImputer(estimator=xgb.XGBRegressor(), max_iter=3)
result = iArt.test(G=XGBoost,Z=Z, X=X, Y=Y, S=S, L=L, verbose=verbose, randomization_design='cluster', threshold_covariate_median_imputation=0.0, random_state=random_state)
with open(file_path, 'a') as file:
    file.write("XGBoost: " + str(result) + '\n')"""

RidgeRegression = IterativeImputer(estimator=linear_model.BayesianRidge(), max_iter=5)
result = iArt.test(G=RidgeRegression,Z=Z, X=X, Y=Y, S=S, L=L, verbose=verbose, randomization_design='cluster', threshold_covariate_median_imputation=0.0, random_state=random_state)
with open(file_path, 'a') as file:
    file.write("RidgeRegression5: " + str(result) + '\n')

LinearRegression = IterativeImputer(estimator=linear_model.LinearRegression(), max_iter=3)
result = iArt.test(G=LinearRegression,Z=Z, X=X, Y=Y, S=S, L=L, verbose=verbose, randomization_design='cluster', threshold_covariate_median_imputation=0.0, random_state=random_state)
with open(file_path, 'a') as file:
    file.write("LinearRegression3: " + str(result) + '\n')

RidgeRegression = IterativeImputer(estimator=linear_model.BayesianRidge())
result = iArt.test(G=RidgeRegression,Z=Z, X=X, Y=Y, S=S, L=L, verbose=verbose, randomization_design='cluster', threshold_covariate_median_imputation=0.0, random_state=random_state)
with open(file_path, 'a') as file:
    file.write("RidgeRegression10: " + str(result) + '\n')

LightGBM = IterativeImputer(estimator=lgb.LGBMRegressor(), max_iter=5)
result = iArt.test(G=LightGBM,Z=Z, X=X, Y=Y, S=S, L=L, verbose=verbose, randomization_design='cluster', threshold_covariate_median_imputation=0.0, random_state=random_state)
with open(file_path, 'a') as file:
    file.write("LightGBM5: " + str(result) + '\n')

LightGBM = IterativeImputer(estimator=lgb.LGBMRegressor(), max_iter=10)
result = iArt.test(G=LightGBM,Z=Z, X=X, Y=Y, S=S, L=L, verbose=verbose, randomization_design='cluster', threshold_covariate_median_imputation=0.0, random_state=random_state)
with open(file_path, 'a') as file:
    file.write("LightGBM10: " + str(result) + '\n')




LightGBM = IterativeImputer(estimator=lgb.LGBMRegressor())
result = iArt.test(G=LightGBM,Z=Z, X=X, Y=Y, S=S, L=L, verbose=verbose, randomization_design='cluster', threshold_covariate_median_imputation=0.0, random_state=random_state)
with open(file_path, 'a') as file:
    file.write("LightGBM: " + str(result) + '\n')

"""result = iArt.test(G=RidgeRegression,Z=Z, X=X, Y=Y, S=S, L=L, verbose=verbose, randomization_design='cluster', threshold_covariate_median_imputation=0.0, random_state=random_state, covariate_adjustment='linear')
with open(file_path, 'a') as file:
    file.write("RidgeRegression with covariate adjustment: " + str(result) + '\n')

result = iArt.test(G=LightGBM,Z=Z, X=X, Y=Y, S=S, L=L, verbose=verbose, randomization_design='cluster', threshold_covariate_median_imputation=0.0, random_state=random_state, covariate_adjustment='lightgbm')
with open(file_path, 'a') as file:
    file.write("LightGBM with covariate adjustment: " + str(result) + '\n')

result = iArt.test(G=XGBoost,Z=Z, X=X, Y=Y, S=S, L=L, verbose=verbose, randomization_design='cluster', threshold_covariate_median_imputation=0.0, random_state=random_state, covariate_adjustment='xgboost')
with open(file_path, 'a') as file:
    file.write("XGBoost with covariate adjustment: " + str(result) + '\n')"""