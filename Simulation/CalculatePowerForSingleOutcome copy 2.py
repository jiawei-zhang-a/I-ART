import sys
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer
from sklearn.impute import IterativeImputer
from sklearn import linear_model
import SingleOutcomeModelGenerator as Generator
import RandomizationTestModelBased as RandomizationTest
import os
import lightgbm as lgb
import xgboost as xgb
import iArt as iArt

# Do not change this parameter
beta_coef = None
task_id = 1


def run(Nsize, filepath,  Missing_lambda,adjust = 0, model = 0, verbose=1, small_size = True, multiple = False):

    Missing_lambda = None
    # Simulate data
    if multiple == False:
        DataGen = Generator.DataGenerator(N = Nsize, strata_size=10,beta = beta_coef,model = model, MaskRate=0.5, verbose=verbose,Missing_lambda = Missing_lambda)
        X, Z, U, Y, M, S = DataGen.GenerateData()

    Framework = RandomizationTest.RandomizationTest(N = Nsize)
    reject, p_values= Framework.test(Z, X, M, Y,strata_size = 10, L=10000, G = None,verbose=verbose)
    # Append p-values to corresponding lists
    values_oracle = [ *p_values, reject]

    
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
    lambda_value = None
    # Model 1
    for coef in np.arange(0.8,1.6,0.1):
        beta_coef = coef    
        run(1000, filepath = "Result/HPC_power_1000_model1", adjust = 0, model = 1, Missing_lambda = None, small_size=False)

    for coef in np.arange(0.3,8,0.5):
        beta_coef = coef
        run(50, filepath = "Result/HPC_power_50_model1", adjust = 0, model = 1, Missing_lambda = lambda_value, small_size=True)
    # Model 2
    for coef in np.arange(1,2,0.1):
        beta_coef = coef
        run(1000, filepath = "Result/HPC_power_1000_model2", adjust = 0, model = 2, Missing_lambda = lambda_value, small_size=False)

    for coef in np.arange(2,5,0.1):
        beta_coef = coef
        run(50, filepath = "Result/HPC_power_50_model2", adjust = 0, model = 2, Missing_lambda = lambda_value, small_size=True)
          
    # Model 3
    for coef in np.arange(0.5,0.8,0.1):
        beta_coef = coef
        run(1000, filepath = "Result/HPC_power_1000_model3", adjust = 0, model = 3, Missing_lambda = lambda_value, small_size=False)

    for coef in np.arange(3,9,0.1):
        beta_coef = coef
        run(50, filepath = "Result/HPC_power_50_model3", adjust = 0, model = 3, Missing_lambda = lambda_value, small_size=True)
  
    # Model 4
    for coef in np.arange(0.5,1,0.1):
        beta_coef = coef
        run(1000, filepath = "Result/HPC_power_1000_model4", adjust = 0, model = 4, Missing_lambda = lambda_value, small_size=False)

    for coef in np.arange(3,9,0.1):
        beta_coef = coef
        run(50, filepath = "Result/HPC_power_50_model4", adjust = 0, model = 4, Missing_lambda = lambda_value, small_size=True)
