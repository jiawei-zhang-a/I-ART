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

# Set the default values
max_iter = 3

def run(Nsize, filepath, model = 0,adjust=0, verbose=1, small_size = True):
    
    Missing_lambda = None
        
    # Simulate data
    DataGen = Generator.DataGenerator(N = Nsize, strata_size=10,beta = beta_coef,model = model, MaskRate=0.5, verbose=verbose,Missing_lambda = Missing_lambda)
    X, Z, U, Y, M, S = DataGen.GenerateData()

    Framework = RandomizationTest.RandomizationTest(N = Nsize)
    reject, p_values= Framework.test(Z, X, M, Y,strata_size = 10, L=10000, G = None,verbose=verbose)
    
    # Append p-values to corresponding lists
    values_TZM = [ *p_values]

    os.makedirs("%s/%f"%(filepath,beta_coef), exist_ok=True)
    
    np.save('%s/%f/p_values_TestMissing_%d.npy' % (filepath, beta_coef, task_id), values_TZM)


task_id_origin = 0
if __name__ == '__main__':
    if len(sys.argv) == 2:
        task_id_origin = int(sys.argv[1])
    else:
        print("Please add the job number like this\nEx.python Power.py 1")
        exit()

    task_id = task_id_origin
    # Model 1
    for coef in np.arange(0.0,4,0.5):
        beta_coef = coef
        run(1000, filepath = "Result/HPC_power_1000_model1_TestMissing", adjust = 0, model = 1, small_size=False)
    for coef in np.arange(0.0,4,0.5):
        beta_coef = coef
        run(50, filepath = "Result/HPC_power_50_model1_TestMissing", adjust = 0, model = 1, small_size=True)
    for coef in np.arange(0.0,4,0.5):
        beta_coef = coef
        run(1000, filepath = "Result/HPC_power_1000_model2_TestMissing", adjust = 0, model = 2, small_size=False)
    for coef in np.arange(0.0,4,0.5):
        beta_coef = coef
        run(50, filepath = "Result/HPC_power_50_model2_TestMissing", adjust = 0, model = 2, small_size=True)
    for coef in np.arange(0.0,4,0.5):
        beta_coef = coef
        run(1000, filepath = "Result/HPC_power_1000_model3_TestMissing", adjust = 0, model = 3, small_size=False)
    for coef in np.arange(0.0,4,0.5):
        beta_coef = coef
        run(50, filepath = "Result/HPC_power_50_model3_TestMissing", adjust = 0, model = 3, small_size=True)
    for coef in np.arange(0.0,4,0.5):
        beta_coef = coef
        run(1000, filepath = "Result/HPC_power_1000_model4_TestMissing", adjust = 0, model = 4, small_size=False)
    for coef in np.arange(0.0,4,0.5):
        beta_coef = coef
        run(50, filepath = "Result/HPC_power_50_model4_TestMissing", adjust = 0, model = 4, small_size=True)

