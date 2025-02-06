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

def calculate_missing_rates(Z, M):
    """
    Calculate the missing rates for treated and control groups.

    Parameters:
    - Z: numpy array, where 1 represents the treated group and 0 represents the control group
    - M: numpy array, where 1 means the measurement is present and 0 means it's missing

    Returns:
    - A dictionary with missing rates for treated and control groups
    """
    # Ensure Z and M have the same length
    if len(Z) != len(M):
        raise ValueError("Z and M must have the same length")

    # Total counts for treated and control groups
    total_treated = np.sum(Z == 1)
    total_control = np.sum(Z == 0)

    # Missing counts for treated and control groups
    missing_treated = np.sum((Z == 1) & (M == 0))
    missing_control = np.sum((Z == 0) & (M == 0))

    # Calculate missing rates, handling cases where there are no treated or control samples
    missing_rate_treated = missing_treated / total_treated if total_treated > 0 else None
    missing_rate_control = missing_control / total_control if total_control > 0 else None

    # Return the results as a dictionary
    return {"treated": missing_rate_treated, "control": missing_rate_control}

def run(Nsize, filepath,  Missing_lambda,adjust = 0, model = 0, verbose=1, small_size = True, multiple = False):

    Missing_lambda = None
    # Simulate data
    if multiple == False:
        DataGen = Generator.DataGenerator(N = Nsize, strata_size=10,beta = beta_coef,model = model, MaskRate=0.5, verbose=verbose,Missing_lambda = Missing_lambda)
        X, Z, U, Y, M, S = DataGen.GenerateData()

    Framework = RandomizationTest.RandomizationTest(N = Nsize)

    #print("Model, beta_coef",model, beta_coef)
    X = calculate_missing_rates(Z, M)


    os.makedirs("%s/%f"%(filepath,beta_coef), exist_ok=True)
    
    np.save('%s/%f/p_values_oracle_%d.npy' % (filepath, beta_coef, task_id), X["treated"])

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

    for coef in np.arange(0,9,0.5):
        beta_coef = coef
        run(50, filepath = "Result/HPC_power_50_model3", adjust = 0, model = 3, Missing_lambda = lambda_value, small_size=True)
  
    # Model 4
    for coef in np.arange(0.5,1,0.1):
        beta_coef = coef
        run(1000, filepath = "Result/HPC_power_1000_model4", adjust = 0, model = 4, Missing_lambda = lambda_value, small_size=False)

    for coef in np.arange(0,9,0.5):
        beta_coef = coef
        run(50, filepath = "Result/HPC_power_50_model4", adjust = 0, model = 4, Missing_lambda = lambda_value, small_size=True)
