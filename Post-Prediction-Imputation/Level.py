
import xgboost as xgb
from sklearn.neural_network import MLPRegressor
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer
from sklearn import linear_model
from sklearn.impute import SimpleImputer
import multiprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.kernel_approximation import Nystroem
import Simulation as Generator
import OneShot
import warnings

#from cuml import XGBRegressor
 #   XGBRegressor(tree_method='gpu_hist')



if __name__ == '__main__':
    multiprocessing.freeze_support() # This is necessary and important, not sure why 
    # Mask Rate

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    
    # Create an instance of the OneShot class
    Framework = OneShot.OneShotTest(N = 1000)

    # level initialization
    level_median = 0
    level_LR = 0
    level_xgboost = 0
    
    # Fixed X, Z, change beta to make different Y,M
    for i in range(200):
        
        print("Iteration: ", i)
        # Simulate data
        DataGen = Generator.DataGenerator(N = 1000, N_T = 500, N_S = 50, beta_11 = 0, beta_12 = 0, beta_21 = 0, beta_22 = 0, beta_23 = 0, beta_31 = 0, MaskRate=0.3,Unobserved=0)

        X, Z, U, Y, M, S = DataGen.GenerateData()

        #test Median imputer
        median_imputer_1 = SimpleImputer(missing_values=np.nan, strategy='median')
        median_imputer_2 = SimpleImputer(missing_values=np.nan, strategy='median')
        p11, p12, p21, p22, p31, p32, corr1, corr2, reject = Framework.one_shot_test_parallel(Z, X, M, Y, G1=median_imputer_1, G2=median_imputer_2,verbose=0)
        if p31 <= 0.05 or p32 <= 0.05:
            level_median += 1

        #test LR imputer
        BayesianRidge_1 = IterativeImputer(estimator = linear_model.BayesianRidge(),max_iter=10, random_state=0)
        BayesianRidge_2 = IterativeImputer(estimator = linear_model.BayesianRidge(),max_iter=10, random_state=0)
        p11, p12, p21, p22, p31, p32, corr1, corr2, reject = Framework.one_shot_test_parallel(Z, X, M, Y, G1=median_imputer_1, G2=median_imputer_2,verbose=0)
        if p31 <= 0.05 or p32 <= 0.05:
            level_LR += 1


        #XGBoost
        XGBRegressor_1 = xgb.XGBRegressor()
        XGBRegressor_2 = xgb.XGBRegressor()

        XGBoost_1= IterativeImputer(estimator = XGBRegressor_1 ,max_iter=10, random_state=0)
        XGBoost_2= IterativeImputer(estimator = XGBRegressor_2 ,max_iter=10, random_state=0)
        p11, p12, p21, p22, p31, p32, corr1, corr2, reject = Framework.one_shot_test(Z, X, M, Y, G1=XGBoost_1, G2=XGBoost_2,verbose=0)
        if p31 <= 0.05 or p32 <= 0.05:
            level_xgboost += 1
    
    print("level of Median Imputer: ", level_median/200)
    print("level of LR Imputer: ", level_LR/200)
    print("level of XGBoost Imputer: ", level_xgboost/200)








        


