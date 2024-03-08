import iArt
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer

# Define the path to the TSV file
file_path = 'Data/36158-0001-Data.tsv'

# Load the TSV file into a DataFrame
df = pd.read_csv(file_path, sep='\t')

# Filter the DataFrame for rows where 'WAVE' equals 2
# Adjust the CONDITION column in the whole DataFrame before filtering

# Assuming df is your combined DataFrame with all necessary columns including 'CONDITION'

# Adjust the CONDITION column in the whole DataFrame before filtering
df['CONDITION'] = df['CONDITION'].replace(2, 0)

# Filter to include the CONDITION column for simplicity
df_filtered = df[['ADMINLINK', 'WAVE', 'SCWM_CWH', 'RMZFN', 'STUDYGROUP', 'CONDITION', 'SCEM_DISTI','SCEM_STRSI','SCWM_FTWCI', 'SCWM_WTFCI', 'SCWM_TIMEALLI' ]]

# Separate DataFrames by wave
wave1_df = df_filtered[df_filtered['WAVE'] == 1]
wave2_df = df_filtered[df_filtered['WAVE'] == 2].set_index('ADMINLINK')

# Initialize empty lists to store your matched records
matched_Y = []
matched_X = []
matched_S = []
matched_Z = [] # For CONDITION

# Iterate through unique ADMINLINK identifiers in Wave 1
for adminlink in wave1_df['ADMINLINK'].unique():
    # Get covariate, study group, and condition from Wave 1
    covariate_record = wave1_df[wave1_df['ADMINLINK'] == adminlink].iloc[0]
    
    # Attempt to get the corresponding outcome from Wave 2
    outcome_record = wave2_df.loc[adminlink, 'SCWM_CWH'] if adminlink in wave2_df.index else np.nan
    
    # Add the outcome or NaN to matched_Y
    matched_Y.append(outcome_record)
    
    # Add the covariates to matched_X
    matched_X.append(covariate_record[['SCWM_CWH', 'RMZFN', 'SCEM_DISTI','SCWM_FTWCI', 'SCWM_TIMEALLI', 'SCEM_STRS']].values)
    
    # Add the study group to matched_S
    matched_S.append(covariate_record['STUDYGROUP'])
    
    # Add the condition to matched_Z
    matched_Z.append(covariate_record['CONDITION'])

# Function to convert values, replacing '-8' and empty strings with np.nan
def convert_to_float(value):
    try:
        # Convert value to float
        float_value = float(value)
        # Replace '-8' with np.nan
        if float_value == -8.0:
            return np.nan
        else:
            return float_value
    except ValueError:
        # Handle empty strings or other non-numeric values
        return np.nan

# Convert lists to numpy arrays and reshape as needed
Y = np.array(matched_Y).reshape(-1, 1)
Y = np.array([convert_to_float(y) for y in Y.flatten()]).reshape(Y.shape)
X = np.array(matched_X)  # Should already be in the correct shape (n, 2)
X = np.array([[convert_to_float(x) for x in row] for row in X])
S = np.array(matched_S).reshape(-1, 1)
Z = np.array(matched_Z).reshape(-1, 1)


"""
# Print the description of the data
cluster_sizes = np.bincount(S.flatten())
cluster_sizes.pop(0)
print("cluster_sizes",cluster_sizes)
# largest cluster size
max_size = np.max(cluster_sizes)
# smallest cluster size
min_size = np.min(cluster_sizes)

print("max_size",max_size)
print("min_size",min_size)

#print total number of individuals
print("Total number of individuals: ", len(Y))

# print the missing percentage of the outcome
missing_percentage = np.mean(np.isnan(Y))
print("Missing percentage of the outcome: ", missing_percentage)

# print the missing percentage of the covariates SCWM_CWH
missing_percentage = np.mean(np.isnan(X[:,0]))
print("Missing percentage of the covariates: ", missing_percentage)

"""

file_path = "p_values.txt"
L = 10000
verbose = 0
random_state = 0

median_imputer = SimpleImputer(missing_values=np.nan, strategy='median')
result = iArt.test(Z=Z, X=X, Y=Y, S=S,L=L,G= median_imputer, verbose=verbose,mode = 'cluster',random_state=random_state)
with open(file_path, 'a') as file:
    file.write("median: " + str(result) + '\n')

result = iArt.test(Z=Z, X=X, Y=Y, S=S,L=L, verbose = verbose,mode = 'cluster',random_state=random_state)
with open(file_path, 'a') as file:
    file.write("RidgeRegression: " + str(result) + '\n')

result = iArt.test(Z=Z, X=X, Y=Y, S=S,L=L, verbose=verbose,mode = 'cluster', covariate_adjustment=1,random_state=random_state)
with open(file_path, 'a') as file:
    file.write("RidgeRegression with covariate adjustment: " + str(result) + '\n')

LightGBM = IterativeImputer(estimator=lgb.LGBMRegressor(n_jobs = 26,verbosity=-1), max_iter=1)
result = iArt.test(Z=Z, X=X, Y=Y,G=LightGBM,S=S,L=L, verbose=verbose,mode = 'cluster',random_state=random_state)
with open(file_path, 'a') as file:
    file.write("LightGBM: " + str(result) + '\n')

result = iArt.test(Z=Z, X=X, Y=Y,G=LightGBM,S=S,L=L, verbose=verbose,mode = 'cluster', covariate_adjustment=3,random_state=random_state)
with open(file_path, 'a') as file:
    file.write("LightGBM with covariate adjustment: " + str(result) + '\n')