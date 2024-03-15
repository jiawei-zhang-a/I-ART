import numpy as np
import os

results_dir = 'Result'
files = os.listdir(results_dir)

# Initialize a dictionary to store results by imputer type
results_by_imputer = {}

# Loop through each file, load the result, and organize by imputer type
for file in files:
    if file.startswith('test_') and file.endswith('.npy'):
        imputer_type = file.split('_')[1]  # Extract imputer type from the file name
        result = np.load(os.path.join(results_dir, file))
        
        # If the imputer type is not in the dictionary, add it with an empty list
        if imputer_type not in results_by_imputer:
            results_by_imputer[imputer_type] = []
        
        # Append the result to the correct list based on imputer type
        results_by_imputer[imputer_type].append(result)

# Process each imputer type to calculate confidence sets or other statistics
for imputer_type, results in results_by_imputer.items():
    results = np.array(results)  # Convert list to numpy array for easier processing
    
    # Assuming the second element in the result indicates whether the null hypothesis is rejected
    confidence_set = results[results[:, 1] == False, 0]
    
    print(f"One-way confidence set for beta using {imputer_type} imputer:", confidence_set)

# Optionally, you can save or further process the `results_by_imputer` dictionary as needed