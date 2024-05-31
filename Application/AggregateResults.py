import numpy as np
import os
import re  # Import the regex module
import matplotlib.pyplot as plt

results_dir = 'ConfidenceRegions'
files = os.listdir(results_dir)

# Initialize a dictionary to store results by imputer type
results_by_imputer = {}

# Define a regex pattern to match the imputer type more accurately.
# This pattern assumes that the imputer type is followed by an underscore and possibly a negative sign before the numeric value.
pattern = re.compile(r'test_([a-zA-Z]+)_?-?\d')

# Loop through each file, load the result, and organize by imputer type
for file in files:
    if file.startswith('test_') and file.endswith('.npy'):
        match = pattern.search(file)  # Search for the pattern in the filename
        if match:
            imputer_type = match.group(1)  # Extract imputer type from the regex match
            result = np.load(os.path.join(results_dir, file))
            
            # If the imputer type is not in the dictionary, add it with an empty list
            if imputer_type not in results_by_imputer:
                results_by_imputer[imputer_type] = []
            
            # Append the result to the correct list based on imputer type
            results_by_imputer[imputer_type].append(result)

# Process each imputer type to calculate confidence sets or other statistics
for imputer_type, results in results_by_imputer.items():
    results = np.array(results)  # Convert list to numpy array for easier processing

    # make a plot of the p-values over the beta values
    # Assuming the first element in the result is the beta value
    beta_values = results[:, 0]
    p_values = results[:, 2]
    """plt.scatter(beta_values, p_values, label=imputer_type)
    plt.xlabel('Beta')
    plt.ylabel('p-value')
    plt.title('p-values over beta values')
    plt.legend()
    plt.show()"""

    # Create the plot smoother
    # Add the metric to the X axis    plt.scatter(beta_values, p_values, label=imputer_type,s=1) 

    # Assuming beta_values and p_values are already defined, along with imputer_type
    plt.scatter(beta_values, p_values, label=imputer_type, s=1)

    # Add a vertical line at beta = 0.125
    plt.axvline(x=0.125, color='r', linestyle='--', label='Metric 0.125')

    plt.xlabel('Beta')
    plt.ylabel('p-value')
    plt.title('p-values over beta values')
    plt.legend()
    #plt.show()


    # Assuming the last element in the result is the p-value
    # We consider the null hypothesis rejected if p-value <= 0.1
    confidence_set = results[results[:, 2] > 0.05, 0].round(5)

    # Sort the confidence set
    confidence_set = np.sort(confidence_set)
    
    print(f"One-way confidence set for beta using {imputer_type} imputer:", confidence_set)

