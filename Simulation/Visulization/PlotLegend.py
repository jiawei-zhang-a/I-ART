import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Define the line styles for each method with updated colors and labels
legend_lines = [
    Line2D([0], [0], color='blue', lw=2, linestyle='-'),
    Line2D([0], [0], color='red', lw=2, linestyle='-'),
    Line2D([0], [0], color='green', lw=2, linestyle='-'),
    Line2D([0], [0], color='purple', lw=2, linestyle='-')
]

# Define labels for each method
legend_labels = [
    'Method 1 (Non-Informative Imputation)',
    'Method 2 (Algo 3 – Logistic)',
    'Method 3 (Algo 3 – Boosting)',
    'Method 4 (Oracle)'
]

# Create a figure specifically for the legend
fig, ax = plt.subplots(figsize=(10, 1))  # Adjust size for spacing as needed
ax.axis('off')  # Hide axes for a cleaner look

# Create the legend
ax.legend(legend_lines, legend_labels, loc='center', fontsize='large', ncol=4)

# Save the legend to a PDF file
output_filename = "legend_survival.pdf"
fig.savefig(output_filename, format='pdf', bbox_inches='tight')

# Close the figure after saving
plt.close(fig)

# Provide the file to the user
output_filename
