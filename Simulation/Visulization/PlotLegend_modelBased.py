import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Create a new figure for the custom_lines_types legend
fig_leg1, ax_leg1 = plt.subplots(figsize=(18, 2))  # Increased size
ax_leg1.axis('off')

# Define custom styles with only cross markers (no lines)
custom_lines_types = [
    Line2D([0], [0], color='red', marker='x', markersize=10, linestyle='None',markeredgewidth=2),
    Line2D([0], [0], color='green', marker='x', markersize=10, linestyle='None',markeredgewidth=2)
]

# Create the legend
legend1 = ax_leg1.legend(custom_lines_types, 
                         ['Model-Based Imputation (Linear)', 'Model-Based Imputation (Boosting)'],
                         loc='center', 
                         ncol=2,  
                         fontsize='large')

# Save the legend with a larger figure size
fig_leg1.savefig('legend_typeone_error.pdf', format='pdf', bbox_inches='tight')

