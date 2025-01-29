import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LogisticRegression

# Example dataset with binary missing values (0/1)
X = np.array([[1, 0, np.nan], 
              [0, 1, 1], 
              [1, np.nan, 0], 
              [0, 1, np.nan]])

# IterativeImputer using Logistic Regression for binary missing values
imputer = IterativeImputer(estimator=LogisticRegression(), max_iter=10, random_state=42)

# Impute missing values
X_imputed = imputer.fit_transform(X)

print(X_imputed)
