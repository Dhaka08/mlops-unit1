import pandas as pd

# Load dataset
# Using built-in sample dataset from sklearn
from sklearn.datasets import load_iris

data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)

# Print basic statistics
print("First 5 Rows:")
print(df.head())

print("\nBasic Statistics:")
print(df.describe())

print("\nDataset Shape:")
print(df.shape)