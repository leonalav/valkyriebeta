import pandas as pd

# Replace with your actual file path
file_path = "/root/nanogpt/datasets/processed/WildChat_russian.parquet"
df = pd.read_parquet(file_path)

# Basic info
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# View first few rows
print("\nFirst few rows:")
print(df.head())

# Check for null values
print("\nNull value counts:")
print(df.isnull().sum())