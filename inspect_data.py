
from pathlib import Path
import pandas as pd

# Locate the mathematics dataset
project_dir = Path(__file__).parent
csv_path = project_dir / "data" / "student-mat.csv"

# The UCI dataset uses semicolons as separators
df = pd.read_csv(csv_path, sep=";")

print("=" * 50)
print("STUDENT PERFORMANCE DATASET")
print("=" * 50)

print("\nDataset shape:")
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")

print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst five students:")
print(df.head())

print("\nMissing values per column:")
print(df.isnull().sum())

print("\nFinal grade (G3) summary:")
print(df["G3"].describe())
