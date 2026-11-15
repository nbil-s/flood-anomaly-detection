
from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/raw")

csv_files = list(DATA_DIR.glob("*.csv"))

if not csv_files:
    print("No CSV files found in data/raw")
    raise SystemExit

for file in csv_files:
    print("\n" + "=" * 70)
    print(f"DATASET: {file.name}")
    print("=" * 70)

    try:
        df = pd.read_csv(file)

        print("\nShape:")
        print(df.shape)

        print("\nColumns:")
        print(df.columns.tolist())

        print("\nData types:")
        print(df.dtypes)

        print("\nMissing values:")
        print(df.isnull().sum())

        print("\nFirst 5 rows:")
        print(df.head())

        print("\nDuplicate rows:")
        print(df.duplicated().sum())

    except Exception as error:
        print(f"Could not inspect {file.name}: {error}")
