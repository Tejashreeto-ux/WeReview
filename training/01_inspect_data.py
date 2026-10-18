
from pathlib import Path
import pandas as pd

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"

hollenbeck_path = DATA_DIR / "public_reviews_dataset_cleaned.csv"
amazon_path = DATA_DIR / "amazon_review_2023" / "All_Beauty.jsonl.gz"

# Inspect Hollenbeck dataset
print("\n===== HOLLENBECK DATASET =====")

if hollenbeck_path.exists():
    df = pd.read_csv(hollenbeck_path)

    print("Shape:", df.shape)
    print("\nColumns:")
    print(df.columns.tolist())
    print("\nMissing values:")
    print(df.isnull().sum())
    print("\nFirst 3 rows:")
    print(df.head(3).to_string())
else:
    print("CSV not found. Check the exact filename in data/.")

# Inspect Amazon Reviews 2023
print("\n===== AMAZON REVIEWS 2023 =====")

if amazon_path.exists():
    amazon = pd.read_json(
        amazon_path,
        lines=True,
        compression="gzip",
        nrows=5
    )

    print("First 5 rows:")
    print(amazon.head().to_string())
    print("\nColumns:")
    print(amazon.columns.tolist())
else:
    print("Amazon file not found. Check the folder and filename.")
