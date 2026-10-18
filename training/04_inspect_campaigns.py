
from pathlib import Path
import pandas as pd

project_dir = Path(__file__).resolve().parent.parent
files = list((project_dir / "data").rglob(
    "public_reviews_dataset_cleaned.csv"
))

if not files:
    raise FileNotFoundError("Hollenbeck CSV not found.")

df = pd.read_csv(files[0], low_memory=False)

columns = [
    "fake_review_campaign_start_date",
    "fake_review_product",
    "reviewer_classified_fake",
    "reviewer_classified_honest",
    "reviewer_labeled_fake",
    "reviewer_labeled_honest",
    "review_is_removed_by_amazon",
]

print("=== CAMPAIGN DATE ===")
print("Missing:", df["fake_review_campaign_start_date"].isna().sum())
print("Non-missing:", df["fake_review_campaign_start_date"].notna().sum())
print(df["fake_review_campaign_start_date"].dropna().head(10).to_string(index=False))

print("\n=== REVIEWER LABEL COMBINATIONS ===")
print(
    df.groupby(
        ["reviewer_classified_fake", "reviewer_classified_honest"],
        dropna=False
    ).size().to_string()
)

print("\n=== LABEL FIELD VALUES ===")
for col in columns[4:]:
    print(f"\n{col}")
    print(df[col].value_counts(dropna=False).head(10).to_string())

print("\n=== REVIEW DATE RANGE ===")
print("First:", df["review_date"].min())
print("Last:", df["review_date"].max())
