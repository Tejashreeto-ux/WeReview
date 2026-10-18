
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    ROOT / "data"
    / "public_reviews_dataset_cleaned.csv_ (6)"
    / "public_reviews_dataset_cleaned.csv"
)

CANDIDATE_PATH = (
    ROOT / "data" / "processed"
    / "hollenbeck_baseline_candidates.csv"
)

OUTPUT_PATH = (
    ROOT / "data" / "processed"
    / "hollenbeck_hybrid_features_v3.csv"
)

print("Loading original reviews...")
df = pd.read_csv(DATA_PATH, low_memory=False)

df["review_date"] = pd.to_datetime(
    df["review_date"], errors="coerce"
)

# Ensure each review has a usable ID.
df = df.dropna(subset=["review_id"]).copy()

# Calculate reviewer history in chronological order.
reviewer_history = df.sort_values(
    ["reviewer_id", "review_date", "review_id"],
    na_position="first"
).copy()

reviewer_history["reviewer_prior_review_count"] = (
    reviewer_history.groupby("reviewer_id").cumcount()
)

reviewer_features = reviewer_history[
    ["review_id", "reviewer_prior_review_count"]
].copy()

# Calculate product history independently.
product_history = df.sort_values(
    ["asin", "review_date", "review_id"],
    na_position="first"
).copy()

product_history["product_prior_review_count"] = (
    product_history.groupby("asin").cumcount()
)

product_features = product_history[
    ["review_id", "product_prior_review_count"]
].copy()

# Merge the two historical feature sets.
features = reviewer_features.merge(
    product_features,
    on="review_id",
    how="inner",
    validate="one_to_one"
)

# Attach the existing provisional labels.
candidates = pd.read_csv(CANDIDATE_PATH)

candidates = candidates[
    ["review_id", "label"]
].drop_duplicates(subset=["review_id"])

result = candidates.merge(
    features,
    on="review_id",
    how="left",
    validate="one_to_one"
)

# Missing history means no usable historical record was found.
count_columns = [
    "reviewer_prior_review_count",
    "product_prior_review_count",
]

result[count_columns] = (
    result[count_columns].fillna(0).astype("int64")
)

OUTPUT_PATH.parent.mkdir(
    parents=True, exist_ok=True
)

result.to_csv(OUTPUT_PATH, index=False)

print("\nFeature preparation complete.")
print("Rows:", len(result))
print("Columns:", result.columns.tolist())
print("Missing values:")
print(result.isna().sum())
print("Saved to:", OUTPUT_PATH)
print(
    "\nImportant: labels remain provisional. "
    "Historical counts still require leakage-safe evaluation."
)
