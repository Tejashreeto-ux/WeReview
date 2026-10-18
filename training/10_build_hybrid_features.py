
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
    / "hollenbeck_hybrid_features.csv"
)

print("Loading original review data...")
df = pd.read_csv(DATA_PATH, low_memory=False)

print("Loading candidate labels...")
candidates = pd.read_csv(CANDIDATE_PATH)

# Join on the unique review ID, not reviewer ID or text.
df = df.merge(
    candidates[["review_id", "label", "reviewer_id"]],
    on="review_id",
    how="inner",
    validate="one_to_one",
    suffixes=("", "_candidate"),
)

if "reviewer_id" not in df.columns:
    df["reviewer_id"] = df["reviewer_id_candidate"]
print("Matched candidate reviews:", len(df))

# Combine review title and review text.
df["review_title"] = df["review_title"].fillna("").astype(str)
df["review_text"] = df["review_text"].fillna("").astype(str)

df["combined_text"] = (
    df["review_title"].str.strip()
    + " "
    + df["review_text"].str.strip()
).str.strip()

# Basic text features.
df["text_length"] = df["combined_text"].str.len()
df["word_count"] = df["combined_text"].str.split().str.len()
df["exclamation_count"] = df["combined_text"].str.count("!")
df["uppercase_count"] = df["combined_text"].str.count(r"[A-Z]")

# Numeric metadata.
for col in ["review_rating", "number_of_helpful", "number_of_photos"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

review_dates = pd.to_datetime(df["review_date"], errors="coerce")
df["review_month"] = review_dates.dt.month
df["review_year"] = review_dates.dt.year

feature_columns = [
    "review_rating",
    "number_of_helpful",
    "number_of_photos",
    "review_month",
    "review_year",
    "text_length",
    "word_count",
    "exclamation_count",
    "uppercase_count",
]

# Preserve IDs for splitting and auditing, not as model features.
result = df[
    ["review_id", "reviewer_id", "asin", "combined_text", "label"]
    + feature_columns
].copy()

result = result.drop_duplicates(subset=["review_id"])

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
result.to_csv(OUTPUT_PATH, index=False)

print("\nFeature preparation complete.")
print("Saved to:", OUTPUT_PATH)
print("Rows:", len(result))
print("\nLabel counts:")
print(result["label"].value_counts().sort_index())
print("\nFeature columns:")
print(feature_columns)
print("\nReminder: labels are provisional.")
print("Review IDs and reviewer IDs are retained for auditing.")
