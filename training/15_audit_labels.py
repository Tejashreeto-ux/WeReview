
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

RAW_PATH = (
    ROOT / "data"
    / "public_reviews_dataset_cleaned.csv_ (6)"
    / "public_reviews_dataset_cleaned.csv"
)
CANDIDATE_PATH = (
    ROOT / "data" / "processed"
    / "hollenbeck_baseline_candidates.csv"
)

print("Loading candidate labels...")
candidates = pd.read_csv(CANDIDATE_PATH)

print("Loading original label-related columns...")
columns = [
    "review_id",
    "review_rating",
    "fake_review_product",
    "reviewer_classified_fake",
    "reviewer_classified_honest",
    "reviewer_labeled_fake",
    "reviewer_labeled_honest",
    "review_is_removed_by_amazon",
]

raw = pd.read_csv(
    RAW_PATH,
    usecols=columns,
    low_memory=False,
)

df = candidates.merge(
    raw,
    on="review_id",
    how="left",
    validate="one_to_one",
)

print("\nTotal candidates:", len(df))
print("\nCandidate label counts:")
print(df["label"].value_counts().sort_index())

# Reconstruct the provisional fake-candidate rule.
fake_rule = (
    df["review_rating"].eq(5)
    & df["fake_review_product"].eq(True)
    & df["reviewer_classified_fake"].eq(True)
)

# Reconstruct the provisional genuine-candidate rule.
genuine_rule = (
    df["reviewer_labeled_honest"].eq(1)
    & df["reviewer_classified_honest"].eq(True)
    & ~fake_rule
)

print("\n--- Label rule audit ---")
print(
    "Label 1 rows satisfying fake rule:",
    int((df["label"].eq(1) & fake_rule).sum()),
)
print(
    "Label 1 rows NOT satisfying fake rule:",
    int((df["label"].eq(1) & ~fake_rule).sum()),
)
print(
    "Label 0 rows satisfying genuine rule:",
    int((df["label"].eq(0) & genuine_rule).sum()),
)
print(
    "Label 0 rows NOT satisfying genuine rule:",
    int((df["label"].eq(0) & ~genuine_rule).sum()),
)

print("\n--- Rating by candidate label ---")
print(pd.crosstab(df["label"], df["review_rating"]))

for col in [
    "fake_review_product",
    "reviewer_classified_fake",
    "reviewer_classified_honest",
    "reviewer_labeled_fake",
    "reviewer_labeled_honest",
    "review_is_removed_by_amazon",
]:
    print(f"\n--- {col} by label ---")
    print(pd.crosstab(
        df["label"],
        df[col].fillna("Missing").astype(str),
        normalize="index",
    ).round(3))

print("\nAudit complete. No files were changed.")
print("Reminder: candidate labels are provisional.")
