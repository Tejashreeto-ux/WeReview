
from pathlib import Path
import pandas as pd

project_dir = Path(__file__).resolve().parent.parent
data_dir = project_dir / "data"

# Find the CSV even if it is inside an extracted subfolder
files = list(data_dir.rglob("public_reviews_dataset_cleaned.csv"))

if not files:
    raise FileNotFoundError(
        "Could not find public_reviews_dataset_cleaned.csv inside data/"
    )

csv_path = files[0]
print("Dataset:", csv_path)

df = pd.read_csv(csv_path)

print("\nDataset shape:", df.shape)
print("\nReview ratings:")
print(df["review_rating"].value_counts(dropna=False).sort_index())

print("\nFake-review product counts:")
print(df["fake_review_product"].value_counts(dropna=False))

print("\nReviewer classification counts:")
print(df["reviewer_classified_fake"].value_counts(dropna=False))
print(df["reviewer_classified_honest"].value_counts(dropna=False))

# Candidate fake-review rule from the dataset documentation.
candidate_fake = (
    (df["review_rating"] == 5)
    & (df["fake_review_product"] == True)
    & (df["reviewer_classified_fake"] == True)
)

print("\nCandidate fake reviews:", int(candidate_fake.sum()))
print("Candidate fake percentage:", round(candidate_fake.mean() * 100, 2), "%")

print("\nMissing values in important fields:")
print(df[
    [
        "review_rating",
        "review_text",
        "fake_review_product",
        "reviewer_classified_fake",
        "reviewer_classified_honest",
        "reviewer_labeled_fake",
        "reviewer_labeled_honest",
    ]
].isna().sum())
