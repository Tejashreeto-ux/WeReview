
from pathlib import Path
import pandas as pd

project_dir = Path(__file__).resolve().parent.parent
files = list((project_dir / "data").rglob(
    "public_reviews_dataset_cleaned.csv"
))

if not files:
    raise FileNotFoundError("Hollenbeck CSV not found.")

df = pd.read_csv(files[0], low_memory=False)

fake_candidate = (
    df["review_rating"].eq(5)
    & df["fake_review_product"].eq(True)
    & df["reviewer_classified_fake"].eq(True)
)

genuine_candidate = (
    df["reviewer_labeled_honest"].eq(1)
    & df["reviewer_classified_honest"].eq(True)
    & ~fake_candidate
)

# Preview samples, without assigning labels to uncertain rows
print("=== FAKE CANDIDATE EXAMPLES ===")
print(
    df.loc[fake_candidate, [
        "review_rating", "review_title", "review_text",
        "fake_review_product", "reviewer_classified_fake"
    ]].head(5).to_string(index=False)
)

print("\n=== GENUINE CANDIDATE EXAMPLES ===")
print(
    df.loc[genuine_candidate, [
        "review_rating", "review_title", "review_text",
        "reviewer_labeled_honest", "reviewer_classified_honest"
    ]].head(5).to_string(index=False)
)

print("\nFake candidates:", int(fake_candidate.sum()))
print("Genuine candidates:", int(genuine_candidate.sum()))
