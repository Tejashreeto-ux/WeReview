
from pathlib import Path
import pandas as pd

project_dir = Path(__file__).resolve().parent.parent
files = list((project_dir / "data").rglob(
    "public_reviews_dataset_cleaned.csv"
))

if not files:
    raise FileNotFoundError("Hollenbeck CSV not found.")

df = pd.read_csv(files[0], low_memory=False)

# Reviewer-level labels supplied by the dataset
fake_labeled = df["reviewer_labeled_fake"].eq(1)
honest_labeled = df["reviewer_labeled_honest"].eq(1)

# Strict fake candidate rule
fake_candidate = (
    df["review_rating"].eq(5)
    & df["fake_review_product"].eq(True)
    & df["reviewer_classified_fake"].eq(True)
)

# Strict honest candidate rule using explicit honest labels
honest_candidate = (
    honest_labeled
    & df["reviewer_classified_honest"].eq(True)
    & ~fake_candidate
)

# Identify conflicting or unlabeled records
conflict = fake_candidate & honest_candidate
uncertain = ~(fake_candidate | honest_candidate)

print("Total reviews:", len(df))
print("Explicit fake reviewer labels:", int(fake_labeled.sum()))
print("Explicit honest reviewer labels:", int(honest_labeled.sum()))
print("Fake candidates:", int(fake_candidate.sum()))
print("Honest candidates:", int(honest_candidate.sum()))
print("Conflicting candidates:", int(conflict.sum()))
print("Uncertain reviews:", int(uncertain.sum()))

print("\nReminder: reviewer labels are not independently verified review-level truth.")
