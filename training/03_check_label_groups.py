
from pathlib import Path
import pandas as pd

project_dir = Path(__file__).resolve().parent.parent
files = list((project_dir / "data").rglob(
    "public_reviews_dataset_cleaned.csv"
))

if not files:
    raise FileNotFoundError("Hollenbeck CSV not found.")

df = pd.read_csv(files[0])

# Provisional high-confidence fake candidate rule
fake_candidate = (
    (df["review_rating"] == 5)
    & df["fake_review_product"]
    & df["reviewer_classified_fake"]
)

# Conservative genuine candidate rule:
# reviewer is classified as honest AND product is not classified as fake
genuine_candidate = (
    df["reviewer_classified_honest"]
    & (~df["fake_review_product"])
)

# Exclude any overlap rather than assigning contradictory labels
genuine_candidate = genuine_candidate & (~fake_candidate)

uncertain = ~(fake_candidate | genuine_candidate)

print("Total reviews:", len(df))
print("Fake candidates:", int(fake_candidate.sum()))
print("Genuine candidates:", int(genuine_candidate.sum()))
print("Uncertain reviews:", int(uncertain.sum()))

print("\nPercentages:")
print("Fake:", round(fake_candidate.mean() * 100, 2), "%")
print("Genuine:", round(genuine_candidate.mean() * 100, 2), "%")
print("Uncertain:", round(uncertain.mean() * 100, 2), "%")

print("\nLabel groups are provisional, not independently verified.")
