
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = (
    PROJECT_DIR
    / "data"
    / "public_reviews_dataset_cleaned.csv_ (6)"
    / "public_reviews_dataset_cleaned.csv"
)
OUTPUT_DIR = PROJECT_DIR / "data" / "processed"
OUTPUT_PATH = OUTPUT_DIR / "hollenbeck_baseline_candidates.csv"

print("Reading dataset:", DATA_PATH)

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

df = pd.read_csv(DATA_PATH, low_memory=False)

print("Total reviews:", len(df))

# Provisional fake-review candidates
fake_candidate = (
    df["review_rating"].eq(5)
    & df["fake_review_product"].eq(True)
    & df["reviewer_classified_fake"].eq(True)
)

# More conservative genuine-review candidates
genuine_candidate = (
    df["reviewer_labeled_honest"].eq(1)
    & df["reviewer_classified_honest"].eq(True)
    & ~fake_candidate
)

selected = df.loc[fake_candidate | genuine_candidate].copy()
selected["label"] = fake_candidate.loc[selected.index].astype(int)

selected = selected.rename(
    columns={
        "review_text": "text",
        "reviewer_id": "reviewer_id",
    }
)

selected = selected[
    ["review_id", "text", "label", "reviewer_id"]
]
selected["text"] = selected["text"].fillna("").astype(str).str.strip()
selected = selected[selected["text"].str.len() > 0]
selected = selected.dropna(subset=["reviewer_id"])

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
selected.to_csv(OUTPUT_PATH, index=False)

print("\nCandidate dataset prepared.")
print("Saved to:", OUTPUT_PATH)
print("Selected reviews:", len(selected))
print("\nCandidate label counts (0 = genuine, 1 = fake):")
print(selected["label"].value_counts().sort_index())
print("\nImportant: these are provisional labels, not verified ground truth.")
