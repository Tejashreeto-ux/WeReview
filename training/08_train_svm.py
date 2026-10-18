
from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "processed" / "hollenbeck_baseline_candidates.csv"
MODEL_DIR = PROJECT_DIR / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=["text", "label", "reviewer_id"]).copy()
df["text"] = df["text"].astype(str).str.strip()
df = df[df["text"].str.len() > 0].copy()
df["label"] = df["label"].astype(int)

print("Dataset size:", len(df))
print("\nLabel counts:")
print(df["label"].value_counts().sort_index())

# Split by reviewer to reduce reviewer overlap between sets.
splitter = GroupShuffleSplit(
    n_splits=1, test_size=0.20, random_state=42
)
train_val_idx, test_idx = next(
    splitter.split(df, df["label"], groups=df["reviewer_id"])
)

train_val = df.iloc[train_val_idx].copy()
test = df.iloc[test_idx].copy()

splitter2 = GroupShuffleSplit(
    n_splits=1, test_size=0.20, random_state=43
)
train_idx, val_idx = next(
    splitter2.split(
        train_val,
        train_val["label"],
        groups=train_val["reviewer_id"],
    )
)

train = train_val.iloc[train_idx].copy()
val = train_val.iloc[val_idx].copy()

for a, b, name in [
    (train, val, "train-validation"),
    (train, test, "train-test"),
    (val, test, "validation-test"),
]:
    overlap = set(a["reviewer_id"]) & set(b["reviewer_id"])
    print(f"{name} reviewer overlap:", len(overlap))

for name, part in [
    ("Train", train),
    ("Validation", val),
    ("Test", test),
]:
    print(f"\n{name} size: {len(part)}")
    print(part["label"].value_counts().sort_index())

# Stop if a split does not contain both classes.
for name, part in [
    ("train", train),
    ("validation", val),
    ("test", test),
]:
    if part["label"].nunique() < 2:
        raise ValueError(
            f"The {name} split contains only one class. "
            "We must adjust the group split before training."
        )

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            max_features=50000,
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,
        ),
    ),
    (
        "svm",
        LinearSVC(class_weight="balanced", random_state=42),
    ),
])

print("\nTraining TF-IDF + Linear SVM...")
model.fit(train["text"], train["label"])

print("\nVALIDATION RESULTS")
val_pred = model.predict(val["text"])
print(classification_report(
    val["label"],
    val_pred,
    labels=[0, 1],
    target_names=["Genuine candidate", "Fake candidate"],
    zero_division=0,
))
print("Validation confusion matrix:")
print(confusion_matrix(val["label"], val_pred, labels=[0, 1]))

# Save the baseline model for later comparison.
joblib.dump(model, MODEL_DIR / "tfidf_svm_baseline.joblib")

# Evaluate the held-out test set once for this baseline.
test_pred = model.predict(test["text"])

results = {
    "model": "TF-IDF + Linear SVM",
    "label_source": "Provisional Hollenbeck candidate labels",
    "test_size": int(len(test)),
    "accuracy": float(accuracy_score(test["label"], test_pred)),
    "fake_precision": float(
        precision_score(test["label"], test_pred, pos_label=1, zero_division=0)
    ),
    "fake_recall": float(
        recall_score(test["label"], test_pred, pos_label=1, zero_division=0)
    ),
    "fake_f1": float(
        f1_score(test["label"], test_pred, pos_label=1, zero_division=0)
    ),
    "confusion_matrix_labels": ["genuine_candidate", "fake_candidate"],
    "confusion_matrix": confusion_matrix(
        test["label"], test_pred, labels=[0, 1]
    ).tolist(),
}

with open(MODEL_DIR / "svm_baseline_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nHELD-OUT TEST RESULTS")
print(classification_report(
    test["label"],
    test_pred,
    labels=[0, 1],
    target_names=["Genuine candidate", "Fake candidate"],
    zero_division=0,
))
print("Test confusion matrix (rows=true, columns=predicted; labels 0, 1):")
print(confusion_matrix(test["label"], test_pred, labels=[0, 1]))
print("\nSaved model:", MODEL_DIR / "tfidf_svm_baseline.joblib")
print("Saved metrics:", MODEL_DIR / "svm_baseline_results.json")
print("\nReminder: results measure agreement with provisional labels,")
print("not verified real-world fake-review detection.")
