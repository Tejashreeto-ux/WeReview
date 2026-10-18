
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
    average_precision_score,
    confusion_matrix,
    classification_report,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = (
    ROOT / "data" / "processed"
    / "hollenbeck_baseline_candidates.csv"
)
MODEL_DIR = ROOT / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=["text", "label", "reviewer_id"]).copy()
df["text"] = df["text"].astype(str).str.strip()
df = df[df["text"].str.len() > 0].copy()
df["label"] = df["label"].astype(int)

print("Dataset size:", len(df))
print("Label counts:")
print(df["label"].value_counts().sort_index())

# Hold out entire reviewers.
splitter = GroupShuffleSplit(
    n_splits=1, test_size=0.20, random_state=52
)
train_idx, test_idx = next(
    splitter.split(df, df["label"], groups=df["reviewer_id"])
)

train = df.iloc[train_idx].copy()
test = df.iloc[test_idx].copy()

overlap = set(train["reviewer_id"]) & set(test["reviewer_id"])
print("Reviewer overlap:", len(overlap))

if overlap:
    raise ValueError("Reviewer leakage detected.")

if train["label"].nunique() < 2 or test["label"].nunique() < 2:
    raise ValueError("Training and test splits must contain both classes.")

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
        LinearSVC(class_weight="balanced", random_state=52),
    ),
])

print("\nTraining revised text-only SVM...")
model.fit(train["text"], train["label"])

pred = model.predict(test["text"])
scores = model.decision_function(test["text"])

cm = confusion_matrix(test["label"], pred, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()

results = {
    "model": "TF-IDF + Linear SVM revised",
    "label_source": "Provisional Hollenbeck candidate labels",
    "test_size": int(len(test)),
    "reviewer_overlap": len(overlap),
    "accuracy": float(accuracy_score(test["label"], pred)),
    "fake_precision": float(
        precision_score(test["label"], pred, zero_division=0)
    ),
    "fake_recall": float(
        recall_score(test["label"], pred, zero_division=0)
    ),
    "fake_f1": float(
        f1_score(test["label"], pred, zero_division=0)
    ),
    "pr_auc": float(
        average_precision_score(test["label"], scores)
    ),
    "false_positive_rate": float(
        fp / (fp + tn) if (fp + tn) else 0
    ),
    "confusion_matrix": cm.tolist(),
    "note": (
        "Metrics measure agreement with provisional labels, "
        "not independently verified fake-review truth."
    ),
}

joblib.dump(model, MODEL_DIR / "tfidf_svm_explained.joblib")

with open(
    MODEL_DIR / "svm_explained_results.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(results, f, indent=2)

print("\nTEST RESULTS")
print(json.dumps(results, indent=2))
print("\nClassification report:")
print(classification_report(
    test["label"],
    pred,
    labels=[0, 1],
    target_names=["Genuine candidate", "Fake candidate"],
    zero_division=0,
))
print("\nModel saved:", MODEL_DIR / "tfidf_svm_explained.joblib")
print("Metrics saved:", MODEL_DIR / "svm_explained_results.json")
