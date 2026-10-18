
from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed" / "hollenbeck_baseline_candidates.csv"
MODEL_DIR = ROOT / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA)
df = df.dropna(subset=["text", "label", "reviewer_id"]).copy()
df["text"] = df["text"].astype(str).str.strip()
df = df[df["text"].str.len() > 0].copy()
df["label"] = df["label"].astype(int)

# Recreate the same reviewer-grouped split as the baseline.
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
train_idx, _ = next(
    splitter2.split(
        train_val, train_val["label"],
        groups=train_val["reviewer_id"]
    )
)
train = train_val.iloc[train_idx].copy()

print("Training reviews:", len(train))
print("Test reviews:", len(test))

if train["label"].nunique() < 2 or test["label"].nunique() < 2:
    raise ValueError("Train or test set contains only one class.")

classifiers = {
    "Logistic Regression": LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=42
    ),
    "Linear SVM": LinearSVC(
        class_weight="balanced", random_state=42
    ),
}

results = []

for name, classifier in classifiers.items():
    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=50000,
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,
        )),
        ("classifier", classifier),
    ])

    print(f"\nTraining {name}...")
    model.fit(train["text"], train["label"])
    predictions = model.predict(test["text"])

    result = {
        "model": name,
        "accuracy": round(
            accuracy_score(test["label"], predictions) * 100, 2
        ),
        "fake_precision": round(
            precision_score(
                test["label"], predictions,
                pos_label=1, zero_division=0
            ) * 100, 2
        ),
        "fake_recall": round(
            recall_score(
                test["label"], predictions,
                pos_label=1, zero_division=0
            ) * 100, 2
        ),
        "fake_f1": round(
            f1_score(
                test["label"], predictions,
                pos_label=1, zero_division=0
            ) * 100, 2
        ),
    }
    results.append(result)
    print(result)

    if name == "Logistic Regression":
        joblib.dump(
            model, MODEL_DIR / "tfidf_logistic_regression.joblib"
        )

with open(MODEL_DIR / "model_comparison.json", "w") as file:
    json.dump(results, file, indent=2)

print("\nFINAL COMPARISON")
print(pd.DataFrame(results).to_string(index=False))
print("\nSaved results to model/model_comparison.json")
print("Original SVM model remains unchanged.")
