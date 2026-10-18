
from pathlib import Path
import json
import pandas as pd
import joblib

from catboost import CatBoostClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

text_path = DATA_DIR / "hollenbeck_hybrid_features.csv"
behaviour_path = DATA_DIR / "hollenbeck_hybrid_features_v2.csv"

text_df = pd.read_csv(text_path)
behaviour_df = pd.read_csv(behaviour_path)

df = text_df.merge(
    behaviour_df.drop(columns=["label"]),
    on=["review_id", "reviewer_id", "asin"],
    how="inner",
    validate="one_to_one",
)

if len(df) != len(text_df):
    raise ValueError("Feature merge lost rows; stop and investigate.")

df = df.dropna(subset=["label"]).copy()
df["label"] = df["label"].astype(int)

print("Merged rows:", len(df))
print("Label counts:")
print(df["label"].value_counts().sort_index())

# Group by reviewer to reduce reviewer overlap.
splitter = GroupShuffleSplit(
    n_splits=1, test_size=0.20, random_state=42
)
train_idx, test_idx = next(
    splitter.split(df, df["label"], groups=df["reviewer_id"])
)
train = df.iloc[train_idx].copy()
test = df.iloc[test_idx].copy()

# Keep IDs and text out of the raw CatBoost input.
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
    "reviewer_prior_review_count",
    "product_prior_review_count",
    "reviewer_prior_rating_mean",
    "product_prior_rating_mean",
]

X_train = train[feature_columns].copy()
X_test = test[feature_columns].copy()

# Fill missing numeric values using training-set medians only.
medians = X_train.median()
X_train = X_train.fillna(medians)
X_test = X_test.fillna(medians)

model = CatBoostClassifier(
    iterations=600,
    depth=7,
    learning_rate=0.05,
    loss_function="Logloss",
    eval_metric="AUC",
    verbose=100,
    random_seed=42,
    allow_writing_files=False,
)

print("\nTraining CatBoost...")
model.fit(X_train, train["label"])

predictions = model.predict(X_test).astype(int).ravel()

results = {
    "model": "CatBoost structured features",
    "label_source": "Provisional Hollenbeck candidate labels",
    "test_size": int(len(test)),
    "accuracy": float(accuracy_score(test["label"], predictions)),
    "fake_precision": float(
        precision_score(test["label"], predictions, zero_division=0)
    ),
    "fake_recall": float(
        recall_score(test["label"], predictions, zero_division=0)
    ),
    "fake_f1": float(
        f1_score(test["label"], predictions, zero_division=0)
    ),
    "confusion_matrix": confusion_matrix(
        test["label"], predictions, labels=[0, 1]
    ).tolist(),
    "features": feature_columns,
}

model.save_model(str(MODEL_DIR / "catboost_hybrid.cbm"))
joblib.dump(medians.to_dict(), MODEL_DIR / "catboost_feature_medians.joblib")

with open(MODEL_DIR / "catboost_hybrid_results.json", "w") as file:
    json.dump(results, file, indent=2)

print("\nTEST RESULTS")
print(json.dumps(results, indent=2))
print("\nSaved model and metrics in the model folder.")
print("Reminder: scores reflect provisional labels.")
