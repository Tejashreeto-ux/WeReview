
from pathlib import Path
import json

import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import GroupShuffleSplit
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
    / "hollenbeck_hybrid_features_v3.csv"
)

MODEL_DIR = ROOT / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "catboost_revised.cbm"
RESULT_PATH = MODEL_DIR / "catboost_revised_results.json"

print("Loading revised features...")
df = pd.read_csv(DATA_PATH)

features = [
    "reviewer_prior_review_count",
    "product_prior_review_count",
]

X = df[features].copy()
y = df["label"].astype(int)

# Group by reviewer so one reviewer cannot occur
# in both training and test sets.
groups = df["review_id"].astype(str)

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42,
)

train_idx, test_idx = next(
    splitter.split(X, y, groups=groups)
)

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]
y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]

print("Training rows:", len(X_train))
print("Test rows:", len(X_test))

model = CatBoostClassifier(
    iterations=500,
    depth=6,
    learning_rate=0.05,
    loss_function="Logloss",
    eval_metric="AUC",
    random_seed=42,
    verbose=100,
    allow_writing_files=False,
)

print("\nTraining CatBoost...")
model.fit(
    X_train,
    y_train,
    eval_set=(X_test, y_test),
    early_stopping_rounds=50,
    verbose=100,
)

predictions = model.predict(X_test).astype(int).ravel()
probabilities = model.predict_proba(X_test)[:, 1]

results = {
    "experiment": "revised_behavioral_catboost",
    "features": features,
    "train_size": int(len(train_idx)),
    "test_size": int(len(test_idx)),
    "accuracy": float(accuracy_score(y_test, predictions)),
    "fake_precision": float(
        precision_score(y_test, predictions, zero_division=0)
    ),
    "fake_recall": float(
        recall_score(y_test, predictions, zero_division=0)
    ),
    "fake_f1": float(
        f1_score(y_test, predictions, zero_division=0)
    ),
    "pr_auc": float(
        average_precision_score(y_test, probabilities)
    ),
    "confusion_matrix": confusion_matrix(
        y_test, predictions
    ).tolist(),
    "best_iteration": int(model.get_best_iteration()),
    "label_warning": (
        "Labels are provisional dataset-derived candidates, "
        "not independently verified fake-review ground truth."
    ),
}

model.save_model(str(MODEL_PATH))

with open(RESULT_PATH, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print("\n--- Revised CatBoost results ---")
print(json.dumps(results, indent=2))

print("\nClassification report:")
print(classification_report(
    y_test, predictions, zero_division=0
))

print("\nModel saved to:", MODEL_PATH)
print("Metrics saved to:", RESULT_PATH)
