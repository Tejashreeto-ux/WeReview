
from pathlib import Path
import pandas as pd

from catboost import CatBoostClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = (
    ROOT / "data" / "processed"
    / "hollenbeck_hybrid_features.csv"
)
BEHAVIOUR_PATH = (
    ROOT / "data" / "processed"
    / "hollenbeck_hybrid_features_v2.csv"
)

text_df = pd.read_csv(DATA_PATH)
behaviour_df = pd.read_csv(BEHAVIOUR_PATH)

df = text_df.merge(
    behaviour_df.drop(columns=["label"]),
    on=["review_id", "reviewer_id", "asin"],
    how="inner",
    validate="one_to_one",
)

base_features = [
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

# Hold out entire products from training.
splitter = GroupShuffleSplit(
    n_splits=1, test_size=0.20, random_state=123
)
train_idx, test_idx = next(
    splitter.split(df, df["label"], groups=df["asin"])
)

train = df.iloc[train_idx].copy()
test = df.iloc[test_idx].copy()

print("Train rows:", len(train))
print("Test rows:", len(test))
print(
    "Product overlap:",
    len(set(train["asin"]) & set(test["asin"]))
)

experiments = {
    "All features": base_features,
    "Without rating and product rating mean": [
        col for col in base_features
        if col not in [
            "review_rating",
            "product_prior_rating_mean",
        ]
    ],
}

for name, features in experiments.items():
    X_train = train[features].copy()
    X_test = test[features].copy()

    medians = X_train.median()
    X_train = X_train.fillna(medians)
    X_test = X_test.fillna(medians)

    model = CatBoostClassifier(
        iterations=600,
        depth=7,
        learning_rate=0.05,
        loss_function="Logloss",
        verbose=False,
        random_seed=42,
        allow_writing_files=False,
    )

    model.fit(X_train, train["label"])
    pred = model.predict(X_test).astype(int).ravel()

    print(f"\n{name}")
    print(f"Accuracy: {accuracy_score(test['label'], pred):.4f}")
    print(
        "Fake precision:",
        f"{precision_score(test['label'], pred, zero_division=0):.4f}",
    )
    print(
        "Fake recall:",
        f"{recall_score(test['label'], pred, zero_division=0):.4f}",
    )
    print(
        "Fake F1:",
        f"{f1_score(test['label'], pred, zero_division=0):.4f}",
    )

print("\nLabels are provisional; interpret these as audit experiments.")
