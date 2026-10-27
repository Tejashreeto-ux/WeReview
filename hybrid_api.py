
from pathlib import Path
from datetime import datetime
import re
import string

import joblib
import numpy as np
import torch
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from textblob import TextBlob

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models" / "final_hybrid_model"

app = Flask(__name__)

print("Loading DistilBERT tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR / "tokenizer")

print("Loading DistilBERT model...")
text_model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
text_model.eval()

print("Loading XGBoost classifier...")
classifier = joblib.load(
    MODEL_DIR / "hybrid_distilbert_behavior_xgboost.joblib"
)

THRESHOLD = float(
    (MODEL_DIR / "classification_threshold.txt").read_text().strip()
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
text_model.to(DEVICE)

FEATURES = [
    "rating", "is_1_star", "is_5_star", "helpful_votes",
    "has_helpful_votes", "photo_count", "has_photos",
    "review_text_length", "review_word_count", "review_title_length",
    "review_title_word_count", "review_year", "review_month",
    "review_day_of_week", "punctuation_count", "exclamation_count",
    "question_count", "comma_count", "digit_count", "uppercase_count",
    "letter_count", "uppercase_ratio", "digit_ratio", "avg_word_length",
    "unique_word_ratio", "title_text_length_ratio",
    "sentiment_polarity", "sentiment_subjectivity",
    "reviewer_review_count", "reviewer_avg_rating",
    "reviewer_rating_std", "reviewer_avg_helpful"
]

def make_features(text, title, data):
    words = re.findall(r"\b\w+\b", text.lower())
    letters = [c for c in text if c.isalpha()]
    now = datetime.now()
    sentiment = TextBlob(text)

    rating = float(data.get("rating", 5))
    helpful = float(data.get("helpful_votes", 0))
    photos = float(data.get("photo_count", 0))

    # These reviewer-history values are only estimates unless
    # calculated from real stored review history.
    values = {
        "rating": rating,
        "is_1_star": int(rating == 1),
        "is_5_star": int(rating == 5),
        "helpful_votes": helpful,
        "has_helpful_votes": int(helpful > 0),
        "photo_count": photos,
        "has_photos": int(photos > 0),
        "review_text_length": len(text),
        "review_word_count": len(words),
        "review_title_length": len(title),
        "review_title_word_count": len(title.split()),
        "review_year": now.year,
        "review_month": now.month,
        "review_day_of_week": now.weekday(),
        "punctuation_count": sum(c in string.punctuation for c in text),
        "exclamation_count": text.count("!"),
        "question_count": text.count("?"),
        "comma_count": text.count(","),
        "digit_count": sum(c.isdigit() for c in text),
        "uppercase_count": sum(c.isupper() for c in text),
        "letter_count": len(letters),
        "uppercase_ratio": sum(c.isupper() for c in letters) / max(len(letters), 1),
        "digit_ratio": sum(c.isdigit() for c in text) / max(len(text), 1),
        "avg_word_length": sum(map(len, words)) / max(len(words), 1),
        "unique_word_ratio": len(set(words)) / max(len(words), 1),
        "title_text_length_ratio": len(title) / max(len(text), 1),
        "sentiment_polarity": sentiment.sentiment.polarity,
        "sentiment_subjectivity": sentiment.sentiment.subjectivity,
        "reviewer_review_count": float(data.get("reviewer_review_count", 1)),
        "reviewer_avg_rating": float(data.get("reviewer_avg_rating", rating)),
        "reviewer_rating_std": float(data.get("reviewer_rating_std", 0)),
        "reviewer_avg_helpful": float(data.get("reviewer_avg_helpful", 0)),
    }

    return np.array([[values[f] for f in FEATURES]], dtype=np.float32)


def get_embedding(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        output = text_model.base_model(**inputs).last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1)
        embedding = (output * mask).sum(dim=1) / mask.sum(dim=1)

    return embedding.cpu().numpy().astype(np.float32)


@app.get("/health")
def health():
    return jsonify({
        "status": "ready",
        "model": "DistilBERT + behavioral features + XGBoost",
        "feature_count": classifier.n_features_in_,
        "threshold": THRESHOLD
    })


@app.post("/predict")
def predict():
    data = request.get_json(silent=True) or {}
    text = data.get("review_text", "").strip()
    title = data.get("review_title", "")

    if not text:
        return jsonify({"error": "review_text is required"}), 400

    try:
        behavior = make_features(text, title, data)
        embedding = get_embedding(text)
        combined = np.concatenate([embedding, behavior], axis=1)

        if combined.shape[1] != classifier.n_features_in_:
            return jsonify({
                "error": "Feature count mismatch",
                "expected": int(classifier.n_features_in_),
                "received": int(combined.shape[1])
            }), 500

        probability = float(classifier.predict_proba(combined)[0][1])
        prediction = "Potentially Fake" if probability >= THRESHOLD else "Genuine"

        return jsonify({
            "prediction": prediction,
            "fake_probability": round(probability, 4),
            "genuine_probability": round(1 - probability, 4),
            "threshold": THRESHOLD,
            "model": "DistilBERT + behavior + XGBoost",
            "note": "Reviewer-history features use defaults unless supplied from actual history."
        })

    except Exception as exc:
        app.logger.exception("Prediction failed")
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    print("Hybrid prediction API: http://127.0.0.1:5001")
    app.run(host="127.0.0.1", port=5001, debug=False)
