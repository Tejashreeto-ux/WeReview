
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = FastAPI(title="WeReview AI Review Detection API")

# Allow requests from the React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Path to your downloaded DistilBERT model
MODEL_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "model"
    / "weReview_distilbert"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = None
model = None


@app.on_event("startup")
def load_model():
    global tokenizer, model

    if not (MODEL_PATH / "config.json").exists():
        raise RuntimeError(
            f"Model files not found. Expected folder: {MODEL_PATH}"
        )

    print("Loading WeReview DistilBERT model...")

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))
    model = AutoModelForSequenceClassification.from_pretrained(
        str(MODEL_PATH)
    ).to(device)

    model.eval()

    print(f"Model loaded successfully on {device}")


class ReviewRequest(BaseModel):
    review: str = Field(min_length=10, max_length=5000)


@app.get("/")
def home():
    return {
        "message": "WeReview AI API is running",
        "model_path": str(MODEL_PATH),
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "device": str(device),
    }


@app.post("/predict")
def predict(request: ReviewRequest):
    if model is None or tokenizer is None:
        raise HTTPException(
            status_code=503,
            detail="The model has not loaded yet."
        )

    review_text = request.review.strip()

    inputs = tokenizer(
        review_text,
        return_tensors="pt",
        truncation=True,
        max_length=256,
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=1)[0]

    predicted_class = int(torch.argmax(probabilities).item())
    fake_score = float(probabilities[1].item())

    if predicted_class == 1:
        prediction = "potentially_suspicious"
        message = "This review may warrant further inspection."
    else:
        prediction = "likely_genuine"
        message = "The model classified this as a likely genuine candidate."

    return {
        "prediction": prediction,
        "fake_candidate_score": round(fake_score, 4),
        "message": message,
        "notice": (
            "This is a model prediction, not proof of fraud or authenticity. "
            "The score is not necessarily a calibrated probability."
        ),
    }
