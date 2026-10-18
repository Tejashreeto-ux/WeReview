
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = FastAPI(title="WeReview API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = (
    Path(__file__).resolve().parent
    / ".."
    / "model"
    / "weReview_distilbert"
).resolve()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = None
model = None


@app.on_event("startup")
def load_model():
    global tokenizer, model

    if not (MODEL_PATH / "config.json").exists():
        raise RuntimeError(f"Model files not found at: {MODEL_PATH}")

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))
    model = AutoModelForSequenceClassification.from_pretrained(
        str(MODEL_PATH)
    ).to(device)

    model.eval()


class ReviewRequest(BaseModel):
    review: str = Field(min_length=10, max_length=5000)


@app.get("/")
def home():
    return {"message": "WeReview API is running"}


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
        raise HTTPException(status_code=503, detail="Model is not loaded")

    text = request.review.strip()

    if len(text) < 10:
        raise HTTPException(
            status_code=400,
            detail="Please provide a longer review."
        )

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256,
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
        probabilities = torch.softmax(logits, dim=1)[0]

    predicted_class = int(torch.argmax(probabilities).item())
    fake_score = float(probabilities[1].item())

    return {
        "prediction": (
            "potentially_suspicious"
            if predicted_class == 1
            else "likely_genuine"
        ),
        "fake_candidate_score": round(fake_score, 4),
        "message": (
            "This review may warrant further inspection."
            if predicted_class == 1
            else "The model classified this review as a likely genuine candidate."
        ),
        "notice": (
            "This is a model prediction, not proof of fraud or authenticity. "
            "The score is not necessarily a calibrated probability."
        ),
    }
