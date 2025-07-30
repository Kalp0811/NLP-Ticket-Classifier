# src/api.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
import torch
from . import config

# App & Model Loading
app = FastAPI(
    title="Support Ticket Classifier API",
    description="API to classify support tickets into Billing, Technical, or Other.",
    version="1.0.0"
)

# Load the fine-tuned transformer model at startup
try:
    print("Loading transformer model...")
    device = 0 if torch.cuda.is_available() else -1
    classifier = pipeline(
        "text-classification",
        model=str(config.TRANSFORMER_MODEL_DIR),
        tokenizer=str(config.TRANSFORMER_MODEL_DIR),
        device=device,
        return_all_scores=True
    )
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    classifier = None

# --- Pydantic Models ---
class TicketInput(BaseModel):
    text: str

class Prediction(BaseModel):
    label: str
    confidence: float

class TicketOutput(BaseModel):
    predictions: list[Prediction]

# API Endpoints
@app.get("/", tags=["General"])
def read_root():
    return {"message": "Welcome to the Support Ticket Classifier API!"}

@app.get("/health", tags=["General"])
def health_check():
    return {"status": "ok" if classifier else "error", "model_loaded": bool(classifier)}

@app.post("/predict", response_model=TicketOutput, tags=["Classification"])
def predict_ticket(ticket: TicketInput):
    if not classifier:
        raise HTTPException(status_code=503, detail="Model is not available.")
    
    raw_predictions = classifier(ticket.text)[0]
    
    formatted_predictions = [
        Prediction(label=p['label'], confidence=round(p['score'], 4))
        for p in raw_predictions
    ]
    
    # Sort predictions by confidence
    formatted_predictions.sort(key=lambda x: x.confidence, reverse=True)
    
    return TicketOutput(predictions=formatted_predictions)