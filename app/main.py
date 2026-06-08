
import pickle
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

# ── Load model artifacts on startup ─────────────────────────────
# These load once when the server starts and stay in memory.
# Much faster than reloading on every request.
with open("app/model.pkl", "rb") as f:
    model = pickle.load(f)

with open("app/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("app/features.pkl", "rb") as f:
    feature_cols = pickle.load(f)

# ── Define the app ───────────────────────────────────────────────
app = FastAPI(
    title="QB Churn Prediction API",
    description="Predicts churn probability for QuickBooks customers",
    version="1.0.0"
)

# ── Define what a customer record must look like ─────────────────
# Pydantic validates every incoming request against this schema.
# If a field is missing or the wrong type, FastAPI auto-rejects
# the request with a clear error message before it touches the model.
class CustomerFeatures(BaseModel):
    monthly_price:          float   # subscription price in dollars
    days_since_last_txn:    int     # days since last transaction
    total_transactions:     int     # total number of transactions
    total_spend:            float   # total spend in dollars
    avg_transaction_amount: float   # average transaction amount
    total_tickets:          int     # number of support tickets

# ── Define the risk tier based on probability ────────────────────
def get_risk_tier(prob: float) -> str:
    if prob >= 0.70:
        return "HIGH"
    elif prob >= 0.40:
        return "MEDIUM"
    else:
        return "LOW"

# ── Health check endpoint ────────────────────────────────────────
# GET /health — returns a simple status message.
# Standard practice — monitoring systems ping this to confirm
# the server is alive.
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model":  "LogisticRegression",
        "auc":    0.941,
        "features": feature_cols
    }

# ── Prediction endpoint ──────────────────────────────────────────
# POST /predict — accepts a CustomerFeatures object,
# runs it through the scaler and model, returns prediction.
@app.post("/predict")
def predict(customer: CustomerFeatures):

    # Step 1: arrange features in the exact order the model expects
    feature_values = [[
        customer.monthly_price,
        customer.days_since_last_txn,
        customer.total_transactions,
        customer.total_spend,
        customer.avg_transaction_amount,
        customer.total_tickets,
    ]]

    # Step 2: scale using the same scaler fitted during training
    scaled = scaler.transform(feature_values)

    # Step 3: get churn probability (index 1 = probability of churned=True)
    churn_prob = model.predict_proba(scaled)[0][1]

    # Step 4: return results
    return {
        "churn_probability": round(float(churn_prob), 4),
        "churn_percentage":  f"{churn_prob:.1%}",
        "risk_tier":         get_risk_tier(churn_prob),
        "input_received":    customer.dict(),
    }
