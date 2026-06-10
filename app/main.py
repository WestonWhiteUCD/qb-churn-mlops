import pickle
import numpy as np
import logging
import json
from datetime import datetime, timezone
from fastapi import FastAPI
from pydantic import BaseModel

# ── Configure prediction logger ──────────────────────────────────
# Writes one JSON line per prediction to predictions.log
# CloudWatch agent ships this file to AWS in real time
logging.basicConfig(level=logging.INFO)
prediction_logger = logging.getLogger("predictions")
file_handler = logging.FileHandler("/var/log/qb-churn-api/predictions.log")
file_handler.setFormatter(logging.Formatter("%(message)s"))
prediction_logger.addHandler(file_handler)
prediction_logger.propagate = False

# ── Load model artifacts on startup ─────────────────────────────
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
class CustomerFeatures(BaseModel):
    monthly_price:          float
    days_since_last_txn:    int
    total_transactions:     int
    total_spend:            float
    avg_transaction_amount: float
    total_tickets:          int

# ── Define the risk tier ─────────────────────────────────────────
def get_risk_tier(prob: float) -> str:
    if prob >= 0.70:
        return "HIGH"
    elif prob >= 0.40:
        return "MEDIUM"
    else:
        return "LOW"

# ── Health check endpoint ────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status":   "healthy",
        "model":    "LogisticRegression",
        "auc":      0.941,
        "features": feature_cols
    }

# ── Prediction endpoint ──────────────────────────────────────────
@app.post("/predict")
def predict(customer: CustomerFeatures):

    # Step 1: arrange features in correct order
    feature_values = [[
        customer.monthly_price,
        customer.days_since_last_txn,
        customer.total_transactions,
        customer.total_spend,
        customer.avg_transaction_amount,
        customer.total_tickets,
    ]]

    # Step 2: scale features
    scaled = scaler.transform(feature_values)

    # Step 3: get churn probability
    churn_prob = model.predict_proba(scaled)[0][1]
    risk_tier  = get_risk_tier(churn_prob)

    # Step 4: log prediction to CloudWatch
    # Each line is a JSON object — easy to query in CloudWatch Insights
    log_entry = {
        "timestamp":        datetime.now(timezone.utc).isoformat(),
        "churn_probability": round(float(churn_prob), 4),
        "risk_tier":         risk_tier,
        "inputs":            customer.dict(),
    }
    prediction_logger.info(json.dumps(log_entry))

    # Step 5: return results
    return {
        "churn_probability": round(float(churn_prob), 4),
        "churn_percentage":  f"{churn_prob:.1%}",
        "risk_tier":         risk_tier,
        "input_received":    customer.dict(),
    }