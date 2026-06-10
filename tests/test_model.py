import pickle
import numpy as np
import pytest

# ── Load artifacts once for all tests ───────────────────────────
with open("app/model.pkl", "rb") as f:
    model = pickle.load(f)

with open("app/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("app/features.pkl", "rb") as f:
    feature_cols = pickle.load(f)

# ── Test 1: correct number of features ──────────────────────────
# Verifies the feature list has exactly 6 items.
# If someone accidentally adds or removes a feature, this catches it.
def test_feature_count():
    assert len(feature_cols) == 6, f"Expected 6 features, got {len(feature_cols)}"

# ── Test 2: correct feature names ───────────────────────────────
# Verifies the exact feature names haven't changed.
# Order matters — the model expects features in a specific sequence.
def test_feature_names():
    expected = [
        "monthly_price",
        "days_since_last_txn",
        "total_transactions",
        "total_spend",
        "avg_transaction_amount",
        "total_tickets",
    ]
    assert feature_cols == expected, f"Feature mismatch: {feature_cols}"

# ── Test 3: prediction is a valid probability ────────────────────
# Verifies the model returns a number between 0 and 1.
# A probability outside that range means something is badly wrong.
def test_prediction_is_probability():
    sample = np.array([[79.0, 50, 30, 15000.0, 500.0, 3]])
    scaled = scaler.transform(sample)
    prob   = model.predict_proba(scaled)[0][1]
    assert 0.0 <= prob <= 1.0, f"Probability out of range: {prob}"

# ── Test 4: high risk customer scores high ───────────────────────
# Verifies the model's direction is correct — a clearly at-risk
# customer should score above 0.5. If this fails the model logic
# is inverted or broken.
def test_high_risk_customer_scores_high():
    high_risk = np.array([[29.0, 280, 3, 500.0, 166.0, 12]])
    scaled    = scaler.transform(high_risk)
    prob      = model.predict_proba(scaled)[0][1]
    assert prob > 0.5, f"High risk customer scored too low: {prob:.2%}"

# ── Test 5: low risk customer scores low ────────────────────────
# Verifies a healthy customer scores below 0.5.
def test_low_risk_customer_scores_low():
    low_risk = np.array([[149.0, 2, 80, 95000.0, 1187.0, 1]])
    scaled   = scaler.transform(low_risk)
    prob     = model.predict_proba(scaled)[0][1]
    assert prob < 0.5, f"Low risk customer scored too high: {prob:.2%}"