# QB Churn MLOps Pipeline

A production ML system that extends the [QB Customer Intelligence](https://github.com/WestonWhiteUCD/qb-customer-intelligence) project from a research notebook into a fully deployed inference service.

The original project built and evaluated a churn prediction model. This project ships it; replacing SQLite with Snowflake, wrapping the model in a REST API, containerizing with Docker, deploying to AWS EC2, and adding CI and monitoring.

---

## Live API

```
GET  http://52.14.84.197:8000/health   — server status
POST http://52.14.84.197:8000/predict  — churn prediction
GET  http://52.14.84.197:8000/docs     — interactive documentation
```

**Example request:**
```bash
curl -X POST http://52.14.84.197:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "monthly_price": 29.0,
    "days_since_last_txn": 280,
    "total_transactions": 3,
    "total_spend": 500.0,
    "avg_transaction_amount": 166.0,
    "total_tickets": 12
  }'
```

**Example response:**
```json
{
  "churn_probability": 0.9955,
  "churn_percentage": "99.6%",
  "risk_tier": "HIGH",
  "input_received": { ... }
}
```

---

## Architecture

```
Snowflake (QB_INTELLIGENCE)
    ↓ Python connector
Feature matrix (2,000 customers × 6 features)
    ↓ StandardScaler → Logistic Regression (AUC 0.941)
Pickle artifacts (model.pkl, scaler.pkl, features.pkl)
    ↓ FastAPI /predict endpoint
Docker container (qb-churn-api)
    ↓ port 8000
AWS EC2 (t3.micro, us-east-1)
    ↓ structured JSON logs
CloudWatch (qb-churn-api log group + drift alarm)
```

GitHub Actions runs 5 pytest unit tests on every push to main before deployment.

---

## Project Structure

```
qb-churn-mlops/
├── app/
│   ├── main.py          — FastAPI app with /health and /predict endpoints
│   ├── model.pkl        — trained Logistic Regression (AUC 0.941)
│   ├── scaler.pkl       — StandardScaler fitted on training data
│   └── features.pkl     — ordered feature list
├── data/
│   └── generate_mock_data.py  — reproducible mock data (seed=42)
├── notebooks/
│   ├── 01_Snowflake_ETL.ipynb      — data loading into Snowflake
│   ├── 02_Model_Training.ipynb     — model training and artifact export
│   ├── 03_FastAPI.ipynb            — API development and CloudWatch setup
│   └── 04_Docker.ipynb             — containerization and CI/CD
├── tests/
│   └── test_model.py    — 5 pytest unit tests
├── .github/
│   └── workflows/
│       └── deploy.yml   — GitHub Actions CI/CD pipeline
├── Dockerfile
└── requirements.txt
```

---

## Model

| Metric | Value |
|---|---|
| Model | Logistic Regression |
| AUC | 0.941 |
| Recall (Churned) | 74% |
| Precision (Churned) | 97% |
| False Negatives | 22 / 400 test customers |
| Training data | 1,600 customers from Snowflake |

**Features:** `monthly_price` · `days_since_last_txn` · `total_transactions` · `total_spend` · `avg_transaction_amount` · `total_tickets`

Logistic Regression was chosen over Random Forest (AUC 0.942) for interpretability where each feature has an explicit coefficient and it had a lower false positive rate (2 vs 4), important for targeting retention interventions efficiently.

---

## Stack

| Layer | Tool |
|---|---|
| Data warehouse | Snowflake (QB_INTELLIGENCE database) |
| ML framework | scikit-learn |
| API framework | FastAPI + Uvicorn |
| Containerization | Docker |
| Cloud compute | AWS EC2 (t3.micro) |
| CI/CD | GitHub Actions |
| Monitoring | AWS CloudWatch (logs + alarm) |
| Language | Python 3.11 |

---

## CI Pipeline

Every push to `main` triggers a GitHub Actions workflow that:

1. Checks out the latest code on a fresh Ubuntu runner
2. Installs dependencies
3. Runs 5 pytest unit tests where we were verifying feature integrity, probability range, and directional correctness
4. Deploys to EC2 if all tests pass (SSH → git pull → docker rebuild → container restart)

---

## Setup

**Requirements:** Python 3.11+, Docker, Snowflake account, AWS account

**Snowflake:** Run `snowflake_setup.sql` in the Snowflake UI to create the warehouse, database, schema, and tables. Then run `01_Snowflake_ETL.ipynb` to load data.

**Environment variables required:**
```
SNOWFLAKE_ACCOUNT
SNOWFLAKE_USER
SNOWFLAKE_PASSWORD
SNOWFLAKE_WAREHOUSE
SNOWFLAKE_DATABASE
SNOWFLAKE_SCHEMA
```

**Run locally with Docker:**
```bash
docker build -t qb-churn-api .
docker run -p 8000:8000 qb-churn-api
```

**Run tests:**
```bash
pytest tests/ -v
```

---

## Related

[QB Customer Intelligence](https://github.com/WestonWhiteUCD/qb-customer-intelligence) — the original research project: ETL, segmentation, churn prediction, anomaly detection, NLP, A/B testing, and a recommender system across 2,000 accounts and 59,000+ transactions.

---

## Author

**Weston White** — M.S. Applied Mathematics, University of Colorado Denver  
[whitewest19@gmail.com](mailto:whitewest19@gmail.com) · [GitHub](https://github.com/WestonWhiteUCD)
