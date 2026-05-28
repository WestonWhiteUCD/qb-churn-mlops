
import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import timedelta

fake = Faker()
np.random.seed(42)
random.seed(42)

N = 2000
plans      = ["Basic", "Pro", "Advanced"]
industries = ["Retail", "Healthcare", "Finance", "Tech", "Education"]
sizes      = ["1-10", "11-50", "51-200", "201-500", "500+"]
regions    = ["West", "East", "Midwest", "South"]

# ── Customers ──
customers = []
for i in range(N):
    churned    = random.random() < 0.214
    signup     = fake.date_between(start_date="-3y", end_date="-6m")
    churn_date = fake.date_between(start_date=signup, end_date="today") if churned else None
    customers.append({
        "customer_id":  f"C{i+1:04d}",
        "name":         fake.company(),
        "email":        fake.email(),
        "signup_date":  signup,
        "plan":         random.choice(plans),
        "industry":     random.choice(industries),
        "company_size": random.choice(sizes),
        "region":       random.choice(regions),
        "churned":      churned,
        "churn_date":   churn_date,
    })
df_customers = pd.DataFrame(customers)
df_customers.to_csv("data/customers.csv", index=False)

# ── Subscriptions ──
plan_prices = {"Basic": 29, "Pro": 79, "Advanced": 149}
subs = []
for _, c in df_customers.iterrows():
    subs.append({
        "subscription_id": f"S{c.customer_id[1:]}",
        "customer_id":     c.customer_id,
        "plan":            c.plan,
        "monthly_price":   plan_prices[c.plan],
        "start_date":      c.signup_date,
        "end_date":        c.churn_date,
        "status":          "cancelled" if c.churned else "active",
    })
pd.DataFrame(subs).to_csv("data/subscriptions.csv", index=False)

# ── Transactions ──
txns = []
txn_id = 1
for _, c in df_customers.iterrows():
    n_txns = random.randint(10, 50)
    end    = pd.to_datetime(c.churn_date) if c.churned and c.churn_date else pd.Timestamp.today()
    for _ in range(n_txns):
        txns.append({
            "transaction_id":   f"T{txn_id:06d}",
            "customer_id":      c.customer_id,
            "transaction_date": fake.date_between(start_date=pd.to_datetime(c.signup_date), end_date=end),
            "amount":           round(random.uniform(50, 2000), 2),
            "category":         random.choice(["Software", "Services", "Hardware", "Consulting"]),
            "description":      fake.bs(),
        })
        txn_id += 1
pd.DataFrame(txns).to_csv("data/transactions.csv", index=False)

# ── Support tickets ──
tickets = []
ticket_id = 1
issue_types = ["Billing", "Technical", "Account", "Feature Request", "Cancellation"]
for _, c in df_customers.iterrows():
    n_tickets = random.randint(1, 5)
    end = pd.to_datetime(c.churn_date) if c.churned and c.churn_date else pd.Timestamp.today()
    for _ in range(n_tickets):
        tickets.append({
            "ticket_id":       f"TK{ticket_id:05d}",
            "customer_id":     c.customer_id,
            "created_date":    fake.date_between(start_date=pd.to_datetime(c.signup_date), end_date=end),
            "issue_type":      random.choice(issue_types),
            "priority":        random.choice(["Low", "Medium", "High"]),
            "status":          "resolved",
            "resolution_days": round(random.uniform(0.5, 14), 1),
            "ticket_text":     fake.sentence(nb_words=12),
        })
        ticket_id += 1
pd.DataFrame(tickets).to_csv("data/support_tickets.csv", index=False)

print(f"customers:       {len(df_customers):,}")
print(f"subscriptions:   {len(subs):,}")
print(f"transactions:    {len(txns):,}")
print(f"support_tickets: {len(tickets):,}")
print("✓ Mock data generated")
