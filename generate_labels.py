"""
generate_labels.py  —  Syria Campaign · Realistic Fraud Label Generation

Replaces the randomly-assigned Is_Fraud labels with rule-based labels that
encode genuine patterns a compliance team would flag. Each rule maps to a
real-world donation threat vector.

Run BEFORE prepare.py:
    uv run python generate_labels.py
    uv run python prepare.py

Output:
    Bank_Transaction_Fraud_Detection_labeled.csv   (drop-in replacement)

The rules are intentionally imperfect and overlapping — just like real
compliance criteria — so models learn soft probabilistic boundaries rather
than memorising hard cutoffs.
"""

import numpy as np
import pandas as pd

RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)

INPUT_CSV  = "Bank_Transaction_Fraud_Detection.csv"
OUTPUT_CSV = "Bank_Transaction_Fraud_Detection_labeled.csv"

# ── Target fraud rate (realistic for humanitarian campaigns: ~4–6%) ───────────
TARGET_FRAUD_RATE = 0.055

print("Loading data…")
df = pd.read_csv(INPUT_CSV)

# ── Derived columns needed for rules ─────────────────────────────────────────
df["Hour"]       = pd.to_datetime(df["Transaction_Time"], format="%H:%M:%S").dt.hour
df["Date_parsed"]= pd.to_datetime(df["Transaction_Date"], dayfirst=True, errors="coerce")
df["DayOfWeek"]  = df["Date_parsed"].dt.dayofweek          # 0=Mon … 6=Sun
df["IsWeekend"]  = (df["DayOfWeek"] >= 5).astype(int)
df["AmtBalRatio"]= df["Transaction_Amount"] / (df["Account_Balance"] + 1)

# Percentile anchors (computed from full dataset so rules generalise)
amt_p75  = df["Transaction_Amount"].quantile(0.75)   # ~74k INR
amt_p90  = df["Transaction_Amount"].quantile(0.90)   # ~89k INR
amt_p95  = df["Transaction_Amount"].quantile(0.95)   # ~94k INR
bal_p10  = df["Account_Balance"].quantile(0.10)      # ~15k INR
ratio_p90= df["AmtBalRatio"].quantile(0.90)          # ~3.5×

print(f"Amount p75={amt_p75:.0f}  p90={amt_p90:.0f}  p95={amt_p95:.0f}")
print(f"Balance p10={bal_p10:.0f}")
print(f"Amt/Bal ratio p90={ratio_p90:.2f}")

# ─────────────────────────────────────────────────────────────────────────────
# FRAUD RULES — each returns a float score [0, 1] representing suspicion level.
# Scores are combined; a noisy threshold determines the final binary label.
# This mimics a real scoring system far better than hard binary rules.
# ─────────────────────────────────────────────────────────────────────────────

scores = pd.Series(np.zeros(len(df)), index=df.index)

# ── Rule 1 · LARGE ROUND-NUMBER DONATION ─────────────────────────────────────
# Classic structuring signal: suspiciously large donations ending in 000 or 00.
# In a charity context this can indicate money laundering via donation.
is_large       = df["Transaction_Amount"] >= amt_p90
is_round_1000  = (df["Transaction_Amount"] % 1000 == 0)
is_round_500   = (df["Transaction_Amount"] % 500  == 0)
scores += (is_large & is_round_1000).astype(float) * 1.8
scores += (is_large & is_round_500).astype(float)  * 1.0

# ── Rule 2 · NIGHT-TIME LARGE TRANSFER ───────────────────────────────────────
# Large transfers initiated between midnight and 5am — automated/scripted.
is_night    = df["Hour"].between(0, 4)
is_transfer = df["Transaction_Type"] == "Transfer"
is_large_t  = df["Transaction_Amount"] >= amt_p75
scores += (is_night & is_transfer & is_large_t).astype(float) * 2.0
scores += (is_night & is_large_t).astype(float) * 0.6

# ── Rule 3 · HIGH-RISK ANONYMOUS PLATFORM ────────────────────────────────────
# Virtual cards, QR codes, and chatbot-initiated donations are harder to trace.
# Elevated risk when combined with large amounts.
high_risk_devices = ["Virtual Card", "QR Code Scanner", "Banking Chatbot",
                     "Wearable Device", "Voice Assistant"]
is_anon_device = df["Transaction_Device"].isin(high_risk_devices)
scores += is_anon_device.astype(float) * 0.7
scores += (is_anon_device & is_large_t).astype(float) * 1.2

# ── Rule 4 · DONATION EXCEEDS ACCOUNT BALANCE ────────────────────────────────
# Donation amount >> account balance suggests a pass-through account —
# funds deposited specifically to make this donation and withdraw remainder.
scores += (df["AmtBalRatio"] > ratio_p90).astype(float) * 1.4
scores += (df["AmtBalRatio"] > ratio_p90 * 2).astype(float) * 1.0   # extra for extreme

# ── Rule 5 · VERY LOW BALANCE + VERY LARGE DONATION ──────────────────────────
# Donor with thin account making a large donation: classic mule/pass-through.
is_low_balance = df["Account_Balance"] < bal_p10
scores += (is_low_balance & is_large_t).astype(float) * 1.9

# ── Rule 6 · WEEKEND NIGHT TRANSFER ──────────────────────────────────────────
# Weekend + night + transfer is an unusual combination suggesting automated
# or coordinated activity timed to avoid weekday monitoring.
scores += (df["IsWeekend"] & is_night & is_transfer).astype(float) * 1.5

# ── Rule 7 · ELECTRONICS / HIGH-VALUE MERCHANT + LARGE AMOUNT ────────────────
# In donation fraud, electronics merchants are used to convert cash donations
# to goods. Suspicious when combined with transfers or large amounts.
is_electronics = df["Merchant_Category"] == "Electronics"
scores += (is_electronics & is_large_t).astype(float) * 0.9
scores += (is_electronics & is_transfer).astype(float) * 0.8

# ── Rule 8 · VERY LARGE WITHDRAWAL ───────────────────────────────────────────
# Immediate large withdrawal post-donation is a layering signal.
is_withdrawal    = df["Transaction_Type"] == "Withdrawal"
is_very_large    = df["Transaction_Amount"] >= amt_p95
scores += (is_withdrawal & is_very_large).astype(float) * 1.6

# ── Rule 9 · NEAR-LIMIT AMOUNTS ("just under threshold") ─────────────────────
# Structuring to avoid reporting thresholds: transactions just below
# a round number ceiling (within 1% of 50k, 75k, 90k, 100k boundaries).
thresholds = [50000, 75000, 90000, 99000]
for t in thresholds:
    near_below = (df["Transaction_Amount"] >= t * 0.99) & (df["Transaction_Amount"] < t)
    scores += near_below.astype(float) * 1.1

# ── Rule 10 · BUSINESS ACCOUNT + NIGHT TRANSFER ──────────────────────────────
# Business accounts making large night transfers — potential corporate
# money laundering routed through campaign donations.
is_business = df["Account_Type"] == "Business"
scores += (is_business & is_night & is_transfer & is_large_t).astype(float) * 1.7

# ── Rule 11 · ATM + LARGE AMOUNT ─────────────────────────────────────────────
# ATM-initiated donations above p75 are atypical — ATMs are for withdrawals.
is_atm = df["Transaction_Device"] == "ATM"
scores += (is_atm & is_large_t).astype(float) * 1.2

# ── Rule 12 · YOUNG DONOR + VERY LARGE AMOUNT ────────────────────────────────
# Young donors (under 25) making very large donations disproportionate to
# typical income — potential use of recruited mules.
is_young = df["Age"] < 25
scores += (is_young & is_very_large).astype(float) * 1.3

# ─────────────────────────────────────────────────────────────────────────────
# SCORE → LABEL CONVERSION
# Use a sigmoid-like transformation + calibrated threshold to hit target rate.
# Add noise so the boundary is soft (models can't just memorise a cutoff).
# ─────────────────────────────────────────────────────────────────────────────
print("\nScore distribution (pre-noise):")
print(scores.describe().round(3).to_string())

# Normalise scores to [0, 1]
score_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)

# Add calibrated Gaussian noise — enough to blur hard edges but not drown signal
noise = rng.normal(loc=0, scale=0.12, size=len(df))
score_noisy = np.clip(score_norm + noise, 0, 1)

# Find threshold that produces ~TARGET_FRAUD_RATE
threshold = np.percentile(score_noisy, (1 - TARGET_FRAUD_RATE) * 100)
print(f"\nLabel threshold: {threshold:.4f}  →  target rate {TARGET_FRAUD_RATE:.1%}")

labels = (score_noisy >= threshold).astype(int)
actual_rate = labels.mean()
print(f"Actual fraud rate achieved: {actual_rate:.2%}  (n={labels.sum():,})")

# ─────────────────────────────────────────────────────────────────────────────
# SANITY CHECK — verify rules have real discriminating power
# ─────────────────────────────────────────────────────────────────────────────
df["Is_Fraud_New"] = labels

print("\n=== Signal validation — fraud rate by rule trigger ===")
checks = {
    "Large round amount (≥p90, mod 1000)":  (is_large & is_round_1000),
    "Night + Transfer + Large":             (is_night & is_transfer & is_large_t),
    "High-risk platform":                   is_anon_device,
    "High-risk platform + Large":           (is_anon_device & is_large_t),
    "Amt/Bal ratio > p90":                  (df["AmtBalRatio"] > ratio_p90),
    "Low balance + Large donation":         (is_low_balance & is_large_t),
    "Weekend + Night + Transfer":           (df["IsWeekend"] & is_night & is_transfer),
    "Electronics + Large":                  (is_electronics & is_large_t),
    "Very large withdrawal":                (is_withdrawal & is_very_large),
    "Near-threshold structuring":           pd.concat([
        (df["Transaction_Amount"] >= t*0.99) & (df["Transaction_Amount"] < t)
        for t in thresholds], axis=1).any(axis=1),
    "Young donor + Very large":             (is_young & is_very_large),
    "Business + Night + Transfer":          (is_business & is_night & is_transfer),
    "Baseline (all)":                       pd.Series(True, index=df.index),
}

for desc, mask in checks.items():
    n    = mask.sum()
    rate = df.loc[mask, "Is_Fraud_New"].mean() if n > 0 else 0
    lift = rate / actual_rate if actual_rate > 0 else 0
    print(f"  {desc:<45s}  n={n:>7,}  rate={rate:.3f}  lift={lift:.2f}×")

# ─────────────────────────────────────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────────────────────────────────────
out_df = df.drop(columns=["Hour", "Date_parsed", "DayOfWeek", "IsWeekend",
                           "AmtBalRatio", "Is_Fraud_New"])
out_df["Is_Fraud"] = labels

out_df.to_csv(OUTPUT_CSV, index=False)
print(f"\n✅  Saved → {OUTPUT_CSV}")
print(f"   {len(out_df):,} rows  |  {labels.sum():,} fraud ({labels.mean():.2%})")
print("\nNext step:  uv run python prepare.py")
