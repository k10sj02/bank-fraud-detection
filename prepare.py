"""
prepare.py  —  DonorGuard · Train & Cache Artifacts

Run order:
    1. uv run python reshape_dataset.py
    2. uv run python generate_labels.py
    3. uv run python prepare.py              ← this script
    4. uv run streamlit run app.py

Input:  Donations_DonorGuard_Labeled.csv
Output: artifacts/
"""

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print('WARNING: shap not installed — SHAP artifacts will be skipped.')
import pandas as pd
import scipy.stats as stats
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, classification_report,
    confusion_matrix, precision_recall_curve, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

ARTIFACTS    = Path("artifacts")
RANDOM_STATE = 42
CSV_PATH     = "Donations_DonorGuard_Labeled.csv"

CAT_ENCODE = [
    "Gender", "Donor_Country", "Donor_Segment", "Donation_Frequency",
    "Donation_Type", "Donation_Platform", "Payment_Processor",
    "Campaign_ID", "Acquisition_Channel", "Currency",
    "CVV_Check", "Device_Fingerprint", "Webhook_Event",
]

MODEL_FEATURES = [
    "Age", "Donor_Since_Days", "Log_Donor_Lifetime_Value", "Is_Recurring",
    "Is_Anonymous", "Matched_Giving", "Gift_Aid_Eligible",
    "Log_Donation_Amount", "Donation_Amount_Zscore", "Amt_LTV_Ratio",
    "Hour", "Is_Night_Donation", "Is_Weekend", "Is_Business_Hours",
    "Is_Round_Amount", "Near_Threshold",
    "High_Risk_Platform", "Is_Cold_Acquisition",
    "Is_Bank_Transfer", "Is_Refund_Type",
    "CVV_Failed", "Device_Flagged", "Webhook_Chargeback", "Webhook_Reversed",
    "Refund_Requested", "Is_First_Time_Large", "Is_Anon_Large",
    "Is_Corporate_Night", "Matched_Anon_Flag",
    "IP_Country_Match", "Is_VPN_Or_Proxy", "IP_Velocity_24h",
    "IP_Mismatch_Large", "VPN_Anon_Flag", "High_Velocity_Flag",
] + CAT_ENCODE

HIGH_RISK_PLATFORMS = [
    "Virtual Card", "QR Code", "Voice Assistant", "Chatbot", "Wearable", "Biometric"
]


# ── Helper functions (defined before use) ────────────────────────────────────

def optimal_threshold(proba, y_true, beta=0.5):
    """
    Find the decision threshold maximising F-beta score.

    beta=0.5 weights precision twice as heavily as recall —
    correct for this use case because flagging an innocent donor
    is more costly than missing a bad actor.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, proba)
    p = precisions[:-1]
    r = recalls[:-1]
    denom = (beta ** 2 * p) + r
    fbeta = np.where(denom > 0, (1 + beta ** 2) * p * r / denom, 0.0)
    best  = fbeta.argmax()
    return {
        "threshold": float(round(thresholds[best], 4)),
        "precision": float(round(p[best], 4)),
        "recall":    float(round(r[best], 4)),
        "fbeta":     float(round(fbeta[best], 4)),
        "beta":      beta,
    }


def model_metrics(name, proba, y_true, beta=0.5):
    """
    Compute all metrics at the optimal F-beta threshold.
    Returns dict with standard metrics + optimal threshold info.
    """
    opt  = optimal_threshold(proba, y_true, beta=beta)
    pred = (proba >= opt["threshold"]).astype(int)
    rep  = classification_report(
        y_true, pred,
        target_names=["Legitimate", "Anomalous"],
        output_dict=True,
    )
    return {
        "name":          name,
        "auc":           round(roc_auc_score(y_true, proba), 4),
        "ap":            round(average_precision_score(y_true, proba), 4),
        "precision":     round(rep["Anomalous"]["precision"], 4),
        "recall":        round(rep["Anomalous"]["recall"], 4),
        "f1":            round(rep["Anomalous"]["f1-score"], 4),
        "cm":            confusion_matrix(y_true, pred).tolist(),
        "opt_threshold": opt["threshold"],
        "opt_precision": opt["precision"],
        "opt_recall":    opt["recall"],
        "opt_fbeta":     opt["fbeta"],
    }


# ── 1. Load ───────────────────────────────────────────────────────────────────
print("Loading dataset…")
df = pd.read_csv(CSV_PATH)
print(f"  {len(df):,} rows, {df.shape[1]} columns")
print(f"  Anomaly rate: {df['Is_Anomalous'].mean():.2%}")

# ── 2. Feature engineering ────────────────────────────────────────────────────
print("Engineering features…")

df["Donation_Date"] = pd.to_datetime(df["Donation_Date"], dayfirst=True, errors="coerce")
df["Hour"]          = pd.to_datetime(df["Donation_Time"], format="%H:%M:%S", errors="coerce").dt.hour
df["DayOfWeek"]     = df["Donation_Date"].dt.day_name()

df["Log_Donation_Amount"]      = np.log1p(df["Donation_Amount"])
df["Log_Donor_Lifetime_Value"] = np.log1p(df["Donor_Lifetime_Value"])
df["Donation_Amount_Zscore"]   = stats.zscore(df["Donation_Amount"])
df["Amt_LTV_Ratio"]            = df["Donation_Amount"] / (df["Donor_Lifetime_Value"] + 1)
df["Is_Night_Donation"] = df["Hour"].between(0, 4).astype(int)
df["Is_Weekend"]        = df["DayOfWeek"].isin(["Saturday", "Sunday"]).astype(int)
df["Is_Business_Hours"] = df["Hour"].between(9, 17).astype(int)

amt_p90 = df["Donation_Amount"].quantile(0.90)
amt_p75 = df["Donation_Amount"].quantile(0.75)
ltv_p10 = df["Donor_Lifetime_Value"].quantile(0.10)
structuring_thresholds = [500, 750, 900, 950]

df["Is_Round_Amount"] = (df["Donation_Amount"] % 50 == 0).astype(int)
near_flags = pd.DataFrame({
    str(t): (df["Donation_Amount"] >= t * 0.99) & (df["Donation_Amount"] < t)
    for t in structuring_thresholds
})
df["Near_Threshold"]     = near_flags.any(axis=1).astype(int)
df["High_Risk_Platform"] = df["Donation_Platform"].isin(HIGH_RISK_PLATFORMS).astype(int)
df["Is_Cold_Acquisition"]= (df["Acquisition_Channel"] == "Cold Outreach").astype(int)
df["Is_Bank_Transfer"]   = (df["Donation_Type"] == "Bank Transfer").astype(int)
df["Is_Refund_Type"]     = (df["Donation_Type"] == "Refund").astype(int)
df["CVV_Failed"]         = (df["CVV_Check"] == "Fail").astype(int)
df["Device_Flagged"]     = (df["Device_Fingerprint"] == "Known Flagged").astype(int)
df["Webhook_Chargeback"] = (df["Webhook_Event"] == "chargeback.received").astype(int)
df["Webhook_Reversed"]   = (df["Webhook_Event"] == "payment.reversed").astype(int)

df["Is_First_Time_Large"] = (
    (df["Donation_Frequency"] == "First-time") &
    (df["Donation_Amount"] >= amt_p90)
).astype(int)
df["Is_Anon_Large"] = (
    (df["Is_Anonymous"] == 1) & (df["Donation_Amount"] >= amt_p75)
).astype(int)
df["Is_Corporate_Night"] = (
    (df["Donor_Segment"] == "Corporate") &
    df["Hour"].between(0, 4) &
    (df["Donation_Type"] == "Bank Transfer")
).astype(int)
df["Matched_Anon_Flag"] = (
    (df["Matched_Giving"] == 1) & (df["Is_Anonymous"] == 1)
).astype(int)
df["IP_Mismatch_Large"] = (
    (df["IP_Country_Match"] == 0) & (df["Donation_Amount"] >= amt_p75)
).astype(int)
df["VPN_Anon_Flag"]      = (
    (df["Is_VPN_Or_Proxy"] == 1) & (df["Is_Anonymous"] == 1)
).astype(int)
df["High_Velocity_Flag"] = (df["IP_Velocity_24h"] >= 3).astype(int)

# ── 3. Encode categoricals ────────────────────────────────────────────────────
print("Encoding categoricals…")
encoders = {}
model_df = df[MODEL_FEATURES + ["Is_Anomalous"]].copy()
for col in CAT_ENCODE:
    le = LabelEncoder()
    model_df[col] = le.fit_transform(model_df[col].astype(str))
    encoders[col] = le

model_df = model_df.dropna()
X = model_df[MODEL_FEATURES]
y = model_df["Is_Anomalous"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)
scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)
print(f"  Train: {X_train.shape}  |  Test: {X_test.shape}")
print(f"  Train anomaly rate: {y_train.mean():.2%}")

# ── 4. Train models ───────────────────────────────────────────────────────────
print("Training Random Forest…")
rf = RandomForestClassifier(
    n_estimators=300, class_weight="balanced",
    max_depth=16, min_samples_leaf=4,
    max_features="sqrt", random_state=RANDOM_STATE, n_jobs=-1,
)
rf.fit(X_train, y_train)
rf_proba = rf.predict_proba(X_test)[:, 1]

print("Training Logistic Regression (with probability calibration)…")
_lr_base = LogisticRegression(
    class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE
)
lr = CalibratedClassifierCV(_lr_base, cv=5, method="isotonic")
lr.fit(X_train_sc, y_train)
lr_proba = lr.predict_proba(X_test_sc)[:, 1]

print("Training Isolation Forest…")
iso = IsolationForest(
    contamination=0.055, n_estimators=200,
    random_state=RANDOM_STATE, n_jobs=-1,
)
iso.fit(X_train_sc)
iso_scores = -iso.score_samples(X_test_sc)

# ── 5. Compute optimal thresholds ────────────────────────────────────────────
print("Computing optimal thresholds (F-beta, β=0.5)…")
rf_opt  = optimal_threshold(rf_proba,  y_test, beta=0.5)
lr_opt  = optimal_threshold(lr_proba,  y_test, beta=0.5)
iso_opt = optimal_threshold(iso_scores, y_test, beta=0.5)

print(f"  RF  threshold={rf_opt['threshold']}  "
      f"P={rf_opt['precision']:.1%}  R={rf_opt['recall']:.1%}  F0.5={rf_opt['fbeta']:.3f}")
print(f"  LR  threshold={lr_opt['threshold']}  "
      f"P={lr_opt['precision']:.1%}  R={lr_opt['recall']:.1%}  F0.5={lr_opt['fbeta']:.3f}")
print(f"  ISO threshold={iso_opt['threshold']}  "
      f"P={iso_opt['precision']:.1%}  R={iso_opt['recall']:.1%}  F0.5={iso_opt['fbeta']:.3f}")

# Predictions at optimal thresholds
rf_pred  = (rf_proba  >= rf_opt["threshold"]).astype(int)
lr_pred  = (lr_proba  >= lr_opt["threshold"]).astype(int)
iso_pred = (iso_scores >= iso_opt["threshold"]).astype(int)

# ── 6. Metrics ────────────────────────────────────────────────────────────────
print("Computing metrics…")
metrics = {
    "Random Forest":       model_metrics("Random Forest",       rf_proba,   y_test),
    "Logistic Regression": model_metrics("Logistic Regression", lr_proba,   y_test),
    "Isolation Forest":    model_metrics("Isolation Forest",    iso_scores, y_test),
}

print("\n=== Results (at optimal F-0.5 threshold) ===")
for name, m in metrics.items():
    print(f"  {name:<22s}  AUC={m['auc']}  "
          f"P={m['precision']:.1%}  R={m['recall']:.1%}  "
          f"threshold={m['opt_threshold']}")

# ── 7. Risk scoring DataFrame ─────────────────────────────────────────────────
risk_df = X_test.copy()
risk_df["True_Label"]    = y_test.values
risk_df["Anomaly_Score"] = rf_proba
risk_df["LR_Score"]      = lr_proba
risk_df["ISO_Score"]     = iso_scores
risk_df["RF_Pred"]       = rf_pred
risk_df["Risk_Tier"]     = pd.cut(
    risk_df["Anomaly_Score"],
    bins=[0, 0.30, 0.60, 0.80, 1.01],
    labels=["Low", "Medium", "High", "Critical"],
)

importances = pd.Series(
    rf.feature_importances_, index=MODEL_FEATURES
).sort_values(ascending=False)

# ── 8. SHAP explainer ────────────────────────────────────────────────────────
shap_explainer = None
shap_df        = None
if SHAP_AVAILABLE:
    print('Computing SHAP explainer…')
    shap_background = X_train.sample(200, random_state=RANDOM_STATE)
    shap_explainer  = shap.TreeExplainer(rf, shap_background)
    shap_sample     = X_test.sample(min(500, len(X_test)), random_state=RANDOM_STATE)
    shap_values_pos = shap_explainer.shap_values(shap_sample, check_additivity=False)
    # Handle both (n, features) and (n, features, n_classes) shapes
    if isinstance(shap_values_pos, list):
        shap_values_pos = shap_values_pos[1]
    elif shap_values_pos.ndim == 3:
        shap_values_pos = shap_values_pos[:, :, 1]
    shap_df = pd.DataFrame(shap_values_pos, columns=MODEL_FEATURES)
    print(f'  SHAP values computed — shape: {shap_df.shape}')
else:
    print("Skipping SHAP — run: uv add 'shap>=0.44.0'")

# ── 9. EDA cache ──────────────────────────────────────────────────────────────
eda_cols = [
    "Is_Anomalous", "Donation_Amount", "Donor_Lifetime_Value", "Hour", "DayOfWeek",
    "Donor_Country", "Donation_Type", "Campaign_ID", "Campaign_Name",
    "Donation_Platform", "Payment_Processor", "Currency",
    "Is_Night_Donation", "High_Risk_Platform", "Age",
    "Is_Round_Amount", "Near_Threshold", "Refund_Requested",
    "Is_Anonymous", "Donation_Frequency", "Donor_Segment",
    "CVV_Check", "Device_Fingerprint", "Webhook_Event",
    "Matched_Giving", "Gift_Aid_Eligible", "Is_Recurring",
    "Acquisition_Channel", "IP_Country_Match", "Is_VPN_Or_Proxy", "IP_Velocity_24h",
]
eda_df = df[eda_cols].copy()

# ── 10. Meta / summary stats ───────────────────────────────────────────────────
summary_stats = {
    "amount_mean":            float(df["Donation_Amount"].mean()),
    "amount_std":             float(df["Donation_Amount"].std()),
    "anomaly_rate":           float(df["Is_Anomalous"].mean()),
    "total":                  int(len(df)),
    "n_anomalous":            int(df["Is_Anomalous"].sum()),
    "night_rate":             float(df[df["Is_Night_Donation"] == 1]["Is_Anomalous"].mean()),
    "high_risk_n":            int(df[df["High_Risk_Platform"] == 1]["Is_Anomalous"].sum()),
    "refund_rate":            float(df["Refund_Requested"].mean()),
    "anon_rate":              float(df["Is_Anonymous"].mean()),
    "features":               MODEL_FEATURES,
    "cat_encode":             CAT_ENCODE,
    "amt_p75":                float(amt_p75),
    "amt_p90":                float(amt_p90),
    "ltv_p10":                float(ltv_p10),
    "structuring_thresholds": structuring_thresholds,
    "high_risk_platforms":    HIGH_RISK_PLATFORMS,
    "ip_mismatch_rate":       float((df["IP_Country_Match"] == 0).mean()),
    "vpn_rate":               float(df["Is_VPN_Or_Proxy"].mean()),
    "high_velocity_rate":     float((df["IP_Velocity_24h"] >= 3).mean()),
    "opt_thresholds": {
        "Random Forest":       rf_opt["threshold"],
        "Logistic Regression": lr_opt["threshold"],
        "Isolation Forest":    iso_opt["threshold"],
    },
    "campaigns": [
        {"id": "SC-2024-EMG", "name": "Emergency Relief Fund 2024"},
        {"id": "SC-2024-WIN", "name": "Winter Giving Campaign 2024"},
        {"id": "SC-2024-EDU", "name": "Education & Scholarship Fund 2024"},
        {"id": "SC-2024-MED", "name": "Medical & Humanitarian Aid Appeal"},
        {"id": "SC-2024-REF", "name": "Refugee & Displacement Support"},
        {"id": "SC-2023-EMG", "name": "Emergency Relief Fund 2023"},
        {"id": "SC-2024-RAM", "name": "Ramadan Giving Campaign 2024"},
        {"id": "SC-2024-MAT", "name": "Year-End Matched Giving"},
    ],
}

# ── 11. Save ──────────────────────────────────────────────────────────────────
ARTIFACTS.mkdir(exist_ok=True)
print("\nSaving artifacts…")
joblib.dump(rf,          ARTIFACTS / "rf_model.joblib",    compress=3)
joblib.dump(lr,          ARTIFACTS / "lr_model.joblib",    compress=3)
joblib.dump(iso,         ARTIFACTS / "iso_model.joblib",   compress=3)
joblib.dump(scaler,      ARTIFACTS / "scaler.joblib",      compress=3)
joblib.dump(encoders,    ARTIFACTS / "encoders.joblib",    compress=3)
joblib.dump(importances,    ARTIFACTS / "importances.joblib",  compress=3)
if shap_explainer is not None:
    joblib.dump(shap_explainer, ARTIFACTS / "shap_explainer.joblib", compress=3)
    shap_df.to_parquet(ARTIFACTS / "shap_values.parquet", index=False)
risk_df.to_parquet(ARTIFACTS / "risk_df.parquet",  index=False)
eda_df.to_parquet( ARTIFACTS / "eda_cache.parquet", index=False)
with open(ARTIFACTS / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
with open(ARTIFACTS / "meta.json", "w") as f:
    json.dump(summary_stats, f, indent=2)

print(f"\n✅  Done → ./artifacts/")
for name, m in metrics.items():
    print(f"   {name:<22s}  AUC={m['auc']}  "
          f"P={m['precision']:.1%}  R={m['recall']:.1%}  "
          f"threshold={m['opt_threshold']}")
