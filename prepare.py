"""
prepare.py  —  Syria Campaign · Train & cache all artifacts

Run order:
    1. uv run python generate_labels.py
    2. uv run python prepare.py
    3. uv run streamlit run app.py

Outputs → ./artifacts/
    rf_model.joblib       Random Forest
    lr_model.joblib       Logistic Regression
    iso_model.joblib      Isolation Forest
    scaler.joblib         StandardScaler (for LR / ISO paths)
    encoders.joblib       dict of LabelEncoders
    importances.joblib    RF feature importances (Series)
    risk_df.parquet       scored test-set rows
    eda_cache.parquet     lightweight EDA dataframe
    metrics.json          per-model evaluation metrics
    meta.json             dataset statistics + feature list
"""

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
CSV_PATH     = "Bank_Transaction_Fraud_Detection_labeled.csv"
ARTIFACTS    = Path("artifacts")
RANDOM_STATE = 42

COLUMN_MAP = {
    "Customer_ID": "Donor_ID", "Customer_Name": "Donor_Name",
    "Gender": "Gender", "Age": "Age", "State": "Region", "City": "City",
    "Bank_Branch": "Donation_Branch", "Account_Type": "Donation_Channel",
    "Transaction_ID": "Donation_ID", "Transaction_Date": "Donation_Date",
    "Transaction_Time": "Donation_Time", "Transaction_Amount": "Donation_Amount",
    "Merchant_ID": "Campaign_ID", "Transaction_Type": "Donation_Type",
    "Merchant_Category": "Campaign_Category", "Account_Balance": "Cumulative_Given",
    "Transaction_Device": "Donation_Platform", "Transaction_Location": "Donation_Location",
    "Device_Type": "Device_Type", "Is_Fraud": "Is_Anomalous",
    "Transaction_Currency": "Currency", "Customer_Contact": "Donor_Contact",
    "Transaction_Description": "Donation_Description", "Customer_Email": "Donor_Email",
}

HIGH_RISK_PLATFORMS = [
    "Virtual Card", "QR Code Scanner", "Banking Chatbot",
    "Wearable Device", "Voice Assistant",
]

# Categoricals to label-encode
CAT_ENCODE = [
    "Gender", "Region", "Donation_Channel", "Donation_Type",
    "Campaign_Category", "Device_Type", "Currency",
]

# Full feature set — matches generate_labels.py threat vectors
MODEL_FEATURES = [
    # Core amounts
    "Age",
    "Log_Donation_Amount",
    "Log_Cumulative_Given",
    "Amt_Bal_Ratio",
    "Amount_Zscore",
    # Temporal
    "Hour",
    "Is_Night_Donation",
    "Is_Weekend",
    "Is_Business_Hours",
    # Amount pattern signals
    "Is_Round_Amount",
    "Near_Threshold",
    # Platform / device risk
    "High_Risk_Platform",
    # Transaction type flags
    "Is_Transfer",
    "Is_Withdrawal",
    # Account signals
    "Is_Business_Account",
    "Low_Balance_Large_Gift",
    # Demographic
    "Young_Large",
    # Encoded categoricals
] + CAT_ENCODE


# ── 1. Load & rename ──────────────────────────────────────────────────────────
print("Loading CSV…")
df = pd.read_csv(CSV_PATH).rename(columns=COLUMN_MAP)

# ── 2. Feature engineering ─────────────────────────────────────────────────--
print("Engineering features…")
df["Donation_Date"] = pd.to_datetime(df["Donation_Date"], dayfirst=True, errors="coerce")
df["Hour"]          = pd.to_datetime(df["Donation_Time"], format="%H:%M:%S", errors="coerce").dt.hour
df["DayOfWeek"]     = df["Donation_Date"].dt.day_name()

# Temporal
df["Is_Night_Donation"]  = df["Hour"].between(0, 4).astype(int)
df["Is_Weekend"]         = df["DayOfWeek"].isin(["Saturday", "Sunday"]).astype(int)
df["Is_Business_Hours"]  = df["Hour"].between(9, 17).astype(int)

# Amount / balance
df["Log_Donation_Amount"]  = np.log1p(df["Donation_Amount"])
df["Log_Cumulative_Given"] = np.log1p(df["Cumulative_Given"])
df["Amount_Zscore"]        = stats.zscore(df["Donation_Amount"])
df["Amt_Bal_Ratio"]        = df["Donation_Amount"] / (df["Cumulative_Given"] + 1)

# Amount pattern signals
df["Is_Round_Amount"] = (df["Donation_Amount"] % 500 == 0).astype(int)

# Near-threshold structuring (just below 50k / 75k / 90k / 99k)
structuring_thresholds = [50000, 75000, 90000, 99000]
near_flags = pd.DataFrame({
    str(t): (df["Donation_Amount"] >= t * 0.99) & (df["Donation_Amount"] < t)
    for t in structuring_thresholds
})
df["Near_Threshold"] = near_flags.any(axis=1).astype(int)

# Platform risk
df["High_Risk_Platform"]  = df["Donation_Platform"].isin(HIGH_RISK_PLATFORMS).astype(int)

# Transaction type flags
df["Is_Transfer"]   = (df["Donation_Type"] == "Transfer").astype(int)
df["Is_Withdrawal"] = (df["Donation_Type"] == "Withdrawal").astype(int)

# Account signals
df["Is_Business_Account"] = (df["Donation_Channel"] == "Business").astype(int)

amt_p75 = df["Donation_Amount"].quantile(0.75)
bal_p10 = df["Cumulative_Given"].quantile(0.10)
df["Low_Balance_Large_Gift"] = (
    (df["Cumulative_Given"] < bal_p10) & (df["Donation_Amount"] >= amt_p75)
).astype(int)

# Demographic risk combination
df["Young_Large"] = (
    (df["Age"] < 25) & (df["Donation_Amount"] >= df["Donation_Amount"].quantile(0.95))
).astype(int)

# ── 3. Encode categoricals ────────────────────────────────────────────────────
print("Encoding categoricals…")
encoders  = {}
model_df  = df[MODEL_FEATURES + ["Is_Anomalous"]].copy()
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

# ── 4. Train models ───────────────────────────────────────────────────────────
print("Training Random Forest…")
rf = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    max_depth=16,
    min_samples_leaf=4,
    max_features="sqrt",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
rf.fit(X_train, y_train)
rf_proba = rf.predict_proba(X_test)[:, 1]
rf_pred  = (rf_proba >= 0.45).astype(int)

print("Training Logistic Regression…")
lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)
lr.fit(X_train_sc, y_train)
lr_proba = lr.predict_proba(X_test_sc)[:, 1]
lr_pred  = (lr_proba >= 0.45).astype(int)

print("Training Isolation Forest…")
iso = IsolationForest(
    contamination=0.055, n_estimators=200,
    random_state=RANDOM_STATE, n_jobs=-1,
)
iso.fit(X_train_sc)
iso_scores = -iso.score_samples(X_test_sc)
iso_pred   = (iso.predict(X_test_sc) == -1).astype(int)

# ── 5. Metrics ────────────────────────────────────────────────────────────────
def model_metrics(name, proba, pred, y_true):
    rep = classification_report(
        y_true, pred, target_names=["Legitimate", "Anomalous"], output_dict=True
    )
    return {
        "name":      name,
        "auc":       round(roc_auc_score(y_true, proba), 4),
        "ap":        round(average_precision_score(y_true, proba), 4),
        "precision": round(rep["Anomalous"]["precision"], 4),
        "recall":    round(rep["Anomalous"]["recall"], 4),
        "f1":        round(rep["Anomalous"]["f1-score"], 4),
        "cm":        confusion_matrix(y_true, pred).tolist(),
    }

metrics = {
    "Random Forest":       model_metrics("Random Forest",       rf_proba,   rf_pred,  y_test),
    "Logistic Regression": model_metrics("Logistic Regression", lr_proba,   lr_pred,  y_test),
    "Isolation Forest":    model_metrics("Isolation Forest",    iso_scores, iso_pred, y_test),
}

print("\n=== Model Results ===")
for name, m in metrics.items():
    print(f"  {name:<22s}  AUC={m['auc']}  AP={m['ap']}  P={m['precision']}  R={m['recall']}")

# ── 6. Risk scoring DataFrame ─────────────────────────────────────────────────
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

# ── 7. Lightweight EDA cache ──────────────────────────────────────────────────
eda_cols = [
    "Is_Anomalous", "Donation_Amount", "Cumulative_Given", "Hour", "DayOfWeek",
    "Region", "Donation_Type", "Campaign_Category", "Donation_Channel",
    "Device_Type", "Currency", "Is_Night_Donation", "High_Risk_Platform",
    "Age", "Is_Round_Amount", "Near_Threshold", "Low_Balance_Large_Gift",
    "Amt_Bal_Ratio",
]
eda_df = df[eda_cols].copy()

# ── 8. Meta / summary stats ───────────────────────────────────────────────────
summary_stats = {
    "amount_mean":  float(df["Donation_Amount"].mean()),
    "amount_std":   float(df["Donation_Amount"].std()),
    "anomaly_rate": float(df["Is_Anomalous"].mean()),
    "total":        int(len(df)),
    "n_anomalous":  int(df["Is_Anomalous"].sum()),
    "night_rate":   float(df[df["Is_Night_Donation"] == 1]["Is_Anomalous"].mean()),
    "high_risk_n":  int(df[df["High_Risk_Platform"] == 1]["Is_Anomalous"].sum()),
    "features":     MODEL_FEATURES,
    "cat_encode":   CAT_ENCODE,
    "amt_p75":      float(amt_p75),
    "bal_p10":      float(bal_p10),
    "structuring_thresholds": structuring_thresholds,
}

# ── 9. Save ───────────────────────────────────────────────────────────────────
ARTIFACTS.mkdir(exist_ok=True)
print("\nSaving artifacts…")
joblib.dump(rf,          ARTIFACTS / "rf_model.joblib",    compress=3)
joblib.dump(lr,          ARTIFACTS / "lr_model.joblib",    compress=3)
joblib.dump(iso,         ARTIFACTS / "iso_model.joblib",   compress=3)
joblib.dump(scaler,      ARTIFACTS / "scaler.joblib",      compress=3)
joblib.dump(encoders,    ARTIFACTS / "encoders.joblib",    compress=3)
joblib.dump(importances, ARTIFACTS / "importances.joblib", compress=3)
risk_df.to_parquet(ARTIFACTS / "risk_df.parquet",  index=False)
eda_df.to_parquet( ARTIFACTS / "eda_cache.parquet", index=False)
with open(ARTIFACTS / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
with open(ARTIFACTS / "meta.json", "w") as f:
    json.dump(summary_stats, f, indent=2)

print(f"\n✅  Done.  Artifacts → ./artifacts/")
print(f"   RF   AUC={metrics['Random Forest']['auc']}  AP={metrics['Random Forest']['ap']}")
print(f"   LR   AUC={metrics['Logistic Regression']['auc']}  AP={metrics['Logistic Regression']['ap']}")
print(f"   ISO  AUC={metrics['Isolation Forest']['auc']}  AP={metrics['Isolation Forest']['ap']}")
