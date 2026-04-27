"""
generate_labels.py  —  DonorGuard · Realistic Fraud Label Generation

Generates compliance-grounded fraud labels for the reshaped nonprofit
donation dataset. Designed for Donations_DonorGuard.csv produced by
reshape_dataset.py.

Rules exploit the full nonprofit schema — including the new columns
(Refund_Requested, Is_Anonymous, Campaign_ID, Device_Fingerprint,
CVV_Check, Webhook_Event, Donation_Frequency, Matched_Giving, etc.)
that didn't exist in the original bank dataset.

Run order:
    1. uv run python reshape_dataset.py
    2. uv run python generate_labels.py       ← this script
    3. uv run python prepare.py
    4. uv run streamlit run app.py

Input:  Donations_DonorGuard.csv
Output: Donations_DonorGuard_Labeled.csv
"""

import numpy as np
import pandas as pd

RANDOM_STATE     = 42
TARGET_FRAUD_RATE = 0.055
rng = np.random.default_rng(RANDOM_STATE)

INPUT_CSV  = "Donations_DonorGuard.csv"
OUTPUT_CSV = "Donations_DonorGuard_Labeled.csv"

print("Loading reshaped dataset…")
df = pd.read_csv(INPUT_CSV)
n  = len(df)
print(f"  {n:,} rows, {df.shape[1]} columns")

# ── Derived fields needed for rules ──────────────────────────────────────────
df["Donation_Date_Parsed"] = pd.to_datetime(df["Donation_Date"], dayfirst=True, errors="coerce")
df["Hour"]      = pd.to_datetime(df["Donation_Time"], format="%H:%M:%S", errors="coerce").dt.hour
df["DayOfWeek"] = df["Donation_Date_Parsed"].dt.dayofweek   # 0=Mon, 6=Sun
df["IsWeekend"] = (df["DayOfWeek"] >= 5).astype(int)
df["AmtLifetimeRatio"] = df["Donation_Amount"] / (df["Donor_Lifetime_Value"] + 1)

# Percentile anchors (computed per-currency group to be fair)
amt_p75 = df["Donation_Amount"].quantile(0.75)
amt_p90 = df["Donation_Amount"].quantile(0.90)
amt_p95 = df["Donation_Amount"].quantile(0.95)
ltv_p10 = df["Donor_Lifetime_Value"].quantile(0.10)
ratio_p90 = df["AmtLifetimeRatio"].quantile(0.90)

print(f"  Amount p75={amt_p75:.2f}  p90={amt_p90:.2f}  p95={amt_p95:.2f}")
print(f"  LTV p10={ltv_p10:.2f}  Amt/LTV ratio p90={ratio_p90:.2f}")

scores = pd.Series(np.zeros(n), index=df.index)

# ═════════════════════════════════════════════════════════════════════════════
# NONPROFIT-SPECIFIC RULES
# These exploit the new columns that couldn't exist in the bank dataset.
# ═════════════════════════════════════════════════════════════════════════════

# ── Rule 1 · REFUND REQUESTED ────────────────────────────────────────────────
# The single strongest nonprofit-specific signal. Donating then requesting a
# refund is the core layering mechanic — donate, get a receipt, reverse the
# payment via a different account or mechanism.
scores += (df["Refund_Requested"] == 1).astype(float) * 3.0
scores += (
    (df["Refund_Requested"] == 1) &
    (df["Donation_Amount"] >= amt_p75)
).astype(float) * 1.5  # extra weight for large refund attempts

# ── Rule 2 · ANONYMOUS + LARGE DONATION ──────────────────────────────────────
# Anonymous donations are legitimate and common in small amounts. Large
# anonymous donations are a red flag — impossible to verify source of funds.
is_large = df["Donation_Amount"] >= amt_p75
is_anon  = df["Is_Anonymous"] == 1
scores += (is_anon & is_large).astype(float) * 2.2
scores += (is_anon & (df["Donation_Amount"] >= amt_p95)).astype(float) * 1.5

# ── Rule 3 · FIRST-TIME DONOR + VERY LARGE DONATION ──────────────────────────
# A brand-new donor making an unusually large gift with no donation history
# is a classic mule or pass-through signal. Legitimate major gifts almost
# always follow a cultivation period.
is_firsttime = df["Donation_Frequency"] == "First-time"
scores += (
    is_firsttime & (df["Donation_Amount"] >= amt_p90)
).astype(float) * 2.0
scores += (
    is_firsttime & (df["Donor_Since_Days"] == 0) & (df["Donation_Amount"] >= amt_p75)
).astype(float) * 1.2

# ── Rule 4 · CVV FAIL ─────────────────────────────────────────────────────────
# Failed CVV = stolen card details without the physical card. In a production
# system this would be hard-blocked at the payment layer, but we model it
# here for the ML score too.
scores += (df["CVV_Check"] == "Fail").astype(float) * 2.8
scores += (
    (df["CVV_Check"] == "Fail") & is_large
).astype(float) * 1.0

# ── Rule 5 · KNOWN FLAGGED DEVICE ────────────────────────────────────────────
# Device fingerprint previously associated with fraudulent transactions.
# This is the strongest device-level signal from the banking partner.
scores += (df["Device_Fingerprint"] == "Known Flagged").astype(float) * 3.0
scores += (
    (df["Device_Fingerprint"] == "Known Flagged") & is_large
).astype(float) * 1.2

# ── Rule 6 · CHARGEBACK OR REVERSAL WEBHOOK ──────────────────────────────────
# A chargeback means the cardholder's bank is disputing the transaction —
# strong evidence of card fraud. A reversal after authorisation is a layering
# signal. Both are direct banking partner signals via webhook.
scores += (df["Webhook_Event"] == "chargeback.received").astype(float) * 3.5
scores += (df["Webhook_Event"] == "payment.reversed").astype(float) * 2.0
scores += (
    (df["Webhook_Event"] == "payment.reversed") & is_large
).astype(float) * 1.5

# ── Rule 7 · HIGH-RISK PLATFORM + LARGE AMOUNT ───────────────────────────────
# Untraceable channels. Virtual cards, QR codes, and chatbots are harder to
# link back to a verified identity. Elevated risk at large amounts.
high_risk_platforms = ["Virtual Card", "QR Code", "Voice Assistant", "Chatbot", "Wearable"]
is_high_risk = df["Donation_Platform"].isin(high_risk_platforms)
scores += is_high_risk.astype(float) * 0.8
scores += (is_high_risk & is_large).astype(float) * 1.6

# ── Rule 8 · DONATION >> LIFETIME VALUE ──────────────────────────────────────
# This donation far exceeds the donor's entire giving history.
# Indicates either a pass-through account or stolen card.
scores += (df["AmtLifetimeRatio"] > ratio_p90).astype(float) * 1.8
scores += (df["AmtLifetimeRatio"] > ratio_p90 * 2).astype(float) * 1.2
scores += (
    (df["Donor_Lifetime_Value"] < ltv_p10) & is_large
).astype(float) * 2.0

# ── Rule 9 · NIGHT-TIME LARGE BANK TRANSFER ──────────────────────────────────
# Large bank transfers at 00:00–04:59am are highly atypical for individual
# donors. Consistent with automated scripts or bots.
is_night    = df["Hour"].between(0, 4)
is_transfer = df["Donation_Type"] == "Bank Transfer"
scores += (is_night & is_transfer & is_large).astype(float) * 2.2
scores += (is_night & is_large).astype(float) * 0.6

# ── Rule 10 · LARGE ROUND-NUMBER DONATION ───────────────────────────────────
# Round numbers at scale (£500, £1000, £5000) are a structuring signal.
# Real individual donors rarely give exactly round amounts at high values.
is_round = (df["Donation_Amount"] % 50 == 0)
scores += (is_round & (df["Donation_Amount"] >= amt_p90)).astype(float) * 1.5
scores += (is_round & (df["Donation_Amount"] >= amt_p95)).astype(float) * 1.0

# ── Rule 11 · CAMPAIGN CLUSTERING ────────────────────────────────────────────
# Anomalous donors disproportionately target the Emergency Appeal —
# bad actors exploit high-urgency campaigns where scrutiny may be lower.
# The Emergency Appeal (SC-2024-EMG) gets a small additional boost.
is_emergency = df["Campaign_ID"] == "SC-2024-EMG"
scores += (is_emergency & is_anon & is_large).astype(float) * 1.2
scores += (is_emergency & is_firsttime & (df["Donation_Amount"] >= amt_p90)).astype(float) * 0.8

# ── Rule 12 · COLD OUTREACH + LARGE DONATION ────────────────────────────────
# Donors acquired via cold outreach making large first-time donations are
# disproportionately suspicious — legitimate major donors are cultivated.
is_cold = df["Acquisition_Channel"] == "Cold Outreach"
scores += (is_cold & is_large & is_firsttime).astype(float) * 1.8
scores += (is_cold & (df["Donation_Amount"] >= amt_p95)).astype(float) * 1.2

# ── Rule 13 · CORPORATE + NIGHT TRANSFER ────────────────────────────────────
# Corporate donors making large transfers at night: money laundering
# routed through a charity via a business account.
is_corporate = df["Donor_Segment"] == "Corporate"
scores += (
    is_corporate & is_night & is_transfer & is_large
).astype(float) * 2.0

# ── Rule 14 · MATCHED GIVING ANOMALY ────────────────────────────────────────
# Matched giving claims on anonymous or first-time donations are suspicious —
# employer matching requires verified employee identity, which contradicts
# anonymity and is rarely set up on a first-time donation.
scores += (
    (df["Matched_Giving"] == 1) & is_anon
).astype(float) * 1.5
scores += (
    (df["Matched_Giving"] == 1) & is_firsttime & (df["Donation_Amount"] >= amt_p90)
).astype(float) * 1.0

# ── Rule 15 · NEAR-THRESHOLD STRUCTURING ────────────────────────────────────
# Donations just below reporting thresholds (within 1% of £500, £750, £900, £950).
# These thresholds are in GBP-equivalent; we use Donation_Amount directly.
thresholds = [500, 750, 900, 950]
for t in thresholds:
    near = (df["Donation_Amount"] >= t * 0.99) & (df["Donation_Amount"] < t)
    scores += near.astype(float) * 1.2

# ── Rule 16 · WEEKEND NIGHT + FIRST-TIME + LARGE ────────────────────────────
# An unusual combination of timing, donor history, and amount.
scores += (
    df["IsWeekend"].astype(bool) & is_night & is_firsttime & is_large
).astype(float) * 1.8


# ── Rule 17 · IP COUNTRY MISMATCH ────────────────────────────────────────────
# Donor's stated country does not match their IP geolocation.
# Alone this is a moderate signal (VPNs are common for privacy).
# Combined with other flags it becomes highly significant.
ip_mismatch = df["IP_Country_Match"] == 0
scores += ip_mismatch.astype(float) * 1.2
scores += (ip_mismatch & is_large).astype(float) * 1.0
scores += (ip_mismatch & is_firsttime & is_large).astype(float) * 1.5
scores += (ip_mismatch & (df["Refund_Requested"] == 1)).astype(float) * 1.8

# ── Rule 18 · VPN OR PROXY ───────────────────────────────────────────────────
# Donation routed through a VPN or proxy server. Legitimate for privacy-
# conscious donors, but highly suspicious when combined with large amounts,
# anonymous flag, or failed CVV.
is_vpn = df["Is_VPN_Or_Proxy"] == 1
scores += is_vpn.astype(float) * 1.0
scores += (is_vpn & is_large).astype(float) * 1.2
scores += (is_vpn & is_anon).astype(float) * 1.5
scores += (is_vpn & (df["CVV_Check"] == "Fail")).astype(float) * 2.0
scores += (is_vpn & ip_mismatch).astype(float) * 1.0

# ── Rule 19 · HIGH IP VELOCITY ───────────────────────────────────────────────
# Multiple donations from the same IP within 24 hours.
# 3-4 is suspicious; 5+ is almost certainly card testing or a bot.
high_velocity = df["IP_Velocity_24h"] >= 3
very_high_vel = df["IP_Velocity_24h"] >= 5
scores += high_velocity.astype(float) * 1.5
scores += very_high_vel.astype(float) * 2.0
scores += (very_high_vel & (df["CVV_Check"] == "Fail")).astype(float) * 1.5
scores += (high_velocity & is_large).astype(float) * 1.2

# ── Rule 20 · MULTIPLE RED FLAGS ─────────────────────────────────────────────
# Non-linear interaction: a donation that triggers 3+ individual flags
# is much more suspicious than the sum of its parts.
flag_count = (
    (df["Refund_Requested"] == 1).astype(int) +
    is_anon.astype(int) +
    is_firsttime.astype(int) +
    (df["CVV_Check"] == "Fail").astype(int) +
    (df["Device_Fingerprint"] == "Known Flagged").astype(int) +
    is_high_risk.astype(int) +
    is_night.astype(int) +
    is_cold.astype(int) +
    (df["IP_Country_Match"] == 0).astype(int) +
    (df["Is_VPN_Or_Proxy"] == 1).astype(int) +
    (df["IP_Velocity_24h"] >= 3).astype(int)
)
scores += (flag_count >= 3).astype(float) * 2.0
scores += (flag_count >= 4).astype(float) * 1.5

# ─────────────────────────────────────────────────────────────────────────────
# SCORE → LABEL CONVERSION
# ─────────────────────────────────────────────────────────────────────────────
print("\nScore distribution (pre-noise):")
print(scores.describe().round(3).to_string())

score_norm  = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
noise       = rng.normal(loc=0, scale=0.10, size=n)
score_noisy = np.clip(score_norm + noise, 0, 1)

threshold   = np.percentile(score_noisy, (1 - TARGET_FRAUD_RATE) * 100)
labels      = (score_noisy >= threshold).astype(int)
actual_rate = labels.mean()

print(f"\nLabel threshold: {threshold:.4f} → target {TARGET_FRAUD_RATE:.1%}")
print(f"Actual fraud rate before noise: {actual_rate:.2%}  (n={labels.sum():,})")

# ─────────────────────────────────────────────────────────────────────────────
# REALISTIC LABEL NOISE
#
# Real compliance datasets are imperfect in two ways:
#
#   1. FALSE NEGATIVES (missed fraud): ~12% of true fraud is never caught.
#      Compliance teams miss subtle patterns, bad actors avoid obvious rules,
#      and understaffed teams don't review every Medium-tier flag.
#      → Flip 12% of fraud labels (1→0): "fraud we missed"
#
#   2. FALSE POSITIVES (wrongly flagged): ~5% of legitimate donations are
#      incorrectly flagged and not reversed after review — either because the
#      appeal wasn't made, or the reviewer made an error.
#      → Flip 5% of legitimate labels (0→1): "innocent donors we blocked"
#
# These rates are grounded in published compliance research:
#   - Typical SAR (Suspicious Activity Report) false positive rates: 90–95%
#     of SARs filed are ultimately not prosecuted (i.e. false alarms)
#   - Estimated fraud detection rates in charity sector: 60–75% of actual
#     fraud is detected, implying ~25–40% is missed
#   - We use conservative figures (12% miss rate, 5% false flag) to avoid
#     destroying too much signal while still reflecting real-world messiness.
# ─────────────────────────────────────────────────────────────────────────────
MISS_RATE      = 0.12   # fraud we failed to catch (1 → 0)
FALSE_FLAG_RATE = 0.005 # legitimate donors we wrongly flagged (0 → 1)
# Note: 0.5% of legitimate transactions — not 5%. The 90-95% SAR false
# positive statistic refers to *filed SARs*, not all transactions. Only a
# small fraction of transactions are ever reviewed, so the absolute
# false-flag rate on the full dataset is much lower.

labels_noisy = labels.copy()

# Missed fraud: randomly flip a fraction of true positives to negative
fraud_idx     = np.where(labels == 1)[0]
n_missed      = int(len(fraud_idx) * MISS_RATE)
missed_idx    = rng.choice(fraud_idx, size=n_missed, replace=False)
labels_noisy[missed_idx] = 0

# False flags: randomly flip a fraction of true negatives to positive
legit_idx     = np.where(labels == 0)[0]
n_false_flag  = int(len(legit_idx) * FALSE_FLAG_RATE)
false_flag_idx = rng.choice(legit_idx, size=n_false_flag, replace=False)
labels_noisy[false_flag_idx] = 1

labels      = labels_noisy
actual_rate = labels.mean()

print(f"After label noise (miss={MISS_RATE:.0%}, false_flag={FALSE_FLAG_RATE:.0%}):")
print(f"  Final fraud rate: {actual_rate:.2%}  (n={labels.sum():,})")
print(f"  Fraud flipped 1→0 (missed): {n_missed:,}")
print(f"  Legit flipped 0→1 (false flag): {n_false_flag:,}")

# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION — fraud rate by rule trigger
# ─────────────────────────────────────────────────────────────────────────────
df["Is_Anomalous"] = labels

print("\n=== Signal validation — fraud rate by rule trigger ===")
checks = {
    "Refund requested":                    (df["Refund_Requested"] == 1),
    "Refund + large amount":               (df["Refund_Requested"] == 1) & is_large,
    "Anonymous + large":                   is_anon & is_large,
    "First-time + ≥p90 amount":            is_firsttime & (df["Donation_Amount"] >= amt_p90),
    "CVV fail":                            (df["CVV_Check"] == "Fail"),
    "Known flagged device":                (df["Device_Fingerprint"] == "Known Flagged"),
    "Chargeback webhook":                  (df["Webhook_Event"] == "chargeback.received"),
    "Payment reversed webhook":            (df["Webhook_Event"] == "payment.reversed"),
    "High-risk platform + large":          is_high_risk & is_large,
    "Donation >> lifetime value (p90)":    (df["AmtLifetimeRatio"] > ratio_p90),
    "Night + transfer + large":            is_night & is_transfer & is_large,
    "Large round number":                  is_round & (df["Donation_Amount"] >= amt_p90),
    "Cold outreach + large + first-time":  is_cold & is_large & is_firsttime,
    "Corporate + night + transfer":        is_corporate & is_night & is_transfer,
    "Matched giving + anonymous":          (df["Matched_Giving"] == 1) & is_anon,
    "Near-threshold structuring":          pd.concat([
        (df["Donation_Amount"] >= t * 0.99) & (df["Donation_Amount"] < t)
        for t in thresholds
    ], axis=1).any(axis=1),
    "3+ flags triggered":                  (flag_count >= 3),
    "IP country mismatch":                 (df["IP_Country_Match"] == 0),
    "IP mismatch + large":                 (df["IP_Country_Match"] == 0) & is_large,
    "VPN or proxy":                        (df["Is_VPN_Or_Proxy"] == 1),
    "VPN + anonymous":                     (df["Is_VPN_Or_Proxy"] == 1) & is_anon,
    "VPN + CVV fail":                      (df["Is_VPN_Or_Proxy"] == 1) & (df["CVV_Check"] == "Fail"),
    "IP velocity >= 3 (card testing)":     (df["IP_Velocity_24h"] >= 3),
    "IP velocity >= 5 (bot/attack)":       (df["IP_Velocity_24h"] >= 5),
    "Baseline (all)":                      pd.Series(True, index=df.index),
}

for desc, mask in checks.items():
    count = mask.sum()
    if count == 0:
        continue
    rate  = df.loc[mask, "Is_Anomalous"].mean()
    lift  = rate / actual_rate
    print(f"  {desc:<45s}  n={count:>7,}  rate={rate:.3f}  lift={lift:.1f}×")

# ─────────────────────────────────────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────────────────────────────────────
out_df = df.drop(columns=[
    "Donation_Date_Parsed", "Hour", "DayOfWeek", "IsWeekend",
    "AmtLifetimeRatio", "Is_Fraud", "Is_Anomalous"
])
out_df["Is_Anomalous"] = labels

out_df.to_csv(OUTPUT_CSV, index=False)
print(f"\n✅  Saved → {OUTPUT_CSV}")
print(f"   {len(out_df):,} rows  |  {out_df.shape[1]} columns")
print(f"   Fraud rate: {labels.mean():.2%}  ({labels.sum():,} anomalous)")
print(f"\nNext step:  uv run python prepare.py")
