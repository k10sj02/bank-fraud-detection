# DonorGuard · Donation Anomaly Detection

**DonorGuard** is a donation compliance platform for nonprofits and charitable organisations. It detects anomalous and potentially bad-actor donations in real time — flagging money laundering, structuring, sanctions evasion, chargeback abuse, and coordinated fraud before they cause financial or reputational harm.

Built on a reshaped financial transaction dataset with domain-specific feature engineering, realistic synthetic compliance labels, sanctions screening via Watchman, and a Streamlit compliance dashboard.

> **Dataset note:** The original Kaggle `Is_Fraud` labels are randomly assigned and carry zero predictive signal. `generate_labels.py` replaces them with rule-based compliance labels. The original bank schema has also been fully remapped to a nonprofit donation domain. See [The Label Problem](#️-the-label-problem) and [Dataset Reshaping](#dataset-reshaping) below.

---

## Table of Contents

- [Overview](#overview)
- [Why This Matters — The Market Gap](#why-this-matters--the-market-gap)
- [Quickstart](#quickstart)
- [Project Structure](#project-structure)
- [The Label Problem](#️-the-label-problem)
- [Dataset Reshaping](#dataset-reshaping)
- [Synthetic Label Generation](#synthetic-label-generation)
- [IP Feature Design & AUC Calibration](#ip-feature-design--auc-calibration)
- [Pipeline](#pipeline)
- [Features](#features)
- [Sanctions Screening](#sanctions-screening)
- [Dashboard](#dashboard)
- [Build Log — Issues Encountered](#build-log--issues-encountered)
- [Ethical Notes](#ethical-notes)
- [Dataset Credit](#dataset-credit)

---

## Overview

This system detects suspicious donation patterns for the DonorGuard — flagging potential money laundering via charity, structuring, pass-through accounts, card testing, chargeback abuse, and sanctions evasion. It operates across two layers:

**Layer 1 · Prevention** (at the payment gateway)
- CVV verification
- Device fingerprinting
- Velocity checks
- IP & geolocation intelligence
- Real-time webhooks

**Layer 2 · Detection** (on every donation that passes through)
- Sanctions screening via [Watchman](https://github.com/moov-io/watchman) (OFAC, EU, UN, UK)
- Ensemble ML anomaly detection
- Risk triage queue (Critical / High / Medium / Low)
- Audit logs & explainability
- Manual overrides

The pipeline runs in four steps:

| Script | What it does | When to run |
|---|---|---|
| `reshape_dataset.py` | Remaps bank schema → nonprofit donation schema | Once |
| `generate_labels.py` | Builds realistic compliance labels | Once, after reshape |
| `prepare.py` | Trains models, writes `artifacts/` | Once, or when retraining |
| `app.py` | Streamlit compliance dashboard | Any time |

---

## Why This Matters — The Market Gap

Nonprofits and charities are significant targets for financial crime. The UK's Charity Commission and HMRC both impose Anti-Money Laundering (AML) obligations on charitable organisations, and the FATF (Financial Action Task Force) explicitly identifies the nonprofit sector as vulnerable to terrorist financing and sanctions evasion. Despite this, the compliance tooling available to most nonprofits ranges from inadequate to nonexistent.

**What exists today:**

- **Generic payment fraud tools** (Stripe Radar, PayPal fraud detection) — handle card-level fraud well but have no nonprofit-specific logic. They don't understand donation context: Gift Aid, recurring giving, campaign clustering, anonymous donations, or the difference between a £50,000 major gift and a £50,000 structured payment.
- **Enterprise AML platforms** (ComplyAdvantage, Hawk AI, Unit21) — built for banks and fintechs, priced accordingly. A mid-sized charity cannot justify £50,000+/year for transaction monitoring software.
- **Manual processes** — the reality for most small and mid-sized nonprofits. A spreadsheet, an occasional OFAC PDF check, and a compliance officer who also does three other jobs.

**The gap:**

No purpose-built, accessible compliance tool exists for the nonprofit sector that combines donation behavioural analysis, sanctions screening, banking partner signals, and explainable risk scoring in a single platform. The specific threat vectors that matter for charities — layering via refund requests, pass-through accounts posing as major donors, coordinated structuring across campaigns, Gift Aid abuse — are invisible to generic fraud tools.

**Why it's growing:**

Regulatory pressure on nonprofits is increasing. The UK's Economic Crime and Corporate Transparency Act (2023), updated Charity Commission AML guidance, and post-2022 sanctions enforcement around Russia and Belarus have all raised the bar for what "adequate" compliance looks like. Smaller organisations that previously flew under the radar are now expected to demonstrate proportionate controls — and most have nothing to show.

DonorGuard is a proof-of-concept for what purpose-built nonprofit compliance tooling could look like: sector-aware, explainable, and accessible without an enterprise contract.

---

## Quickstart

**Requirements:** Python 3.11+, [`uv`](https://github.com/astral-sh/uv), [Docker](https://www.docker.com)

```bash
# Install dependencies
uv add pandas numpy scipy scikit-learn joblib pyarrow streamlit matplotlib requests

# Terminal 1 — start Watchman sanctions screening (optional but recommended)
docker run -p 8084:8084 moov/watchman
# Wait for: msg="binding to :8084 for HTTP server"

# Terminal 2 — run the pipeline
uv run python reshape_dataset.py      # ~30 seconds
uv run python generate_labels.py      # ~15 seconds
uv run python prepare.py              # ~3 minutes
uv run streamlit run app.py
```

> `prepare.py` is the only heavy step. It runs once locally and writes compressed joblib + parquet artifacts to `artifacts/`. The app loads these instantly — no training happens in the browser.

---

## Project Structure

```
bank-fraud-detection/
├── reshape_dataset.py                             # Step 1: remap bank → nonprofit schema
├── generate_labels.py                             # Step 2: build compliance labels
├── prepare.py                                     # Step 3: train models, write artifacts/
├── watchman.py                                    # Sanctions screening helper
├── app.py                                         # Step 4: Streamlit dashboard
├── Bank_Transaction_Fraud_Detection.csv           # Original Kaggle dataset
├── Donations_DonorGuard.csv                   ← generated by step 1
├── Donations_DonorGuard_Labeled.csv           ← generated by step 2
├── artifacts/                                     ← generated by step 3
│   ├── rf_model.joblib
│   ├── lr_model.joblib
│   ├── iso_model.joblib
│   ├── scaler.joblib
│   ├── encoders.joblib
│   ├── importances.joblib
│   ├── risk_df.parquet
│   ├── eda_cache.parquet
│   ├── metrics.json
│   └── meta.json
├── pyproject.toml
└── uv.lock
```

---

## ⚠️ The Label Problem

After the initial pipeline was built and Precision-Recall curves were plotted, all three models performed at essentially random chance — Average Precision ~0.052, ROC-AUC ~0.50.

**Root cause:** The `Is_Fraud` column in the original Kaggle dataset is randomly assigned. Every diagnostic confirmed zero signal:

| Signal check | Result |
|---|---|
| Correlation: `Transaction_Amount` vs `Is_Fraud` | −0.0021 |
| Correlation: `Age` vs `Is_Fraud` | −0.0015 |
| Correlation: `Account_Balance` vs `Is_Fraud` | +0.0001 |
| Fraud rate across all `Transaction_Type` values | 4.93% – 5.19% (completely flat) |
| Fraud rate across all `Device_Type` values | 4.99% – 5.10% (completely flat) |
| Fraud rate variance across 34 states | std dev 0.0027 (noise-level) |
| Top 1% largest transactions | 5.85% vs 5.04% baseline (negligible) |

This is a Kaggle dataset designed for *practising the ML pipeline*, not one where labels were generated from feature values. No model can learn from noise.

**Fix:** `generate_labels.py` rebuilds labels from scratch using 17 weighted compliance rules.

---

## Dataset Reshaping

The original dataset models a retail bank ledger — every row is a debit, credit, or transfer from a customer's bank account. A nonprofit's donation data looks nothing like this.

### Columns dropped (no nonprofit equivalent)

| Column | Reason |
|---|---|
| `Bank_Branch` | Donors aren't assigned to branches |
| `Merchant_ID` | There's only one merchant (the nonprofit) |
| `Merchant_Category` | Restaurant/Electronics/Groceries — wrong domain |
| `Transaction_Description` | Taxi fares, laundry — wrong domain |
| `Customer_Contact` / `Customer_Email` | Anonymised and not useful |

### Columns remapped

| Original | Renamed to | Notes |
|---|---|---|
| `Account_Type` | `Donor_Segment` | Savings/Checking/Business → Individual/Corporate/Major Donor |
| `Account_Balance` | `Donor_Lifetime_Value` | Reframed as historical giving, not bank balance |
| `Transaction_Type` | `Donation_Type` | Debit/Transfer → Card/Bank Transfer/Direct Debit/Gift/Refund |
| `Transaction_Device` | `Donation_Platform` | ATM/Kiosk → JustGiving/Stripe/GoCardless/Mobile App |
| `State` | `Donor_Country` | 34 Indian states → 14 international donor geographies |
| `Transaction_Currency` | `Currency` | All-INR → GBP/USD/EUR/SEK/DKK/CAD/AUD |

### Columns generated synthetically

| New column | Why it matters |
|---|---|
| `Campaign_ID` / `Campaign_Name` | 8 named appeals (Emergency, Winter, Ramadan, etc.) |
| `Donation_Frequency` | First-time / Occasional / Regular / Lapsed |
| `Donor_Since_Days` | Account age — strong fraud signal |
| `Is_Anonymous` | Anonymous donations are higher risk |
| `Gift_Aid_Eligible` | UK-specific — 72% of UK donors eligible |
| `Refund_Requested` | Classic layering signal |
| `Is_Recurring` | Recurring donors are almost never bad actors |
| `Matched_Giving` | Employer-matched donations are very low risk |
| `Acquisition_Channel` | Cold outreach channels have higher fraud rates |
| `CVV_Check` | Banking partner signal |
| `Device_Fingerprint` | Banking partner signal |
| `Webhook_Event` | payment.authorised / reversed / declined / chargeback |
| `IP_Country_Match` | IP geolocation vs stated country |
| `Is_VPN_Or_Proxy` | VPN/proxy detection |
| `IP_Velocity_24h` | Donations from same IP in 24 hours |

---

## Synthetic Label Generation

Labels are generated in four stages:

1. **Rule scoring** — each row scored across 17 compliance rules with weighted severity
2. **Noise injection** — Gaussian noise (σ=0.10) blurs hard boundaries so models learn soft probabilities
3. **Threshold calibration** — percentile threshold achieves ~5.5% anomaly rate
4. **Realistic label noise** — 12% of fraud flipped to 0 (missed fraud), 0.5% of legitimate flipped to 1 (false flags)

### Compliance rules

| Rule | Threat modelled |
|---|---|
| Refund requested + large amount | Layering — donate, get receipt, reverse |
| Anonymous + large donation | Unverifiable source of funds |
| First-time donor + very large gift | Mule / pass-through account |
| CVV check failed | Stolen card details |
| Known flagged device | Previously linked to fraud |
| Chargeback webhook | Card fraud or deliberate exploitation |
| Payment reversed webhook | Funds cycling |
| High-risk platform + large amount | Untraceable channel |
| Donation >> lifetime value | Pass-through account |
| Night + transfer + large | Automated / scripted activity |
| Cold outreach + large + first-time | Cultivated mule |
| Corporate + night + transfer | Corporate money laundering via charity |
| Matched giving + anonymous | Contradiction — matching requires verified identity |
| Near-threshold structuring | Threshold avoidance |
| IP country mismatch + large | Misrepresented location |
| VPN/proxy + anonymous | Coordinated anonymity |
| IP velocity ≥ 3 in 24h | Card testing / bot attack |

### Results

| Metric | Before (random labels) | After (rule labels + noise) |
|---|---|---|
| ROC-AUC | 0.502 | **0.880** |
| Average Precision | 0.052 | **0.403** |
| Lift over baseline | 1.0× | **7.6×** |

---

## IP Feature Design & AUC Calibration

An early version of the IP features produced an AUC of 0.951 — suspiciously high. The cause was **circular dependency**: IP features were generated with probabilities conditioned on `CVV_Check == "Fail"` and `Device_Fingerprint == "Known Flagged"`, which are themselves inputs to the fraud label rules. The model was memorising the generative process rather than learning real patterns.

**Fix:** IP features generated using only plausible ambient correlations to donor *behaviour* (anonymity preference, platform choice, country routing patterns) — not to banking partner fraud signals. The correlation is then *discovered* by the model through the label rules, not pre-encoded.

Additionally, **realistic label noise** (12% missed fraud, 0.5% false flags) was added to simulate real-world compliance dataset imperfection.

Final AUC: **0.880** — strong, defensible, and realistic.

---

## Pipeline

### `reshape_dataset.py`

Transforms the raw bank CSV into a nonprofit donation dataset. Drops bank-specific columns, remaps geography to international donor base (UK 35%, US 23%, Gulf 15%, EU 17%, Canada/Australia 10%), converts amounts to local currencies, and generates 15 new nonprofit-specific columns.

### `generate_labels.py`

Scores each row across 17 compliance rules, injects calibrated noise, applies label noise (12% missed fraud, 0.5% false flags). Prints a validation table showing fraud rate and lift per rule on exit.

### `prepare.py`

Trains three models and computes optimal decision thresholds via F-beta optimisation (β=0.5):

| Model | Config | Notes |
|---|---|---|
| **Logistic Regression** | Calibrated with `CalibratedClassifierCV`, `class_weight='balanced'` | Default — AUC 0.905, P 69%, R 38%, threshold 0.47 |
| **Random Forest** | 300 trees, `max_depth=16`, `class_weight='balanced'` | AUC 0.902, P 68%, R 32%, threshold 0.79 |
| **Isolation Forest** | `contamination=0.055`, 200 trees | Unsupervised — no labels needed |

Optimal thresholds are saved to `meta.json` and loaded as sidebar defaults in the app.

### `watchman.py`

Sanctions screening helper. Handles Watchman v0.31.x response format (flat per-list keys). Degrades gracefully when offline.

### `app.py`

Three-page Streamlit dashboard:
- **🏠 Context** — two-layer protection architecture, threat guide, risk tier guide
- **📊 Dashboard** — KPIs, model performance, EDA, risk triage queue, OFAC screening, donation scorer
- **📖 Glossary** — 30+ plain-English definitions for compliance reviewers

---

## Features

| Feature | Description |
|---|---|
| `Log_Donation_Amount` | log1p of donation amount |
| `Log_Donor_Lifetime_Value` | log1p of donor lifetime giving |
| `Amt_LTV_Ratio` | Donation / (Lifetime Value + 1) — pass-through signal |
| `Donation_Amount_Zscore` | Z-score relative to full dataset |
| `Is_Night_Donation` | 1 if hour 00:00–04:59 |
| `Is_Weekend` | 1 if Saturday or Sunday |
| `Is_Round_Amount` | 1 if amount % 50 = 0 |
| `Near_Threshold` | 1 if within 1% below £500/£750/£900/£950 |
| `High_Risk_Platform` | 1 if Virtual Card, QR Code, Chatbot, Wearable |
| `CVV_Failed` | 1 if CVV check failed |
| `Device_Flagged` | 1 if known flagged device fingerprint |
| `Webhook_Chargeback` / `Webhook_Reversed` | Banking partner event flags |
| `Refund_Requested` | 1 if donor requested a refund |
| `Is_First_Time_Large` | First-time donor + amount ≥ p90 |
| `Is_Anon_Large` | Anonymous + amount ≥ p75 |
| `Is_Corporate_Night` | Corporate + night + bank transfer |
| `Matched_Anon_Flag` | Matched giving + anonymous |
| `IP_Country_Match` | 1 if IP country matches stated country |
| `Is_VPN_Or_Proxy` | 1 if VPN/proxy detected |
| `IP_Velocity_24h` | Donations from same IP in 24 hours |
| `IP_Mismatch_Large` / `VPN_Anon_Flag` / `High_Velocity_Flag` | IP composite signals |
| Encoded categoricals | Donor_Country, Donor_Segment, Donation_Frequency, Donation_Type, Donation_Platform, Payment_Processor, Campaign_ID, Acquisition_Channel, Currency, CVV_Check, Device_Fingerprint, Webhook_Event |

---

## Sanctions Screening

Watchman runs locally via Docker and screens against OFAC SDN (18,863 entities), EU Consolidated (5,993), UK CSL (5,135), BIS Entity List (3,420), plus DPL, SSI, DTC, ISN and more.

```bash
docker run -p 8084:8084 moov/watchman
# Ready when you see: msg="binding to :8084 for HTTP server"
```

A confirmed sanctions match is a **hard block** — ML scoring is bypassed entirely. The app degrades gracefully when Watchman is offline.

> **Watchman version note:** v0.31.x uses `/search` (no version prefix) and returns a flat per-list response. `watchman.py` handles this automatically — `/health` returns 404 on this version and should not be used as a liveness check.

---

## Dashboard

**Overview** — KPIs: total donations, anomaly count and rate, AUC, average precision, night anomaly rate, high-risk platform count.

**Model Performance** — confusion matrix and PR curves at optimal F-0.5 threshold. Threshold slider in sidebar to explore precision-recall trade-off.

**Exploratory Analysis** — five tabs: Amounts, Temporal, Geography (by country), Categories, Feature Importance.

**Risk Triage Queue** — two tabs:
- *Triage Queue* — filterable by tier (Critical/High/Medium/Low), sortable by score, with recommended actions per tier
- *OFAC Bulk Screen* — paste donor names, screen against all loaded sanctions lists, see detailed hits

**Single Donation Scorer** — enter donation details, OFAC screen runs first (hard block on hit), then ML anomaly score with fired compliance rules explained in plain English.

---

## Build Log — Issues Encountered

### 1. Random labels — zero model signal

**Problem:** All models at random chance (AUC ~0.50). Zero correlation between any feature and `Is_Fraud`.

**Cause:** Kaggle dataset labels are randomly assigned.

**Fix:** `generate_labels.py` — 17 weighted compliance rules, calibrated noise, label noise injection.

**Result:** AUC 0.502 → 0.880, lift 1× → 7.6×

---

### 2. Bank schema mismatch for nonprofit use case

**Problem:** Bank balances, branch assignments, merchant categories, Indian geography, INR currency, and descriptions like "Taxi fare" — none applicable to a nonprofit.

**Fix:** `reshape_dataset.py` — drops 6 irrelevant columns, remaps geography to 14 international donor countries, converts to multi-currency, generates 15 new nonprofit-specific columns.

---

### 3. AUC inflation from circular IP feature dependency (0.951)

**Problem:** After adding IP features, AUC jumped from 0.919 to 0.951. Model was memorising the generative process.

**Cause:** IP features conditioned on `CVV_Check == "Fail"` and `Device_Fingerprint == "Known Flagged"` — same columns feeding the fraud label rules.

**Fix:** IP features regenerated using only donor behaviour signals (anonymity, platform choice, country routing). Label noise added (12% missed fraud, 0.5% false flags).

**Result:** AUC 0.951 → 0.880

---

### 4. Low precision at default threshold (38.5%)

**Problem:** At threshold 0.45, 62% of flagged donations were legitimate donors.

**Cause:** 0.45 was hardcoded from the old dataset. Score distribution shifted after reshaping.

**Fix:** `prepare.py` computes F-beta optimal threshold (β=0.5) per model and saves to `meta.json`. App loads as sidebar default.

**Result:** Precision 38.5% → 67–69%

---

### 5. LR threshold at 0.946 (uncalibrated probabilities)

**Problem:** Raw LR probabilities squashed toward extremes — optimal threshold landed at 0.946.

**Fix:** Wrapped LR in `CalibratedClassifierCV` with 5-fold isotonic calibration.

**Result:** Threshold 0.946 → 0.470, recall improved, precision held at 69%

---

### 6. `KeyError: 'Region'` after dataset reshape

**Problem:** App crashed on geography tab after column rename `Region` → `Donor_Country`.

**Fix:** Updated all stale column references in `app.py` — geography, categories, scorer sample dict, and fired rule descriptions.

---

### 7. `NameError: 'arts' is not defined` in sidebar

**Problem:** Sidebar slider read `OPT_THRESHOLDS` before artifact loading code had run.

**Fix:** Moved `arts = load_artifacts()` and all derived variables before the sidebar block.

---

### 8. `optimal_threshold()` called before it was defined

**Problem:** Function called at line 216, defined at line 244. Python doesn't hoist — script silently used old threshold of 0.45.

**Fix:** `prepare.py` rewritten from scratch with helper functions defined at the top.

---

### 9. Watchman showing offline despite Docker running

**Problem:** Sidebar showed "Watchman offline" even with `docker run -p 8084:8084 moov/watchman` running.

**Cause 1:** `_is_alive()` probed `/health` first — returns 404 on v0.31.3.

**Cause 2:** Response parser expected `{ entities: [...] }` (v2 API) but v0.31.x returns flat per-list keys: `SDNs`, `euConsolidatedSanctionsList`, `ukConsolidatedSanctionsList`, etc.

**Fix:** `watchman.py` updated to probe `/search` directly and parse the flat v0.31.x response format with field name handling per list.

---

### 10. `use_container_width` deprecation warnings

**Problem:** Streamlit logging repeated deprecation warnings.

**Fix:** `use_container_width=True` → `width='stretch'`, `use_container_width=False` → `width='content'` across all `st.image()` and `st.dataframe()` calls.

---

### 11. Squashed section labels in model performance area

**Problem:** Section labels rendering as full uppercase monospace banners, wrapping across multiple lines in narrow columns.

**Fix:** Replaced `section_label()` calls with lightweight `<p>` inline headers. Threshold note condensed to one sentence with inline code chip.

---

### 12. Fuzzy/blurry matplotlib charts

**Problem:** All charts rendered blurry, especially on Retina displays.

**Fix 1:** `figure.dpi: 300`, `savefig.dpi: 300` in `plt.rcParams`.

**Fix 2 (definitive):** `show_fig()` helper saves figures to `BytesIO` at 300 DPI and serves via `st.image()`, bypassing Streamlit's pyplot renderer entirely.

---

## Ethical Notes

> Labels are synthetically generated from domain rules, not real compliance outcomes. All Critical and High flags should go through human review before any action is taken against a donor.

- **Human-in-the-loop** — Critical/High flags require compliance officer sign-off
- **Appeals mechanism** — donors incorrectly flagged need a clear dispute process
- **Bias audits** — regularly test for systematic over-flagging of any demographic group
- **Feedback loop** — compliance review outcomes should retrain the model monthly
- **SHAP explanations** — add for explainable audit trails required by regulators
- **Real labels** — replace synthetic labels with compliance-reviewed historical cases over time
- **Sanctions ≠ full KYC** — OFAC screening is a hard-block layer, not a substitute for KYC on large donations

---

## Dataset Credit

**Training data** — adapted from a synthetic bank transaction dataset ([Kaggle](https://www.kaggle.com/datasets/marusagar/bank-transaction-fraud-detection)), reshaped into a nonprofit donation schema via `reshape_dataset.py`.

The original fraud labels carried no predictive signal (randomly assigned). DonorGuard's label pipeline replaces them with 17 compliance rules encoding real donation fraud patterns. See [The Label Problem](#️-the-label-problem) and [Dataset Reshaping](#dataset-reshaping) for the full diagnostic.