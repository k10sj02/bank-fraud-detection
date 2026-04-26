"""
reshape_dataset.py  —  Syria Campaign · Dataset Reshaping

Transforms the LOL Bank transaction dataset into a realistic nonprofit
donation dataset. Run this BEFORE generate_labels.py.

Pipeline order:
    1. uv run python reshape_dataset.py
    2. uv run python generate_labels.py
    3. uv run python prepare.py
    4. uv run streamlit run app.py

Input:  Bank_Transaction_Fraud_Detection.csv
Output: Donations_Syria_Campaign.csv
"""

import numpy as np
import pandas as pd

RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)

INPUT_CSV  = "Bank_Transaction_Fraud_Detection.csv"
OUTPUT_CSV = "Donations_Syria_Campaign.csv"

# ── 1. Load ───────────────────────────────────────────────────────────────────
print("Loading raw dataset…")
df = pd.read_csv(INPUT_CSV)
n  = len(df)
print(f"  {n:,} rows, {df.shape[1]} columns")

# ── 2. Drop bank-specific columns ─────────────────────────────────────────────
print("Dropping bank-specific columns…")
DROP_COLS = [
    "Bank_Branch",        # branches don't exist in nonprofits
    "Merchant_ID",        # there's only one merchant (the campaign)
    "Merchant_Category",  # Restaurant/Electronics/Groceries — wrong domain
    "Transaction_Description",  # taxi fares, laundry etc.
    "Customer_Contact",   # not useful for fraud detection
    "Customer_Email",     # anonymised anyway
]
df = df.drop(columns=DROP_COLS)

# ── 3. Rename columns to donation domain ──────────────────────────────────────
print("Renaming columns to donation domain…")
RENAME = {
    "Customer_ID":        "Donor_ID",
    "Customer_Name":      "Donor_Name",
    "State":              "Donor_Region",
    "City":               "Donor_City",
    "Account_Type":       "Donor_Segment",
    "Transaction_ID":     "Donation_ID",
    "Transaction_Date":   "Donation_Date",
    "Transaction_Time":   "Donation_Time",
    "Transaction_Amount": "Donation_Amount_INR",
    "Transaction_Type":   "Donation_Type_Raw",    # will be replaced
    "Account_Balance":    "Donor_Lifetime_Value_INR",
    "Transaction_Device": "Device_Raw",           # will be replaced
    "Transaction_Location": "Donation_Location",
    "Device_Type":        "Device_Category",
    "Is_Fraud":           "Is_Fraud",             # kept for generate_labels.py
}
df = df.rename(columns=RENAME)

# ── 4. Remap geography to international donor base ────────────────────────────
# Syria Campaign donors are predominantly UK, US, EU, Gulf, and diaspora communities.
# Map Indian states to realistic donor geographies weighted by campaign's actual
# supporter base profile.
print("Remapping geography to international donor base…")

REGION_MAP = {
    # UK regions (largest donor base for Syria Campaign)
    "Nagaland":           "London, UK",
    "Meghalaya":          "Manchester, UK",
    "Uttar Pradesh":      "Birmingham, UK",
    "Uttarakhand":        "Edinburgh, UK",
    "Lakshadweep":        "Bristol, UK",
    "Telangana":          "Leeds, UK",
    "Haryana":            "Glasgow, UK",
    "Delhi":              "Liverpool, UK",
    "Maharashtra":        "Sheffield, UK",
    "Karnataka":          "Cardiff, UK",
    # US regions
    "Gujarat":            "New York, US",
    "Rajasthan":          "Washington DC, US",
    "Tamil Nadu":         "Los Angeles, US",
    "West Bengal":        "Chicago, US",
    "Punjab":             "Houston, US",
    "Kerala":             "San Francisco, US",
    "Andhra Pradesh":     "Seattle, US",
    "Madhya Pradesh":     "Boston, US",
    # Gulf / MENA (significant Syria Campaign donors)
    "Goa":                "Dubai, UAE",
    "Odisha":             "Abu Dhabi, UAE",
    "Assam":              "Doha, Qatar",
    "Jharkhand":          "Kuwait City, Kuwait",
    "Himachal Pradesh":   "Riyadh, Saudi Arabia",
    "Chhattisgarh":       "Amman, Jordan",
    # EU
    "Chandigarh":         "Berlin, Germany",
    "Manipur":            "Paris, France",
    "Tripura":            "Amsterdam, Netherlands",
    "Mizoram":            "Stockholm, Sweden",
    "Arunachal Pradesh":  "Copenhagen, Denmark",
    # Canada / Australia
    "Jammu and Kashmir":  "Toronto, Canada",
    "Sikkim":             "Sydney, Australia",
    "Dadra and Nagar Haveli and Daman and Diu": "Melbourne, Australia",
    "Andaman and Nicobar Islands": "Vancouver, Canada",
}

# Extract region from Donation_Location (format: "City, State")
df["Donor_Region_Raw"] = df["Donation_Location"].str.split(", ").str[-1]
df["Donor_Country_City"] = df["Donor_Region_Raw"].map(REGION_MAP).fillna("London, UK")
df["Donor_Country"] = df["Donor_Country_City"].str.split(", ").str[-1]
df["Donor_City_Mapped"] = df["Donor_Country_City"].str.split(", ").str[0]
df = df.drop(columns=["Donation_Location", "Donor_Region_Raw", "Donor_Country_City",
                       "Donor_Region", "Donor_City"])
df = df.rename(columns={"Donor_City_Mapped": "Donor_City",
                         "Donor_Country": "Donor_Country"})

# ── 5. Remap currency (INR → realistic multi-currency) ────────────────────────
print("Remapping currency to multi-currency…")

# Exchange rates relative to INR (approximate 2024 rates)
CURRENCY_BY_COUNTRY = {
    "UK":           ("GBP", 0.0095),
    "US":           ("USD", 0.012),
    "UAE":          ("USD", 0.012),   # Gulf donations often in USD
    "Qatar":        ("USD", 0.012),
    "Kuwait":       ("USD", 0.012),
    "Saudi Arabia": ("USD", 0.012),
    "Jordan":       ("USD", 0.012),
    "Germany":      ("EUR", 0.011),
    "France":       ("EUR", 0.011),
    "Netherlands":  ("EUR", 0.011),
    "Sweden":       ("SEK", 0.125),
    "Denmark":      ("DKK", 0.082),
    "Canada":       ("CAD", 0.016),
    "Australia":    ("AUD", 0.018),
}

currencies, amounts = [], []
for _, row in df[["Donor_Country", "Donation_Amount_INR"]].iterrows():
    currency, rate = CURRENCY_BY_COUNTRY.get(row["Donor_Country"], ("GBP", 0.0095))
    currencies.append(currency)
    amounts.append(round(row["Donation_Amount_INR"] * rate, 2))

df["Currency"] = currencies
df["Donation_Amount"] = amounts
df = df.drop(columns=["Donation_Amount_INR"])

# Remap Donor_Lifetime_Value similarly
df["Donor_Lifetime_Value"] = (
    df["Donor_Lifetime_Value_INR"] *
    df["Currency"].map({
        "GBP": 0.0095, "USD": 0.012, "EUR": 0.011,
        "SEK": 0.125,  "DKK": 0.082, "CAD": 0.016, "AUD": 0.018,
    })
).round(2)
df = df.drop(columns=["Donor_Lifetime_Value_INR"])

# ── 6. Remap Donor_Segment ────────────────────────────────────────────────────
print("Remapping donor segments…")
SEGMENT_MAP = {
    "Savings":   "Individual",
    "Checking":  "Individual",
    "Business":  "Corporate",
}
df["Donor_Segment"] = df["Donor_Segment"].map(SEGMENT_MAP)
# Add a small slice of Major Donors
major_mask = (df["Donation_Amount"] > df["Donation_Amount"].quantile(0.95))
df.loc[major_mask & (rng.random(n) < 0.3), "Donor_Segment"] = "Major Donor"

# ── 7. Remap Donation_Type ────────────────────────────────────────────────────
print("Remapping donation types…")
DONATION_TYPE_MAP = {
    "Transfer":     "Bank Transfer",
    "Bill Payment": "Direct Debit",
    "Debit":        "Card",
    "Withdrawal":   "Refund",       # withdrawal → refund request in nonprofit context
    "Credit":       "Gift",
}
df["Donation_Type"] = df["Donation_Type_Raw"].map(DONATION_TYPE_MAP)
df = df.drop(columns=["Donation_Type_Raw"])

# ── 8. Remap Donation_Platform (device) ──────────────────────────────────────
print("Remapping donation platforms…")
PLATFORM_MAP = {
    "Web Browser":              "JustGiving",
    "Mobile Device":            "Mobile App",
    "Desktop/Laptop":           "Campaign Website",
    "Tablet":                   "Campaign Website",
    "Debit/Credit Card":        "Card (Phone)",
    "Payment Gateway Device":   "Stripe Checkout",
    "POS Terminal":             "Card (Phone)",
    "POS Mobile Device":        "Card (Phone)",
    "POS Mobile App":           "Card (Phone)",
    "Smart Card":               "Card (Phone)",
    "ATM":                      "Bank Transfer",
    "ATM Booth Kiosk":          "Bank Transfer",
    "Self-service Banking Machine": "Bank Transfer",
    "Bank Branch":              "Bank Transfer",
    "Virtual Card":             "Virtual Card",
    "QR Code Scanner":          "QR Code",
    "Voice Assistant":          "Voice Assistant",
    "Banking Chatbot":          "Chatbot",
    "Wearable Device":          "Wearable",
    "Biometric Scanner":        "Biometric",
}
df["Donation_Platform"] = df["Device_Raw"].map(PLATFORM_MAP).fillna("Campaign Website")
df = df.drop(columns=["Device_Raw", "Device_Category"])

# ── 9. Remap Payment Processor ────────────────────────────────────────────────
print("Generating payment processor…")
PROCESSOR_BY_PLATFORM = {
    "JustGiving":       "JustGiving",
    "Mobile App":       "Stripe",
    "Campaign Website": "Stripe",
    "Card (Phone)":     "WorldPay",
    "Stripe Checkout":  "Stripe",
    "Bank Transfer":    "GoCardless",
    "Direct Debit":     "GoCardless",
    "Virtual Card":     "PayPal",
    "QR Code":          "PayPal",
    "Voice Assistant":  "PayPal",
    "Chatbot":          "PayPal",
    "Wearable":         "Stripe",
    "Biometric":        "Stripe",
}
df["Payment_Processor"] = df["Donation_Platform"].map(PROCESSOR_BY_PLATFORM).fillna("Stripe")

# ── 10. Generate Campaign_ID ──────────────────────────────────────────────────
print("Generating campaign IDs…")

CAMPAIGNS = [
    ("SC-2024-EMG", "Syria Emergency Appeal 2024",     0.28),
    ("SC-2024-WIN", "Winter Appeal 2024",               0.18),
    ("SC-2024-EDU", "Education Fund 2024",              0.14),
    ("SC-2024-MED", "Medical Supplies Appeal",          0.12),
    ("SC-2024-REF", "Refugee Support Programme",        0.10),
    ("SC-2023-EMG", "Syria Emergency Appeal 2023",      0.08),
    ("SC-2024-RAM", "Ramadan Appeal 2024",               0.06),
    ("SC-2024-MAT", "Matched Giving December",          0.04),
]

campaign_ids, campaign_names, campaign_weights = zip(*CAMPAIGNS)
campaign_weights = np.array(campaign_weights)
campaign_weights /= campaign_weights.sum()

chosen = rng.choice(len(CAMPAIGNS), size=n, p=campaign_weights)
df["Campaign_ID"]   = [campaign_ids[i]   for i in chosen]
df["Campaign_Name"] = [campaign_names[i] for i in chosen]

# ── 11. Generate Donation_Frequency ──────────────────────────────────────────
print("Generating donation frequency…")
FREQUENCY_OPTIONS = ["First-time", "Occasional", "Regular", "Lapsed"]
FREQUENCY_WEIGHTS = [0.42, 0.28, 0.22, 0.08]
df["Donation_Frequency"] = rng.choice(
    FREQUENCY_OPTIONS, size=n, p=FREQUENCY_WEIGHTS
)

# ── 12. Generate Donor_Since_Days ─────────────────────────────────────────────
print("Generating donor tenure…")
# Most donors are relatively new, a long tail of established donors
donor_since = rng.exponential(scale=400, size=n).astype(int)
donor_since = np.clip(donor_since, 0, 3650)

# Override: Regular donors must be older
regular_mask = df["Donation_Frequency"] == "Regular"
donor_since[regular_mask] = rng.integers(365, 3650, size=regular_mask.sum())

# Override: First-time donors are always 0
firsttime_mask = df["Donation_Frequency"] == "First-time"
donor_since[firsttime_mask] = 0

df["Donor_Since_Days"] = donor_since

# ── 13. Generate Is_Anonymous ─────────────────────────────────────────────────
print("Generating anonymity flag…")
# ~8% anonymous overall; higher for large donations and virtual card
anon_base = rng.random(n) < 0.08
anon_highval = (df["Donation_Amount"] > df["Donation_Amount"].quantile(0.90)) & (rng.random(n) < 0.15)
anon_platform = df["Donation_Platform"].isin(["Virtual Card", "QR Code", "Chatbot"]) & (rng.random(n) < 0.25)
df["Is_Anonymous"] = (anon_base | anon_highval | anon_platform).astype(int)

# ── 14. Generate Gift_Aid_Eligible ───────────────────────────────────────────
print("Generating Gift Aid eligibility…")
# Only UK donors can claim Gift Aid; ~72% of UK donors are eligible
uk_mask = df["Donor_Country"] == "UK"
gift_aid = np.zeros(n, dtype=int)
gift_aid[uk_mask] = (rng.random(uk_mask.sum()) < 0.72).astype(int)
df["Gift_Aid_Eligible"] = gift_aid

# ── 15. Generate Refund_Requested ────────────────────────────────────────────
print("Generating refund flag…")
# ~2% overall; significantly higher for anomalous donations (set later by generate_labels)
# For now, generate independently — generate_labels.py will use this as a signal
refund_base = rng.random(n) < 0.02
# Refunds more common on large donations and virtual card / chatbot platforms
refund_highval = (df["Donation_Amount"] > df["Donation_Amount"].quantile(0.90)) & (rng.random(n) < 0.04)
refund_platform = df["Donation_Platform"].isin(["Virtual Card", "Chatbot"]) & (rng.random(n) < 0.06)
df["Refund_Requested"] = (refund_base | refund_highval | refund_platform).astype(int)

# ── 16. Generate Is_Recurring ────────────────────────────────────────────────
print("Generating recurring donation flag…")
# Regular donors are almost always recurring; others occasionally set up standing orders
recurring_prob = df["Donation_Frequency"].map({
    "Regular":    0.85,
    "Occasional": 0.20,
    "First-time": 0.08,
    "Lapsed":     0.00,
})
df["Is_Recurring"] = (rng.random(n) < recurring_prob).astype(int)

# ── 17. Generate Matched_Giving ──────────────────────────────────────────────
print("Generating matched giving flag…")
# ~5% of donations are employer-matched; mostly Corporate segment and December campaign
matched_base = rng.random(n) < 0.05
matched_corporate = (df["Donor_Segment"] == "Corporate") & (rng.random(n) < 0.18)
matched_december  = (df["Campaign_ID"] == "SC-2024-MAT") & (rng.random(n) < 0.60)
df["Matched_Giving"] = (matched_base | matched_corporate | matched_december).astype(int)

# ── 18. Generate Acquisition_Channel ────────────────────────────────────────
print("Generating acquisition channel…")
CHANNELS = ["Email", "Social Media", "Direct Mail", "Organic Search",
            "Referral", "Corporate Partnership", "Events", "Cold Outreach"]
CHANNEL_WEIGHTS = [0.28, 0.24, 0.14, 0.12, 0.10, 0.06, 0.04, 0.02]
df["Acquisition_Channel"] = rng.choice(CHANNELS, size=n, p=CHANNEL_WEIGHTS)

# ── 19. Generate CVV_Check ────────────────────────────────────────────────────
print("Generating CVV check result…")
# Matches banking partner signals in the app
# Most pass; ~3% fail; N/A for non-card platforms
card_platforms = ["JustGiving", "Mobile App", "Campaign Website",
                  "Card (Phone)", "Stripe Checkout", "Wearable", "Biometric"]
cvv_result = np.where(
    df["Donation_Platform"].isin(card_platforms),
    rng.choice(["Pass", "Fail", "Pass", "Pass", "Pass", "Pass", "Pass",
                "Pass", "Pass", "Pass", "Pass", "Pass", "Pass", "Pass",
                "Pass", "Pass", "Pass", "Pass", "Pass", "Pass"],
               size=n),   # ~5% fail rate via random choice weighting
    "N/A"
)
# Make it cleaner
cvv_choices = rng.random(n)
cvv_result = np.where(
    ~df["Donation_Platform"].isin(card_platforms), "N/A",
    np.where(cvv_choices < 0.94, "Pass", "Fail")
)
df["CVV_Check"] = cvv_result

# ── 20. Generate Device_Fingerprint_Status ───────────────────────────────────
print("Generating device fingerprint status…")
fp_choices = rng.random(n)
df["Device_Fingerprint"] = np.where(
    fp_choices < 0.78, "Known Clean",
    np.where(fp_choices < 0.97, "First Seen", "Known Flagged")
)

# ── 21. Generate Webhook_Event ───────────────────────────────────────────────
print("Generating webhook event type…")
webhook_choices = rng.random(n)
df["Webhook_Event"] = np.where(
    webhook_choices < 0.91, "payment.authorised",
    np.where(webhook_choices < 0.955, "payment.declined",
    np.where(webhook_choices < 0.983, "payment.reversed",
             "chargeback.received"))
)


# ── 22. Generate IP-derived features ─────────────────────────────────────────
print("Generating IP-derived features…")

# ── IP_Country_Match ──────────────────────────────────────────────────────────
# Probability that IP country matches stated donor country.
# Mismatches are more likely for anonymous donors, high-risk platforms,
# failed CVV, and known flagged devices.
mismatch_prob = np.full(n, 0.08)
mismatch_prob[df["Is_Anonymous"] == 1]                                       += 0.15
mismatch_prob[df["Donation_Platform"].isin(
    ["Virtual Card", "QR Code", "Chatbot", "Voice Assistant"])]              += 0.12
mismatch_prob[df["Donation_Frequency"] == "First-time"]                      += 0.05
mismatch_prob[df["Refund_Requested"] == 1]                                   += 0.10
mismatch_prob[df["CVV_Check"] == "Fail"]                                     += 0.12
mismatch_prob[df["Device_Fingerprint"] == "Known Flagged"]                   += 0.18
mismatch_prob = np.clip(mismatch_prob, 0, 0.95)

ip_mismatch = rng.random(n) < mismatch_prob
df["IP_Country_Match"] = (~ip_mismatch).astype(int)

# ── Is_VPN_Or_Proxy ───────────────────────────────────────────────────────────
# VPN/proxy usage probability. Correlated with mismatch and platform.
# In production this comes from MaxMind GeoIP2 or IPQualityScore API.
vpn_prob = np.full(n, 0.04)
vpn_prob[ip_mismatch]                                                         += 0.20
vpn_prob[df["Is_Anonymous"] == 1]                                             += 0.12
vpn_prob[df["Donation_Platform"].isin(["Virtual Card", "Chatbot"])]           += 0.10
vpn_prob[df["Device_Fingerprint"] == "Known Flagged"]                         += 0.15
vpn_prob = np.clip(vpn_prob, 0, 0.95)

df["Is_VPN_Or_Proxy"] = (rng.random(n) < vpn_prob).astype(int)

# Force consistency: VPN users are mismatched ~80% of the time
vpn_mask = df["Is_VPN_Or_Proxy"] == 1
df.loc[vpn_mask & (rng.random(n) < 0.80), "IP_Country_Match"] = 0

# ── IP_Velocity_24h ───────────────────────────────────────────────────────────
# Donations from the same IP in the past 24 hours.
# High velocity (3+) indicates card testing or coordinated attack.
# Simulated from risk signals; in production this is a real-time counter.
velocity = np.ones(n, dtype=int)

high_risk_mask = (
    (df["CVV_Check"] == "Fail") |
    (df["Device_Fingerprint"] == "Known Flagged") |
    (df["Is_VPN_Or_Proxy"] == 1)
)
velocity[high_risk_mask] = rng.choice(
    [1, 2, 3, 4, 5, 8, 12, 20],
    size=high_risk_mask.sum(),
    p=[0.30, 0.20, 0.15, 0.12, 0.10, 0.07, 0.04, 0.02],
)
# Occasional velocity bumps for normal donors (shared office IPs etc.)
normal_bump = (rng.random(n) < 0.05) & ~high_risk_mask
velocity[normal_bump] = rng.integers(2, 4, size=normal_bump.sum())

df["IP_Velocity_24h"] = velocity

print(f"  IP mismatch rate:   {(df['IP_Country_Match']==0).mean():.1%}")
print(f"  VPN/proxy rate:     {df['Is_VPN_Or_Proxy'].mean():.1%}")
print(f"  High velocity (>=3):{(df['IP_Velocity_24h']>=3).mean():.1%}")

# ── 23. Reorder and tidy columns ─────────────────────────────────────────────
print("Tidying column order…")
FINAL_COLS = [
    # Donor identity
    "Donor_ID", "Donor_Name", "Gender", "Age",
    "Donor_City", "Donor_Country", "Donor_Segment",
    "Donor_Since_Days", "Donor_Lifetime_Value",
    "Donation_Frequency", "Acquisition_Channel",
    # Donation details
    "Donation_ID", "Donation_Date", "Donation_Time",
    "Donation_Amount", "Currency",
    "Donation_Type", "Donation_Platform", "Payment_Processor",
    # Campaign
    "Campaign_ID", "Campaign_Name",
    # Donor behaviour flags
    "Is_Recurring", "Is_Anonymous", "Matched_Giving",
    "Gift_Aid_Eligible", "Refund_Requested",
    # Banking partner signals
    "CVV_Check", "Device_Fingerprint", "Webhook_Event",
    # IP-derived signals
    "IP_Country_Match", "Is_VPN_Or_Proxy", "IP_Velocity_24h",
    # Label (to be replaced by generate_labels.py)
    "Is_Fraud",
]
df = df[FINAL_COLS]

# ── 24. Validation ────────────────────────────────────────────────────────────
print("\n=== Validation ===")
print(f"Shape: {df.shape}")
print(f"\nColumn dtypes:")
print(df.dtypes.to_string())
print(f"\nSample row:")
print(df.iloc[0].to_string())
print(f"\nKey distributions:")
print(f"  Countries:     {df['Donor_Country'].value_counts().to_dict()}")
print(f"  Currencies:    {df['Currency'].value_counts().to_dict()}")
print(f"  Segments:      {df['Donor_Segment'].value_counts().to_dict()}")
print(f"  Frequency:     {df['Donation_Frequency'].value_counts().to_dict()}")
print(f"  Platforms:     {df['Donation_Platform'].value_counts().head(6).to_dict()}")
print(f"  Processors:    {df['Payment_Processor'].value_counts().to_dict()}")
print(f"  Campaigns:     {df['Campaign_ID'].value_counts().to_dict()}")
print(f"  Anonymous:     {df['Is_Anonymous'].mean():.1%}")
print(f"  Gift Aid:      {df['Gift_Aid_Eligible'].mean():.1%} (overall), "
      f"{df[df['Donor_Country']=='UK']['Gift_Aid_Eligible'].mean():.1%} (UK only)")
print(f"  Refund req:    {df['Refund_Requested'].mean():.1%}")
print(f"  Recurring:     {df['Is_Recurring'].mean():.1%}")
print(f"  Matched:       {df['Matched_Giving'].mean():.1%}")
print(f"  CVV fail:      {(df['CVV_Check']=='Fail').mean():.1%}")
print(f"  Device flagged:{(df['Device_Fingerprint']=='Known Flagged').mean():.1%}")
print(f"  Webhook ok:    {(df['Webhook_Event']=='payment.authorised').mean():.1%}"
    f"  IP mismatch:   {(df['IP_Country_Match']==0).mean():.1%}\n"
    f"  VPN/proxy:     {df['Is_VPN_Or_Proxy'].mean():.1%}\n"
    f"  High velocity: {(df['IP_Velocity_24h']>=3).mean():.1%}")
print(f"\nDonation_Amount (GBP equivalent sample):")
print(f"  {df[df['Currency']=='GBP']['Donation_Amount'].describe().round(2).to_string()}")

# ── 25. Save ──────────────────────────────────────────────────────────────────
df.to_csv(OUTPUT_CSV, index=False)
print(f"\n✅  Saved → {OUTPUT_CSV}")
print(f"   {len(df):,} rows  |  {df.shape[1]} columns")
print(f"\nNext step:  uv run python generate_labels.py")
