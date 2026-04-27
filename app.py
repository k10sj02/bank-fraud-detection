"""
app.py  —  DonorGuard · Donation Anomaly Detection
Requires artifacts/ produced by:  uv run python prepare.py
Run:  uv run streamlit run app.py
"""

import io
import json
from pathlib import Path

import joblib
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import precision_recall_curve
from watchman import (
    screen_donor, screen_bulk, inject_demo_hit,
    DEMO_SDN_NAMES, ScreeningResult, _is_alive, DEFAULT_MIN_MATCH,
    WATCHMAN_URL,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DonorGuard · Donation Anomaly Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design tokens — Vanta-inspired ───────────────────────────────────────────
BG     = "#f4f4f9"   # Vanta light lavender-white background
PANEL  = "#ffffff"   # white cards
CREAM  = "#ffffff"   # alias kept for compat
SAND   = "#f4f4f9"   # alias kept for compat
INK    = "#1b1f3b"   # Vanta deep navy
SLATE  = "#4b5068"   # secondary text
RULE   = "#e3e4ee"   # lavender-tinted dividers
PURPLE = "#6b4fbb"   # Vanta violet CTA
VIOLET = "#9b84e0"   # lighter purple
RED    = "#d63e4a"   # anomalous
CORAL  = "#e8735a"   # high risk
BLUE   = "#2c5f8a"   # legitimate / safe (kept for charts)
TEAL   = "#1a7a6e"   # good / low risk
GOLD   = "#c47f1a"   # medium / warning
MUTED  = "#8b8fa8"   # muted labels

plt.rcParams.update({
    "figure.facecolor": CREAM, "axes.facecolor": CREAM,
    "axes.edgecolor": RULE, "axes.labelcolor": SLATE,
    "axes.titlecolor": INK, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.labelsize": 9,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "grid.color": RULE, "grid.linewidth": 0.6,
    "text.color": INK, "font.family": "sans-serif",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False, "axes.grid": True,
    "axes.grid.axis": "x", "figure.dpi": 300, "savefig.dpi": 300,
})

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=DM+Mono:wght@400&display=swap');
html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; background-color: {BG}; color: {INK}; }}
.stApp {{ background-color: {BG}; }}
[data-testid="stSidebar"] {{ background-color: {INK}; border-right: none; }}
[data-testid="stSidebar"] * {{ color: #a8adc4 !important; }}
[data-testid="stSidebar"] h2 {{ color: #ffffff !important; font-family: 'Inter', sans-serif; font-size: 0.68rem !important; font-weight: 600 !important; letter-spacing: 0.16em; text-transform: uppercase; margin-top: 1.6rem !important; }}
[data-testid="stSidebar"] label {{ color: #6a6e87 !important; font-size: 0.68rem !important; letter-spacing: 0.1em; text-transform: uppercase; }}
[data-testid="stSidebar"] [data-testid="stRadio"] label {{ color: #c8cce0 !important; font-size: 0.85rem !important; letter-spacing: 0 !important; text-transform: none !important; }}
[data-testid="metric-container"] {{ background: {PANEL}; border: 1px solid {RULE}; border-radius: 8px; padding: 18px 20px 14px; box-shadow: 0 1px 3px rgba(27,31,59,0.06); }}
[data-testid="metric-container"] label {{ color: {MUTED} !important; font-size: 0.65rem !important; letter-spacing: 0.12em; text-transform: uppercase; font-family: 'Inter', sans-serif; font-weight: 500; }}
[data-testid="metric-container"] [data-testid="stMetricValue"] {{ color: {INK} !important; font-size: 1.9rem !important; font-weight: 700 !important; font-family: 'Inter', sans-serif; }}
[data-testid="metric-container"] [data-testid="stMetricDelta"] {{ font-size: 0.72rem !important; }}
h1 {{ font-family: 'Inter', sans-serif !important; font-size: 2.4rem !important; font-weight: 700 !important; color: {INK} !important; letter-spacing: -0.03em; line-height: 1.15; }}
h2 {{ font-family: 'Inter', sans-serif !important; font-size: 0.62rem !important; font-weight: 600 !important; color: {MUTED} !important; letter-spacing: 0.2em; text-transform: uppercase; border-bottom: 1px solid {RULE}; padding-bottom: 10px; margin-top: 2.8rem !important; }}
h3 {{ font-family: 'Inter', sans-serif !important; font-size: 1.05rem !important; color: {INK} !important; font-weight: 600 !important; }}
[data-baseweb="tab-list"] {{ background: transparent; border-bottom: 1px solid {RULE}; gap: 0; }}
[data-baseweb="tab"] {{ font-size: 0.72rem; letter-spacing: 0.06em; text-transform: uppercase; color: {MUTED}; padding: 10px 20px; border-bottom: 2px solid transparent; font-weight: 500; }}
[data-baseweb="tab"][aria-selected="true"] {{ color: {PURPLE}; border-bottom: 2px solid {PURPLE}; font-weight: 600; }}
.stButton > button {{ background: {PURPLE}; color: #ffffff; border: none; border-radius: 6px; font-family: 'Inter', sans-serif; font-size: 0.82rem; font-weight: 600; padding: 0.55rem 1.6rem; transition: background 0.15s; }}
.stButton > button:hover {{ background: #7c60cc; color: #ffffff; }}
[data-baseweb="select"] > div {{ background: {PANEL}; border-color: {RULE}; border-radius: 6px; }}
[data-testid="stDataFrame"] {{ border: 1px solid {RULE}; border-radius: 8px; box-shadow: 0 1px 3px rgba(27,31,59,0.06); }}
</style>
""", unsafe_allow_html=True)

HIGH_RISK_PLATFORMS = ["Virtual Card", "QR Code Scanner", "Banking Chatbot"]
TIER_COLORS = {"Low": TEAL, "Medium": GOLD, "High": CORAL, "Critical": RED}
ARTIFACTS = Path("artifacts")

TIER_ACTIONS = {
    "Critical": "Block and escalate to senior compliance officer immediately.",
    "High":     "Queue for compliance review within 24 hours.",
    "Medium":   "Monitor donor activity; review if pattern repeats.",
    "Low":      "No action required. Log and continue.",
}

# ── Helper: render matplotlib fig as sharp PNG ───────────────────────────────
def show_fig(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    buf.seek(0)
    st.image(buf, width='stretch')
    plt.close(fig)

# ── Helper: insight callout ───────────────────────────────────────────────────
def insight(text, level="info"):
    colors = {
        "info":    (PURPLE, "#f0edf9"),
        "warning": (GOLD,  "#fdf6ed"),
        "alert":   (RED,   "#fdf0ee"),
        "good":    (TEAL,  "#eaf4f2"),
    }
    icons = {"info": "ℹ️", "warning": "⚠️", "alert": "🚨", "good": "✅"}
    border_c, bg = colors.get(level, colors["info"])
    icon = icons.get(level, "ℹ️")
    st.markdown(
        f'<div style="background:{bg};border-left:4px solid {border_c};'
        f'padding:12px 16px;border-radius:2px;margin:12px 0 20px;'
        f'font-size:0.85rem;color:{SLATE};line-height:1.6;">'
        f'<strong style="color:{border_c};">{icon} Insight</strong>'
        f'&nbsp;&nbsp;{text}</div>',
        unsafe_allow_html=True,
    )

def section_label(text):
    st.markdown(
        f'<p style="font-family:\'DM Mono\',monospace;font-size:0.68rem;'
        f'color:{MUTED};letter-spacing:0.14em;text-transform:uppercase;'
        f'margin-bottom:4px;">{text}</p>',
        unsafe_allow_html=True,
    )

# ── Artifact loading ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_artifacts():
    if not ARTIFACTS.exists():
        return None
    d = {
        "rf":             joblib.load(ARTIFACTS / "rf_model.joblib"),
        "lr":             joblib.load(ARTIFACTS / "lr_model.joblib"),
        "iso":            joblib.load(ARTIFACTS / "iso_model.joblib"),
        "scaler":         joblib.load(ARTIFACTS / "scaler.joblib"),
        "encoders":       joblib.load(ARTIFACTS / "encoders.joblib"),
        "importances":    joblib.load(ARTIFACTS / "importances.joblib"),
    }
    shap_path = ARTIFACTS / "shap_explainer.joblib"
    if shap_path.exists():
        d["shap_explainer"] = joblib.load(shap_path)
    return d

@st.cache_data(show_spinner=False)
def load_data():
    risk_df   = pd.read_parquet(ARTIFACTS / "risk_df.parquet")
    eda_df    = pd.read_parquet(ARTIFACTS / "eda_cache.parquet")
    metrics   = json.loads((ARTIFACTS / "metrics.json").read_text())
    meta      = json.loads((ARTIFACTS / "meta.json").read_text())
    shap_vals = None
    shap_path = ARTIFACTS / "shap_values.parquet"
    if shap_path.exists():
        shap_vals = pd.read_parquet(shap_path)
    return risk_df, eda_df, metrics, meta, shap_vals

# ── Confusion matrix ──────────────────────────────────────────────────────────
def confusion_matrix_fig(cm, title):
    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    fig.patch.set_facecolor(PANEL); ax.set_facecolor(PANEL)
    cell_labels = [["True Neg\nLegit → Legit", "False Pos\nLegit → Anom"],
                   ["False Neg\nAnom → Legit", "True Pos\nAnom → Anom"]]
    cell_colors = [[BLUE, CORAL], [CORAL, TEAL]]
    cell_alphas = [[0.12, 0.22], [0.22, 0.16]]
    for i in range(2):
        for j in range(2):
            ax.add_patch(plt.Rectangle([j, 1 - i], 1, 1,
                facecolor=cell_colors[i][j], alpha=cell_alphas[i][j],
                edgecolor=RULE, linewidth=1.8))
            ax.text(j + 0.5, 1 - i + 0.60, f"{cm[i][j]:,}", ha="center", va="center",
                    fontsize=24, fontweight="bold", color=cell_colors[i][j], fontfamily="serif")
            ax.text(j + 0.5, 1 - i + 0.28, cell_labels[i][j], ha="center", va="center",
                    fontsize=7.5, color=SLATE, fontfamily="sans-serif", linespacing=1.5)
    ax.set_xlim(0, 2); ax.set_ylim(0, 2)
    ax.set_xticks([0.5, 1.5]); ax.set_yticks([0.5, 1.5])
    ax.set_xticklabels(["Predicted\nLegitimate", "Predicted\nAnomalous"], fontsize=9, color=SLATE)
    ax.set_yticklabels(["Anomalous\n(actual)", "Legitimate\n(actual)"], fontsize=9, color=SLATE)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=14, color=INK)
    ax.grid(False); ax.spines[:].set_visible(False); ax.tick_params(length=0)
    plt.tight_layout()
    return fig


# ── Load ──────────────────────────────────────────────────────────────────────
arts = load_artifacts()
if arts is None and page == "📊  Dashboard":
    st.error("Artifacts not found. Run `python prepare.py` first.")
    st.stop()

if arts is not None:
    risk_df, eda_df, metrics, meta, shap_vals = load_data()
    rf          = arts["rf"]
    importances = arts["importances"]
    avg         = meta["anomaly_rate"]
    # Optimal thresholds computed by prepare.py via F-beta (beta=0.5)
    OPT_THRESHOLDS = meta.get("opt_thresholds", {
        "Logistic Regression": 0.55,
        "Random Forest":       0.55,
        "Isolation Forest":    0.55,
    })

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — PAGE NAVIGATION
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## DonorGuard")
    st.markdown("## Navigate")
    page = st.radio(
        "Page",
        ["🏠  Context", "📊  Dashboard", "📖  Glossary"],
        label_visibility="collapsed",
    )

    if page == "📊  Dashboard":
        st.markdown("## Model")
        st.markdown(
            '<div style="font-size:0.68rem;color:#6a6e87;line-height:1.6;margin-bottom:6px;">'
            'Logistic Regression is the default — highest precision (71%) '
            'and AUC (0.905) with calibrated probabilities.'
            '</div>',
            unsafe_allow_html=True,
        )
        model_choice = st.selectbox("Algorithm",
            ["Logistic Regression", "Random Forest", "Isolation Forest"],
            label_visibility="collapsed")

        # Default to the F-beta optimal threshold computed by prepare.py
        _opt_default = float(OPT_THRESHOLDS.get(model_choice, 0.55)) if arts is not None else 0.55
        threshold = st.slider("Decision threshold", 0.10, 0.90,
            value=_opt_default, step=0.01,
            help=(
                f"Optimal threshold for {model_choice}: **{_opt_default:.2f}** "
                f"(maximises F-0.5 score, weighting precision 2× over recall). "
                "Lower = more flags, higher recall. Higher = fewer flags, higher precision."
            ))

        # Show what the optimal threshold achieves
        if arts is not None and 'metrics' in dir():
            cur_opt = metrics.get(model_choice, {})
            opt_p = cur_opt.get("opt_precision", 0)
            opt_r = cur_opt.get("opt_recall", 0)
            opt_f = cur_opt.get("opt_fbeta", 0)
            st.markdown(
                f'<div style="font-size:0.68rem;color:#a8adc4;line-height:1.7;margin-top:4px;">'

                f'At optimal threshold:<br>'

                f'Precision <strong style="color:#fff;">{opt_p:.1%}</strong> &nbsp;·&nbsp; '

                f'Recall <strong style="color:#fff;">{opt_r:.1%}</strong> &nbsp;·&nbsp; '

                f'F&#8209;0.5 <strong style="color:#fff;">{opt_f:.3f}</strong>'

                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown("## Triage Queue")
        n_risk_rows = st.slider("Rows to show", 10, 100, 30, 5)
        show_tiers  = st.multiselect("Show tiers",
            ["Critical", "High", "Medium", "Low"], default=["Critical", "High"])
    else:
        model_choice = "Random Forest"
        threshold    = 0.45
        n_risk_rows  = 30
        show_tiers   = ["Critical", "High"]

    st.markdown("---")
    # Watchman status
    wm_live = _is_alive()
    if wm_live:
        st.markdown(
            '<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">'
            '<div style="width:8px;height:8px;border-radius:50%;background:#2ecc71;flex-shrink:0;"></div>'
            '<span style="font-size:0.72rem;color:#a8adc4;">Watchman live</span></div>',
            unsafe_allow_html=True)
    else:
        st.markdown(
            '<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">'
            '<div style="width:8px;height:8px;border-radius:50%;background:#e74c3c;flex-shrink:0;"></div>'
            '<span style="font-size:0.72rem;color:#6a6e87;">Watchman offline</span></div>',
            unsafe_allow_html=True)
        st.markdown(
            '<div style="font-size:0.62rem;color:#55596a;line-height:1.7;margin-bottom:8px;">'
            '<code style="background:#222;padding:1px 4px;border-radius:3px;font-size:0.6rem;">'
            'docker run -p 8084:8084 moov/watchman</code></div>',
            unsafe_allow_html=True)

    if page == "📊  Dashboard":
        st.markdown("## Sanctions")
        min_match_ui = st.slider("OFAC match threshold", 0.50, 0.99, 0.75, 0.01,
            help="Minimum score to treat a Watchman result as a sanctions hit. 0.75 is Watchman's recommended default.")
    else:
        min_match_ui = DEFAULT_MIN_MATCH

    st.markdown("---")
    st.markdown(
        '<div style="font-size:0.65rem;color:#44475a;line-height:1.9;">'
        'DonorGuard · Donation Compliance Platform<br>'
        'Run <code style="background:#222;padding:1px 5px;border-radius:2px;">'
        'python prepare.py</code> to retrain.</div>',
        unsafe_allow_html=True,
    )



# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — CONTEXT
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠  Context":
    st.markdown("# DonorGuard")
    st.markdown(
        f'<p style="font-family:\'DM Mono\',monospace;font-size:0.75rem;color:{MUTED};'
        f'letter-spacing:0.14em;margin-top:-10px;">DONATION COMPLIANCE PLATFORM · PRODUCT GUIDE</p>',
        unsafe_allow_html=True)

    st.markdown("## What is this tool?")
    st.markdown("""
**DonorGuard** is a real-time donation compliance platform built for nonprofits and charitable organisations. It helps your compliance team **identify donations that may be coming from bad actors** — including money launderers, sanctioned entities, and individuals attempting to exploit your organisation for financial crime.

DonorGuard analyses behavioural patterns across every donation and flags anything that deviates significantly from what a legitimate donor looks like. Flagged donations are scored and tiered so your team can focus review time where it matters most, without blocking genuine supporters.
    """)

    st.markdown("## Who is it for?")
    col1, col2, col3 = st.columns(3)
    for col, title, desc in [
        (col1, "👩‍💼 Compliance Officers",
         "Review flagged donations in the Risk Triage Queue. DonorGuard provides the rationale — you make the call on whether to accept, escalate, or reject."),
        (col2, "📋 Finance Team",
         "Monitor overall donation health, spot emerging risk patterns by campaign or channel, and maintain audit-ready records for regulators and trustees."),
        (col3, "🔬 Data & Tech Team",
         "Retrain models as new threat patterns emerge, tune thresholds per campaign type, and extend compliance rules without rebuilding the pipeline."),
    ]:
        col.markdown(
            f'<div style="background:{PANEL};border:1px solid {RULE};border-top:3px solid {PURPLE};'
            f'padding:16px;border-radius:2px;">'
            f'<strong style="font-size:0.95rem;">{title}</strong>'
            f'<p style="font-size:0.82rem;color:{SLATE};margin-top:8px;line-height:1.6;">{desc}</p>'
            f'</div>', unsafe_allow_html=True)

    st.markdown("## How protection works: two layers")
    st.markdown(
        f'<p style="font-size:0.88rem;color:{SLATE};line-height:1.7;margin-bottom:1rem;">'
        f'Protection against bad-actor donations operates at two distinct layers — '
        f'<strong>prevention</strong> (stopping fraud before it enters the system) and '
        f'<strong>detection</strong> (identifying suspicious patterns in donations that passed). '
        f'Both layers were built in close partnership with our banking providers.</p>',
        unsafe_allow_html=True,
    )

    lc, rc = st.columns(2)
    with lc:
        st.markdown(
            f'<div style="background:{PANEL};border:1px solid {RULE};border-radius:8px;padding:20px;height:100%;">'
            f'<div style="font-size:0.65rem;font-weight:600;letter-spacing:0.16em;color:{PURPLE};'
            f'text-transform:uppercase;margin-bottom:12px;">Layer 1 · Prevention</div>'
            f'<p style="font-size:0.82rem;color:{SLATE};line-height:1.7;margin-bottom:16px;">'
            f'Deployed at the payment layer in partnership with our banking providers. '
            f'Bad actors are stopped before a donation record is even created.</p>'
            f'<div style="display:flex;flex-direction:column;gap:12px;">'
            f'<div style="padding:10px 14px;background:#f0edf9;border-radius:6px;">'
            f'<strong style="font-size:0.85rem;color:{INK};">💳 CVV Verification</strong>'
            f'<p style="font-size:0.79rem;color:{SLATE};margin:4px 0 0;line-height:1.6;">'
            f'Real-time card security code validation on every donation. Cards that fail CVV '
            f'checks are declined immediately — this blocks the most common stolen-card fraud '
            f'before any funds move.</p></div>'
            f'<div style="padding:10px 14px;background:#f0edf9;border-radius:6px;">'
            f'<strong style="font-size:0.85rem;color:{INK};">🖥️ Device Fingerprinting</strong>'
            f'<p style="font-size:0.79rem;color:{SLATE};margin:4px 0 0;line-height:1.6;">'
            f'Cryptographic fingerprints of the donor browser and device environment, '
            f'provided by our payment gateway partner. Known bad devices — previously used '
            f'in fraudulent transactions — are flagged or blocked before payment is processed.</p></div>'
            f'<div style="padding:10px 14px;background:#f0edf9;border-radius:6px;">'
            f'<strong style="font-size:0.85rem;color:{INK};">🔔 Real-Time Webhooks</strong>'
            f'<p style="font-size:0.79rem;color:{SLATE};margin:4px 0 0;line-height:1.6;">'
            f'Event-driven notifications from our banking partner fire the moment a payment '
            f'is authorised, declined, or reversed — enabling near-instant anomaly scoring '
            f'rather than batch processing hours later.</p></div>'
            f'<div style="padding:10px 14px;background:#f0edf9;border-radius:6px;">'
            f'<strong style="font-size:0.85rem;color:{INK};">🌍 IP Geolocation & Velocity</strong>'
            f'<p style="font-size:0.79rem;color:{SLATE};margin:4px 0 0;line-height:1.6;">'
            f'Every donation IP is checked against MaxMind GeoIP2 for country match and '
            f'VPN/proxy detection, and against a 24-hour velocity counter for card-testing '
            f'patterns. A mismatch between stated and IP country, or 3+ donations from the '
            f'same IP in 24 hours, are automatic escalation triggers.</p></div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )
    with rc:
        st.markdown(
            f'<div style="background:{PANEL};border:1px solid {RULE};border-radius:8px;padding:20px;height:100%;">'
            f'<div style="font-size:0.65rem;font-weight:600;letter-spacing:0.16em;color:{RED};'
            f'text-transform:uppercase;margin-bottom:12px;">Layer 2 · Detection</div>'
            f'<p style="font-size:0.82rem;color:{SLATE};line-height:1.7;margin-bottom:16px;">'
            f'Applied to donations that passed the payment layer. Catches more sophisticated '
            f'threats: structuring, pass-through accounts, sanctions evasion, and coordinated bad actors.</p>'
            f'<div style="display:flex;flex-direction:column;gap:12px;">'
            f'<div style="padding:10px 14px;background:#fdf0ee;border-radius:6px;">'
            f'<strong style="font-size:0.85rem;color:{INK};">🛡️ Sanctions Screening (Watchman)</strong>'
            f'<p style="font-size:0.79rem;color:{SLATE};margin:4px 0 0;line-height:1.6;">'
            f'Every donor name is checked against OFAC SDN, EU Consolidated, UN, and UK OFSI '
            f'lists via Watchman. A match is a hard block — the ML score is not consulted.</p></div>'
            f'<div style="padding:10px 14px;background:#fdf0ee;border-radius:6px;">'
            f'<strong style="font-size:0.85rem;color:{INK};">🤖 ML Anomaly Detection</strong>'
            f'<p style="font-size:0.79rem;color:{SLATE};margin:4px 0 0;line-height:1.6;">'
            f'Random Forest, Logistic Regression, and Isolation Forest models score every '
            f'donation across 17 behavioural features — amount patterns, timing, platform '
            f'risk, and account signals.</p></div>'
            f'<div style="padding:10px 14px;background:#fdf0ee;border-radius:6px;">'
            f'<strong style="font-size:0.85rem;color:{INK};">📋 Risk Triage Queue</strong>'
            f'<p style="font-size:0.79rem;color:{SLATE};margin:4px 0 0;line-height:1.6;">'
            f'Donations are tiered Critical / High / Medium / Low with recommended actions '
            f'for compliance reviewers, ensuring human sign-off on the highest-risk cases.</p></div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("## What threats does it detect?")
    threats = [
        ("💸 Money Laundering via Nonprofit",
         "Large donations — especially round numbers — that are quickly followed by refund requests or withdrawals. Criminals use charities to 'clean' dirty money by donating it and requesting receipts."),
        ("🏦 Pass-Through Accounts",
         "Donors whose account balance is very low relative to their donation size. This suggests funds were deposited specifically to make this donation — a classic money-mule pattern."),
        ("🌙 Automated / Scripted Donations",
         "Large transfers made between midnight and 5am. Legitimate individual donors rarely donate at these hours; this pattern is more consistent with automated scripts or bots."),
        ("📉 Threshold Avoidance (Structuring)",
         "Donations made just below reporting thresholds (e.g. £49,900 instead of £50,000). This is a deliberate attempt to avoid triggering mandatory financial reporting."),
        ("📱 Untraceable Payment Channels",
         "Donations via Virtual Cards, QR codes, or chatbot payments — channels that are harder to trace back to a verified identity, especially when combined with large amounts."),
        ("👤 Recruited Mules",
         "Young donors (under 25) making very large donations disproportionate to typical income for that age group. May indicate individuals recruited to move money on someone else's behalf."),
        ("↩️ Chargeback Abuse",
         "A donor makes a large donation, receives a receipt or tax benefit, then files a chargeback with their bank to reclaim the funds. Your organisation loses the funds but the donor retains the documentation. Combined with high anomaly scores, chargebacks are a strong indicator of deliberate exploitation."),
        ("🃏 Card Testing",
         "Fraudsters use a nonprofit's donation form to validate stolen card numbers by making a rapid series of small donations. Indicators include multiple declined attempts from the same device or IP, or many round-number micro-donations in quick succession."),
    ]
    for title, desc in threats:
        st.markdown(
            f'<div style="display:flex;gap:16px;padding:14px 0;border-bottom:1px solid {RULE};">'
            f'<div style="font-size:1.4rem;flex-shrink:0;padding-top:2px;">{title.split()[0]}</div>'
            f'<div><strong style="font-size:0.88rem;color:{INK};">{" ".join(title.split()[1:])}</strong>'
            f'<p style="font-size:0.82rem;color:{SLATE};margin:4px 0 0;line-height:1.6;">{desc}</p></div>'
            f'</div>', unsafe_allow_html=True)

    st.markdown("## How to use this dashboard")
    steps = [
        ("1", "Start with the Overview KPIs", "Get a quick read on how many donations have been flagged today and whether the anomaly rate is within expected bounds."),
        ("2", "Check Model Performance", "Verify the model is performing well. A healthy ROC-AUC is above 0.80 and Average Precision above 0.40."),
        ("3", "Review the Triage Queue", "Work through Critical and High tier donations first. Each row shows the risk score and which signals triggered the flag."),
        ("4", "Use the Donation Scorer", "If a specific donation has been reported to you, enter its details directly to get an instant risk score and explanation."),
        ("5", "Explore the EDA tabs", "Use the analysis tabs to spot systemic patterns — e.g. a specific region or platform with unusually high anomaly rates."),
    ]
    for num, title, desc in steps:
        st.markdown(
            f'<div style="display:flex;gap:16px;align-items:flex-start;padding:12px 0;border-bottom:1px solid {RULE};">'
            f'<div style="background:{PURPLE};color:#ffffff;width:28px;height:28px;border-radius:50%;'
            f'display:flex;align-items:center;justify-content:center;font-size:0.78rem;'
            f'font-weight:700;flex-shrink:0;">{num}</div>'
            f'<div><strong style="font-size:0.88rem;color:{INK};">{title}</strong>'
            f'<p style="font-size:0.82rem;color:{SLATE};margin:4px 0 0;line-height:1.6;">{desc}</p></div>'
            f'</div>', unsafe_allow_html=True)

    st.markdown("## Risk Tier Guide")
    tier_guide = [
        ("Critical", RED,  "Score 0.80–1.00", "Block and escalate to senior compliance officer immediately. Do not process until reviewed."),
        ("High",     CORAL,"Score 0.60–0.79", "Queue for compliance review within 24 hours. Monitor donor for further activity."),
        ("Medium",   GOLD, "Score 0.30–0.59", "Log and monitor. Review if the donor appears again with additional flags."),
        ("Low",      TEAL, "Score 0.00–0.29", "No action required. Donation is within normal parameters."),
    ]
    for tier, clr, score_range, action in tier_guide:
        st.markdown(
            f'<div style="display:flex;gap:16px;align-items:center;padding:12px 16px;'
            f'margin:6px 0;background:{PANEL};border-left:4px solid {clr};border-radius:2px;">'
            f'<div style="min-width:80px;font-weight:700;font-size:0.88rem;color:{clr};">{tier}</div>'
            f'<div style="min-width:120px;font-family:monospace;font-size:0.75rem;color:{MUTED};">{score_range}</div>'
            f'<div style="font-size:0.82rem;color:{SLATE};line-height:1.5;">{action}</div>'
            f'</div>', unsafe_allow_html=True)

    st.markdown("## Important Caveats")
    st.warning("""
**DonorGuard is a decision-support tool, not an automated enforcement system.**

- Anomaly scores are based on behavioural patterns and compliance rules, not verified case outcomes. A High or Critical score means *investigate*, not *block automatically*.
- Watchman handles name-based sanctions screening, but is not a substitute for full KYC on large or complex gifts.
- Every flagged donor should have access to a clear appeals process — being incorrectly flagged is a serious matter for a genuine supporter.
- Retrain models quarterly or whenever your fraud patterns change significantly.
    """)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊  Dashboard":
    cur_m = metrics[model_choice]

    # Header
    hc1, hc2 = st.columns([3, 1])
    with hc1:
        st.markdown("# DonorGuard")
        st.markdown(
            f'<p style="font-family:\'DM Mono\',monospace;font-size:0.75rem;color:{MUTED};'
            f'letter-spacing:0.14em;margin-top:-10px;">DONATION COMPLIANCE PLATFORM · ANOMALY DETECTION & RISK TRIAGE</p>',
            unsafe_allow_html=True)
    with hc2:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            f'<div style="text-align:right;font-family:\'DM Mono\',monospace;font-size:0.7rem;color:{MUTED};">'
            f'Model: <strong style="color:{INK};">{model_choice}</strong><br>'
            f'Threshold: <strong style="color:{INK};">{threshold}</strong></div>',
            unsafe_allow_html=True)

    # ── KPI Bar ───────────────────────────────────────────────────────────────
    st.markdown("## Overview")
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Total Donations",            f"{meta['total']:,}")
    k2.metric("Flagged as Anomalous",        f"{meta['n_anomalous']:,}",  f"{meta['anomaly_rate']:.1%} of total")
    k3.metric("Detection Accuracy (AUC)",    f"{cur_m['auc']}")
    k4.metric("Finds Real Anomalies (AP)",   f"{cur_m['ap']}")
    k5.metric("Late-Night Anomaly Rate",     f"{meta['night_rate']:.1%}", "Midnight–5am donations")
    k6.metric("Via Untraceable Platforms",   f"{meta['high_risk_n']:,}",  "Virtual Card / QR / Chatbot")

    insight(
        f"<strong>{meta['n_anomalous']:,} donations ({meta['anomaly_rate']:.1%})</strong> have been flagged across the dataset. "
        f"The model's detection accuracy (AUC) is <strong>{cur_m['auc']}</strong> — "
        f"{'well above' if cur_m['auc'] > 0.80 else 'approaching'} the 0.80 threshold considered reliable for compliance use. "
        f"Late-night donations show a <strong>{meta['night_rate']:.1%}</strong> anomaly rate, "
        f"significantly above the {meta['anomaly_rate']:.1%} baseline — a key watch area.",
        level="info" if cur_m['auc'] > 0.75 else "warning"
    )

    # ── Model Performance ─────────────────────────────────────────────────────
    st.markdown("## Model Performance")
    section_label("How well is the selected model detecting anomalies?")

    pm1, pm2, pm3, pm4 = st.columns(4)
    pm1.metric("Accuracy When It Flags",  f"{cur_m['precision']:.1%}",
               help="Of all donations the model flags as anomalous, what % are actually anomalous? Computed at the optimal F-0.5 threshold.")
    pm2.metric("Anomalies It Catches",    f"{cur_m['recall']:.1%}",
               help="Of all truly anomalous donations, what % does the model find? Computed at the optimal F-0.5 threshold.")
    pm3.metric("Balanced Score (F1)",     f"{cur_m['f1']:.3f}",
               help="Harmonic mean of accuracy-when-it-flags and anomalies-it-catches.")
    pm4.metric("Overall Accuracy (AUC)",  f"{cur_m['auc']:.4f}",
               help="1.0 = perfect. 0.5 = no better than random. Above 0.80 is reliable for compliance use.")

    # Explain threshold optimisation
    opt_t = cur_m.get("opt_threshold", threshold)
    st.markdown(
        f'<p style="font-size:0.75rem;color:{MUTED};margin:4px 0 16px;">'
        f'Metrics shown at optimal threshold '
        f'<code style="background:#e8e9f0;padding:1px 5px;border-radius:3px;font-size:0.72rem;">{opt_t:.2f}</code>. '
        f'Use the sidebar slider to explore the precision–recall trade-off.</p>',
        unsafe_allow_html=True,
    )

    col_cm, col_pr = st.columns([1, 1.7])

    with col_cm:
        st.markdown(
            f'<p style="font-size:0.8rem;font-weight:600;color:{SLATE};margin:0 0 8px;">'
            f'Confusion Matrix</p>',
            unsafe_allow_html=True,
        )
        fig = confusion_matrix_fig(cur_m["cm"], model_choice)
        show_fig(fig)
        tn, fp, fn, tp = cur_m["cm"][0][0], cur_m["cm"][0][1], cur_m["cm"][1][0], cur_m["cm"][1][1]
        with st.expander("What do these numbers mean?"):
            st.markdown(f"""
- **{tn:,} True Negatives** — legitimate donations correctly passed through ✅
- **{tp:,} True Positives** — real anomalies correctly caught ✅
- **{fp:,} False Positives** — legitimate donations wrongly flagged ⚠️ *(these donors need an appeals path)*
- **{fn:,} False Negatives** — real anomalies the model missed ⚠️ *(adjust threshold to catch more)*
            """)

    with col_pr:
        st.markdown(
            f'<p style="font-size:0.8rem;font-weight:600;color:{SLATE};margin:0 0 8px;">'
            f'Precision-Recall Curve</p>',
            unsafe_allow_html=True,
        )
        fig, ax = plt.subplots(figsize=(7, 4.5))
        fig.patch.set_facecolor(PANEL); ax.set_facecolor(PANEL)
        for mname, scol, clr in [
            ("Random Forest",       "Anomaly_Score", BLUE),
            ("Logistic Regression", "LR_Score",      TEAL),
            ("Isolation Forest",    "ISO_Score",      CORAL),
        ]:
            p, r, _ = precision_recall_curve(risk_df["True_Label"], risk_df[scol])
            ap  = metrics[mname]["ap"]
            lw  = 2.5 if mname == model_choice else 1.2
            alf = 1.0  if mname == model_choice else 0.30
            ax.plot(r, p, color=clr, lw=lw, alpha=alf, label=f"{mname}  (AP {ap:.3f})")
        ax.axhline(meta["anomaly_rate"], color=RULE, lw=1.2, ls="--", label="Random baseline")
        ax.set_xlabel("How many anomalies are caught (Recall)")
        ax.set_ylabel("How accurate the flags are (Precision)")
        ax.set_ylim(0, 1.02); ax.set_xlim(0, 1)
        ax.legend(fontsize=8, frameon=False)
        ax.grid(True, axis="both", color=RULE, linewidth=0.5)
        ax.spines["bottom"].set_visible(True); ax.spines["left"].set_visible(True)
        plt.tight_layout()
        show_fig(fig)
        with st.expander("How to read this chart"):
            st.markdown("""
The curve shows the trade-off between **catching more anomalies** (moving right) and **being more precise** (moving up).
A curve that stays high across the full width means the model is both accurate and comprehensive.
The dashed line is what a random classifier would achieve — any real model should be well above it.
The **threshold slider** in the sidebar moves your operating point along this curve.
            """)

    # ── EDA Tabs ──────────────────────────────────────────────────────────────
    st.markdown("## Exploratory Analysis")
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "💰  Amounts", "📅  Temporal", "🗺️  Geography", "🏷️  Categories", "🔬  Features", "🧠  SHAP",
    ])

    with tab1:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        fig.patch.set_facecolor(PANEL)
        for label, clr, name, alpha in [(0, BLUE, "Legitimate", 0.45), (1, RED, "Anomalous", 0.65)]:
            sub = eda_df[eda_df["Is_Anomalous"] == label]["Donation_Amount"]
            axes[0].hist(sub, bins=70, alpha=alpha, color=clr, label=name,
                         density=True, edgecolor="white", linewidth=0.3)
        axes[0].set_title("Donation Amount Distribution")
        axes[0].set_xlabel("Amount"); axes[0].set_ylabel("Density")
        axes[0].legend(frameon=False, fontsize=8)
        axes[0].spines["bottom"].set_visible(True); axes[0].grid(True, axis="y")

        bp_data = [eda_df[eda_df["Is_Anomalous"] == i]["Donation_Amount"] for i in [0, 1]]
        bp = axes[1].boxplot(bp_data, patch_artist=True, widths=0.45, notch=True,
                              medianprops=dict(color=INK, linewidth=2),
                              whiskerprops=dict(color=SLATE, linewidth=1),
                              capprops=dict(color=SLATE),
                              flierprops=dict(marker=".", color=MUTED, alpha=0.3, markersize=3))
        for patch, clr in zip(bp["boxes"], [BLUE, RED]):
            patch.set_facecolor(clr); patch.set_alpha(0.22)
        axes[1].set_xticklabels(["Legitimate", "Anomalous"])
        axes[1].set_title("Donation Amount by Label")
        axes[1].spines["bottom"].set_visible(True); axes[1].grid(True, axis="y")
        plt.tight_layout()
        show_fig(fig)

        anom_median  = eda_df[eda_df["Is_Anomalous"]==1]["Donation_Amount"].median()
        legit_median = eda_df[eda_df["Is_Anomalous"]==0]["Donation_Amount"].median()
        insight(
            f"Anomalous donations have a median amount of <strong>£{anom_median:,.0f}</strong> vs "
            f"<strong>£{legit_median:,.0f}</strong> for legitimate ones — "
            f"<strong>{anom_median/legit_median:.1f}× higher</strong>. "
            f"Large donation size is one of the strongest individual signals in this dataset. "
            f"The spike at the top of the anomalous distribution reflects large round-number structuring.",
            level="warning"
        )

    with tab2:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        fig.patch.set_facecolor(PANEL)

        hourly = eda_df.groupby("Hour")["Is_Anomalous"].mean()
        peak_hour = int(hourly.idxmax())
        bar_c = [RED if v > avg else BLUE for v in hourly.values]
        axes[0].bar(hourly.index, hourly.values * 100, color=bar_c, alpha=0.72, width=0.8)
        axes[0].axhline(avg * 100, color=GOLD, lw=1.5, ls="--", label=f"Baseline {avg:.1%}")
        axes[0].set_title("Anomaly Rate by Hour of Day")
        axes[0].set_xlabel("Hour (0 = midnight, 12 = noon)"); axes[0].set_ylabel("% of donations flagged")
        axes[0].legend(frameon=False, fontsize=8); axes[0].spines["bottom"].set_visible(True)

        day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        daily = eda_df.groupby("DayOfWeek")["Is_Anomalous"].mean().reindex(
            [d for d in day_order if d in eda_df["DayOfWeek"].unique()])
        peak_day = daily.idxmax() if len(daily) > 0 else "N/A"
        bar_c2 = [RED if v > avg else BLUE for v in daily.values]
        axes[1].bar(daily.index, daily.values * 100, color=bar_c2, alpha=0.72, width=0.6)
        axes[1].axhline(avg * 100, color=GOLD, lw=1.5, ls="--", label=f"Baseline {avg:.1%}")
        axes[1].set_title("Anomaly Rate by Day of Week")
        axes[1].set_ylabel("% of donations flagged")
        axes[1].tick_params(axis="x", rotation=30)
        axes[1].legend(frameon=False, fontsize=8); axes[1].spines["bottom"].set_visible(True)
        plt.tight_layout()
        show_fig(fig)

        peak_rate = hourly.max()
        insight(
            f"The highest-risk hour is <strong>{peak_hour:02d}:00</strong> with a "
            f"<strong>{peak_rate:.1%}</strong> anomaly rate — "
            f"{peak_rate/avg:.1f}× above the {avg:.1%} baseline. "
            f"Donations made between midnight and 5am are a strong red flag and should be "
            f"automatically queued for review regardless of amount. "
            f"<strong>{peak_day}</strong> shows the highest day-of-week risk.",
            level="warning"
        )

    with tab3:
        top = (eda_df.groupby("Donor_Country")["Is_Anomalous"]
               .agg(["mean", "count"])
               .rename(columns={"mean": "Rate", "count": "Count"})
               .sort_values("Rate", ascending=True).tail(20))
        highest_region = top["Rate"].idxmax()
        highest_rate   = top["Rate"].max()

        fig, ax = plt.subplots(figsize=(10, 6.5))
        fig.patch.set_facecolor(PANEL); ax.set_facecolor(PANEL)
        clrs = [RED if r > avg else BLUE for r in top["Rate"]]
        ax.barh(top.index, top["Rate"] * 100, color=clrs, alpha=0.72, height=0.62)
        ax2 = ax.twiny()
        ax2.scatter(top["Count"], top.index, color=SLATE, s=20, alpha=0.45, zorder=5)
        ax2.set_xlabel("Number of donations (dots)", fontsize=8, color=MUTED)
        ax2.tick_params(labelsize=7, colors=MUTED)
        ax2.spines[:].set_visible(False)
        ax.axvline(avg * 100, color=GOLD, lw=1.5, ls="--", label=f"Baseline {avg:.1%}")
        ax.set_title("Anomaly Rate by Country — Top 20 Highest Risk")
        ax.set_xlabel("% of donations flagged as anomalous")
        ax.legend(frameon=False, fontsize=8)
        ax.grid(True, axis="x"); ax.spines["bottom"].set_visible(True)
        plt.tight_layout()
        show_fig(fig)

        insight(
            f"<strong>{highest_region}</strong> has the highest regional anomaly rate at "
            f"<strong>{highest_rate:.1%}</strong> — {highest_rate/avg:.1f}× above baseline. "
            f"Countries shown in red exceed the overall average. In a production system, "
            f"high-risk regions should be cross-referenced against applicable sanctions lists "
            f"(OFAC, UN, EU) before processing. The dot overlay shows donation volume — "
            f"small dots in high-risk regions may indicate targeted, low-volume probing.",
            level="warning"
        )

    with tab4:
        cat_cols = ["Donation_Type", "Campaign_ID", "Donation_Platform", "Payment_Processor", "Acquisition_Channel", "Donor_Segment"]
        fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
        axes = axes.flatten(); fig.patch.set_facecolor(PANEL)
        top_cats = {}
        for i, col in enumerate(cat_cols):
            rates = eda_df.groupby(col)["Is_Anomalous"].mean().sort_values()
            top_cats[col] = (rates.idxmax(), rates.max())
            clrs = [RED if v > avg else BLUE for v in rates.values]
            axes[i].barh(rates.index, rates.values * 100, color=clrs, alpha=0.72, height=0.6)
            axes[i].axvline(avg * 100, color=GOLD, lw=1.2, ls="--")
            axes[i].set_title(f"Anomaly Rate by {col.replace('_', ' ')}")
            axes[i].set_xlabel("% flagged", fontsize=8)
            axes[i].grid(True, axis="x"); axes[i].spines["bottom"].set_visible(True)
        axes[-1].set_visible(False)
        plt.suptitle("Category Breakdown", fontsize=12, fontweight="bold", color=INK, y=1.01)
        plt.tight_layout()
        show_fig(fig)

        riskiest_type, riskiest_rate = top_cats["Donation_Type"]
        riskiest_device, device_rate = top_cats["Donation_Platform"]
        insight(
            f"<strong>{riskiest_type}</strong> is the riskiest donation type "
            f"({riskiest_rate:.1%} anomaly rate). "
            f"For devices, <strong>{riskiest_device}</strong> shows the highest risk "
            f"({device_rate:.1%}). "
            f"Bars shown in red are above the overall baseline — these categories "
            f"warrant closer scrutiny, especially when combined with large amounts or off-hours timing.",
            level="info"
        )

    with tab5:
        fig, ax = plt.subplots(figsize=(9, 7.5))
        fig.patch.set_facecolor(PANEL); ax.set_facecolor(PANEL)
        imp  = importances
        clrs = [RED if i < 5 else (SLATE if i < 10 else MUTED) for i in range(len(imp))]
        ax.barh(imp.index[::-1], imp.values[::-1], color=clrs[::-1], alpha=0.8, height=0.65)
        for i, (feat, val) in enumerate(imp.head(5).items()):
            ax.text(val + 0.0005, len(imp) - 1 - i, f"{val:.4f}",
                    va="center", fontsize=7.5, color=RED, fontfamily="monospace")
        ax.set_title("Random Forest — Feature Importance (reference model)")
        ax.set_xlabel("Relative importance (higher = more influential)")
        ax.grid(True, axis="x"); ax.spines["bottom"].set_visible(True)
        ax.legend(handles=[
            mpatches.Patch(color=RED,   alpha=0.8, label="Top 5 features"),
            mpatches.Patch(color=SLATE, alpha=0.8, label="Features 6–10"),
            mpatches.Patch(color=MUTED, alpha=0.8, label="Remaining features"),
        ], frameon=False, fontsize=8)
        plt.tight_layout()
        show_fig(fig)

        top_feat = imp.index[0]
        top_val  = imp.iloc[0]
        insight(
            f"<strong>{top_feat.replace('_', ' ')}</strong> is the single most influential signal "
            f"(importance score: {top_val:.4f}), followed by "
            f"<strong>{imp.index[1].replace('_', ' ')}</strong> and "
            f"<strong>{imp.index[2].replace('_', ' ')}</strong>. "
            f"Features in red are the top 5 contributors — these are the data points your "
            f"compliance team should prioritise when manually reviewing flagged donations. "
            f"If a feature you'd expect to matter isn't near the top, it may be a signal "
            f"to add more targeted compliance rules.",
            level="info"
        )

    with tab6:
        if shap_vals is None:
            st.info("SHAP values not found. Re-run `prepare.py` to generate them.")
        else:
            feat_names = meta["features"]
            mean_abs_shap = shap_vals.abs().mean().sort_values(ascending=False)

            st.markdown("### Global Feature Impact")
            st.markdown(
                f'<p style="font-size:0.84rem;color:{SLATE};line-height:1.6;margin-bottom:1rem;">'                f'Mean absolute SHAP value across 500 test donations — how much each feature '                f'shifts the model anomaly score on average. Higher = more influential.</p>',
                unsafe_allow_html=True,
            )

            # SHAP summary bar chart
            fig, ax = plt.subplots(figsize=(9, 8))
            fig.patch.set_facecolor(PANEL); ax.set_facecolor(PANEL)
            top_n = 20
            top_feats = mean_abs_shap.head(top_n)
            colors = [PURPLE if i < 5 else (SLATE if i < 10 else MUTED)
                      for i in range(len(top_feats))]
            ax.barh(top_feats.index[::-1], top_feats.values[::-1],
                    color=colors[::-1], alpha=0.82, height=0.65)
            for i, (feat, val) in enumerate(top_feats.head(5).items()):
                ax.text(val + 0.0002, len(top_feats) - 1 - i,
                        f"{val:.4f}", va="center", fontsize=7.5,
                        color=PURPLE, fontfamily="monospace")
            ax.set_title("Top 20 Features by Mean |SHAP Value|")
            ax.set_xlabel("Mean |SHAP value| (impact on anomaly score)")
            ax.grid(True, axis="x"); ax.spines["bottom"].set_visible(True)
            ax.legend(handles=[
                mpatches.Patch(color=PURPLE, alpha=0.82, label="Top 5"),
                mpatches.Patch(color=SLATE,  alpha=0.82, label="6–10"),
                mpatches.Patch(color=MUTED,  alpha=0.82, label="Rest"),
            ], frameon=False, fontsize=8)
            plt.tight_layout()
            show_fig(fig)

            insight(
                f"<strong>{mean_abs_shap.index[0].replace('_', ' ')}</strong> is the most "
                f"influential feature (mean |SHAP| = {mean_abs_shap.iloc[0]:.4f}), followed by "
                f"<strong>{mean_abs_shap.index[1].replace('_', ' ')}</strong> and "
                f"<strong>{mean_abs_shap.index[2].replace('_', ' ')}</strong>. "
                f"Unlike feature importance from the tree structure, SHAP values measure "
                f"actual contribution to each prediction — positive SHAP = pushes score up "
                f"(more suspicious), negative SHAP = pushes score down (more legitimate).",
                level="info"
            )

            # SHAP direction chart — mean signed SHAP (positive vs negative)
            st.markdown("### Feature Direction — Does it flag or clear donations?")
            st.markdown(
                f'<p style="font-size:0.84rem;color:{SLATE};line-height:1.6;margin-bottom:1rem;">'                f'Mean signed SHAP value — red bars push the anomaly score up (suspicious signal), '                f'blue bars push it down (legitimacy signal).</p>',
                unsafe_allow_html=True,
            )
            mean_signed = shap_vals.mean().reindex(mean_abs_shap.head(top_n).index)
            fig, ax = plt.subplots(figsize=(9, 8))
            fig.patch.set_facecolor(PANEL); ax.set_facecolor(PANEL)
            clrs = [RED if v > 0 else BLUE for v in mean_signed.values[::-1]]
            ax.barh(mean_signed.index[::-1], mean_signed.values[::-1],
                    color=clrs, alpha=0.78, height=0.65)
            ax.axvline(0, color=RULE, lw=1.2)
            ax.set_title("Mean Signed SHAP — Red = Flags Donations, Blue = Clears Donations")
            ax.set_xlabel("Mean SHAP value")
            ax.grid(True, axis="x"); ax.spines["bottom"].set_visible(True)
            plt.tight_layout()
            show_fig(fig)

    # ── Risk Triage Queue ─────────────────────────────────────────────────────
    st.markdown("## Risk Triage Queue")
    section_label("Donations sorted by anomaly score — work top to bottom")

    triage_tab, ofac_tab = st.tabs(["📋  Triage Queue", "🛡️  OFAC Bulk Screen"])
    with ofac_tab:
        st.markdown("### Bulk Sanctions Screening")
        st.markdown(
            "Screen donor names against global sanctions lists via "
            "[Watchman](https://github.com/moov-io/watchman) "
            "(OFAC SDN, EU Consolidated, UN Consolidated, UK OFSI). "
            "Paste one name per line below."
        )

        wm_live_ofac = _is_alive()
        if not wm_live_ofac:
            st.warning(
                "Watchman is not running. Start it with: "
                "`docker run -p 8084:8084 moov/watchman` — "
                "screening will be skipped until Watchman is reachable."
            )

        col_inp, col_opts = st.columns([2, 1])
        with col_inp:
            default_names = "\n".join(DEMO_SDN_NAMES[:3]) + "\nAhmed Al-Rashid\nSarah Johnson"
            raw_names = st.text_area(
                "Donor names (one per line)",
                value=default_names,
                height=180,
                help="In production these come from your donor CRM or payment processor.",
            )
        with col_opts:
            st.markdown("<br>", unsafe_allow_html=True)
            ofac_type = st.selectbox(
                "Entity type", ["individual", "organization", "vessel", "aircraft"]
            )
            demo_mode = st.checkbox(
                "Demo mode",
                value=True,
                help="Injects synthetic OFAC hits for known SDN names. Disable with real donor data.",
            )
            run_screening = st.button("Run Sanctions Screen →")

        if run_screening:
            names = [n.strip() for n in raw_names.strip().splitlines() if n.strip()]
            if not names:
                st.warning("Enter at least one name.")
            else:
                results_out = []
                with st.spinner(f"Screening {len(names)} names against sanctions lists…"):
                    for name in names:
                        is_sdn = any(
                            sdn.lower() in name.lower() or name.lower() in sdn.lower()
                            for sdn in DEMO_SDN_NAMES
                        )
                        if demo_mode and is_sdn:
                            res = inject_demo_hit(name)
                        else:
                            res = screen_donor(
                                name, min_match=min_match_ui, entity_type=ofac_type
                            )

                        if res.is_blocked:
                            status = "🚨 BLOCKED"
                            action = "Hard block — escalate + file SAR"
                        elif not res.watchman_live:
                            status = "⚠️ OFFLINE"
                            action = "Manual review required"
                        else:
                            status = "✅ CLEAR"
                            action = "Clear to process"

                        top_score = f"{res.hits[0].match_score:.2f}" if res.hits else "—"
                        top_list  = res.hits[0].list_name if res.hits else "—"
                        top_match = res.hits[0].name if res.hits else "—"
                        results_out.append({
                            "Name Screened": name,
                            "Status":        status,
                            "Top Match":     top_match,
                            "List":          top_list,
                            "Score":         top_score,
                            "Action":        action,
                        })

                result_df = pd.DataFrame(results_out)
                blocked = (result_df["Status"] == "🚨 BLOCKED").sum()
                clear   = (result_df["Status"] == "✅ CLEAR").sum()

                rc1, rc2, rc3 = st.columns(3)
                rc1.metric("Names Screened", len(names))
                rc2.metric("Blocked",        blocked)
                rc3.metric("Clear",          clear)

                st.dataframe(
                    result_df, width='stretch', height=300,
                    column_config={
                        "Score": st.column_config.NumberColumn("Match Score", format="%.2f"),
                    },
                )

                if blocked > 0:
                    st.error(
                        f"{blocked} donor(s) matched sanctions lists. "
                        "Hard-block these donations and escalate to your compliance officer. "
                        "A Suspicious Activity Report (SAR) may be required."
                    )
                    st.markdown("#### Matched Entries")
                    for name in names:
                        is_sdn = any(sdn.lower() in name.lower() for sdn in DEMO_SDN_NAMES)
                        res = inject_demo_hit(name) if (demo_mode and is_sdn) else screen_donor(name, min_match=min_match_ui)
                        if not res.is_blocked:
                            continue
                        for hit in res.hits:
                            programs_str = ", ".join(hit.programs) if hit.programs else "—"
                            st.markdown(
                                f'<div style="padding:12px 16px;margin:8px 0;'
                                f'background:#fdf0ee;border-left:4px solid {RED};'
                                f'border-radius:6px;">'
                                f'<strong style="color:{RED};">{hit.name}</strong>'
                                f'<span style="float:right;font-family:monospace;'
                                f'font-size:0.8rem;color:{RED};">'
                                f'Match: {hit.match_score:.2f}</span><br>'
                                f'<span style="font-size:0.8rem;color:{SLATE};">'
                                f'List: <strong>{hit.list_name}</strong> &nbsp;·&nbsp; '
                                f'Type: {hit.entity_type} &nbsp;·&nbsp; '
                                f'Programs: {programs_str}</span>'
                                + (f'<br><span style="font-size:0.75rem;color:{MUTED};">'
                                   f'{hit.remarks}</span>' if hit.remarks else "")
                                + "</div>",
                                unsafe_allow_html=True,
                            )
                else:
                    st.success("All screened names returned clear.")

    with triage_tab:
        col_sum, col_tbl = st.columns([1, 3])

    with col_sum:
        st.markdown("### Tier Breakdown")
        tier_order = ["Critical", "High", "Medium", "Low"]
        tc = risk_df["Risk_Tier"].value_counts().reindex(tier_order).fillna(0).astype(int)
        for tier in tier_order:
            count = int(tc[tier]); pct = count / len(risk_df) * 100; clr = TIER_COLORS[tier]
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:10px;margin:8px 0;">'
                f'<div style="width:7px;height:7px;border-radius:50%;background:{clr};flex-shrink:0;"></div>'
                f'<span style="font-size:0.8rem;font-weight:600;color:{clr};width:68px;">{tier}</span>'
                f'<span style="font-size:0.8rem;color:{SLATE};">{count:,}</span>'
                f'<span style="font-size:0.72rem;color:{MUTED};margin-left:auto;">{pct:.1f}%</span>'
                f'</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("##### Recommended Actions")
        for tier, clr in [(t, TIER_COLORS[t]) for t in tier_order]:
            st.markdown(
                f'<div style="padding:8px 12px;margin:4px 0;background:{PANEL};'
                f'border-left:3px solid {clr};border-radius:2px;">'
                f'<strong style="font-size:0.78rem;color:{clr};">{tier}</strong>'
                f'<p style="font-size:0.75rem;color:{SLATE};margin:3px 0 0;line-height:1.5;">'
                f'{TIER_ACTIONS[tier]}</p></div>', unsafe_allow_html=True)

    with col_tbl:
        tiers_to_show = show_tiers if show_tiers else tier_order
        filtered = risk_df[risk_df["Risk_Tier"].isin(tiers_to_show)].copy()
        filtered = filtered.sort_values("Anomaly_Score", ascending=False).head(n_risk_rows)
        display_map = {
            "Anomaly_Score":         "Risk Score",
            "Risk_Tier":             "Tier",
            "True_Label":            "Confirmed",
            "RF_Pred":               "Flagged",
            "Log_Donation_Amount":   "Log Amount",
            "Amt_Bal_Ratio":         "Amt/Balance",
            "Is_Night_Donation":     "Night",
            "High_Risk_Platform":    "Hi-Risk Platform",
            "Is_Transfer":           "Transfer",
            "Low_Balance_Large_Gift":"Low Bal+Large",
        }
        show = filtered[[c for c in display_map if c in filtered.columns]].rename(columns=display_map)
        show["Risk Score"] = show["Risk Score"].round(4)
        if "Amt/Balance" in show.columns:
            show["Amt/Balance"] = show["Amt/Balance"].round(2)
        st.dataframe(
            show, width='stretch', height=430,
            column_config={
                "Risk Score": st.column_config.ProgressColumn(
                    "Risk Score", min_value=0, max_value=1, format="%.4f"),
            },
        )
        insight(
            f"Showing <strong>{len(filtered)}</strong> donations in the selected tiers. "
            f"The Risk Score column shows the model's confidence that a donation is anomalous "
            f"(0 = definitely legitimate, 1 = definitely suspicious). "
            f"'Night' and 'Hi-Risk Platform' columns are binary flags — 1 means the signal was present. "
            f"'Amt/Balance' above 3.5 suggests the donation far exceeds the donor's account balance.",
            level="info"
        )

    # ── Single Donation Scorer ────────────────────────────────────────────────
    st.markdown("## Score a Donation")
    section_label("Score any donation — enter details for an instant risk assessment")
    st.markdown(
        f'<p style="font-size:0.8rem;color:{MUTED};">'
        f'DonorGuard screens sanctions first via Watchman — a confirmed match is a hard block regardless of anomaly score. '
        f'Demo SDN names for testing: <code>{", ".join(DEMO_SDN_NAMES[:3])}</code></p>',
        unsafe_allow_html=True,
    )

    MODEL_FEATURES = meta["features"]
    CAT_ENCODE     = meta["cat_encode"]
    amt_p75        = meta.get("amt_p75", 74315)
    bal_p10        = meta.get("bal_p10", 14532)
    struct_thresh  = meta.get("structuring_thresholds", [50000, 75000, 90000, 99000])
    amt_p95        = meta["amount_mean"] + 2 * meta["amount_std"]

    with st.form("score_form"):

        # ── Section 1: Donor identity ─────────────────────────────────────────
        st.markdown(
            f'<div style="background:{PANEL};border:1px solid {RULE};border-radius:8px;'
            f'padding:16px 20px 8px;margin-bottom:12px;">'
            f'<p style="font-size:0.65rem;font-weight:600;letter-spacing:0.14em;'
            f'color:{PURPLE};text-transform:uppercase;margin-bottom:12px;">👤 Donor Identity</p>',
            unsafe_allow_html=True,
        )
        fn1, fn2 = st.columns([3, 1])
        s_name      = fn1.text_input("Donor Name", value="",
                          placeholder="Enter full name for sanctions screening…",
                          help="Screened against OFAC SDN, EU, UN, and UK sanctions lists via Watchman.")
        demo_scorer = fn2.checkbox("Demo mode", value=True,
                          help="Injects a synthetic OFAC hit if you type a known SDN name. Disable with real data.")
        d1, d2, d3 = st.columns(3)
        s_age      = d1.number_input("Donor Age", 18, 90, 35)
        s_account  = d2.selectbox("Donor Segment", ["Individual", "Corporate", "Major Donor"])
        s_freq     = d3.selectbox("Donation Frequency",
            ["First-time", "Occasional", "Regular", "Lapsed"])
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Section 2: Donation details ───────────────────────────────────────
        st.markdown(
            f'<div style="background:{PANEL};border:1px solid {RULE};border-radius:8px;'
            f'padding:16px 20px 8px;margin-bottom:12px;">'
            f'<p style="font-size:0.65rem;font-weight:600;letter-spacing:0.14em;'
            f'color:{PURPLE};text-transform:uppercase;margin-bottom:12px;">💷 Donation Details</p>',
            unsafe_allow_html=True,
        )
        da1, da2, da3, da4 = st.columns(4)
        s_amount    = da1.number_input("Amount (£)", 1.0, 10000.0, 50.0, 10.0)
        s_balance   = da2.number_input("Donor Lifetime Value (£)", 0.0, 20000.0, 200.0, 50.0)
        s_platform  = da3.selectbox("Donation Platform",
            ["Campaign Website", "JustGiving", "Mobile App", "Stripe Checkout",
             "Card (Phone)", "Bank Transfer", "Virtual Card", "QR Code",
             "Voice Assistant", "Chatbot", "Wearable"])
        s_don_type  = da4.selectbox("Donation Type",
            ["Bank Transfer", "Direct Debit", "Card", "Gift", "Refund"])
        da5, da6, da7, da8 = st.columns(4)
        s_hour      = da5.slider("Hour Donated (0 = midnight)", 0, 23, 14)
        s_weekend   = da6.checkbox("Weekend donation", value=False)
        s_anon      = da7.checkbox("Anonymous", value=False)
        s_matched   = da8.checkbox("Matched giving", value=False)
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Section 3: Banking partner & IP signals ───────────────────────────
        st.markdown(
            f'<div style="background:#f0edf9;border:1px solid #d8d0f0;border-radius:8px;'
            f'padding:16px 20px 8px;margin-bottom:16px;">'
            f'<p style="font-size:0.65rem;font-weight:600;letter-spacing:0.14em;'
            f'color:{PURPLE};text-transform:uppercase;margin-bottom:12px;">'
            f'🔌 Banking Partner & IP Signals</p>',
            unsafe_allow_html=True,
        )
        bp1, bp2, bp3 = st.columns(3)
        s_cvv_pass      = bp1.selectbox("CVV Check",
            ["Pass ✅", "Fail ❌", "Not applicable"],
            help="Result of real-time card CVV verification from the payment gateway.")
        s_device_known  = bp2.selectbox("Device Fingerprint",
            ["Known clean device", "First-seen device", "Known flagged device ⚠️"],
            help="Device fingerprint status from the payment gateway.")
        s_webhook_event = bp3.selectbox("Webhook Event",
            ["payment.authorised", "payment.reversed", "payment.declined", "chargeback.received"],
            help="Real-time event received from the banking partner webhook.")
        ip1, ip2, ip3 = st.columns(3)
        s_ip_match      = ip1.selectbox("IP Country Match",
            ["Match ✅", "Mismatch ⚠️"],
            help="Does the donor IP geolocation match their stated country?")
        s_vpn           = ip2.selectbox("VPN / Proxy",
            ["Not detected", "Detected ⚠️"],
            help="Whether the donation IP is a known VPN, proxy, or datacenter range.")
        s_ip_velocity   = ip3.number_input("IP Velocity (24h)",
            min_value=1, max_value=50, value=1, step=1,
            help="Donations from this IP in 24h. 3+ suspicious; 5+ indicates card testing.")
        st.markdown('</div>', unsafe_allow_html=True)

        submitted = st.form_submit_button("Calculate Risk Score →", use_container_width=True)
    if submitted:
        amount_mean = meta["amount_mean"]; amount_std = meta["amount_std"]

        # ── Step 1: OFAC sanctions screen ─────────────────────────────────
        section_label("Step 1 of 2 — Sanctions screening")
        is_sdn_demo = any(
            sdn.lower() in s_name.lower() or s_name.lower() in sdn.lower()
            for sdn in DEMO_SDN_NAMES
        ) if demo_scorer else False

        if demo_scorer and is_sdn_demo:
            ofac_result = inject_demo_hit(s_name)
        else:
            ofac_result = screen_donor(s_name, min_match=min_match_ui)

        if ofac_result.is_blocked:
            st.error(
                f"🚨 **HARD BLOCK — Sanctions Match** · "
                f"**{ofac_result.hits[0].name}** matched on "
                f"**{ofac_result.hits[0].list_name}** "
                f"(score {ofac_result.hits[0].match_score:.2f}). "
                f"Do not process this donation. Escalate to compliance and consider filing a SAR."
            )
            for hit in ofac_result.hits:
                programs_str = ", ".join(hit.programs) if hit.programs else "—"
                st.markdown(
                    f'<div style="padding:12px 16px;margin:6px 0;'
                    f'background:#fdf0ee;border-left:4px solid {RED};border-radius:6px;">'
                    f'<strong style="color:{RED};">{hit.name}</strong>'
                    f'<span style="float:right;font-family:monospace;font-size:0.8rem;color:{RED};">'
                    f'Match: {hit.match_score:.2f}</span><br>'
                    f'<span style="font-size:0.8rem;color:{SLATE};">'
                    f'List: <strong>{hit.list_name}</strong> &nbsp;·&nbsp; '
                    f'Programs: {programs_str}</span></div>',
                    unsafe_allow_html=True,
                )
            st.stop()
        elif not ofac_result.watchman_live:
            st.warning(
                "⚠️ Watchman is offline — sanctions screening was skipped. "
                "Start Watchman before processing this donation in production: "
                "`docker run -p 8084:8084 moov/watchman`"
            )
        else:
            st.success(f"✅ **Sanctions clear** — no matches above {min_match_ui:.0%} threshold on any loaded list.")

        section_label("Step 2 of 2 — ML anomaly score")
        is_near_threshold = any(s_amount >= t * 0.99 and s_amount < t for t in struct_thresh)
        sample = {f: 0 for f in MODEL_FEATURES}
        sample["Age"]                    = s_age
        sample["Log_Donation_Amount"]    = np.log1p(s_amount)
        sample["Log_Donor_Lifetime_Value"] = np.log1p(s_balance)
        sample["Amt_LTV_Ratio"]          = s_amount / (s_balance + 1)
        sample["Donation_Amount_Zscore"] = (s_amount - amount_mean) / (amount_std + 1e-9)
        sample["Hour"]                   = s_hour
        sample["Is_Night_Donation"]      = int(s_hour <= 4)
        sample["Is_Weekend"]             = int(s_weekend)
        sample["Is_Business_Hours"]      = int(9 <= s_hour <= 17)
        sample["Is_Round_Amount"]        = int(s_amount % 500 == 0)
        sample["Near_Threshold"]         = int(is_near_threshold)
        sample["High_Risk_Platform"]     = int(s_platform in HIGH_RISK_PLATFORMS)
        sample["Is_Bank_Transfer"]       = int(s_don_type == "Bank Transfer")
        sample["Is_Refund_Type"]         = int(s_don_type == "Refund")
        sample["Is_Corporate_Night"]     = int(s_account == "Corporate" and s_hour <= 4 and s_don_type == "Bank Transfer")
        sample["Is_First_Time_Large"]    = int(s_freq == "First-time" and s_amount >= amt_p90)
        sample["Is_Anon_Large"]          = int(s_anon and s_amount >= amt_p75)
        sample["Matched_Anon_Flag"]      = int(s_matched and s_anon)
        for col in CAT_ENCODE:
            sample[col] = 0
        sample["IP_Country_Match"]  = int(s_ip_match == "Match ✅")
        sample["Is_VPN_Or_Proxy"]   = int(s_vpn == "Detected ⚠️")
        sample["IP_Velocity_24h"]   = s_ip_velocity
        sample["IP_Mismatch_Large"] = int(s_ip_match == "Mismatch ⚠️" and s_amount >= amt_p75)
        sample["VPN_Anon_Flag"]     = int(s_vpn == "Detected ⚠️" and s_anon)
        sample["High_Velocity_Flag"]= int(s_ip_velocity >= 3)

        # Banking partner signal adjustments
        banking_boost = 0.0
        banking_rules = []

        if s_cvv_pass == "Fail ❌":
            banking_boost += 0.25
            banking_rules.append(("💳", "CVV check failed",
                "The card security code did not match. In production this would be hard-blocked "
                "at the payment layer before reaching this system."))
        if s_device_known == "Known flagged device ⚠️":
            banking_boost += 0.30
            banking_rules.append(("🖥️", "Known flagged device",
                "This device fingerprint has been associated with prior fraudulent transactions "
                "by the payment gateway. Treat any donation from this device with high suspicion."))
        elif s_device_known == "First-seen device":
            banking_boost += 0.05
            banking_rules.append(("🖥️", "First-seen device",
                "This device has no prior transaction history. Slightly elevated caution warranted "
                "for high-value donations from previously-unseen devices."))
        if s_webhook_event == "payment.reversed":
            banking_boost += 0.20
            banking_rules.append(("🔔", "Payment reversed",
                "The donation was reversed after authorisation — a strong layering signal. "
                "Funds may have been cycled through the charity account intentionally."))
        elif s_webhook_event == "chargeback.received":
            banking_boost += 0.35
            banking_rules.append(("🔔", "Chargeback received",
                "A chargeback was filed on this donation. Combined with high anomaly score, "
                "this may indicate card fraud or deliberate exploitation of the donation system."))
        elif s_webhook_event == "payment.declined":
            banking_boost += 0.10
            banking_rules.append(("🔔", "Payment declined by bank",
                "The payment was declined at the bank level. Repeated declined attempts "
                "are a card-testing signal."))

        fired_rules = []
        if sample["Is_Night_Donation"] and sample["Is_Transfer"] and s_amount >= amt_p75:
            fired_rules.append(("🌙", "Night-time large transfer", "Large transfers between midnight and 5am are unusual for individual donors and may indicate automated activity."))
        if sample["High_Risk_Platform"] and s_amount >= amt_p75:
            fired_rules.append(("📱", "High-risk platform + large amount", "Virtual cards, QR codes, and chatbot payments are harder to trace. Large amounts via these channels are elevated risk."))
        if sample["Amt_LTV_Ratio"] > 3.5:
            fired_rules.append(("⚖️", "Donation far exceeds account balance", f"Donation is {sample['Amt_LTV_Ratio']:.1f}× the donor's total lifetime giving — suggests a pass-through account."))
        if sample["Low_Balance_Large_Gift"]:
            fired_rules.append(("🏦", "Low balance + large gift", "Account balance is in the bottom 10% of donors, yet donation is in the top 25%. Classic mule account pattern."))
        if sample["Is_Round_Amount"] and s_amount >= amt_p75:
            fired_rules.append(("🔢", "Large round-number amount", "Large round numbers (£500, £1,000, £5,000) are a structuring signal — money launderers often use clean figures."))
        if sample["Near_Threshold"]:
            fired_rules.append(("📉", "Just below reporting threshold", f"Amount is within 1% of a reporting ceiling (£{', £'.join(str(t) for t in struct_thresh)})."))
        if sample["Young_Large"]:
            fired_rules.append(("👤", "Young donor + very large amount", "Donor under 25 making a very large donation — disproportionate to typical income for this age group."))
        if sample["Is_Withdrawal"] and s_amount >= amt_p95:
            fired_rules.append(("💸", "Very large withdrawal", "A very large withdrawal may indicate layering — funds cycling through the charity account."))

        X_single = pd.DataFrame([sample])[MODEL_FEATURES]
        ml_score  = float(rf.predict_proba(X_single)[0][1])
        score     = min(ml_score + banking_boost, 1.0)   # banking signals boost the score
        tier      = "Low" if score < 0.30 else "Medium" if score < 0.60 else "High" if score < 0.80 else "Critical"
        flagged   = score >= threshold
        clr       = TIER_COLORS[tier]

        rc1, rc2, rc3, rc4 = st.columns(4)
        rc1.metric("Risk Score",            f"{score:.4f}")
        rc2.metric("Risk Tier",             tier)
        rc3.metric("Flagged at Threshold",  "Yes ⚠️" if flagged else "No ✓")
        rc4.metric("Amount / Lifetime Value", f"{sample['Amt_LTV_Ratio']:.2f}×")
        st.progress(min(score, 1.0))

        if score >= 0.80:
            st.error(f"🚨 **Critical** — {TIER_ACTIONS['Critical']}")
        elif score >= 0.60:
            st.warning(f"⚠️ **High Risk** — {TIER_ACTIONS['High']}")
        elif score >= 0.30:
            st.info(f"ℹ️ **Medium Risk** — {TIER_ACTIONS['Medium']}")
        else:
            st.success(f"✅ **Low Risk** — {TIER_ACTIONS['Low']}")

        all_rules = banking_rules + fired_rules
        if all_rules:
            section_label("Triggered signals")
            if banking_boost > 0:
                st.markdown(
                    f'<div style="padding:8px 14px;margin:0 0 8px;background:#f0edf9;'
                    f'border-left:3px solid {PURPLE};border-radius:6px;'
                    f'font-size:0.8rem;color:{SLATE};">'
                    f'<strong style="color:{PURPLE};">Banking partner signals</strong> '
                    f'added <strong>+{banking_boost:.0%}</strong> to the ML base score of '
                    f'<strong>{ml_score:.4f}</strong> → final score '
                    f'<strong>{score:.4f}</strong></div>',
                    unsafe_allow_html=True,
                )
            for emoji, title, explanation in all_rules:
                st.markdown(
                    f'<div style="padding:10px 14px;margin:6px 0;background:{PANEL};'
                    f'border-left:3px solid {CORAL};border-radius:2px;">'
                    f'<strong style="font-size:0.85rem;color:{INK};">{emoji} {title}</strong>'
                    f'<p style="font-size:0.79rem;color:{SLATE};margin:4px 0 0;line-height:1.5;">{explanation}</p>'
                    f'</div>', unsafe_allow_html=True)
        elif score < 0.30:
            st.markdown(
                f'<div style="padding:10px 14px;margin:6px 0;background:#eef6f4;'
                f'border-left:3px solid {TEAL};border-radius:2px;font-size:0.82rem;color:{SLATE};">'
                f'No specific compliance rules triggered. Donation appears within normal parameters.</div>',
                unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — GLOSSARY
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📖  Glossary":
    st.markdown("# Glossary")
    st.markdown(
        f'<p style="font-family:\'DM Mono\',monospace;font-size:0.75rem;color:{MUTED};'
        f'letter-spacing:0.14em;margin-top:-10px;">PLAIN-ENGLISH DEFINITIONS FOR COMPLIANCE REVIEWERS</p>',
        unsafe_allow_html=True)

    glossary = [
        ("Model Terms", [
            ("Anomaly Score",
             "A number between 0 and 1 that represents how suspicious a donation looks. 0 means almost certainly legitimate; 1 means almost certainly anomalous. Think of it as a confidence level.",
             "A score of 0.85 means the model is 85% confident this donation is anomalous."),
            ("ROC-AUC (Detection Accuracy)",
             "A measure of how well the model separates legitimate donations from anomalous ones overall. Ranges from 0.5 (no better than a coin flip) to 1.0 (perfect). Above 0.80 is considered reliable for compliance use.",
             "An AUC of 0.88 means that if you randomly pick one anomalous and one legitimate donation, the model will rank the anomalous one higher 88% of the time."),
            ("Average Precision (AP)",
             "Measures how accurate the model's flags are, taking into account the rarity of anomalies. More meaningful than AUC when anomalies are rare. Higher is better; 1.0 is perfect.",
             "An AP of 0.49 means the model's flags are roughly 9× more likely to be real anomalies than random chance would produce."),
            ("Precision",
             "Of all the donations the model flags as anomalous, what percentage actually are anomalous. High precision means fewer false alarms for your compliance team to chase.",
             "Precision of 0.65 means 65 out of every 100 flagged donations are genuine anomalies."),
            ("Recall",
             "Of all the truly anomalous donations in the dataset, what percentage does the model catch. High recall means fewer real bad actors slipping through undetected.",
             "Recall of 0.70 means the model catches 70 out of every 100 real anomalies."),
            ("F1 Score",
             "A single number that balances precision and recall. Useful when you care about both catching anomalies and not generating too many false alarms. Ranges from 0 to 1.",
             "An F1 of 0.67 is a reasonable balance between precision and recall."),
            ("Decision Threshold",
             "The cut-off point above which a donation is flagged. Lowering it catches more anomalies but also flags more legitimate donations. Raising it reduces false alarms but misses more real threats.",
             "At threshold 0.45, any donation with an anomaly score above 0.45 gets flagged."),
            ("Confusion Matrix",
             "A 2×2 table showing how many donations the model got right and wrong: True Positives (real anomalies caught), True Negatives (legitimate donations passed), False Positives (legitimate donations wrongly flagged), False Negatives (real anomalies missed).",
             "A False Positive means a legitimate donor was flagged — these need an appeals process."),
            ("Random Forest",
             "A machine learning model that makes decisions by combining hundreds of simple decision trees. Good at capturing complex patterns and providing reliable probability estimates.",
             "The default and most accurate model in this dashboard."),
            ("Logistic Regression",
             "A simpler, more interpretable model that estimates probability based on a weighted sum of input features. Useful as a sanity check against the Random Forest.",
             "If Logistic Regression and Random Forest agree on a flag, it's a stronger signal."),
            ("Isolation Forest",
             "An unsupervised model that identifies anomalies without needing labelled examples. It finds donations that are structurally unusual — useful for catching new, previously-unseen fraud patterns.",
             "Particularly useful when you suspect a new type of bad actor that wasn't in the training data."),
            ("Precision-Recall Curve",
             "A chart showing the trade-off between precision and recall at every possible decision threshold. A curve that stays high and to the right means the model is both accurate and comprehensive.",
             "The area under this curve (Average Precision) summarises model quality in a single number."),
        ]),
        ("Compliance & Financial Crime Terms", [
            ("Money Laundering via Nonprofit",
             "A technique where criminals donate illegally-obtained funds to a legitimate nonprofit to 'clean' the money — obtaining a receipt that makes the funds appear legitimate. Often involves large round-number donations followed by refund requests.",
             "A donation of £2,000 followed by a refund request a few days later."),
            ("Structuring",
             "Deliberately splitting or sizing transactions to avoid triggering mandatory financial reporting thresholds. In the UK, this is a criminal offence.",
             "Making a donation of £499 instead of £500 to stay below a reporting threshold."),
            ("Layering",
             "Moving money through multiple steps or entities to obscure its origin. In a charity context, this can involve donating money, receiving a tax receipt, then requesting a refund to a different account.",
             "Donate → get receipt → request refund to different bank account."),
            ("Pass-Through Account",
             "A bank account used as a temporary conduit for funds — money is deposited specifically to make a transaction, then withdrawn. Identified by very low account balances relative to donation size.",
             "Account balance of £50 making a donation of £800."),
            ("Mule Account",
             "A bank account (often belonging to a recruited or coerced individual) used to move money on behalf of a criminal. The account holder may or may not know they are facilitating a crime.",
             "A young donor with no prior donation history making an unusually large one-off donation."),
            ("Threshold Avoidance",
             "Deliberately sizing transactions just below regulatory reporting thresholds to avoid mandatory disclosure. Common thresholds in financial crime monitoring include round numbers like £10,000, £50,000.",
             "A donation of £499 when your reporting threshold is £500."),
            ("OFAC Sanctions",
             "The US Office of Foreign Assets Control maintains a list of individuals and entities that US persons and organisations are prohibited from transacting with. Similar lists exist from the UN and EU.",
             "Before accepting a large donation, screen the donor name against OFAC's SDN list."),
            ("AML (Anti-Money Laundering)",
             "A set of laws, regulations, and procedures designed to prevent criminals from disguising illegally obtained funds as legitimate income. Charities are subject to AML obligations in most jurisdictions.",
             "Filing a Suspicious Activity Report (SAR) is an AML obligation when you suspect money laundering."),
            ("SAR (Suspicious Activity Report)",
             "A mandatory report filed with financial intelligence authorities when a regulated organisation suspects a transaction may involve money laundering or other financial crime. Failure to file when required is a criminal offence.",
             "A Critical-tier donation that cannot be explained should prompt a SAR review."),
            ("KYC (Know Your Customer)",
             "The process of verifying the identity of donors before accepting large or high-risk donations. Typically involves collecting ID, proof of address, and understanding the source of funds.",
             "For any Critical-tier donation, KYC should be completed before the donation is accepted."),
        ]),
        ("Banking Partner Signals", [
            ("CVV Verification",
             "Card Verification Value — the 3 or 4 digit security code on a payment card. Real-time CVV checking by the payment gateway confirms the physical card is present and matches the issuing bank's records. A CVV failure means the card details were likely stolen without the physical card.",
             "A donation attempt with a CVV failure should be hard-blocked at the payment layer before it ever reaches this system."),
            ("Device Fingerprinting",
             "A technique used by payment gateways to create a unique cryptographic identifier for a donor's browser and device environment — combining browser version, screen resolution, installed fonts, timezone, and other signals. Known fraudulent devices can be flagged or blocked on sight, even if the card details are new.",
             "A device that was used in 10 fraudulent transactions last month will be flagged when it attempts a new donation, regardless of which card it uses."),
            ("Webhooks",
             "Real-time HTTP notifications sent by the banking partner to our systems the moment a payment event occurs — authorisation, reversal, decline, or chargeback. This enables near-instant anomaly scoring rather than relying on batch processing that might run hours later.",
             "A payment.reversed webhook arriving seconds after authorisation is a strong layering signal — funds may have been cycled through the charity account intentionally."),
            ("Payment Reversal",
             "A transaction that is reversed after initial authorisation, returning funds to the sender. In a charity context this is suspicious when combined with a large donation — it may indicate funds cycling (donate → get receipt → reverse → repeat with different account).",
             "Reversed donations should always trigger a manual review, regardless of their ML anomaly score."),
            ("Chargeback",
             "A forced reversal initiated by the cardholder's bank, typically when a cardholder disputes a transaction as unauthorised. For nonprofits, chargebacks on large donations may indicate card fraud — someone used a stolen card to make a donation, or deliberate exploitation of the organisation's payment system.",
             "A chargeback on a donation that also has a high anomaly score is a strong indicator of card fraud and should prompt a SAR review."),
            ("Card Testing",
             "A fraud technique where criminals use a nonprofit's online donation form to test whether stolen card details are valid, by making a series of small donations. Indicators include: multiple rapid small donations from the same device or IP, many declined payment attempts, or unusual numbers of round-number micro-donations.",
             "Ten declined donation attempts of £1 from the same device fingerprint within a minute is a card testing pattern."),
        ]),
        ("Dashboard & Feature Terms", [
            ("Risk Tier",
             "A simple four-level classification of how suspicious a donation is, based on its anomaly score. Critical (0.80–1.00), High (0.60–0.79), Medium (0.30–0.59), Low (0–0.29).",
             "Use the tier as a priority queue: review all Critical donations before High, and so on."),
            ("High-Risk Platform",
             "Donation platforms that are harder to trace back to a verified identity: Virtual Cards, QR Code Scanners, and Banking Chatbots. These are flagged as elevated risk when combined with large amounts.",
             "An £800 donation via Virtual Card scores significantly higher than the same amount via bank transfer."),
            ("Amt/Balance Ratio",
             "The ratio of the donation amount to the donor's account balance. A ratio above 3.5 means the donation is more than 3.5× the account balance — a strong pass-through account signal.",
             "Ratio of 8.2 means a donor with £100 lifetime giving history is donating £820."),
            ("Night Donation",
             "A donation made between midnight and 4:59am. Individual legitimate donors rarely donate at these hours; this pattern is more consistent with automated scripts or bots.",
             "Any donation with Hour 0–4 is flagged as a night donation."),
            ("Low Balance + Large Gift",
             "A binary flag set when a donor's account balance is in the bottom 10% of all donors AND the donation amount is in the top 25%. The combination suggests a purpose-built mule account.",
             "Lifetime giving £40 (bottom 10%) + donation £800 (top 25%) = this flag fires."),
            ("Near Threshold",
             "A flag set when a donation falls within 1% below a reporting ceiling: £500, £750, £900, or £950. This narrow band is unlikely by chance and suggests deliberate structuring.",
             "A donation of £496 falls within 1% of £500 — this flag fires."),
            ("Feature Importance",
             "A measure of how much each input variable contributes to the model's predictions. Features with high importance are the ones the model relies on most to distinguish anomalous from legitimate donations.",
             "If 'Amt_Bal_Ratio' has the highest importance, the ratio of donation to balance is the strongest single signal."),
        ]),
    ]

    for section_title, terms in glossary:
        st.markdown(f"## {section_title}")
        for term, definition, example in terms:
            with st.expander(term):
                st.markdown(
                    f'<p style="font-size:0.85rem;color:{SLATE};line-height:1.7;">{definition}</p>',
                    unsafe_allow_html=True)
                st.markdown(
                    f'<div style="background:{PANEL};border-left:3px solid {BLUE};'
                    f'padding:8px 12px;margin-top:8px;border-radius:2px;">'
                    f'<span style="font-family:\'DM Mono\',monospace;font-size:0.68rem;'
                    f'color:{MUTED};letter-spacing:0.1em;">EXAMPLE</span><br>'
                    f'<span style="font-size:0.82rem;color:{SLATE};">{example}</span></div>',
                    unsafe_allow_html=True)
