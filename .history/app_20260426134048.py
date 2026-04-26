"""
app.py  —  Syria Campaign · Donation Anomaly Detection
Requires artifacts/ produced by:  uv run python prepare.py

Run:  uv run streamlit run app.py
"""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import precision_recall_curve

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Syria Campaign · Anomaly Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design tokens ─────────────────────────────────────────────────────────────
SAND = "#f5f0e8"
CREAM = "#fdfaf5"
INK = "#1a1a2e"
SLATE = "#4a4e69"
RULE = "#e0d9cc"
RED = "#c0392b"
CORAL = "#e8735a"
BLUE = "#2c5f8a"
TEAL = "#1a7a6e"
GOLD = "#c9913d"
MUTED = "#8a8f9e"

plt.rcParams.update(
    {
        "figure.facecolor": CREAM,
        "axes.facecolor": CREAM,
        "axes.edgecolor": RULE,
        "axes.labelcolor": SLATE,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 9,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "grid.color": RULE,
        "grid.linewidth": 0.6,
        "text.color": INK,
        "font.family": "serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.grid": True,
        "axes.grid.axis": "x",
        "figure.dpi": 200,
        "savefig.dpi": 200,
    }
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=DM+Sans:wght@300;400;500&family=DM+Mono:wght@400&display=swap');

html, body, [class*="css"] {{
    font-family: 'DM Sans', sans-serif;
    background-color: {SAND};
    color: {INK};
}}
.stApp {{ background-color: {SAND}; }}

[data-testid="stSidebar"] {{
    background-color: {INK};
}}
[data-testid="stSidebar"] * {{ color: #c8c4bc !important; }}
[data-testid="stSidebar"] h2 {{
    color: #f5f0e8 !important;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.72rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin-top: 1.4rem !important;
}}
[data-testid="stSidebar"] label {{
    color: #6a6a7a !important;
    font-size: 0.68rem !important;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}}

[data-testid="metric-container"] {{
    background: {CREAM};
    border: 1px solid {RULE};
    border-top: 3px solid {INK};
    border-radius: 2px;
    padding: 16px 18px 12px;
}}
[data-testid="metric-container"] label {{
    color: {MUTED} !important;
    font-size: 0.65rem !important;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    font-family: 'DM Mono', monospace;
}}
[data-testid="metric-container"] [data-testid="stMetricValue"] {{
    color: {INK} !important;
    font-size: 1.8rem !important;
    font-weight: 700 !important;
    font-family: 'Playfair Display', serif;
}}
[data-testid="metric-container"] [data-testid="stMetricDelta"] {{
    font-size: 0.72rem !important;
}}

h1 {{
    font-family: 'Playfair Display', serif !important;
    font-size: 3rem !important;
    font-weight: 700 !important;
    color: {INK} !important;
    letter-spacing: -0.02em;
    line-height: 1.1;
}}
h2 {{
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.65rem !important;
    font-weight: 500 !important;
    color: {MUTED} !important;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    border-bottom: 1px solid {RULE};
    padding-bottom: 10px;
    margin-top: 2.8rem !important;
}}
h3 {{
    font-family: 'Playfair Display', serif !important;
    font-size: 1.1rem !important;
    color: {INK} !important;
    font-weight: 600 !important;
}}

[data-baseweb="tab-list"] {{
    background: transparent;
    border-bottom: 1px solid {RULE};
    gap: 0;
}}
[data-baseweb="tab"] {{
    font-size: 0.72rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: {MUTED};
    padding: 8px 18px;
    border-bottom: 2px solid transparent;
}}
[data-baseweb="tab"][aria-selected="true"] {{
    color: {INK};
    border-bottom: 2px solid {INK};
    font-weight: 500;
}}

.stButton > button {{
    background: {INK};
    color: {SAND};
    border: none;
    border-radius: 2px;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.78rem;
    font-weight: 500;
    letter-spacing: 0.08em;
    padding: 0.55rem 1.6rem;
}}
.stButton > button:hover {{ background: {SLATE}; color: {SAND}; }}

[data-baseweb="select"] > div {{
    background: {CREAM};
    border-color: {RULE};
    border-radius: 2px;
}}
[data-testid="stDataFrame"] {{
    border: 1px solid {RULE};
    border-radius: 2px;
}}
</style>
""",
    unsafe_allow_html=True,
)

HIGH_RISK_PLATFORMS = ["Virtual Card", "QR Code Scanner", "Banking Chatbot"]
TIER_COLORS = {"Low": TEAL, "Medium": GOLD, "High": CORAL, "Critical": RED}
ARTIFACTS = Path("artifacts")


# ── Artifact loading ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_artifacts():
    if not ARTIFACTS.exists():
        return None
    return {
        "rf": joblib.load(ARTIFACTS / "rf_model.joblib"),
        "lr": joblib.load(ARTIFACTS / "lr_model.joblib"),
        "iso": joblib.load(ARTIFACTS / "iso_model.joblib"),
        "scaler": joblib.load(ARTIFACTS / "scaler.joblib"),
        "encoders": joblib.load(ARTIFACTS / "encoders.joblib"),
        "importances": joblib.load(ARTIFACTS / "importances.joblib"),
    }


@st.cache_data(show_spinner=False)
def load_data():
    risk_df = pd.read_parquet(ARTIFACTS / "risk_df.parquet")
    eda_df = pd.read_parquet(ARTIFACTS / "eda_cache.parquet")
    metrics = json.loads((ARTIFACTS / "metrics.json").read_text())
    meta = json.loads((ARTIFACTS / "meta.json").read_text())
    return risk_df, eda_df, metrics, meta


# ── Confusion matrix (clean, no heatmap ugliness) ─────────────────────────────
def confusion_matrix_fig(cm, title):
    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    fig.patch.set_facecolor(CREAM)
    ax.set_facecolor(CREAM)

    cell_labels = [
        ["True Neg\nLegit → Legit", "False Pos\nLegit → Anom"],
        ["False Neg\nAnom → Legit", "True Pos\nAnom → Anom"],
    ]
    cell_colors = [[BLUE, CORAL], [CORAL, TEAL]]
    cell_alphas = [[0.12, 0.22], [0.22, 0.16]]

    for i in range(2):
        for j in range(2):
            rect = plt.Rectangle(
                [j, 1 - i],
                1,
                1,
                facecolor=cell_colors[i][j],
                alpha=cell_alphas[i][j],
                edgecolor=RULE,
                linewidth=1.8,
            )
            ax.add_patch(rect)
            ax.text(
                j + 0.5,
                1 - i + 0.60,
                f"{cm[i][j]:,}",
                ha="center",
                va="center",
                fontsize=24,
                fontweight="bold",
                color=cell_colors[i][j],
                fontfamily="serif",
            )
            ax.text(
                j + 0.5,
                1 - i + 0.28,
                cell_labels[i][j],
                ha="center",
                va="center",
                fontsize=7.5,
                color=SLATE,
                fontfamily="sans-serif",
                linespacing=1.5,
            )

    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_xticks([0.5, 1.5])
    ax.set_yticks([0.5, 1.5])
    ax.set_xticklabels(
        ["Predicted\nLegitimate", "Predicted\nAnomalous"], fontsize=9, color=SLATE
    )
    ax.set_yticklabels(
        ["Anomalous\n(actual)", "Legitimate\n(actual)"], fontsize=9, color=SLATE
    )
    ax.set_title(title, fontsize=10, fontweight="bold", pad=14, color=INK)
    ax.grid(False)
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0)
    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## Syria Campaign")

    if not ARTIFACTS.exists():
        st.error("Run `python prepare.py` first.")
        st.stop()

    st.markdown("## Model")
    model_choice = st.selectbox(
        "Algorithm",
        ["Random Forest", "Logistic Regression", "Isolation Forest"],
        label_visibility="collapsed",
    )
    threshold = st.slider("Decision threshold", 0.10, 0.90, 0.45, 0.01)

    st.markdown("## Display")
    n_risk_rows = st.slider("Triage table rows", 10, 100, 30, 5)
    show_tiers = st.multiselect(
        "Tiers to show",
        ["Critical", "High", "Medium", "Low"],
        default=["Critical", "High"],
    )

    st.markdown("---")
    st.markdown(
        '<div style="font-size:0.65rem;color:#44475a;line-height:1.9;">'
        "Syria Campaign · Anomaly Detection<br>"
        "Adapted from LOL Bank Kaggle dataset.<br>"
        'Run <code style="background:#222;padding:1px 5px;border-radius:2px;">'
        "python prepare.py</code> to retrain."
        "</div>",
        unsafe_allow_html=True,
    )

# ═══════════════════════════════════════════════════════════════════════════════
# LOAD ARTIFACTS
# ═══════════════════════════════════════════════════════════════════════════════
arts = load_artifacts()
if arts is None:
    st.error("Artifacts not found. Run `python prepare.py` first.")
    st.stop()

risk_df, eda_df, metrics, meta = load_data()
rf = arts["rf"]
importances = arts["importances"]
cur_m = metrics[model_choice]

# ═══════════════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════════════
hc1, hc2 = st.columns([3, 1])
with hc1:
    st.markdown("# Syria Campaign")
    st.markdown(
        f"<p style=\"font-family:'DM Mono',monospace;font-size:0.75rem;"
        f'color:{MUTED};letter-spacing:0.14em;margin-top:-10px;">'
        f"DONATION ANOMALY DETECTION · COMPLIANCE DASHBOARD</p>",
        unsafe_allow_html=True,
    )
with hc2:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        f"<div style=\"text-align:right;font-family:'DM Mono',monospace;"
        f'font-size:0.7rem;color:{MUTED};">'
        f'Model: <strong style="color:{INK};">{model_choice}</strong><br>'
        f'Threshold: <strong style="color:{INK};">{threshold}</strong></div>',
        unsafe_allow_html=True,
    )

# ═══════════════════════════════════════════════════════════════════════════════
# KPI BAR
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("## Overview")
k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Total Donations", f"{meta['total']:,}")
k2.metric(
    "Anomalous", f"{meta['n_anomalous']:,}", f"{meta['anomaly_rate']:.1%} of total"
)
k3.metric("ROC-AUC", f"{cur_m['auc']}")
k4.metric("Avg Precision", f"{cur_m['ap']}")
k5.metric("Night Anomaly Rate", f"{meta['night_rate']:.1%}", "00:00–05:59")
k6.metric("High-Risk Platform", f"{meta['high_risk_n']:,}", "Virtual/QR/Chatbot")

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("## Model Performance")

pm1, pm2, pm3, pm4 = st.columns(4)
pm1.metric("Precision", f"{cur_m['precision']:.3f}")
pm2.metric("Recall", f"{cur_m['recall']:.3f}")
pm3.metric("F1", f"{cur_m['f1']:.3f}")
pm4.metric("ROC-AUC", f"{cur_m['auc']:.4f}")

col_cm, col_pr = st.columns([1, 1.7])

with col_cm:
    fig = confusion_matrix_fig(cur_m["cm"], model_choice)
    st.pyplot(fig, use_container_width=True)
    plt.close()

with col_pr:
    st.markdown(
        f"<p style=\"font-family:'DM Mono',monospace;font-size:0.7rem;"
        f'color:{MUTED};letter-spacing:0.1em;">PRECISION-RECALL CURVES</p>',
        unsafe_allow_html=True,
    )
    fig, ax = plt.subplots(figsize=(7, 4.5))
    fig.patch.set_facecolor(CREAM)
    ax.set_facecolor(CREAM)

    model_styles = [
        ("Random Forest", "Anomaly_Score", BLUE),
        ("Logistic Regression", "LR_Score", TEAL),
        ("Isolation Forest", "ISO_Score", CORAL),
    ]
    for mname, scol, clr in model_styles:
        p, r, _ = precision_recall_curve(risk_df["True_Label"], risk_df[scol])
        ap = metrics[mname]["ap"]
        lw = 2.5 if mname == model_choice else 1.2
        alf = 1.0 if mname == model_choice else 0.35
        ax.plot(r, p, color=clr, lw=lw, alpha=alf, label=f"{mname}  (AP {ap:.3f})")

    ax.axhline(meta["anomaly_rate"], color=RULE, lw=1.2, ls="--", label="Baseline")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_ylim(0, 1.02)
    ax.set_xlim(0, 1)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(True, axis="both", color=RULE, linewidth=0.5)
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_visible(True)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# EDA TABS
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("## Exploratory Analysis")
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "💰  Amounts",
        "📅  Temporal",
        "🗺️  Geography",
        "🏷️  Categories",
        "🔬  Features",
    ]
)

with tab1:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    fig.patch.set_facecolor(CREAM)
    for label, clr, name, alpha in [
        (0, BLUE, "Legitimate", 0.45),
        (1, RED, "Anomalous", 0.65),
    ]:
        sub = eda_df[eda_df["Is_Anomalous"] == label]["Donation_Amount"]
        axes[0].hist(
            sub,
            bins=70,
            alpha=alpha,
            color=clr,
            label=name,
            density=True,
            edgecolor="white",
            linewidth=0.3,
        )
    axes[0].set_title("Donation Amount Distribution")
    axes[0].set_xlabel("Amount")
    axes[0].set_ylabel("Density")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].spines["bottom"].set_visible(True)
    axes[0].grid(True, axis="y")

    bp_data = [eda_df[eda_df["Is_Anomalous"] == i]["Donation_Amount"] for i in [0, 1]]
    bp = axes[1].boxplot(
        bp_data,
        patch_artist=True,
        widths=0.45,
        notch=True,
        medianprops=dict(color=INK, linewidth=2),
        whiskerprops=dict(color=SLATE, linewidth=1),
        capprops=dict(color=SLATE),
        flierprops=dict(marker=".", color=MUTED, alpha=0.3, markersize=3),
    )
    for patch, clr in zip(bp["boxes"], [BLUE, RED]):
        patch.set_facecolor(clr)
        patch.set_alpha(0.22)
    axes[1].set_xticklabels(["Legitimate", "Anomalous"])
    axes[1].set_title("Donation Amount by Label")
    axes[1].spines["bottom"].set_visible(True)
    axes[1].grid(True, axis="y")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

with tab2:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    fig.patch.set_facecolor(CREAM)
    avg = meta["anomaly_rate"]

    hourly = eda_df.groupby("Hour")["Is_Anomalous"].mean()
    bar_c = [RED if v > avg else BLUE for v in hourly.values]
    axes[0].bar(hourly.index, hourly.values * 100, color=bar_c, alpha=0.72, width=0.8)
    axes[0].axhline(avg * 100, color=GOLD, lw=1.5, ls="--", label="Overall avg")
    axes[0].set_title("Anomaly Rate by Hour of Day")
    axes[0].set_xlabel("Hour (0–23)")
    axes[0].set_ylabel("Anomaly Rate (%)")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].spines["bottom"].set_visible(True)

    day_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    daily = (
        eda_df.groupby("DayOfWeek")["Is_Anomalous"]
        .mean()
        .reindex([d for d in day_order if d in eda_df["DayOfWeek"].unique()])
    )
    bar_c2 = [RED if v > avg else BLUE for v in daily.values]
    axes[1].bar(daily.index, daily.values * 100, color=bar_c2, alpha=0.72, width=0.6)
    axes[1].axhline(avg * 100, color=GOLD, lw=1.5, ls="--", label="Overall avg")
    axes[1].set_title("Anomaly Rate by Day of Week")
    axes[1].set_ylabel("Anomaly Rate (%)")
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].spines["bottom"].set_visible(True)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

with tab3:
    top = (
        eda_df.groupby("Region")["Is_Anomalous"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "Rate", "count": "Count"})
        .sort_values("Rate", ascending=True)
        .tail(20)
    )

    fig, ax = plt.subplots(figsize=(10, 6.5))
    fig.patch.set_facecolor(CREAM)
    ax.set_facecolor(CREAM)
    clrs = [RED if r > avg else BLUE for r in top["Rate"]]
    ax.barh(top.index, top["Rate"] * 100, color=clrs, alpha=0.72, height=0.62)

    ax2 = ax.twiny()
    ax2.scatter(top["Count"], top.index, color=SLATE, s=20, alpha=0.45, zorder=5)
    ax2.set_xlabel("Donation Count", fontsize=8, color=MUTED)
    ax2.tick_params(labelsize=7, colors=MUTED)
    ax2.spines[:].set_visible(False)

    ax.axvline(avg * 100, color=GOLD, lw=1.5, ls="--", label=f"Avg {avg:.1%}")
    ax.set_title("Regional Anomaly Rate (Top 20)")
    ax.set_xlabel("Anomaly Rate (%)")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, axis="x")
    ax.spines["bottom"].set_visible(True)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

with tab4:
    cat_cols = [
        "Donation_Type",
        "Campaign_Category",
        "Donation_Channel",
        "Device_Type",
        "Currency",
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    axes = axes.flatten()
    fig.patch.set_facecolor(CREAM)

    for i, col in enumerate(cat_cols):
        rates = eda_df.groupby(col)["Is_Anomalous"].mean().sort_values()
        clrs = [RED if v > avg else BLUE for v in rates.values]
        axes[i].barh(
            rates.index, rates.values * 100, color=clrs, alpha=0.72, height=0.6
        )
        axes[i].axvline(avg * 100, color=GOLD, lw=1.2, ls="--")
        axes[i].set_title(f"Anomaly Rate by {col}")
        axes[i].set_xlabel("Anomaly Rate (%)", fontsize=8)
        axes[i].grid(True, axis="x")
        axes[i].spines["bottom"].set_visible(True)

    axes[-1].set_visible(False)
    plt.suptitle(
        "Category Breakdown", fontsize=12, fontweight="bold", color=INK, y=1.01
    )
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

with tab5:
    fig, ax = plt.subplots(figsize=(9, 7.5))
    fig.patch.set_facecolor(CREAM)
    ax.set_facecolor(CREAM)
    imp = importances
    clrs = [RED if i < 5 else (SLATE if i < 10 else MUTED) for i in range(len(imp))]
    ax.barh(imp.index[::-1], imp.values[::-1], color=clrs[::-1], alpha=0.8, height=0.65)

    for i, (feat, val) in enumerate(imp.head(5).items()):
        ax.text(
            val + 0.0005,
            len(imp) - 1 - i,
            f"{val:.4f}",
            va="center",
            fontsize=7.5,
            color=RED,
            fontfamily="monospace",
        )

    ax.set_title("Random Forest — Feature Importance")
    ax.set_xlabel("Importance")
    ax.grid(True, axis="x")
    ax.spines["bottom"].set_visible(True)
    patches = [
        mpatches.Patch(color=RED, alpha=0.8, label="Top 5"),
        mpatches.Patch(color=SLATE, alpha=0.8, label="6–10"),
        mpatches.Patch(color=MUTED, alpha=0.8, label="Rest"),
    ]
    ax.legend(handles=patches, frameon=False, fontsize=8)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# RISK TRIAGE QUEUE
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("## Risk Triage Queue")

col_sum, col_tbl = st.columns([1, 3])

with col_sum:
    st.markdown("### Tier Breakdown")
    tier_order = ["Critical", "High", "Medium", "Low"]
    tc = risk_df["Risk_Tier"].value_counts().reindex(tier_order).fillna(0).astype(int)
    for tier in tier_order:
        count = int(tc[tier])
        pct = count / len(risk_df) * 100
        clr = TIER_COLORS[tier]
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:10px;margin:10px 0;">'
            f'<div style="width:7px;height:7px;border-radius:50%;background:{clr};flex-shrink:0;"></div>'
            f'<span style="font-size:0.8rem;font-weight:600;color:{clr};width:68px;">{tier}</span>'
            f'<span style="font-size:0.8rem;color:{SLATE};">{count:,}</span>'
            f'<span style="font-size:0.72rem;color:{MUTED};margin-left:auto;">{pct:.1f}%</span>'
            f"</div>",
            unsafe_allow_html=True,
        )

    fig, ax = plt.subplots(figsize=(3.2, 2.0))
    fig.patch.set_facecolor(CREAM)
    ax.set_facecolor(CREAM)
    clrs_bar = [TIER_COLORS[t] for t in tier_order]
    ax.barh(
        tier_order[::-1],
        [int(tc[t]) for t in tier_order[::-1]],
        color=clrs_bar[::-1],
        alpha=0.72,
        height=0.5,
    )
    ax.set_xlabel("Count", fontsize=7)
    ax.tick_params(labelsize=7)
    ax.grid(True, axis="x")
    ax.spines["bottom"].set_visible(True)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

with col_tbl:
    tiers_to_show = show_tiers if show_tiers else tier_order
    filtered = risk_df[risk_df["Risk_Tier"].isin(tiers_to_show)].copy()
    filtered = filtered.sort_values("Anomaly_Score", ascending=False).head(n_risk_rows)

    display_map = {
        "Anomaly_Score": "Score",
        "Risk_Tier": "Tier",
        "True_Label": "True",
        "RF_Pred": "Pred",
        "Log_Donation_Amount": "Log(Amount)",
        "Amount_Deviation_from_Donor_Mean": "Dev.from Mean",
        "Is_Night_Donation": "Night",
        "High_Risk_Platform": "Hi-Risk",
        "Donor_Unique_Devices": "Devices",
    }
    show = filtered[[c for c in display_map if c in filtered.columns]].rename(
        columns=display_map
    )
    show["Score"] = show["Score"].round(4)
    if "Dev.from Mean" in show.columns:
        show["Dev.from Mean"] = show["Dev.from Mean"].round(2)

    st.dataframe(
        show,
        use_container_width=True,
        height=430,
        column_config={
            "Score": st.column_config.ProgressColumn(
                "Score", min_value=0, max_value=1, format="%.4f"
            ),
        },
    )

# ═══════════════════════════════════════════════════════════════════════════════
# SINGLE DONATION SCORER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("## Score a Donation")
st.markdown(
    f"<p style=\"font-family:'DM Mono',monospace;font-size:0.75rem;color:{MUTED};\">"
    f"Enter donation details for an instant anomaly score from the pre-trained model.</p>",
    unsafe_allow_html=True,
)

MODEL_FEATURES = meta["features"]
CAT_ENCODE = meta["cat_encode"]
amt_p75 = meta.get("amt_p75", 74315)
bal_p10 = meta.get("bal_p10", 14532)
struct_thresh = meta.get("structuring_thresholds", [50000, 75000, 90000, 99000])
amt_p95 = meta["amount_mean"] + 2 * meta["amount_std"]

with st.form("score_form"):
    fc1, fc2, fc3, fc4 = st.columns(4)
    s_amount = fc1.number_input("Donation Amount (INR)", 100.0, 100000.0, 5000.0, 500.0)
    s_balance = fc2.number_input(
        "Account Balance (INR)", 100.0, 100000.0, 20000.0, 1000.0
    )
    s_age = fc3.number_input("Donor Age", 18, 90, 35)
    s_hour = fc4.slider("Hour of Day", 0, 23, 14)

    fc5, fc6, fc7, fc8 = st.columns(4)
    s_platform = fc5.selectbox(
        "Donation Platform",
        HIGH_RISK_PLATFORMS
        + [
            "Debit/Credit Card",
            "Web Browser",
            "Mobile Device",
            "ATM",
            "Desktop/Laptop",
        ],
    )
    s_don_type = fc6.selectbox(
        "Transaction Type",
        ["Transfer", "Withdrawal", "Debit", "Credit", "Bill Payment"],
    )
    s_account = fc7.selectbox("Account Type", ["Savings", "Business", "Checking"])
    s_weekend = fc8.checkbox("Weekend donation", value=False)

    submitted = st.form_submit_button("Calculate Risk Score →")

if submitted:
    amount_mean = meta["amount_mean"]
    amount_std = meta["amount_std"]

    # Derived booleans mirroring generate_labels.py rules
    is_near_threshold = any(
        s_amount >= t * 0.99 and s_amount < t for t in struct_thresh
    )

    sample = {f: 0 for f in MODEL_FEATURES}
    sample["Age"] = s_age
    sample["Log_Donation_Amount"] = np.log1p(s_amount)
    sample["Log_Cumulative_Given"] = np.log1p(s_balance)
    sample["Amt_Bal_Ratio"] = s_amount / (s_balance + 1)
    sample["Amount_Zscore"] = (s_amount - amount_mean) / (amount_std + 1e-9)
    sample["Hour"] = s_hour
    sample["Is_Night_Donation"] = int(s_hour <= 4)
    sample["Is_Weekend"] = int(s_weekend)
    sample["Is_Business_Hours"] = int(9 <= s_hour <= 17)
    sample["Is_Round_Amount"] = int(s_amount % 500 == 0)
    sample["Near_Threshold"] = int(is_near_threshold)
    sample["High_Risk_Platform"] = int(s_platform in HIGH_RISK_PLATFORMS)
    sample["Is_Transfer"] = int(s_don_type == "Transfer")
    sample["Is_Withdrawal"] = int(s_don_type == "Withdrawal")
    sample["Is_Business_Account"] = int(s_account == "Business")
    sample["Low_Balance_Large_Gift"] = int(s_balance < bal_p10 and s_amount >= amt_p75)
    sample["Young_Large"] = int(s_age < 25 and s_amount >= amt_p95)
    for col in CAT_ENCODE:
        sample[col] = 0  # neutral encoding for unspecified categoricals

    # Show which rules fired
    fired_rules = []
    if sample["Is_Night_Donation"] and sample["Is_Transfer"] and s_amount >= amt_p75:
        fired_rules.append("🌙 Night-time large transfer")
    if sample["High_Risk_Platform"] and s_amount >= amt_p75:
        fired_rules.append("📱 High-risk platform + large amount")
    if sample["Amt_Bal_Ratio"] > 3.5:
        fired_rules.append("⚖️ Donation far exceeds account balance")
    if sample["Low_Balance_Large_Gift"]:
        fired_rules.append("🏦 Low balance + large gift (pass-through signal)")
    if sample["Is_Round_Amount"] and s_amount >= amt_p75:
        fired_rules.append("🔢 Large round-number amount (structuring)")
    if sample["Near_Threshold"]:
        fired_rules.append("📉 Just below reporting threshold (structuring)")
    if sample["Young_Large"]:
        fired_rules.append("👤 Young donor + very large amount (mule signal)")
    if sample["Is_Withdrawal"] and s_amount >= amt_p95:
        fired_rules.append("💸 Very large withdrawal (layering signal)")

    X_single = pd.DataFrame([sample])[MODEL_FEATURES]
    score = float(rf.predict_proba(X_single)[0][1])
    tier = (
        "Low"
        if score < 0.30
        else "Medium" if score < 0.60 else "High" if score < 0.80 else "Critical"
    )
    flagged = score >= threshold

    rc1, rc2, rc3, rc4 = st.columns(4)
    rc1.metric("Anomaly Score", f"{score:.4f}")
    rc2.metric("Risk Tier", tier)
    rc3.metric("Flagged", "Yes ⚠️" if flagged else "No ✓")
    rc4.metric("Amt / Balance", f"{sample['Amt_Bal_Ratio']:.2f}×")

    st.progress(min(score, 1.0))

    if score >= 0.80:
        st.error("🚨 **Critical** — Immediate compliance review recommended.")
    elif score >= 0.60:
        st.warning("⚠️ **High risk** — Elevated indicators. Queue for review.")
    elif score >= 0.30:
        st.info("ℹ️ **Medium risk** — Some unusual signals. Monitor donor activity.")
    else:
        st.success("✅ **Low risk** — Donation within normal parameters.")

    if fired_rules:
        st.markdown(
            f"<p style=\"font-family:'DM Mono',monospace;font-size:0.7rem;"
            f'color:{MUTED};letter-spacing:0.1em;margin-top:1rem;">TRIGGERED RULES</p>',
            unsafe_allow_html=True,
        )
        for rule in fired_rules:
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:10px;padding:6px 12px;'
                f"margin:4px 0;background:{CREAM};border-left:3px solid {CORAL};"
                f'border-radius:2px;font-size:0.82rem;color:{SLATE};">{rule}</div>',
                unsafe_allow_html=True,
            )
