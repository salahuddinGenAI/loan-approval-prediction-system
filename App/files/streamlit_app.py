"""
===============================================================================
 LOAN APPROVAL PREDICTION SYSTEM — PROFESSIONAL BANKING DASHBOARD
===============================================================================
A production-style Streamlit application that serves an ML-powered loan
approval decision engine built on top of a trained scikit-learn pipeline
(Random Forest classifier, ~97.8% test accuracy).

Author : Salahuddin Ayubi
GitHub : https://github.com/salahuddinGenAI/loan-approval-prediction-system
LinkedIn: https://www.linkedin.com/in/salahud-din-ayubi/

Run locally:
    streamlit run streamlit_app.py

Deploy:
    Push this file + loan_approval_prediction_pipeline.pkl + requirements.txt
    to a GitHub repo, then deploy on Streamlit Community Cloud.
===============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
from pathlib import Path

# ------------------------------------------------------------------------- #
#  PAGE CONFIGURATION
# ------------------------------------------------------------------------- #
icon_path = "F:\\Machine Learning projects\\Loan Approval Prediction System\\App\\files\\assets\\bank.png"
st.set_page_config(
    page_title="Loan Approval Prediction | AI Banking Suite",
    page_icon= icon_path,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------------- #
#  CONSTANTS
# ------------------------------------------------------------------------- #
PIPELINE_PATH = Path(__file__).parent / "loan_approval_prediction_pipeline.pkl"

# Exact column order the model was trained on — DO NOT REORDER.
FEATURE_ORDER = [
    "no_of_dependents",
    "education",
    "self_employed",
    "income_annum",
    "loan_amount",
    "loan_term",
    "cibil_score",
    "residential_assets_value",
    "commercial_assets_value",
    "luxury_assets_value",
    "bank_asset_value",
]

# Categorical encodings exactly as produced by sklearn's LabelEncoder
# during training (alphabetical ordering): "Graduate" < "Not Graduate",
# "No" < "Yes".
EDUCATION_MAP = {"Graduate": 0, "Not Graduate": 1}
SELF_EMPLOYED_MAP = {"No": 0, "Yes": 1}

# The trained target LabelEncoder mapped classes alphabetically too:
# " Approved" -> 1, " Rejected" -> 0. The RandomForestClassifier therefore
# predicts class 1 = Approved, class 0 = Rejected.
CLASS_APPROVED = 1
CLASS_REJECTED = 0

MODEL_NAME = "Random Forest Classifier"
MODEL_ACCURACY = 0.9778
MODEL_CV_MEAN = 0.9786
MODEL_CV_STD = 0.0044
DATASET_ROWS = 4241
DATASET_FEATURES = 11


# ------------------------------------------------------------------------- #
#  CUSTOM CSS — PREMIUM BANKING THEME
# ------------------------------------------------------------------------- #
def inject_custom_css():
    """Inject a premium navy/gold banking theme across the app."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Poppins:wght@600;700;800&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        :root {
            --navy-950: #060c1f;
            --navy-900: #0b1530;
            --navy-800: #10204a;
            --navy-700: #16305f;
            --gold-500: #d4af37;
            --gold-400: #e8c766;
            --emerald-500: #1fae6b;
            --crimson-500: #e5484d;
            --text-muted: #9fb0d1;
        }

        .stApp {
            background: radial-gradient(circle at top left, #0b1530 0%, #060c1f 55%, #04060f 100%);
            color: #eaf0ff;
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0b1530 0%, #060c1f 100%);
            border-right: 1px solid rgba(212, 175, 55, 0.25);
        }

        section[data-testid="stSidebar"] * {
            color: #eaf0ff;
        }

        /* Hero banner */
        .hero-banner {
            background: linear-gradient(120deg, #0b1530 0%, #16305f 55%, #0b1530 100%);
            border: 1px solid rgba(212, 175, 55, 0.35);
            border-radius: 18px;
            padding: 2.4rem 2.6rem;
            margin-bottom: 1.6rem;
            box-shadow: 0 20px 45px rgba(0,0,0,0.45);
            position: relative;
            overflow: hidden;
        }
        .hero-banner::after {
            content: "";
            position: absolute;
            top: -60px; right: -60px;
            width: 220px; height: 220px;
            background: radial-gradient(circle, rgba(212,175,55,0.25) 0%, rgba(212,175,55,0) 70%);
        }
        .hero-title {
            font-family: 'Poppins', sans-serif;
            font-size: 2.3rem;
            font-weight: 800;
            background: linear-gradient(90deg, #f5e5b3 0%, #d4af37 60%, #f5e5b3 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 0 0 0.4rem 0;
        }
        .hero-subtitle {
            color: var(--text-muted);
            font-size: 1.02rem;
            max-width: 720px;
            line-height: 1.55;
        }
        .hero-badges { margin-top: 1.1rem; }
        .badge {
            display: inline-block;
            background: rgba(212, 175, 55, 0.12);
            border: 1px solid rgba(212, 175, 55, 0.4);
            color: #f0dfa0;
            padding: 0.28rem 0.8rem;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 600;
            margin-right: 0.5rem;
            letter-spacing: 0.02em;
        }

        /* Section headers */
        .section-header {
            font-family: 'Poppins', sans-serif;
            font-size: 1.15rem;
            font-weight: 700;
            color: #f5e5b3;
            border-left: 4px solid var(--gold-500);
            padding-left: 0.7rem;
            margin: 1.6rem 0 0.9rem 0;
        }

        /* Card container */
        .banking-card {
            background: linear-gradient(160deg, #0e1a3a 0%, #0a1330 100%);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 16px;
            padding: 1.4rem 1.6rem;
            margin-bottom: 1rem;
            box-shadow: 0 10px 30px rgba(0,0,0,0.35);
        }

        /* Decision cards */
        .decision-approved {
            background: linear-gradient(135deg, rgba(31,174,107,0.18) 0%, rgba(31,174,107,0.05) 100%);
            border: 1.5px solid var(--emerald-500);
            border-radius: 18px;
            padding: 1.8rem 2rem;
            text-align: center;
        }
        .decision-rejected {
            background: linear-gradient(135deg, rgba(229,72,77,0.18) 0%, rgba(229,72,77,0.05) 100%);
            border: 1.5px solid var(--crimson-500);
            border-radius: 18px;
            padding: 1.8rem 2rem;
            text-align: center;
        }
        .decision-title {
            font-family: 'Poppins', sans-serif;
            font-size: 1.9rem;
            font-weight: 800;
            margin-bottom: 0.3rem;
        }
        .decision-approved .decision-title { color: var(--emerald-500); }
        .decision-rejected .decision-title { color: var(--crimson-500); }
        .decision-sub { color: var(--text-muted); font-size: 0.95rem; }

        /* Metric tiles */
        .metric-tile {
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 12px;
            padding: 0.9rem 1rem;
            text-align: center;
        }
        .metric-tile .val {
            font-size: 1.4rem;
            font-weight: 800;
            color: #f5e5b3;
            font-family: 'Poppins', sans-serif;
        }
        .metric-tile .lab {
            font-size: 0.78rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        /* Buttons */
        div.stButton > button, div.stFormSubmitButton > button {
            background: linear-gradient(90deg, #d4af37 0%, #e8c766 100%);
            color: #0b1530;
            font-weight: 700;
            border: none;
            border-radius: 10px;
            padding: 0.6rem 1.4rem;
            transition: all 0.2s ease-in-out;
            box-shadow: 0 6px 18px rgba(212,175,55,0.25);
        }
        div.stButton > button:hover, div.stFormSubmitButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 10px 22px rgba(212,175,55,0.4);
        }

        /* Footer */
        .app-footer {
            text-align: center;
            color: var(--text-muted);
            font-size: 0.82rem;
            margin-top: 2.5rem;
            padding-top: 1.2rem;
            border-top: 1px solid rgba(255,255,255,0.08);
        }

        hr { border-color: rgba(255,255,255,0.08) !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------------------- #
#  MODEL LOADING
# ------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def load_pipeline():
    """
    Load the trained joblib pipeline from disk.

    NOTE ON THE SAVED PIPELINE:
    The persisted Pipeline object contains three steps: ('label_encoder',
    'scaler', 'best_model'). The 'label_encoder' step is the LabelEncoder
    instance that was left over from encoding the TARGET column
    ('loan_status') during training — it is not a feature transformer, and
    calling pipeline.predict() directly on raw applicant data will raise an
    error because sklearn's LabelEncoder only accepts 1-D input.

    To serve real predictions correctly, this app bypasses that step and
    talks directly to the 'scaler' (StandardScaler fit on the 11 training
    features) and 'best_model' (the trained RandomForestClassifier) steps,
    after manually applying the same categorical encoding used at training
    time (Graduate=0/Not Graduate=1, No=0/Yes=1).
    """
    
    pipeline = joblib.load(PIPELINE_PATH)
    scaler = pipeline.named_steps["scaler"]
    model = pipeline.named_steps["best_model"]
    return scaler, model


def encode_applicant(
    no_of_dependents,
    education,
    self_employed,
    income_annum,
    loan_amount,
    loan_term,
    cibil_score,
    residential_assets_value,
    commercial_assets_value,
    luxury_assets_value,
    bank_asset_value,
) -> pd.DataFrame:
    """Assemble a single-row DataFrame in the exact feature order/encoding
    the model was trained on."""
    row = {
        "no_of_dependents": no_of_dependents,
        "education": EDUCATION_MAP[education],
        "self_employed": SELF_EMPLOYED_MAP[self_employed],
        "income_annum": income_annum,
        "loan_amount": loan_amount,
        "loan_term": loan_term,
        "cibil_score": cibil_score,
        "residential_assets_value": residential_assets_value,
        "commercial_assets_value": commercial_assets_value,
        "luxury_assets_value": luxury_assets_value,
        "bank_asset_value": bank_asset_value,
    }
    return pd.DataFrame([row])[FEATURE_ORDER]


def run_prediction(scaler, model, applicant_df: pd.DataFrame):
    """Scale features and return (decision_label, prob_approved, prob_rejected)."""
    scaled = scaler.transform(applicant_df)
    proba = model.predict_proba(scaled)[0]
    prob_approved = proba[CLASS_APPROVED]
    prob_rejected = proba[CLASS_REJECTED]
    decision = "Approved" if prob_approved >= prob_rejected else "Rejected"
    return decision, prob_approved, prob_rejected


# ------------------------------------------------------------------------- #
#  VALIDATION
# ------------------------------------------------------------------------- #
def validate_inputs(income_annum, loan_amount, cibil_score, loan_term) -> list:
    """Return a list of human-readable validation warnings, empty if clean."""
    issues = []
    if loan_amount > income_annum * 10:
        issues.append(
            "⚠️ The requested loan amount is more than 10× the annual income — "
            "double-check these figures for accuracy."
        )
    if cibil_score < 300 or cibil_score > 900:
        issues.append("⚠️ CIBIL score should realistically fall between 300 and 900.")
    if loan_term <= 0:
        issues.append("⚠️ Loan term must be greater than zero years.")
    if income_annum <= 0:
        issues.append("⚠️ Annual income must be greater than zero.")
    return issues


# ------------------------------------------------------------------------- #
#  UI COMPONENTS
# ------------------------------------------------------------------------- #
def render_hero_banner():
    st.markdown(
        """
        <div class="hero-banner">
            <div class="hero-title">🏦 AI Loan Approval Prediction Suite</div>
            <div class="hero-subtitle">
                An end-to-end machine learning decision engine that evaluates applicant,
                financial and asset information in real time to deliver instant,
                data-driven loan approval predictions — built for the modern digital bank.
            </div>
            <div class="hero-badges">
                <span class="badge">🤖 Random Forest · 97.8% Accuracy</span>
                <span class="badge">⚡ Real-time Inference</span>
                <span class="badge">🔒 Production-ready Pipeline</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar():
    with st.sidebar:
        st.markdown("### 🏦 About This Project")
        st.markdown(
            """
            An AI-powered loan approval prediction system trained on real-world
            lending data, wrapped in a professional Streamlit banking dashboard.
            The app consumes a persisted scikit-learn pipeline and returns
            instant approval decisions with calibrated confidence scores.
            """
        )

        st.markdown("---")
        st.markdown("### 🤖 Model Information")
        st.markdown(
            f"""
            <div class="metric-tile" style="margin-bottom:0.6rem;">
                <div class="val">{MODEL_NAME}</div>
                <div class="lab">Best Performing Model</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                f"""<div class="metric-tile"><div class="val">{MODEL_ACCURACY*100:.2f}%</div>
                <div class="lab">Test Accuracy</div></div>""",
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f"""<div class="metric-tile"><div class="val">{MODEL_CV_MEAN*100:.2f}%</div>
                <div class="lab">CV Mean (±{MODEL_CV_STD*100:.2f}%)</div></div>""",
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.markdown("### 📊 Dataset Summary")
        st.markdown(
            f"""
            - **Records used:** {DATASET_ROWS:,}
            - **Features:** {DATASET_FEATURES}
            - **Target:** `loan_status` (Approved / Rejected)
            - **Models compared:** Logistic Regression, Decision Tree, Random Forest
            """
        )

        st.markdown("---")
        st.markdown("### 📈 Performance Snapshot")
        perf_df = pd.DataFrame(
            {
                "Model": ["Logistic Regression", "Decision Tree", "Random Forest"],
                "Accuracy": [0.9258, 0.9694, 0.9778],
            }
        ).set_index("Model")
        st.bar_chart(perf_df, height=180)

        st.markdown("---")
        st.markdown("### 🔗 Connect")
        st.markdown(
            """
            [![GitHub](https://img.shields.io/badge/GitHub-View_Repo-181717?logo=github)](https://github.com/salahuddinGenAI/loan-approval-prediction-system)
            [![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?logo=linkedin)](https://www.linkedin.com/in/salahud-din-ayubi/)
            """
        )
 

def render_applicant_section():
    st.markdown('<div class="section-header">👤 Applicant Information</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        no_of_dependents = st.number_input(
            "Number of Dependents", min_value=0, max_value=10, value=0, step=1,
            help="Number of people financially dependent on the applicant."
        )
    with c2:
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    with c3:
        self_employed = st.selectbox("Self Employed", ["No", "Yes"])
    return no_of_dependents, education, self_employed


def render_financial_section():
    st.markdown('<div class="section-header">💰 Financial Information</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        income_annum = st.number_input(
            "Annual Income ($)", min_value=0, max_value=20_000_000,
            value=5_000_000, step=100_000
        )
        loan_amount = st.number_input(
            "Loan Amount ($)", min_value=0, max_value=50_000_000,
            value=10_000_000, step=100_000
        )
    with c2:
        loan_term = st.slider("Loan Term (years)", min_value=2, max_value=20, value=10)
        cibil_score = st.slider("CIBIL Score", min_value=300, max_value=900, value=650)
    return income_annum, loan_amount, loan_term, cibil_score


def render_assets_section():
    st.markdown('<div class="section-header">🏠 Asset Information</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        residential_assets_value = st.number_input(
            "Residential Assets Value ($)", min_value=0, max_value=40_000_000,
            value=5_000_000, step=100_000
        )
        commercial_assets_value = st.number_input(
            "Commercial Assets Value ($)", min_value=0, max_value=25_000_000,
            value=3_000_000, step=100_000
        )
    with c2:
        luxury_assets_value = st.number_input(
            "Luxury Assets Value ($)", min_value=0, max_value=40_000_000,
            value=8_000_000, step=100_000
        )
        bank_asset_value = st.number_input(
            "Bank Assets Value ($)", min_value=0, max_value=15_000_000,
            value=3_000_000, step=100_000
        )
    return (
        residential_assets_value,
        commercial_assets_value,
        luxury_assets_value,
        bank_asset_value,
    )


def render_probability_gauge(prob_approved: float):
    """Simple, dependency-free probability gauge using a styled progress bar."""
    pct = int(round(prob_approved * 100))
    color = "#1fae6b" if prob_approved >= 0.5 else "#e5484d"
    st.markdown(
        f"""
        <div style="margin: 0.6rem 0;">
            <div style="display:flex; justify-content:space-between; font-size:0.85rem; color:#9fb0d1; margin-bottom:0.3rem;">
                <span>Rejection Risk</span><span>Approval Confidence</span>
            </div>
            <div style="background: rgba(255,255,255,0.08); border-radius: 999px; height: 22px; overflow:hidden; border:1px solid rgba(255,255,255,0.1);">
                <div style="width:{pct}%; height:100%; background: linear-gradient(90deg,{color},{color}); display:flex; align-items:center; justify-content:flex-end; padding-right:8px; transition: width 0.6s ease;">
                    <span style="font-size:0.75rem; font-weight:700; color:#0b1530;">{pct}%</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def risk_band(prob_approved: float) -> tuple:
    """Return (label, color, emoji) risk classification from approval probability."""
    if prob_approved >= 0.80:
        return "Low Risk", "#1fae6b", "🟢"
    elif prob_approved >= 0.55:
        return "Moderate Risk", "#e8c766", "🟡"
    elif prob_approved >= 0.30:
        return "Elevated Risk", "#e8853f", "🟠"
    else:
        return "High Risk", "#e5484d", "🔴"


def render_decision_card(decision: str, prob_approved: float, prob_rejected: float):
    css_class = "decision-approved" if decision == "Approved" else "decision-rejected"
    icon = "✅" if decision == "Approved" else "❌"
    confidence = max(prob_approved, prob_rejected)
    st.markdown(
        f"""
        <div class="{css_class}">
            <div class="decision-title">{icon} Loan {decision}</div>
            <div class="decision-sub">Model confidence: {confidence*100:.1f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_results_dashboard(applicant_df: pd.DataFrame, decision, prob_approved, prob_rejected):
    st.markdown('<div class="section-header">📈 Results Dashboard</div>', unsafe_allow_html=True)

    left, right = st.columns([1.1, 1])

    with left:
        render_decision_card(decision, prob_approved, prob_rejected)
        st.write("")
        render_probability_gauge(prob_approved)

        label, color, emoji = risk_band(prob_approved)
        st.markdown(
            f"""
            <div class="banking-card" style="text-align:center; border-color:{color}55;">
                <span style="font-size:1.4rem;">{emoji}</span>
                <div style="font-weight:700; font-size:1.05rem; color:{color};">{label}</div>
                <div style="color:#9fb0d1; font-size:0.82rem;">AI-assessed credit risk classification</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if decision == "Approved":
            st.success(
                "This applicant profile closely matches historically **approved** loans "
                "based on income, credit score, and asset strength."
            )
        else:
            st.warning(
                "This applicant profile closely matches historically **rejected** loans. "
                "Consider improving CIBIL score, reducing loan amount, or increasing "
                "declared assets before reapplying."
            )

    with right:
        st.markdown("**Applicant Profile Summary**")
        display_row = applicant_df.iloc[0]
        summary = pd.DataFrame(
            {
                "Field": [
                    "Dependents", "Education", "Self Employed", "Annual Income ($)",
                    "Loan Amount ($)", "Loan Term (yrs)", "CIBIL Score",
                    "Residential Assets ($)", "Commercial Assets ($)",
                    "Luxury Assets ($)", "Bank Assets ($)",
                ],
                "Value": [
                    display_row["no_of_dependents"],
                    "Graduate" if display_row["education"] == 0 else "Not Graduate",
                    "No" if display_row["self_employed"] == 0 else "Yes",
                    f"{display_row['income_annum']:,.0f}",
                    f"{display_row['loan_amount']:,.0f}",
                    display_row["loan_term"],
                    display_row["cibil_score"],
                    f"{display_row['residential_assets_value']:,.0f}",
                    f"{display_row['commercial_assets_value']:,.0f}",
                    f"{display_row['luxury_assets_value']:,.0f}",
                    f"{display_row['bank_asset_value']:,.0f}",
                ],
            }
        )
        st.dataframe(summary, hide_index=True, use_container_width=True)

        m1, m2 = st.columns(2)
        with m1:
            st.markdown(
                f"""<div class="metric-tile"><div class="val">{prob_approved*100:.1f}%</div>
                <div class="lab">Approval Probability</div></div>""",
                unsafe_allow_html=True,
            )
        with m2:
            st.markdown(
                f"""<div class="metric-tile"><div class="val">{prob_rejected*100:.1f}%</div>
                <div class="lab">Rejection Probability</div></div>""",
                unsafe_allow_html=True,
            )


def render_footer():
    st.markdown(
        """
        <div class="app-footer">
            Built with ❤️ using Streamlit &amp; scikit-learn ·
            <strong>Loan Approval Prediction System</strong> ·
            For educational &amp; portfolio purposes only — not a substitute for
            professional financial or credit advice.<br>
            © 2026 · <a href="https://github.com/salahuddinGenAI/loan-approval-prediction-system" style="color:#d4af37;">GitHub</a> ·
            <a href="https://www.linkedin.com/in/salahud-din-ayubi/" style="color:#d4af37;">LinkedIn</a>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_about_section():
    with st.expander("ℹ️ About this app & how it works"):
        st.markdown(
            """
            **How this app works:** 
            When you fill in the form and click predict, 
            the app takes your details, converts things like Education and Self Employed into numbers, 
            and scales all values the same way the model was trained on. 
            
            This data is then passed to a Random Forest model, 
            which gives back a loan approval prediction along with a confidence score.

            **Training data:**
            The model learned from 4,241 real loan applications.
            Each application included applicant details, income, loan amount and term, 
            CIBIL credit score, and four types of assets (residential, commercial, luxury, and bank).

            **Model selection:**
            Three models were tested — Logistic Regression, Decision Tree, and Random Forest. 
            Random Forest gave the best and most consistent results, 
            with about 97.86% accuracy across multiple tests, and showed no signs of overfitting.
            """
        )


# ------------------------------------------------------------------------- #
#  MAIN APP
# ------------------------------------------------------------------------- #
def main():
    inject_custom_css()
    render_hero_banner()
    render_sidebar()

    if not PIPELINE_PATH.exists():
        st.error(
            f"Could not find the trained pipeline at `{PIPELINE_PATH.name}`. "
            "Please place `loan_approval_prediction_pipeline.pkl` in the same "
            "directory as this script."
        )
        st.stop()

    scaler, model = load_pipeline()

    # Reset support: bump a counter to force widgets back to defaults.
    if "reset_counter" not in st.session_state:
        st.session_state.reset_counter = 0

    with st.form(key=f"loan_form_{st.session_state.reset_counter}"):
        no_of_dependents, education, self_employed = render_applicant_section()
        income_annum, loan_amount, loan_term, cibil_score = render_financial_section()
        (
            residential_assets_value,
            commercial_assets_value,
            luxury_assets_value,
            bank_asset_value,
        ) = render_assets_section()

        st.write("")
        fc1, fc2 = st.columns([1, 1])
        with fc1:
            submitted = st.form_submit_button("🔮 Predict Loan Approval", use_container_width=True)
        with fc2:
            reset_clicked = st.form_submit_button("♻️ Reset Form", use_container_width=True)

    if reset_clicked:
        st.session_state.reset_counter += 1
        st.rerun()

    if submitted:
        issues = validate_inputs(income_annum, loan_amount, cibil_score, loan_term)
        for issue in issues:
            st.warning(issue)

        with st.spinner("🤖 Running applicant data through the AI prediction engine..."):
            time.sleep(0.6)  # brief UX pause so the spinner is perceptible
            applicant_df = encode_applicant(
                no_of_dependents, education, self_employed,
                income_annum, loan_amount, loan_term, cibil_score,
                residential_assets_value, commercial_assets_value,
                luxury_assets_value, bank_asset_value,
            )
            decision, prob_approved, prob_rejected = run_prediction(scaler, model, applicant_df)

        render_results_dashboard(applicant_df, decision, prob_approved, prob_rejected)
    else:
        st.info("👆 Fill in the applicant details above and click **Predict Loan Approval** to get an instant AI-driven decision.")

    render_about_section()
    render_footer()


if __name__ == "__main__":
    main()
