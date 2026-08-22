
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(
    page_title="FraudGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "fraud_model.joblib"
DATA_PATH = BASE_DIR / "fraudTest.csv"
OUTPUT_DIR = BASE_DIR / "fraud_model_output"

# ============================================================
# THEME / CSS
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 85% 10%, rgba(99,102,241,.13), transparent 25%),
        radial-gradient(circle at 10% 20%, rgba(14,165,233,.10), transparent 22%),
        #080b14;
    color: #eef2ff;
}

.block-container {
    max-width: 1450px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1220 0%, #090d17 100%);
    border-right: 1px solid rgba(148,163,184,.12);
}

[data-testid="stSidebar"] * {
    color: #e5e7eb;
}

.hero {
    padding: 30px 34px;
    border-radius: 24px;
    background:
        linear-gradient(135deg, rgba(30,41,59,.94), rgba(15,23,42,.86));
    border: 1px solid rgba(129,140,248,.22);
    box-shadow: 0 20px 70px rgba(0,0,0,.25);
    margin-bottom: 24px;
}

.hero-badge {
    display: inline-block;
    padding: 6px 12px;
    border-radius: 999px;
    background: rgba(99,102,241,.15);
    color: #a5b4fc;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: .08em;
    text-transform: uppercase;
    margin-bottom: 12px;
}

.hero h1 {
    font-size: 42px;
    line-height: 1.1;
    margin: 0;
    color: #f8fafc;
}

.hero p {
    color: #94a3b8;
    font-size: 16px;
    margin: 12px 0 0;
}

.card {
    background: rgba(15,23,42,.78);
    border: 1px solid rgba(148,163,184,.12);
    border-radius: 18px;
    padding: 20px;
    min-height: 110px;
    box-shadow: 0 12px 40px rgba(0,0,0,.16);
}

.card-label {
    color: #94a3b8;
    font-size: 13px;
    margin-bottom: 7px;
}

.card-value {
    color: #f8fafc;
    font-size: 28px;
    font-weight: 800;
}

.section-title {
    font-size: 22px;
    font-weight: 800;
    color: #f8fafc;
    margin: 28px 0 14px;
}

.result-safe {
    padding: 28px;
    border-radius: 20px;
    background: rgba(16,185,129,.10);
    border: 1px solid rgba(16,185,129,.30);
    text-align: center;
}

.result-fraud {
    padding: 28px;
    border-radius: 20px;
    background: rgba(239,68,68,.10);
    border: 1px solid rgba(239,68,68,.35);
    text-align: center;
}

.result-title {
    font-size: 32px;
    font-weight: 800;
    margin-bottom: 8px;
}

.result-sub {
    color: #94a3b8;
}

div[data-testid="stMetric"] {
    background: rgba(15,23,42,.78);
    border: 1px solid rgba(148,163,184,.12);
    padding: 16px;
    border-radius: 16px;
}

.stButton > button {
    border-radius: 12px;
    min-height: 46px;
    font-weight: 700;
}

[data-testid="stFileUploader"] {
    background: rgba(15,23,42,.55);
    border-radius: 16px;
}

img {
    border-radius: 14px;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# HELPERS
# ============================================================
@st.cache_resource
def load_model(path):
    return joblib.load(path)

@st.cache_data
def load_dataset(path):
    return pd.read_csv(path)

def unwrap_model(obj):
    """Supports a raw estimator or common saved-model dictionaries."""
    if isinstance(obj, dict):
        for key in ("model", "pipeline", "estimator", "classifier"):
            if key in obj:
                return obj[key]
    return obj

def get_model_features(model, df):
    # Pipeline/estimator feature_names_in_
    for obj in [model, getattr(model, "named_steps", {}).get("model", None)]:
        if obj is not None and hasattr(obj, "feature_names_in_"):
            return list(obj.feature_names_in_)

    # Fall back to dataset columns, excluding obvious target columns
    candidates = [
        c for c in df.columns
        if c.lower() not in {
            "is_fraud", "fraud", "target", "label", "class", "y"
        }
    ]
    return candidates

def card(label, value):
    return f"""
    <div class="card">
        <div class="card-label">{label}</div>
        <div class="card-value">{value}</div>
    </div>
    """

# ============================================================
# LOAD EXISTING MODEL
# ============================================================
if not MODEL_PATH.exists():
    st.error(
        "Could not find `fraud_model.joblib`. Put this app.py in the same "
        "folder as your trained model."
    )
    st.stop()

try:
    saved_object = load_model(str(MODEL_PATH))
    model = unwrap_model(saved_object)
except Exception as e:
    st.error(f"Unable to load fraud_model.joblib: {e}")
    st.stop()

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## 🛡️ FraudGuard AI")
    st.caption("Trained fraud detection model")

    st.divider()

    page = st.radio(
        "Navigation",
        ["🏠 Dashboard", "🔮 Live Prediction", "📊 Model Performance", "📁 Data Explorer"],
        label_visibility="collapsed",
    )

    st.divider()

    st.markdown("### Model status")
    st.success("● Model loaded")

    if DATA_PATH.exists():
        st.info("● Test dataset found")
    else:
        st.warning("● fraudTest.csv not found")

# ============================================================
# HERO
# ============================================================
st.markdown("""
<div class="hero">
    <div class="hero-badge">AI Fraud Detection</div>
    <h1>FraudGuard AI</h1>
    <p>Real-time transaction risk scoring powered by your trained machine-learning model.</p>
</div>
""", unsafe_allow_html=True)

# Load optional dataset
df = None
if DATA_PATH.exists():
    try:
        df = load_dataset(str(DATA_PATH))
    except Exception:
        df = None

features = get_model_features(model, df if df is not None else pd.DataFrame())

# ============================================================
# DASHBOARD
# ============================================================
if page == "🏠 Dashboard":
    st.markdown('<div class="section-title">Overview</div>', unsafe_allow_html=True)

    n_rows = len(df) if df is not None else "—"
    n_features = len(features)
    model_name = type(model).__name__

    cols = st.columns(4)
    with cols[0]:
        st.markdown(card("Model", model_name), unsafe_allow_html=True)
    with cols[1]:
        st.markdown(card("Features", f"{n_features:,}"), unsafe_allow_html=True)
    with cols[2]:
        st.markdown(card("Test records", f"{n_rows:,}" if isinstance(n_rows, int) else n_rows),
                    unsafe_allow_html=True)
    with cols[3]:
        st.markdown(card("Status", "Ready"), unsafe_allow_html=True)

    st.markdown('<div class="section-title">Quick actions</div>', unsafe_allow_html=True)
    a, b, c = st.columns(3)

    with a:
        st.markdown("### 🔮 Live Prediction")
        st.write("Enter transaction information and get an instant fraud prediction.")
    with b:
        st.markdown("### 📊 Performance")
        st.write("Review accuracy, classification report, confusion matrix and ROC curve.")
    with c:
        st.markdown("### 📁 Data Explorer")
        st.write("Inspect the test dataset and understand the available transaction fields.")

    if OUTPUT_DIR.exists():
        st.markdown('<div class="section-title">Saved model reports</div>', unsafe_allow_html=True)
        files = sorted([p.name for p in OUTPUT_DIR.iterdir() if p.is_file()])
        if files:
            st.dataframe(
                pd.DataFrame({"Report / Asset": files}),
                use_container_width=True,
                hide_index=True,
            )

# ============================================================
# LIVE PREDICTION
# ============================================================
elif page == "🔮 Live Prediction":
    st.markdown('<div class="section-title">Live Transaction Prediction</div>', unsafe_allow_html=True)

    if not features:
        st.warning(
            "No feature names were found in the saved model. "
            "Add feature names to the model or provide fraudTest.csv."
        )
        st.stop()

    st.caption(f"Enter values for {len(features)} model features.")

    input_values = {}
    cols = st.columns(3)

    for i, feature in enumerate(features):
        series = df[feature] if df is not None and feature in df.columns else None

        with cols[i % 3]:
            if series is not None and pd.api.types.is_numeric_dtype(series):
                clean = pd.to_numeric(series, errors="coerce").dropna()
                default = float(clean.median()) if len(clean) else 0.0
                input_values[feature] = st.number_input(
                    feature,
                    value=default,
                    key=f"input_{feature}",
                )
            elif series is not None:
                values = series.dropna().astype(str).unique().tolist()
                if len(values) <= 100 and values:
                    input_values[feature] = st.selectbox(
                        feature,
                        values,
                        key=f"input_{feature}",
                    )
                else:
                    input_values[feature] = st.text_input(
                        feature,
                        key=f"input_{feature}",
                    )
            else:
                input_values[feature] = st.number_input(
                    feature,
                    value=0.0,
                    key=f"input_{feature}",
                )

    st.divider()

    if st.button("🛡️ Analyze Transaction", type="primary", use_container_width=True):
        try:
            input_df = pd.DataFrame([input_values])
            prediction = model.predict(input_df)[0]

            probability = None
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(input_df)[0]
                probability = float(np.max(probabilities))

            # Normalize common fraud labels
            fraud = str(prediction).lower() in {
                "1", "true", "yes", "fraud", "fraudulent"
            }

            if fraud:
                st.markdown("""
                <div class="result-fraud">
                    <div class="result-title">🚨 FRAUD DETECTED</div>
                    <div class="result-sub">This transaction has been classified as potentially fraudulent.</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="result-safe">
                    <div class="result-title">✅ TRANSACTION APPEARS SAFE</div>
                    <div class="result-sub">The model did not classify this transaction as fraudulent.</div>
                </div>
                """, unsafe_allow_html=True)

            st.write("")
            r1, r2 = st.columns(2)
            with r1:
                st.metric("Prediction", str(prediction))
            with r2:
                if probability is not None:
                    st.metric("Model confidence", f"{probability * 100:.2f}%")
                else:
                    st.metric("Model confidence", "N/A")

        except Exception as e:
            st.error(
                "Prediction failed. Make sure the fields and data types match "
                f"the training data.\n\nDetails: {e}"
            )

# ============================================================
# PERFORMANCE
# ============================================================
elif page == "📊 Model Performance":
    st.markdown('<div class="section-title">Model Performance</div>', unsafe_allow_html=True)

    # Existing metrics
    metrics_path = OUTPUT_DIR / "model_metrics.csv"
    report_path = OUTPUT_DIR / "classification_report.csv"
    cm_path = OUTPUT_DIR / "confusion_matrix.png"
    roc_path = OUTPUT_DIR / "roc_curve.png"

    if metrics_path.exists():
        try:
            metrics_df = pd.read_csv(metrics_path)
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)
        except Exception as e:
            st.warning(f"Could not read model_metrics.csv: {e}")

    left, right = st.columns(2)

    with left:
        st.markdown("### Confusion Matrix")
        if cm_path.exists():
            st.image(str(cm_path), use_container_width=True)
        else:
            st.info("confusion_matrix.png not found.")

    with right:
        st.markdown("### ROC Curve")
        if roc_path.exists():
            st.image(str(roc_path), use_container_width=True)
        else:
            st.info("roc_curve.png not found.")

    if report_path.exists():
        st.markdown("### Classification Report")
        try:
            report_df = pd.read_csv(report_path)
            st.dataframe(report_df, use_container_width=True, hide_index=True)
        except Exception as e:
            st.warning(f"Could not read classification_report.csv: {e}")

# ============================================================
# DATA EXPLORER
# ============================================================
elif page == "📁 Data Explorer":
    st.markdown('<div class="section-title">Test Dataset Explorer</div>', unsafe_allow_html=True)

    if df is None:
        st.warning("fraudTest.csv was not found in the app folder.")
        st.stop()

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Rows", f"{len(df):,}")
    with c2:
        st.metric("Columns", f"{len(df.columns):,}")
    with c3:
        st.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")

    st.markdown("### Preview")
    st.dataframe(df.head(200), use_container_width=True, hide_index=True)

    st.markdown("### Column information")
    info = pd.DataFrame({
        "Column": df.columns,
        "Type": df.dtypes.astype(str).values,
        "Missing": df.isna().sum().values,
        "Unique": [df[c].nunique(dropna=True) for c in df.columns],
    })
    st.dataframe(info, use_container_width=True, hide_index=True)

# ============================================================
# FOOTER
# ============================================================
st.markdown("""
<div style="text-align:center;color:#64748b;margin-top:45px;padding-top:20px;border-top:1px solid rgba(148,163,184,.10);">
    🛡️ FraudGuard AI &nbsp;•&nbsp; Powered by your trained ML model
</div>
""", unsafe_allow_html=True)
