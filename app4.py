"""
NeuroScreen — Alzheimer's Risk Model  (Streamlit deployment, loads trained .pkl)
================================================================================
Loads your trained model (alzheimers_model.pkl) and serves live predictions.
Built with native Streamlit components only.

Run:
    pip install streamlit plotly pandas scikit-learn joblib
    streamlit run app.py
"""

from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="NeuroScreen · Alzheimer's Risk Model",
                   page_icon="🧠", layout="wide")

# ---------------------------------------------------------------------------
# Model location.  Put alzheimers_model.pkl next to this file, or set the path.
# ---------------------------------------------------------------------------
MODEL_PATH = r"C:\Users\jrush\project\alzheimers_model.pkl"

# All 32 features the model was trained on, in order, with the dataset mean used
# as the default for any feature not exposed as an input control.
FEATURE_DEFAULTS = {
    "Age": 74.902, "Gender": 0.505, "Ethnicity": 0.703, "EducationLevel": 1.286,
    "BMI": 27.683, "Smoking": 0.292, "AlcoholConsumption": 10.111, "PhysicalActivity": 4.907,
    "DietQuality": 5.033, "SleepQuality": 7.046, "FamilyHistoryAlzheimers": 0.251,
    "CardiovascularDisease": 0.139, "Diabetes": 0.153, "Depression": 0.202, "HeadInjury": 0.092,
    "Hypertension": 0.146, "SystolicBP": 133.937, "DiastolicBP": 90.091, "CholesterolTotal": 225.504,
    "CholesterolLDL": 124.763, "CholesterolHDL": 59.069, "CholesterolTriglycerides": 228.918,
    "MMSE": 14.703, "FunctionalAssessment": 5.117, "MemoryComplaints": 0.205,
    "BehavioralProblems": 0.146, "ADL": 4.979, "Confusion": 0.205, "Disorientation": 0.154,
    "PersonalityChanges": 0.151, "DifficultyCompletingTasks": 0.158, "Forgetfulness": 0.306,
}
FEATURE_ORDER = list(FEATURE_DEFAULTS.keys())

# The features exposed as live controls and how they appear in the breakdown.
EXPOSED = ["FunctionalAssessment", "ADL", "MMSE", "Age", "SleepQuality",
           "MemoryComplaints", "BehavioralProblems"]
LABELS = {"FunctionalAssessment": "Functional Assessment", "ADL": "ADL", "MMSE": "MMSE",
          "Age": "Age", "SleepQuality": "Sleep Quality",
          "MemoryComplaints": "Memory Complaints", "BehavioralProblems": "Behavioral Problems"}

TEAL, AMBER, CORAL = "#34C7B5", "#F2A93B", "#F0526E"


@st.cache_resource
def load_model(path):
    """Load the trained model once. Returns (model, error_message)."""
    candidates = [Path(path), Path(__file__).parent / "alzheimers_model.pkl",
                  Path("alzheimers_model.pkl")]
    for p in candidates:
        try:
            if p.exists():
                return joblib.load(p), None
        except Exception as e:                       # noqa: BLE001
            return None, f"Found {p} but failed to load it: {e}"
    return None, ("Could not find 'alzheimers_model.pkl'. Set MODEL_PATH at the top of "
                  "app.py, or place the file next to app.py.")


def to_frame(state):
    """Build a single-row DataFrame with all 32 features in the trained order."""
    row = dict(FEATURE_DEFAULTS)
    row.update(state)
    return pd.DataFrame([[row[f] for f in FEATURE_ORDER]], columns=FEATURE_ORDER)


def predict(model, state):
    return float(model.predict(to_frame(state))[0, 1])


model, err = load_model(MODEL_PATH)

# ===========================================================================
# HEADER
# ===========================================================================
st.title("🧠 NeuroScreen — Alzheimer's Risk Model")
st.caption("A trained model flags Alzheimer's risk from a patient's clinical profile — "
           "and shows its reasoning. Adjust the controls below for a live prediction.")

if err:
    st.error(err)
    st.stop()

m1, m2, m3, m4 = st.columns(4)
m1.metric("Test Accuracy", "94.7%")
m2.metric("ROC-AUC", "0.946")
m3.metric("Patients", "2,149")
m4.metric("Signals that matter", "5")

st.divider()

# ===========================================================================
# 01 · LIVE RISK CONSOLE
# ===========================================================================
st.subheader("01 · Patient risk console")
st.caption("Predictions come live from the loaded model. Lower cognitive / functional "
           "scores are more concerning.")

inputs_col, output_col = st.columns([1, 1], gap="large")

with inputs_col:
    with st.container(border=True):
        st.markdown("**Patient profile**")
        fa = st.slider("Functional Assessment  (0 low – 10 high)", 0.0, 10.0, 5.1, 0.1)
        adl = st.slider("ADL — daily living  (0 low – 10 high)", 0.0, 10.0, 5.0, 0.1)
        mmse = st.slider("MMSE — cognitive exam  (0 – 30)", 0.0, 30.0, 14.7, 0.1)
        age = st.slider("Age", 60, 90, 75, 1)
        sleep = st.slider("Sleep Quality  (4 – 10)", 4.0, 10.0, 7.0, 0.1)
        c1, c2 = st.columns(2)
        mem = c1.radio("Memory Complaints", ["No", "Yes"], horizontal=True)
        beh = c2.radio("Behavioral Problems", ["No", "Yes"], horizontal=True)

state = {
    "Age": age, "SleepQuality": sleep, "MMSE": mmse, "FunctionalAssessment": fa,
    "ADL": adl, "MemoryComplaints": 1 if mem == "Yes" else 0,
    "BehavioralProblems": 1 if beh == "Yes" else 0,
}

prob = predict_proba(model, state)
pct = round(prob * 100)

# model-agnostic contribution: how much each adjusted feature moves the prediction
# away from where it would sit if that single feature were at the dataset average.
contrib = {}
for f in EXPOSED:
    reset = dict(state)
    reset[f] = FEATURE_DEFAULTS[f]
    contrib[f] = prob - predict_proba(model, reset)

if prob >= 0.66:
    label, color = "High Risk", CORAL
    verdict_fn, vtext = st.error, "Strong indicators of cognitive decline. Comprehensive clinical assessment advised."
elif prob >= 0.40:
    label, color = "Moderate", AMBER
    verdict_fn, vtext = st.warning, "Risk near the population baseline. Routine monitoring."
else:
    label, color = "Low Risk", TEAL
    verdict_fn, vtext = st.success, "Cognitive and functional profile is reassuring on these measures."

with output_col:
    with st.container(border=True):
        st.markdown("**Model prediction**")
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pct,
            number={"suffix": "%", "font": {"size": 44, "color": color}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": color, "thickness": 0.3},
                "steps": [
                    {"range": [0, 40], "color": "rgba(52,199,181,.15)"},
                    {"range": [40, 66], "color": "rgba(242,169,59,.15)"},
                    {"range": [66, 100], "color": "rgba(240,82,110,.15)"},
                ],
            },
        ))
        gauge.update_layout(height=240, margin=dict(l=20, r=20, t=20, b=10),
                            paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(gauge, use_container_width=True, config={"displayModeBar": False})
        verdict_fn(f"**{label}** — {vtext}")

# contribution breakdown — native diverging bar chart
st.markdown("**Why — each factor's pull on this prediction**")
items = sorted(contrib.items(), key=lambda kv: abs(kv[1]))
names = [LABELS[k] for k, _ in items]
vals = [round(v * 100, 2) for _, v in items]           # in percentage points
colors = [CORAL if v > 0 else TEAL for v in vals]
fig = go.Figure(go.Bar(
    x=vals, y=names, orientation="h", marker_color=colors,
    text=[f"{v:+.1f} pp" for v in vals], textposition="outside",
))
fig.update_layout(
    height=300, margin=dict(l=10, r=10, t=10, b=10),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    xaxis=dict(title="◀ lowers risk        raises risk ▶ (percentage points)",
               zeroline=True, zerolinecolor="#888", zerolinewidth=1),
)
st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

st.divider()

# ===========================================================================
# 02 · MODEL BENCHMARK
# ===========================================================================
st.subheader("02 · Five models, one clear winner")
st.caption("Each model validated with 5-fold cross-validation so the result isn't a lucky split.")

b1, b2 = st.columns([1.3, 1], gap="large")
with b1:
    models_df = pd.DataFrame({
        "Model": ["Logistic Regression", "Gradient Boosting", "Random Forest", "XGBoost", "SVM (RBF)"],
        "Accuracy": [0.816, 0.947, 0.949, 0.944, 0.835],
        "F1": [0.739, 0.924, 0.926, 0.919, 0.756],
        "ROC-AUC": [0.885, 0.946, 0.940, 0.941, 0.896],
    })

    def highlight_best(row):
        return ['background-color: rgba(52,199,181,0.18)' if row["Model"] == "Gradient Boosting"
                else '' for _ in row]

    styled = (models_df.style.apply(highlight_best, axis=1)
              .format({"Accuracy": "{:.3f}", "F1": "{:.3f}", "ROC-AUC": "{:.3f}"}))
    st.dataframe(styled, hide_index=True, use_container_width=True)
    st.caption("🟢 Best model: **Gradient Boosting**")
with b2:
    st.markdown("**Held-out test results** &nbsp;·&nbsp; n = 430")
    cm1, cm2 = st.columns(2)
    cm1.metric("True Negatives", 267)
    cm2.metric("False Positives", 11)
    cm1.metric("False Negatives", 12)
    cm2.metric("True Positives", 140)
    st.caption("Only 23 errors in 430 patients — and few missed AD cases, "
               "the costliest error in screening.")

st.divider()

# ===========================================================================
# 03 · EXPLAINABILITY (SHAP)
# ===========================================================================
st.subheader("03 · The model leans on cognition, not lifestyle")
st.caption("SHAP attributes the prediction to individual features. Five clinical measures "
           "carry almost all the weight.")

shap_df = pd.DataFrame({
    "Feature": ["Functional Assessment", "ADL (daily living)", "Memory Complaints",
                "MMSE", "Behavioral Problems", "Cholesterol (total)", "Sleep Quality"],
    "Importance": [1.41, 1.32, 0.97, 0.94, 0.79, 0.08, 0.05],
}).sort_values("Importance")
bar_colors = [TEAL if v >= 0.5 else "#5e7186" for v in shap_df["Importance"]]
shap_fig = go.Figure(go.Bar(
    x=shap_df["Importance"], y=shap_df["Feature"], orientation="h", marker_color=bar_colors,
    text=[f"{v:.2f}" for v in shap_df["Importance"]], textposition="outside",
))
shap_fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10),
                       paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                       xaxis_title="Mean |SHAP| value (higher = more influence)")
st.plotly_chart(shap_fig, use_container_width=True, config={"displayModeBar": False})

st.divider()

# ===========================================================================
# 04 · TAKEAWAYS
# ===========================================================================
st.subheader("04 · What the project shows")
t1, t2, t3 = st.columns(3, gap="medium")
with t1:
    with st.container(border=True):
        st.markdown("**01 · It works — and it's stable**")
        st.write("~95% accuracy and 0.95 AUC, confirmed across 5 cross-validation folds. "
                 "Not a one-split fluke.")
with t2:
    with st.container(border=True):
        st.markdown("**02 · Cognition is the signal**")
        st.write("Five functional & cognitive measures drive nearly every prediction; "
                 "demographics and labs barely move it.")
with t3:
    with st.container(border=True):
        st.markdown("**03 · It can explain itself**")
        st.write("SHAP makes each decision legible — the prerequisite for any model a "
                 "clinician would actually trust.")

st.warning("⚠️ This is an educational machine-learning project, not a validated medical "
           "device. It must not be used for real diagnosis. A clinical deployment would require "
           "prospective validation, calibration, and fairness auditing across patient subgroups.")

st.caption("NeuroScreen — Alzheimer's Disease classification project · Logistic Regression · "
           "Random Forest · Gradient Boosting · XGBoost · SVM · SHAP explainability · "
           "Dataset: 2,149 patients × 32 features. Built for academic presentation.")
