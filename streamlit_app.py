"""
Streamlit app — AI Clinical Decision Support System for Alzheimer's Disease.

A clinician-facing form that takes routinely-collected patient data and
returns a risk flag with a probability score, a recommended action band,
and a plain-language explanation of the top factors driving that specific
patient's prediction (via SHAP).

Run with: streamlit run app/streamlit_app.py   (from the project root)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import MODELS_DIR
from src.models import load_model
from src.feature_engineering import engineer_features
from src.explainability import compute_shap_values, explain_single_patient

st.set_page_config(page_title="Alzheimer's Clinical Decision Support", page_icon="🧠", layout="wide")

MODEL_PATH = MODELS_DIR / "alzheimers_diagnosis_model.joblib"


@st.cache_resource
def get_model():
    return load_model(MODEL_PATH)


def main():
    st.title("🧠 — Alzheimer's Disease Prediction")
    st.caption(
        "Decision-support tool to help prioritise further cognitive assessment. "
        "This is **not** a standalone diagnostic device — always use alongside clinical judgement."
    )

    if not MODEL_PATH.exists():
        st.error(
            f"No trained model found at `{MODEL_PATH}`. "
            "Run notebooks 01-07 first (in particular 06_Model_Development.ipynb) to train and save the model."
        )
        return

    pipeline = get_model()

    with st.sidebar:
        st.header("Patient Data Entry")

        st.subheader("Demographics")
        age = st.slider("Age", 60, 90, 75)
        gender = st.selectbox("Gender", ["Female", "Male"])
        ethnicity = st.selectbox("Ethnicity", ["Group 0", "Group 1", "Group 2", "Group 3"])
        education = st.selectbox("Education Level", ["None (0)", "High School (1)", "Bachelor's (2)", "Higher (3)"])

        st.subheader("Lifestyle")
        bmi = st.slider("BMI", 15.0, 45.0, 26.0)
        smoking = st.checkbox("Current smoker")
        alcohol = st.slider("Alcohol consumption (units/week)", 0.0, 20.0, 5.0)
        physical_activity = st.slider("Physical activity (hrs/week)", 0.0, 10.0, 4.0)
        diet_quality = st.slider("Diet quality (0-10)", 0.0, 10.0, 5.0)
        sleep_quality = st.slider("Sleep quality (4-10)", 4.0, 10.0, 7.0)

        st.subheader("Medical History")
        family_history = st.checkbox("Family history of Alzheimer's")
        cvd = st.checkbox("Cardiovascular disease")
        diabetes = st.checkbox("Diabetes")
        depression = st.checkbox("Depression")
        head_injury = st.checkbox("History of head injury")
        hypertension = st.checkbox("Hypertension")

        st.subheader("Clinical Measurements")
        systolic_bp = st.slider("Systolic BP", 90, 180, 130)
        diastolic_bp = st.slider("Diastolic BP", 60, 120, 80)
        chol_total = st.slider("Total cholesterol", 150, 300, 210)
        chol_ldl = st.slider("LDL cholesterol", 50, 200, 110)
        chol_hdl = st.slider("HDL cholesterol", 20, 100, 55)
        chol_trig = st.slider("Triglycerides", 50, 400, 150)

        st.subheader("Cognitive / Functional Assessment")
        mmse = st.slider("MMSE score (0-30, higher = better)", 0.0, 30.0, 24.0)
        functional = st.slider("Functional Assessment (0-10, higher = better)", 0.0, 10.0, 6.0)
        adl = st.slider("ADL score (0-10, higher = better)", 0.0, 10.0, 6.0)

        st.subheader("Reported Symptoms")
        memory_complaints = st.checkbox("Memory complaints")
        behavioral = st.checkbox("Behavioural problems")
        confusion = st.checkbox("Confusion")
        disorientation = st.checkbox("Disorientation")
        personality_changes = st.checkbox("Personality changes")
        difficulty_tasks = st.checkbox("Difficulty completing tasks")
        forgetfulness = st.checkbox("Forgetfulness")

        submitted = st.button("Run Risk Assessment", type="primary", use_container_width=True)

    if not submitted:
        st.info("⬅️ Enter patient data in the sidebar and click **Run Risk Assessment**.")
        st.markdown("### About this tool")
        st.markdown(
            "- Trained on a 2,149-patient de-identified clinical dataset\n"
            "- Final model: tuned **XGBoost** classifier\n"
            "- Test-set performance: **ROC-AUC ≈ 0.95**, **Recall ≈ 0.92** on the Alzheimer's class\n"
            "- Every prediction is explained using **SHAP** — see the top factors driving each patient's result\n"
        )
        return

    patient_dict = {
        "Age": age, "Gender": 1 if gender == "Male" else 0,
        "Ethnicity": int(ethnicity.split()[-1].strip("()")),
        "EducationLevel": int(education.split("(")[-1].strip(")")),
        "BMI": bmi, "Smoking": int(smoking), "AlcoholConsumption": alcohol,
        "PhysicalActivity": physical_activity, "DietQuality": diet_quality, "SleepQuality": sleep_quality,
        "FamilyHistoryAlzheimers": int(family_history), "CardiovascularDisease": int(cvd),
        "Diabetes": int(diabetes), "Depression": int(depression), "HeadInjury": int(head_injury),
        "Hypertension": int(hypertension), "SystolicBP": systolic_bp, "DiastolicBP": diastolic_bp,
        "CholesterolTotal": chol_total, "CholesterolLDL": chol_ldl, "CholesterolHDL": chol_hdl,
        "CholesterolTriglycerides": chol_trig, "MMSE": mmse, "FunctionalAssessment": functional,
        "MemoryComplaints": int(memory_complaints), "BehavioralProblems": int(behavioral),
        "ADL": adl, "Confusion": int(confusion), "Disorientation": int(disorientation),
        "PersonalityChanges": int(personality_changes), "DifficultyCompletingTasks": int(difficulty_tasks),
        "Forgetfulness": int(forgetfulness),
    }

    patient_df = pd.DataFrame([patient_dict])
    patient_df = engineer_features(patient_df)

    proba = pipeline.predict_proba(patient_df)[0, 1]
    prediction = int(proba >= 0.5)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Risk Assessment Result")
        st.metric("Predicted Alzheimer's Probability", f"{proba:.1%}")

        if proba >= 0.6:
            st.error("🔴 **HIGH RISK** — recommend prioritised cognitive assessment referral.")
        elif proba >= 0.3:
            st.warning("🟡 **MODERATE RISK** — recommend further cognitive screening.")
        else:
            st.success("🟢 **LOWER RISK** — routine monitoring; reassess if symptoms progress.")

        st.caption(
            "Threshold bands are illustrative defaults for this demo, not a clinically validated cutoff — "
            "see notebook 07 threshold analysis to set an evidence-based cutoff for real use."
        )

    with col2:
        st.subheader("Why this result? (SHAP explanation)")
        try:
            shap_values, X_transformed, feature_names = compute_shap_values(pipeline, patient_df)
            explanation = explain_single_patient(shap_values, X_transformed, feature_names, 0, top_n=6)

            fig, ax = plt.subplots(figsize=(7, 3.5))
            colors = ["#C44E52" if d == "increases risk" else "#4C72B0" for d in explanation["direction"]]
            ax.barh(explanation["feature"], explanation["shap_value"], color=colors)
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_xlabel("SHAP value (impact on predicted risk)")
            plt.tight_layout()
            st.pyplot(fig)
            st.caption("Red bars increase predicted risk; blue bars decrease it. Longer bar = bigger impact for this patient.")
        except Exception as e:
            st.warning(f"Explanation unavailable for this input: {e}")

    st.divider()
    st.caption(
        "⚠️ This tool is a decision-support aid for an educational/portfolio project. It is not a validated "
        "medical device and must not be used as a sole basis for diagnosis or treatment decisions."
    )


if __name__ == "__main__":
    main()
