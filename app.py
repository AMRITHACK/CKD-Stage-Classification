import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="CKD Stage Classification", page_icon="🩺", layout="wide")

@st.cache_resource
def load_bundle():
    return joblib.load("model_bundle.joblib")

with open("input_metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)
bundle = load_bundle()

st.title("🩺 Chronic Kidney Disease Stage Classification")
st.write("Enter the clinical and laboratory measurements below to generate an experimental model prediction.")
st.warning("Educational project only — this tool is not a medical device and must not be used for diagnosis, treatment, or medical decisions. Consult a qualified healthcare professional.")

with st.form("patient_form"):
    values = {}
    numeric_cols = [c for c, m in metadata.items() if m["type"] == "number"]
    category_cols = [c for c, m in metadata.items() if m["type"] == "category"]
    st.subheader("Patient measurements")
    cols = st.columns(3)
    for i, name in enumerate(numeric_cols):
        m = metadata[name]
        with cols[i % 3]:
            if m["integer"]:
                values[name] = st.number_input(name.replace("_", " "), min_value=int(m["min"]), max_value=int(m["max"]), value=int(m["default"]), step=1)
            else:
                values[name] = st.number_input(name.replace("_", " "), min_value=float(m["min"]), max_value=float(m["max"]), value=float(m["default"]), step=0.1, format="%.2f")
    st.subheader("Medical history")
    cols = st.columns(2)
    for i, name in enumerate(category_cols):
        with cols[i % 2]:
            values[name] = st.selectbox(name.replace("_", " "), metadata[name]["options"])
    submitted = st.form_submit_button("Predict CKD Category", type="primary", use_container_width=True)

if submitted:
    row = pd.DataFrame([values], columns=bundle["feature_cols"])
    for col in category_cols:
        row[col] = bundle["encoders"][col].transform(row[col].astype(str))
    row = row[bundle["feature_cols"]]
    scaled = bundle["scaler"].transform(row)
    pred_code = int(bundle["model"].predict(scaled)[0])
    predicted_label = bundle["encoders"]["Target"].inverse_transform([pred_code])[0]
    st.divider()
    st.subheader("Model output")
    st.success(f"Predicted category: **{predicted_label}**")
    st.caption("This is a machine-learning output based on the project's training dataset, not a clinical assessment.")
    if hasattr(bundle["model"], "predict_proba"):
        probs = bundle["model"].predict_proba(scaled)[0]
        prob_df = pd.DataFrame({"CKD category": bundle["encoders"]["Target"].inverse_transform(bundle["model"].classes_), "Model score (%)": np.round(probs * 100, 2)})
        st.write("Model class scores (not calibrated medical probabilities):")
        st.dataframe(prob_df.sort_values("Model score (%)", ascending=False), hide_index=True, use_container_width=True)

with st.expander("About this project"):
    st.write("The app uses the tuned Random Forest model from the CKD classification project. Patient_ID, Serum_Creatinine and eGFR are excluded from the model inputs as in the notebook's feature-selection step.")
    st.write(f"Held-out test accuracy recorded when this deployment model was trained: {bundle['accuracy']*100:.2f}%.")
    st.write("High test performance on this synthetic/project dataset does not establish real-world clinical reliability.")
