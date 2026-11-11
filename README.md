# CKD Stage Classification — Streamlit Deployment

## Files
- `app.py` — Streamlit web application
- `model_bundle.joblib` — trained tuned Random Forest model, scaler and label encoders
- `input_metadata.json` — input field ranges and category options
- `requirements.txt` — Python dependencies

## Run locally
1. Install Python 3.10–3.12.
2. Open a terminal in this folder.
3. Install packages: `pip install -r requirements.txt`
4. Start the app: `streamlit run app.py`
5. Open the local URL shown in the terminal (usually http://localhost:8501).

## Publish online with Streamlit Community Cloud
1. Create/sign in to GitHub.
2. Create a new **public** repository and upload all four files in this folder.
3. Open https://share.streamlit.io/ and connect GitHub.
4. Choose the repository, branch `main`, and main file path `app.py`.
5. Click Deploy. Wait for the app URL to become available.

## Important project notes
- The deployment model uses the project's selected features, StandardScaler and tuned Random Forest parameters (`n_estimators=50`, `max_depth=5`, `min_samples_split=2`).
- `Patient_ID`, `Serum_Creatinine` and `eGFR` are excluded, matching the notebook's feature selection.
- The app accepts user-entered values within the ranges observed in the supplied dataset.
- This is an academic demonstration, not a diagnostic or treatment tool. Do not enter real patient information or use the output for medical decisions.
