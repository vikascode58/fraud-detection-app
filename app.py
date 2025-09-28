# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Fraud Detection Demo", layout="centered")

st.title("Fraud Detection — Demo App")
st.markdown("Enter transaction details and the model will predict probability of fraud. "
            "App reproduces training preprocessing using your original dataset so columns align.")

# -------------------------
# Helper: load model
# -------------------------
def load_model():
    # Try common filenames used earlier
    candidates = [
        "random_forest_fraud_model_no_imbalance_handling.joblib",
        "random_forest_fraud_model.joblib",
        "random_forest_fraud_model_tuned.joblib",
        "logistic_regression_fraud_model.joblib"
    ]
    for f in candidates:
        if os.path.exists(f):
            model = joblib.load(f)
            st.info(f"Loaded model: {f}")
            return model, f
    st.error("No model file found. Please place a saved model (joblib) in the app folder.")
    return None, None

# -------------------------
# Helper: build preprocessing from original dataset
# -------------------------
@st.cache_data(show_spinner=False)
def build_preprocessing(df_path="fraud_data.xlsx"):
    # Load original dataset used for training
    if not os.path.exists(df_path):
        st.warning(f"Original dataset '{df_path}' not found in folder. The app will still try to preprocess "
                   "based on input columns, but columns may not exactly match training.")
        return None

    df_raw = pd.read_excel(df_path)

    # Drop TransactionID if exists
    if 'TransactionID' in df_raw.columns:
        df_raw = df_raw.drop(columns=['TransactionID'])

    # Identify numeric and categorical (same logic as notebook)
    num_cols = df_raw.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if 'IsFraud' in num_cols:
        num_cols.remove('IsFraud')
    cat_cols = df_raw.select_dtypes(include=['object', 'category']).columns.tolist()

    # Imputers fit on training-like data
    num_imputer = SimpleImputer(strategy='median')
    cat_imputer = SimpleImputer(strategy='constant', fill_value='Missing')

    # Fit on raw dataset (to replicate pipeline)
    df_raw[num_cols] = num_imputer.fit_transform(df_raw[num_cols])
    df_raw[cat_cols] = cat_imputer.fit_transform(df_raw[cat_cols])

    # Fit scaler on numeric columns
    scaler = StandardScaler()
    scaler.fit(df_raw[num_cols])

    # After imputation create encoded template (one-hot)
    df_encoded = pd.get_dummies(df_raw, columns=cat_cols, drop_first=True)

    # Save the template column order (features used during training)
    feature_columns = [c for c in df_encoded.columns if c != 'IsFraud']

    preprocessing = {
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "num_imputer": num_imputer,
        "cat_imputer": cat_imputer,
        "scaler": scaler,
        "feature_columns": feature_columns,
        "raw_df": df_raw  # keep for reference
    }
    return preprocessing

# -------------------------
# UI: load model & preprocessing
# -------------------------
model, model_name = load_model()
preproc = build_preprocessing("fraud_data.xlsx")

if model is None:
    st.stop()

# -------------------------
# Input form
# -------------------------
st.subheader("Transaction Input")

with st.form("transaction_form"):
    amt = st.number_input("Amount (numeric)", min_value=0.0, value=100.0, step=10.0)
    time = st.number_input("Time (numeric, same unit as training)", min_value=0, value=50000, step=1)
    location = st.text_input("Location (city)", value="New York")
    merchant = st.text_input("MerchantCategory (e.g., Groceries, Electronics)", value="Groceries")
    age = st.number_input("CardHolderAge", min_value=0.0, value=30.0, step=1.0)

    submit = st.form_submit_button("Predict")

# -------------------------
# Preprocess single input
# -------------------------
def preprocess_input_row(preproc, input_dict):
    """
    Recreate preprocessing pipeline for a single-row input:
    - use imputers/scaler fit on original dataset
    - one-hot encode by concatenating with the training dataset so get_dummies creates same columns
    - ensure column order matches the feature_columns learned from training data
    """
    # If we don't have preproc (original dataset), do a fallback minimal preprocessing
    if preproc is None:
        # Minimal: create dataframe with numeric columns only
        df_in = pd.DataFrame([input_dict])
        return df_in

    num_cols = preproc["num_cols"]
    cat_cols = preproc["cat_cols"]
    num_imputer = preproc["num_imputer"]
    cat_imputer = preproc["cat_imputer"]
    scaler = preproc["scaler"]
    feature_columns = preproc["feature_columns"]
    raw_df = preproc["raw_df"].copy()

    # Build input row as dataframe
    input_df = pd.DataFrame([input_dict])

    # Ensure same columns exist: first impute missing for any expected columns present in raw_df
    # For numeric columns not present in input, fill with median from raw_df
    for c in num_cols:
        if c not in input_df.columns:
            input_df[c] = raw_df[c].median()

    for c in cat_cols:
        if c not in input_df.columns:
            input_df[c] = "Missing"

    # Apply imputers (they were fit on raw_df)
    input_df[num_cols] = num_imputer.transform(input_df[num_cols])
    input_df[cat_cols] = cat_imputer.transform(input_df[cat_cols])

    # Concatenate with raw_df (single row appended) so get_dummies will create same columns
    df_concat = pd.concat([raw_df.drop(columns=['IsFraud'], errors='ignore'), input_df], ignore_index=True)

    # One-hot encode (drop_first=True to mirror training)
    df_encoded = pd.get_dummies(df_concat, columns=cat_cols, drop_first=True)

    # The last row is our input encoded
    input_encoded = df_encoded.tail(1).reset_index(drop=True)

    # Align to feature_columns (training features). If some training columns are missing in input_encoded, add them with 0.
    for col in feature_columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0

    # If input created extra dummy columns not present in feature_columns, drop them
    extra_cols = [c for c in input_encoded.columns if c not in feature_columns]
    if len(extra_cols) > 0:
        input_encoded = input_encoded.drop(columns=extra_cols)

    # Reorder columns to feature_columns
    input_encoded = input_encoded[feature_columns]

    # Scale numeric columns (num_cols are originally present in feature_columns)
    # Note: feature_columns contain encoded columns; numeric columns exist by name
    for c in num_cols:
        if c in input_encoded.columns:
            input_encoded[c] = scaler.transform(input_encoded[[c]])

    return input_encoded

# -------------------------
# When user clicks Predict
# -------------------------
if submit:
    input_dict = {
        "Amount": float(amt),
        "Time": float(time),
        "Location": str(location),
        "MerchantCategory": str(merchant),
        "CardHolderAge": float(age)
    }

    with st.spinner("Preprocessing input and predicting..."):
        X_input = preprocess_input_row(preproc, input_dict)

        # Ensure model input columns align
        # Some older sklearn models have attribute feature_names_in_
        if hasattr(model, "feature_names_in_"):
            model_cols = list(model.feature_names_in_)
        else:
            # fallback: use X_input columns
            model_cols = X_input.columns.tolist()

        # Align X_input to model_cols
        for col in model_cols:
            if col not in X_input.columns:
                X_input[col] = 0
        X_input = X_input[model_cols]

        # Predict
        try:
            proba = model.predict_proba(X_input)[:, 1][0]
            score = float(proba)
            default_threshold = 0.5
            st.success(f"Predicted fraud probability: {score:.3f}")
        except Exception as e:
            st.error(f"Model prediction failed: {e}")
            st.stop()

        # Threshold slider
        threshold = st.slider("Probability threshold to classify as Fraud", 0.0, 1.0, 0.5, 0.01)

        pred_label = 1 if score >= threshold else 0
        if pred_label == 1:
            st.error("Model prediction: FRAUD (1)")
        else:
            st.success("Model prediction: Genuine (0)")

        # Explain numbers
        st.write(f"Probability = {score:.3f}, Threshold = {threshold:.2f}")

        # Optional: show top feature importances if model supports it
        if hasattr(model, "feature_importances_"):
            try:
                importances = model.feature_importances_
                feat_names = model.feature_names_in_ if hasattr(model, "feature_names_in_") else X_input.columns
                imp_df = pd.DataFrame({"feature": feat_names, "importance": importances})
                imp_df = imp_df.sort_values("importance", ascending=False).head(10)
                st.subheader("Top features used by the model")
                st.table(imp_df.reset_index(drop=True))
            except Exception:
                pass

        st.info("Note: This app re-creates preprocessing using your original dataset (fraud_data.xlsx). "
                "Ensure the same training dataset and model files are used for best consistency.")
