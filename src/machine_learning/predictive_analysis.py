import streamlit as st
import pandas as pd


def predict_salary(X_live, pipeline_dc_fe, pipeline_model):

    # Extract encoders from the pipeline
    ordinal_map = pipeline_dc_fe.named_steps["OrdinalEncoder"].mappings
    freq_enc = pipeline_dc_fe.named_steps["FrequencyEncoder"]

    # Apply ordinal encoding
    for col, mapping in ordinal_map.items():
        if col in X_live.columns:
            X_live[col] = X_live[col].map(mapping)

    # Apply frequency encoding
    for col in freq_enc.variables:
        if col in X_live.columns:
            X_live[col] = X_live[col].map(
                freq_enc.freq_map_[col]
            ).fillna(0)

    # Predict
    prediction = pipeline_model.predict(X_live)
    salary = prediction[0]

    st.success(f"Predicted Salary: **${salary:,.0f}** USD")

    return prediction
