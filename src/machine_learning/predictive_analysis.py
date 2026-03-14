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
    st.write("### Predicted Salary")
    st.success(f"**${salary:,.0f} USD**")

    return prediction


def predict_cluster(pipeline_cluster, experience_level, company_location,
                    company_size, employee_residence):
    """Predict market segment using the cluster pipeline.

    The pipeline expects 11 features. We fill missing ones with
    sensible defaults (mode values from training data) since the
    cluster is primarily driven by experience_level, company_location,
    and employee_residence — all provided by the user.
    """
    years_map = {"EN": 1, "MI": 3, "SE": 8, "EX": 14}

    live_row = pd.DataFrame({
        "job_title": ["Data Scientist"],
        "experience_level": [experience_level],
        "employment_type": ["FT"],
        "company_location": [company_location],
        "company_size": [company_size],
        "employee_residence": [employee_residence],
        "remote_ratio": [50],
        "education_required": ["Bachelor"],
        "years_experience": [years_map[experience_level]],
        "industry": ["Technology"],
        "benefits_score": [7.0],
    })

    cluster_label = pipeline_cluster.predict(live_row)[0]

    st.write("### Predicted Market Segment")
    st.success(f"**Cluster {cluster_label}**")

    return cluster_label
