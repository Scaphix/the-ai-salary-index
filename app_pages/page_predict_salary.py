import streamlit as st
import pandas as pd
from src.data_management import load_pkl_file
from src.machine_learning.predictive_analysis import predict_salary


def page_predict_salary_body():
    st.write("# Predict AI Salary")

    st.info(
        "**Business Requirement 1** — The client wants to predict the expected "
        "annual salary (USD) for a given AI job posting based on its attributes.\n\n"
        "**Success criterion:** R² ≥ 0.70 on both train and test sets."
    )

    version = "v1"
    path = f"outputs/ml_pipeline/predict_salary/{version}"

    pipeline_dc_fe = load_pkl_file(
        f"{path}/pipeline_data_cleaning_feat_eng.pkl"
    )
    pipeline_model = load_pkl_file(f"{path}/pipeline_regressor.pkl")

    # --- Pipeline overview ---
    st.write("---")
    st.write("## How the Model Works")

    st.success(
        "The prediction is powered by two pipelines trained on over "
        "**11,700 AI job postings**:"
    )

    st.info(
        "### 1. Data Cleaning & Feature Engineering Pipeline\n"
        "* Drops irrelevant columns (free-text fields, company names, "
        "industry) and **years_experience** (Spearman ρ = 0.97 with "
        "**experience_level** — nearly identical information).\n"
        "* Encodes **experience_level**, **company_size**, and "
        "**education_required** as ordinal numbers.\n"
        "* Frequency-encodes **company_location** and **employee_residence**.\n"
        "* One-hot encodes **employment_type** and **job_title**.\n\n"
        "### 2. Regression Pipeline\n"
        "* Scales features with StandardScaler.\n"
        "* Predicts salary using a tuned **GradientBoostingRegressor** "
        "(300 estimators, max depth 3, learning rate 0.2).\n"
        "* Only the **4 most important features** (selected automatically "
        "during training) are used for prediction: **experience_level**, "
        "**company_location**, **employee_residence**, and **company_size**."
    )

    st.write("## Model Performance")

    col_perf1, col_perf2 = st.columns(2)
    with col_perf1:
        st.metric("Train R²", "0.878")
        st.metric("Train MAE", "$15,208")
    with col_perf2:
        st.metric("Test R²", "0.863")
        st.metric("Test MAE", "$16,124")

    st.success(
        "The success criterion (R² ≥ 0.70) is met on both sets. "
        "The small train/test gap (0.015) indicates the model generalises "
        "well with no significant overfitting."
    )

    st.write("## Feature Importance")
    st.info(
        "**Most important predictor:** **experience_level** (H1) "
        "the single most influential feature in the model, confirming "
        "the hypothesis that experience level is the strongest salary driver.\n\n"
        "* **experience_level** alone accounts for over 50% of the model's "
        "predictive power.\n"
        "* **company_location** and **employee_residence** are the next most "
        "important features, reflecting the strong influence of location on"
        " salary expectations."
    )
    if st.checkbox("Show Feature Importance Plot"):
        st.image(f"{path}/feature_importance.png", width=800)

    # --- Live prediction ---
    st.write("---")
    st.write("## Predict on Live Data")
    st.write("Provide the job attributes below to get a salary prediction.")

    col1, col2 = st.columns(2)

    with col1:
        experience_level = st.selectbox(
            "Experience Level",
            options=["EN", "MI", "SE", "EX"],
            format_func=lambda x: {
                "EN": "Entry-level",
                "MI": "Mid-level",
                "SE": "Senior",
                "EX": "Executive",
            }[x],
        )
        company_size = st.selectbox(
            "Company Size",
            options=["S", "M", "L"],
            format_func=lambda x: {
                "S": "Small", "M": "Medium", "L": "Large",
            }[x],
        )

    with col2:
        company_location = st.selectbox(
            "Company Location",
            options=[
                "United States", "United Kingdom", "Germany", "Canada",
                "France", "India", "Japan", "Australia", "Netherlands",
                "Switzerland", "Singapore", "Sweden", "Ireland",
                "South Korea", "Denmark", "Finland", "Austria", "Norway",
                "China",
            ],
        )
        employee_residence = st.selectbox(
            "Employee Residence",
            options=[
                "United States", "United Kingdom", "Germany", "India",
                "Canada", "France", "Brazil", "Spain", "Netherlands",
                "Australia", "Japan", "Italy", "Poland", "Switzerland",
                "Singapore", "Portugal", "Sweden", "Mexico",
                "Ireland", "China", "South Korea", "Belgium", "Austria",
                "Denmark", "Argentina", "Romania", "Turkey", "Finland",
                "Norway", "Czech Republic", "Colombia", "Philippines",
                "Indonesia", "New Zealand", "Nigeria", "South Africa",
                "Hungary", "Chile", "Egypt", "Ukraine", "Thailand",
                "Malaysia", "Vietnam", "Russia", "Kenya", "Ghana",
                "Luxembourg", "Latvia", "Estonia",
            ],
        )

    if st.button("Predict Salary"):
        live_data = pd.DataFrame({
            "experience_level": [experience_level],
            "company_location": [company_location],
            "company_size": [company_size],
            "employee_residence": [employee_residence],
        })

        prediction = predict_salary(live_data, pipeline_dc_fe, pipeline_model)
        salary = prediction[0]

        _render_salary_insights(salary, experience_level, company_size)


def _render_salary_insights(salary, experience_level, company_size):
    """Show contextual insights after a prediction is made."""

    st.write("---")
    st.write("### What This Means")

    # Salary tier classification
    if salary >= 200_000:
        tier = "top-tier"
        tier_detail = (
            "This places the role in the **top salary tier** for AI positions. "
            "Roles at this level are typically executive or highly specialised "
            "positions in major tech hubs."
        )
    elif salary >= 140_000:
        tier = "upper-mid"
        tier_detail = (
            "This is an **above-average salary** for AI roles. It reflects "
            "a senior-level position or a mid-level role in a high-paying market."
        )
    elif salary >= 90_000:
        tier = "mid-range"
        tier_detail = (
            "This falls in the **mid-range** for AI salaries. It is typical "
            "for mid-level professionals or senior roles in lower-cost markets."
        )
    else:
        tier = "entry"
        tier_detail = (
            "This is on the **lower end** of AI salaries. It is common for "
            "entry-level roles, smaller companies, or markets with a lower "
            "cost of living."
        )

    st.info(
        f"**Predicted salary: ${salary:,.0f} USD/year**\n\n"
        f"{tier_detail}"
    )

    # Actionable advice
    tips = []
    if experience_level in ("EN", "MI"):
        tips.append(
            "Moving from entry/mid-level to senior typically results in "
            "a significant salary jump — experience level is the strongest "
            "predictor in the model."
        )
    if company_size == "S":
        tips.append(
            "Larger companies tend to offer higher compensation. Consider "
            "targeting medium or large organisations for better pay."
        )
    if experience_level == "EX":
        tips.append(
            "Executive-level roles command premium salaries. Location choice "
            "becomes the key differentiator at this level."
        )

    tips.append(
        "Company and employee location strongly influence salary. "
        "Roles based in the US, Switzerland, or Norway tend to pay the most."
    )

    st.write("**Key Takeaways**")
    for tip in tips:
        st.write(f"* {tip}")

    st.warning(
        "**Note:** This prediction has a margin of error of approximately "
        "**±$16,000** (based on the model's test MAE). Actual salaries may "
        "vary depending on factors not captured by the model, such as "
        "specific skills, negotiation, benefits, and equity."
    )
