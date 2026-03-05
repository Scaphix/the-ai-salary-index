import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from src.data_management import load_pkl_file
from src.machine_learning.predictive_analysis import predict_salary


def page_predict_salary_body():
    st.write("### Predict AI Salary")

    st.info(
        "**BR1** - The client wants to predict the salary for a "
        "given AI job posting based on its attributes."
    )

    version = "v1"
    path = f"outputs/ml_pipeline/predict_salary/{version}"

    pipeline_dc_fe = load_pkl_file(
        f"{path}/pipeline_data_cleaning_feat_eng.pkl"
    )
    pipeline_model = load_pkl_file(f"{path}/pipeline_regressor.pkl")

    st.write("---")
    st.write("### Model Performance")
    st.write("Feature importance from the trained model:")
    fig, ax = plt.subplots()
    img = mpimg.imread(f"{path}/feature_importance.png")
    ax.imshow(img)
    ax.axis("off")
    st.pyplot(fig)

    st.write("---")
    st.write("### Predict on Live Data")
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
                "Switzerland", "Singapore", "Israel", "Sweden", "Ireland",
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
                "Singapore", "Portugal", "Israel", "Sweden", "Mexico",
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

        predict_salary(live_data, pipeline_dc_fe, pipeline_model)
