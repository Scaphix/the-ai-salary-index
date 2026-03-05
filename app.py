import streamlit as st
from src.machine_learning.custom_transformers import OrdinalMappingEncoder, FrequencyEncoder  # noqa: F401

from app_pages.page_summary import page_summary_body
from app_pages.page_salary_study import page_salary_study_body
from app_pages.page_predict_salary import page_predict_salary_body
from app_pages.page_project_hypothesis import page_project_hypothesis_body

st.set_page_config(
    page_title="The AI Salary Index",
    page_icon="📈",
)

pages = st.navigation([
    st.Page(page_summary_body, title="Quick Project Summary"),
    st.Page(page_project_hypothesis_body, title="Project Hypothesis"),
    st.Page(page_salary_study_body, title="AI Salary Study"),
    st.Page(page_predict_salary_body, title="Predict Salary"),
])

pages.run()
