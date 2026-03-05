import streamlit as st
from app_pages.custom_transformers import OrdinalMappingEncoder, FrequencyEncoder  # noqa: F401

from app_pages.page_summary import page_summary_body
from app_pages.page_salary_study import page_salary_study_body
from app_pages.page_predict_salary import page_predict_salary_body
from app_pages.page_cluster import page_cluster_body

st.set_page_config(
    page_title="The AI Salary Index",
    page_icon="🖥️",
)

pages = st.navigation([
    st.Page(page_summary_body, title="Quick Project Summary"),
    st.Page(page_salary_study_body, title="AI Salary Study"),
    st.Page(page_predict_salary_body, title="Predict Salary"),
    st.Page(page_cluster_body, title="Job Market Clusters"),
])

pages.run()
