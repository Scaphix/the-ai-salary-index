import streamlit as st
import pandas as pd
import joblib


@st.cache_data
def load_data():
    df = pd.read_csv(
        "outputs/datasets/collection/ai_job_dataset1.csv"
    )
    return df


def load_pkl_file(file_path):
    return joblib.load(filename=file_path)
