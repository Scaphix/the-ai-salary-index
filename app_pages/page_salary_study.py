import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def page_salary_study_body():
    st.write("### AI Salary Study")

    st.info(
        "**BR1** - The client wants to understand how AI salaries "
        "correlate with job attributes such as experience level, "
        "company size, location, and education."
    )

    df = load_data()

    if st.checkbox("Inspect Dataset"):
        st.write(
            f"The dataset has {df.shape[0]} rows and {df.shape[1]} columns.\n\n"
            f"Below are the first 10 rows."
        )
        st.write(df.head(10))

    st.write("---")
    st.write("### Salary Distribution")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(data=df, x="salary_usd", bins=40, kde=True, ax=ax)
    ax.set_xlabel("Salary (USD)")
    ax.set_title("Distribution of AI Salaries")
    st.pyplot(fig)

    st.write("---")
    st.write("### Salary by Experience Level")
    fig, ax = plt.subplots(figsize=(10, 5))
    order = ["EN", "MI", "SE", "EX"]
    labels = {"EN": "Entry", "MI": "Mid", "SE": "Senior", "EX": "Executive"}
    plot_df = df.copy()
    plot_df["experience_label"] = plot_df["experience_level"].map(labels)
    label_order = [labels[o] for o in order if o in labels]
    sns.boxplot(data=plot_df, x="experience_label", y="salary_usd",
                order=label_order, ax=ax)
    ax.set_xlabel("Experience Level")
    ax.set_ylabel("Salary (USD)")
    ax.set_title("Salary by Experience Level")
    st.pyplot(fig)

    st.write("---")
    st.write("### Salary by Company Size")
    fig, ax = plt.subplots(figsize=(10, 5))
    size_labels = {"S": "Small", "M": "Medium", "L": "Large"}
    plot_df["size_label"] = plot_df["company_size"].map(size_labels)
    sns.boxplot(data=plot_df, x="size_label", y="salary_usd",
                order=["Small", "Medium", "Large"], ax=ax)
    ax.set_xlabel("Company Size")
    ax.set_ylabel("Salary (USD)")
    ax.set_title("Salary by Company Size")
    st.pyplot(fig)

    st.write("---")
    st.write("### Salary by Education Required")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=df, x="education_required", y="salary_usd", ax=ax)
    ax.set_xlabel("Education Required")
    ax.set_ylabel("Salary (USD)")
    ax.set_title("Salary by Education Required")
    plt.xticks(rotation=45)
    st.pyplot(fig)


@st.cache_data
def load_data():
    return pd.read_csv(
        "outputs/datasets/collection/ai_job_dataset1.csv"
    )
