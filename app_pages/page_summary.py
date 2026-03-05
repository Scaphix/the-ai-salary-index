import streamlit as st


def page_summary_body():
    st.write("### Quick Project Summary")

    st.info(
        f"**General Information**\n\n"
        f"The AI job market has grown rapidly, with salaries varying widely "
        f"based on experience level, company size, location, and role.\n\n"
        f"This project analyzes a dataset of AI job postings to understand "
        f"salary trends and build predictive models.\n\n"
        f"**Project Dataset**\n\n"
        f"The dataset contains AI job postings with attributes such as "
        f"job title, salary, experience level, company location, "
        f"remote ratio, required skills, education, and more."
    )

    st.write("---")

    st.success(
        f"The project has 3 business requirements:\n\n"
        f"* **BR1** - The client wants to understand how AI salaries "
        f"correlate with job attributes such as experience level, "
        f"company size, location, and education.\n\n"
        f"* **BR2** - The client wants to predict the salary for a "
        f"given AI job posting based on its attributes.\n\n"
        f"* **BR3** - The client wants to identify clusters or segments "
        f"within the AI job market."
    )
