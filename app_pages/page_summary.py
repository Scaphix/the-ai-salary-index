import streamlit as st


def page_summary_body():
    st.write("### Quick Project Summary")

    st.info(
        f"**General Information**\n\n"
        f"The AI job market has grown rapidly, with salaries varying widely "
        f"based on experience level, company size, location, and role.\n\n"
        f"This project analyzes a dataset of AI job postings to understand "
        f"salary trends and build predictive models.\n\n"

    )

    st.info(
        f"**Project Terms & Jargon**\n"
        f"* A **job listing** is a posted position in the AI/ML field with associated salary and requirements.\n"
        f"* **Experience level** is coded as EN (Entry), MI (Mid-level), SE (Senior), EX (Executive).\n"
        f"* **Remote ratio** indicates work arrangement: 0 = on-site, 50 = hybrid, 100 = fully remote.\n"
        f"* **Company size** is coded as S (<50 employees), M (50–250), L (>250).\n\n"
        f"**Project Dataset**\n"
        f"* The dataset represents **15,000+ AI job listings** from 50+ countries (2025), "
        f"containing individual listing data on job title, salary (USD), experience level, "
        f"employment type, required skills, company location, remote ratio, and more.\n"
        f"* The dataset was sourced from Kaggle: "
        f"*Global AI Job Market & Salary Trends 2025* by Bisma Sajjad."
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

    st.write(
        f"* For additional information, please visit and **read** the "
        f"[Project README file](https://github.com/Scaphix/the-ai-salary-index)."
    )
