import streamlit as st


def page_summary_body():
    st.write("# Quick Project Summary")

    st.write("## General Information & Dataset Overview")

    st.info(
        "**General Information**\n\n"
        "The AI job market has grown rapidly, with salaries varying widely "
        "based on experience level, company size, location, and role.\n\n"
        "This project analyses a dataset of **15,000+ AI job postings** "
        "from 50+ countries to understand salary trends, build a predictive "
        "model, and segment the market into meaningful groups.\n\n"
        "The dashboard provides interactive tools for recruiters and hiring "
        "managers to explore salary data, validate hypotheses, predict "
        "salaries for new candidates, and understand market segments."
    )

    st.info(
        "**Project Terms & Jargon**\n"
        "* A **job listing** is a posted position in the AI/ML field "
        "with associated salary and requirements.\n"
        "* **Experience level** is coded as EN (Entry), MI (Mid-level), "
        "SE (Senior), EX (Executive).\n"
        "* **Remote ratio** indicates work arrangement: 0 = on-site, "
        "50 = hybrid, 100 = fully remote.\n"
        "* **Company size** is coded as S (<50 employees), M (50-250), "
        "L (>250).\n"
    )

    st.info(
        "**Project Dataset**\n"
        "* The dataset represents **15,000+ AI job listings** from 50+ "
        "countries (2025), containing data on job title, salary (USD), "
        "experience level, employment type, required skills, company "
        "location, remote ratio, and more.\n"
        "* The dataset was sourced from Kaggle: "
        "*Global AI Job Market & Salary Trends 2025* by Bisma Sajjad."
    )

    st.write("---")

    st.write("## Business Requirements")
    st.success(
        "The project has 3 Business Requirements:\n\n"
        "* **BR1 - Data Visualisation & Correlation Study** — "
        "Understand how AI salaries correlate with job attributes such "
        "as experience level, company size, location, and education.\n\n"
        "* **BR2 - Salary Prediction** — "
        "Predict the expected annual salary (USD) for a given AI job "
        "posting based on its attributes (R² ≥ 0.70).\n\n"
        "* **BR3 - Market Segmentation** — "
        "Group AI roles into market-based clusters to reveal distinct "
        "salary segments and labour market patterns."
    )

    st.write("---")

    st.write("## Dashboard Pages")

    st.info(
        "* **AI Salary Study** — Interactive exploration of salary "
        "distributions, location analysis, and correlation study (BR1)\n"
        "* **Project Hypothesis** — Four hypotheses tested with "
        "statistical evidence: 3 confirmed, 1 rejected (BR1)\n"
        "* **Predict Salary** — Live salary prediction from a "
        "candidate profile using a GradientBoostingRegressor (BR2)\n"
        "* **ML Regression Performance** — Pipeline steps, feature "
        "importance, and model evaluation with residual analysis (BR2)\n"
        "* **Cluster Analysis** — PCA + KMeans market segmentation "
        "with cluster profiles and distribution charts (BR3)"
    )

    st.write("## How to Use This Dashboard")

    st.info(
        "1. Use the **sidebar** on the left to navigate between "
        "pages.\n"
        "2. Expand **checkboxes** (e.g. *Inspect Dataset*) to "
        "reveal additional tables and visualisations.\n"
        "3. Switch between **tabs** to compare different plots "
        "within a section.\n"
        "4. On the **Predict Salary** page, select a candidate "
        "profile from the dropdowns and click **Predict Salary** "
        "to get an estimate.\n"
        "5. All plots include a short caption explaining what to "
        "look for."
    )

    st.write(
        "For additional information, please visit and **read** the "
        "[Project README file]"
        "(https://github.com/Scaphix/the-ai-salary-index)."
    )
