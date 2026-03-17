import streamlit as st
import pandas as pd
from src.data_management import load_pkl_file
from src.machine_learning.predictive_analysis import (
    predict_salary, predict_cluster
)


def page_predict_salary_body():
    st.write("# Predict AI Salary")

    st.info(
        "**Business Requirement 2** - The client wants to "
        "predict the expected annual salary (USD) for a "
        "given AI job posting based on its attributes.\n\n"
        "**Business Requirement 3** - The client wants to segment AI job "
        "postings into meaningful market clusters based on shared attributes, "
        "so that each prediction is accompanied by a market segment profile."
    )

    version = "v1"
    path = f"outputs/ml_pipeline/predict_salary/{version}"
    cluster_path = f"outputs/ml_pipeline/cluster_analysis/{version}"

    pipeline_dc_fe = load_pkl_file(
        f"{path}/pipeline_data_cleaning_feat_eng.pkl"
    )
    pipeline_model = load_pkl_file(f"{path}/pipeline_regressor.pkl")
    pipeline_cluster = load_pkl_file(f"{cluster_path}/pipeline_cluster.pkl")

    # --- Pipeline overview ---
    st.write("---")
    st.write("## How the Model Works")

    st.success(
        "The prediction is powered by three pipelines "
        "(cleaning + regressor + cluster) trained on over "
        "**11,700 AI job postings**.\n\n"
        "The system provides the following outputs:\n"
        "1. A **predicted annual salary** in USD "
        "based on the job attributes.\n"
        "2. A **salary tier classification** "
        "(top, above-average, mid-range, or lower end).\n"
        "3. A **market segment assignment** from cluster analysis, describing "
        "the typical profile for similar roles.\n"
        "4. **Actionable career tips** tailored to the selected inputs.\n"
        "5. A **margin of error** estimate based on the model's test MAE."
    )

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

        cluster = predict_cluster(
            pipeline_cluster, experience_level, company_location,
            company_size, employee_residence,
        )

        _render_salary_insights(
            salary, experience_level,
            company_size, cluster
        )


def _render_salary_insights(salary, experience_level, company_size, cluster):
    """Show contextual insights after a prediction is made."""

    st.write("---")
    st.write("### What This Means")

    # Salary tier classification
    if salary >= 200_000:
        tier_detail = (
            "This places the role in the **top salary tier** "
            "for AI positions. Roles at this level are typically "
            "executive or highly specialised positions in major "
            "tech hubs."
        )
    elif salary >= 140_000:
        tier_detail = (
            "This is an **above-average salary** for AI roles. "
            "It reflects a senior-level position or a mid-level "
            "role in a high-paying market."
        )
    elif salary >= 90_000:
        tier_detail = (
            "This falls in the **mid-range** for AI salaries. "
            "It is typical for mid-level professionals or senior "
            "roles in lower-cost markets."
        )
    else:
        tier_detail = (
            "This is on the **lower end** of AI salaries. It is "
            "common for entry-level roles, smaller companies, or "
            "markets with a lower cost of living."
        )

    st.info(
        f"**Predicted salary: ${salary:,.0f} USD/year**\n\n"
        f"{tier_detail}"

    )

    # Market segment from cluster analysis
    cluster_descriptions = {
        0: ("India Market (~4%)",
            "India-centric roles (employee residence: India 74%, "
            "company location: India 100%). Mixed experience levels. "
            "Salary is 93% Low, 7% Mid. Despite having senior "
            "professionals, geography is the dominant salary factor "
            "for this group."),
        1: ("Junior/Entry Developed Markets (~18%)",
            "1–3 years experience, MI/EN level. Located in developed "
            "markets (Ireland, Switzerland, France, Canada). "
            "Salary is 57% Low, 41% Mid, 2% High. Early-career "
            "professionals earning less, as expected."),
        2: ("Senior Professionals (~39%)",
            "7–15 years experience, SE/EX level. Located across "
            "developed markets (China, Singapore, Germany). "
            "Salary is 66% High, 30% Mid, 4% Low. Experience drives "
            "these professionals into premium pay."),
        3: ("Mid-career Emerging Markets (~39%)",
            "2–9 years experience, mixed levels. Located in emerging "
            "markets (Romania, Vietnam, Indonesia). Salary is nearly "
            "uniform: Mid 35%, High 34%, Low 31%. Experience still "
            "plays a role within this geographic group."),
    }
    segment_name, segment_desc = cluster_descriptions.get(
        cluster, ("Unknown Segment", ""))

    st.info(
        f"**Cluster {cluster}: {segment_name}** \n\n"
        f"{segment_desc}"
    )

    # Actionable advice
    tips = []
    if experience_level in ("EN", "MI"):
        tips.append(
            "Moving from entry/mid-level to senior typically "
            "results in a significant salary jump, experience "
            "level is the strongest predictor in the model."
        )
    if company_size == "S":
        tips.append(
            "Larger companies tend to offer higher compensation."
            " Consider targeting medium or large organisations "
            "for better pay."
        )
    if experience_level == "EX":
        tips.append(
            "Executive-level roles command premium salaries. "
            "Location choice becomes the key differentiator "
            "at this level."
        )

    tips.append(
        "Company and employee location strongly influence "
        "salary. Roles based in Switzerland, the US, or "
        "Norway tend to pay the most."
    )

    st.write("**Key Takeaways**")
    for tip in tips:
        st.write(f"* {tip}")

    st.warning(
        "**Note:** This prediction has a margin of error of "
        "approximately **±$16,124** (based on the model's test "
        "MAE). Actual salaries may vary depending on factors "
        "not captured by the model, such as specific skills, "
        "negotiation, benefits, and equity."
    )
