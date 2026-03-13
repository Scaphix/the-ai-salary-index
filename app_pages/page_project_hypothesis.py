import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_management import load_data

sns.set_style("whitegrid")


def page_project_hypothesis_body():

    df = load_data()

    st.write("# Project Hypothesis and Validation")
    st.info(
        "The following hypotheses were formulated after the "
        "study of the dataset and validated using statistical tests "
        "(Kruskal-Wallis, Spearman and Pearson correlation)."
    )

    # --- Hypothesis 1 ---
    st.write("---")
    st.write(
        "## H1: Experience Level is the Dominant Salary Driver "
    )
    st.info(
        "**Statement:** Senior (SE) and Executive (EX) roles earn"
        "significantly more than Entry (EN) and Mid-level (MI) "
        "roles in the AI job market "
    )
    st.info(
        "**Validation:** Kruskal-Wallis H-test across four "
        "experience level groups. Spearman correlation between "
        "ordinally-encoded experience level and salary_usd."
    )
    st.success(
        "**Verdict: Confirmed**\n\n"
        "* **experience_level** is the strongest predictor of "
        "**salary_usd**.\n"
        "* Senior (SE) and Executive (EX) roles dominate the "
        "high salary bands (>$140k), while Entry-level (EN) "
        "roles are concentrated in the lower bands.\n"
        "* Consistent across Spearman correlation, boxplots, "
        "and parallel plot.\n\n"
        "This makes it the primary feature for the regression "
        "model."
    )

    if st.checkbox("Show Salary by Experience Level"):
        fig, ax = plt.subplots(figsize=(5, 2.5))
        order = (df.groupby('experience_level')['salary_usd']
                 .median().sort_values(ascending=False).index)
        sns.boxplot(data=df, x='experience_level', y='salary_usd',
                    order=order, ax=ax)
        ax.set_title('Salary by Experience Level')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=False)

    # --- Hypothesis 2 ---
    st.write("---")
    st.write(
        "## H2: Remote roles have different salary "
        "expectations than hybrid/on-site roles"
    )
    st.info(
        "**Statement:** Jobs with remote_ratio = 100 "
        "(fully remote) pay differently than hybrid (50) or "
        "on-site (0) roles. Geographic flexibility may impact "
        "salary."
    )
    st.info(
        "**Validation:** Spearman correlation between "
        "**remote_ratio** and **salary_usd**. Kruskal-Wallis test "
        "across the three remote-ratio groups."
    )
    st.error(
        "**Verdict: Rejected**\n\n"
        "* **remote_ratio** has no meaningful impact on "
        "**salary_usd**.\n"
        "* All salary bands flow evenly across on-site (0), "
        "hybrid (50), and fully remote (100) categories.\n"
        "* Median salary (~$110k) and interquartile range are "
        "nearly identical across all three work arrangements.\n\n"
        "Work arrangement should not be included as a key "
        "feature in the regression model."
    )

    if st.checkbox("Show Salary by Remote Ratio"):
        fig, ax = plt.subplots(figsize=(5, 2.5))
        sns.boxplot(data=df, x='remote_ratio', y='salary_usd',
                    order=[0, 50, 100], ax=ax)
        ax.set_title('Salary by Remote Ratio')
        ax.set_xlabel('Remote Ratio (0=On-site, 50=Hybrid, 100=Remote)')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=False)

    # --- Hypothesis 3 ---
    st.write("---")
    st.write(
        "## H3: Years of experience is positively "
        "correlated with salary"
    )
    st.info(
        "**Statement:** Roles demanding more years of "
        "experience should offer higher compensation to "
        "attract experienced talent."
    )
    st.info(
        "**Validation:** Scatter plot with regression line. "
        "Pearson correlation coefficient to confirm strength "
        "of relationship."
    )
    st.success(
        "**Verdict: Confirmed**\n\n"
        "* **years_experience** is positively correlated with "
        "**salary_usd**.\n"
        "* This makes it the strongest purely numerical "
        "predictor in the dataset."
    )

    if st.checkbox("Show Salary vs Years of Experience"):
        from scipy import stats
        r_pearson, _ = stats.pearsonr(df['years_experience'], df['salary_usd'])
        fig, ax = plt.subplots(figsize=(5, 2.5))
        sns.regplot(data=df, x='years_experience', y='salary_usd',
                    scatter_kws={'alpha': 0.3, 's': 10},
                    line_kws={'color': 'red'}, ax=ax)
        ax.set_title(
            f'Salary vs Years of Experience '
            f'(Pearson r = {r_pearson:.3f})'
        )
        ax.set_xlabel('Years of Experience')
        ax.set_ylabel('Salary (USD)')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=False)

    # --- Hypothesis 4 ---
    st.write("---")
    st.write(
        "## H4: Company size is a significant salary driver"
    )
    st.info(
        "**Statement:** Large companies pay significantly higher "
        "salaries than small and medium-sized companies, reflecting "
        "greater revenue, bigger budgets, and competitive hiring "
        "pressure in the AI talent market."
    )
    st.info(
        "**Validation:** Kruskal-Wallis H-test across the three "
        "company size groups (S, M, L). Compare mean and median "
        "salary_usd per group. Boxplot comparison."
    )
    st.success(
        "**Verdict: Confirmed**\n\n"
        "* Median salary by company size: S = $95k, "
        " M = $105k, L = $122k : a $27k gap between "
        "small and large companies.\n"
        "* Company size is a secondary predictor after "
        "experience level and years of experience.\n\n"
        "Company size is a relevant feature for the regression "
        "model."
    )

    if st.checkbox("Show Salary by Company Size"):
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.boxplot(data=df, x='company_size', y='salary_usd',
                    order=['S', 'M', 'L'], ax=ax)
        ax.set_title('Salary by Company Size')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=False)

    # --- Conclusion ---
    st.write("---")
    st.write("## Conclusion")
    st.info(
        "Three out of four hypotheses were confirmed:\n\n"
        "* **Experience level** (H1) and **years of experience** "
        "(H3) are the strongest salary predictors, confirming that "
        "seniority (both in title and tenure) is the dominant "
        "driver of AI compensation.\n"
        "* **Company size** (H4) is a secondary but statistically "
        "significant factor, with large companies paying up to "
        "$27k more than small ones.\n"
        "* **Remote ratio** (H2) was rejected : work arrangement "
        "has no meaningful impact on salary, suggesting that the "
        "AI job market has largely normalized remote pay.\n\n"
        "**Implication for modeling:** The regression model should "
        "prioritize **experience_level**, **years_experience**, and "
        "**company_size** as core features, while **remote_ratio** "
        "can be safely excluded or deprioritized."
    )
