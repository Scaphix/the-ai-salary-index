import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_management import load_data

sns.set_style("whitegrid")


def page_project_hypothesis_body():

    df = load_data()

    st.write("# Project Hypothesis and Validation")
    st.info(
        "The following hypotheses were formulated before the "
        "analysis and validated using statistical tests "
        "(Kruskal-Wallis, Spearman and Pearson correlation) "
        "in the Job Market Study notebook."
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
        "* `experience_level` is the strongest predictor of "
        "`salary_usd`.\n"
        "* Senior (SE) and Executive (EX) roles dominate the "
        "high salary bands (>$140k), while Entry-level (EN) "
        "roles are concentrated in the lower bands.\n"
        "* Consistent across Spearman correlation, boxplots, "
        "and parallel plot.\n\n"
        "This makes it the primary feature for the regression "
        "model."
    )
    st.write(
        "**Recommended action:** Compensation benchmarking tools should use "
        "experience level as the primary stratification variable before comparing across "
        "job titles or locations."
    )

    if st.checkbox("Show Salary by Experience Level"):
        fig, ax = plt.subplots(figsize=(10, 5))
        order = (df.groupby('experience_level')['salary_usd']
                 .median().sort_values(ascending=False).index)
        sns.boxplot(data=df, x='experience_level', y='salary_usd',
                    order=order, ax=ax)
        ax.set_title('Salary by Experience Level')
        plt.tight_layout()
        st.pyplot(fig)

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
        "remote_ratio and salary_usd. Kruskal-Wallis test "
        "across the three remote-ratio groups."
    )
    st.error(
        "**Verdict: Rejected**\n\n"
        "* `remote_ratio` has no meaningful impact on "
        "`salary_usd`.\n"
        "* All salary bands flow evenly across on-site (0), "
        "hybrid (50), and fully remote (100) categories.\n"
        "* Median salary (~$110k) and interquartile range are "
        "nearly identical across all three work arrangements.\n"
        "* Spearman correlation is near-zero (0.003).\n\n"
        "Work arrangement should not be included as a key "
        "feature in the regression model."
    )

    if st.checkbox("Show Salary by Remote Ratio"):
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.boxplot(data=df, x='remote_ratio', y='salary_usd',
                    order=[0, 50, 100], ax=ax)
        ax.set_title('Salary by Remote Ratio')
        ax.set_xlabel('Remote Ratio (0=On-site, 50=Hybrid, 100=Remote)')
        plt.tight_layout()
        st.pyplot(fig)

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
        "* `years_experience` is positively correlated with "
        "`salary_usd` (Pearson: ~0.7).\n"
        "* This makes it the strongest purely numerical "
        "predictor in the dataset."
    )

    if st.checkbox("Show Salary vs Years of Experience"):
        from scipy import stats
        r_pearson, _ = stats.pearsonr(df['years_experience'], df['salary_usd'])
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.regplot(data=df, x='years_experience', y='salary_usd',
                    scatter_kws={'alpha': 0.3, 's': 10},
                    line_kws={'color': 'red'}, ax=ax)
        ax.set_title(f'Salary vs Years of Experience (Pearson r = {r_pearson:.3f})')
        ax.set_xlabel('Years of Experience')
        ax.set_ylabel('Salary (USD)')
        plt.tight_layout()
        st.pyplot(fig)

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
        "* Kruskal-Wallis test: H = 544.67, **p ≈ 0** "
        "(highly significant at α = 0.05).\n"
        "* Large companies pay notably higher salaries than "
        "small and medium-sized companies.\n\n"
        "Company size is a relevant feature for the regression "
        "model."
    )

    if st.checkbox("Show Salary by Company Size"):
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(data=df, x='company_size', y='salary_usd',
                    order=['S', 'M', 'L'], ax=ax)
        ax.set_title('Salary by Company Size')
        plt.tight_layout()
        st.pyplot(fig)

    # --- Conclusion ---
    st.write("---")
    st.write("## Conclusion")
    st.info(
        "Three out of four hypotheses were confirmed, showing "
        "the following:\n\n"
        "**Domain intuition is supported:** Patterns expected "
        "from recruitment experience — that seniority, years of "
        "experience, and company size drive salary — are "
        "validated by statistical evidence.\n\n"
        "**Features are predictive:** `experience_level`, "
        "`years_experience`, and `company_size` show strong "
        "links to `salary_usd`, making them valuable inputs "
        "for the salary prediction model.\n\n"
        "**One assumption disproved:** Remote work arrangement "
        "does not influence salary, meaning recruiters should "
        "not adjust salary expectations based on whether a role "
        "is on-site, hybrid, or fully remote."
    )
