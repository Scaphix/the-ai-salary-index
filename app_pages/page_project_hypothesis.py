import streamlit as st


def page_project_hypothesis_body():

    st.write("### Project Hypothesis and Validation")
    st.info(
        "The following hypotheses were formulated before the "
        "analysis and validated using statistical tests "
        "(Kruskal-Wallis, Spearman and Pearson correlation) "
        "in the Job Market Study notebook."
    )

    # --- Hypothesis 1 ---
    st.write("---")
    st.write(
        "#### H1: Experience Level is the Dominant Salary Driver "
    )
    st.write(
        "**Statement:** Senior (SE) and Executive (EX) roles earn"
         "significantly more than Entry (EN) and Mid-level (MI) "
         "roles in the AI job market "
    )
    st.write(
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

    # --- Hypothesis 2 ---
    st.write("---")
    st.write(
        "#### H2: Remote roles have different salary "
        "expectations than hybrid/on-site roles"
    )
    st.write(
        "**Statement:** Jobs with remote_ratio = 100 "
        "(fully remote) pay differently than hybrid (50) or "
        "on-site (0) roles. Geographic flexibility may impact "
        "salary."
    )
    st.write(
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

    # --- Hypothesis 3 ---
    st.write("---")
    st.write(
        "#### H3: Years of experience is positively "
        "correlated with salary"
    )
    st.write(
        "**Statement:** Roles demanding more years of "
        "experience should offer higher compensation to "
        "attract experienced talent."
    )
    st.write(
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

    # --- Hypothesis 4 ---
    st.write("---")
    st.write(
        "#### H4: Job category influences salary levels"
    )
    st.write(
        "**Statement:** Different AI/ML domains (e.g., Data "
        "Science vs. ML Engineering) may have distinct salary "
        "markets."
    )
    st.write(
        "**Validation:** Engineered `job_category` by grouping "
        "20 job titles into 6 broad categories. "
        "Kruskal-Wallis test for differences across categories."
    )
    st.error(
        "**Verdict: Rejected**\n\n"
        "* `job_category` was engineered by grouping 20 job "
        "titles into 6 broad categories: Data Science, Data "
        "Engineering, ML Engineering, Research, AI/Specialized "
        "Engineering, and Leadership & Management.\n"
        "* Kruskal-Wallis test: H = 8.94, **p = 0.112** "
        "(not significant at α = 0.05).\n"
        "* Median salaries are remarkably similar across all "
        "categories (~$105k–$113k).\n\n"
        "Job category does not significantly influence salary "
        "and should not be a key predictor in the model."
    )
