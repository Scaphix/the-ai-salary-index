from feature_engine.discretisation import ArbitraryDiscretiser
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
from src.data_management import load_data

sns.set_style("whitegrid")


def page_salary_study_body():
    st.write("# AI Salary Study")

    st.info(
        "**Business Requirement 1** - The client wants to identify"
        " which attributes correlate most closely with AI/ML salary levels."

    )

    df = load_data()

    if st.checkbox("Inspect Dataset"):
        st.write(
            f"The dataset has {df.shape[0]} rows and {df.shape[1]} columns.\n"
            f"Below are the first 10 rows."
        )
        st.write(df.head(10))

    # --- KPI Metrics ---
    st.write("---")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Jobs", f"{df.shape[0]:,}")
    col2.metric("Median Salary", f"${df['salary_usd'].median():,.0f}")
    col3.metric("Top Country",
                df['company_location'].value_counts().idxmax())
    col4.metric("Most Common Level",
                df['experience_level'].value_counts().idxmax())

    # --- Salary Distribution ---
    st.write("---")
    st.write("## Salary Distribution")
    st.info("The salary distribution is right-skewed, "
            "with most salaries concentrated between $50k and $150k "
            "and a long tail extending beyond $400k. "
            "The median salary (~ $110k) is lower than the mean, "
            "reflecting a small number of high-paying executive roles "
            "that pull the average upward. For benchmarking purposes, "
            "the median is a more reliable measure of a typical AI salary.")
    if st.checkbox("Show Distribution of AI Salaries"):
        fig, ax = plt.subplots(figsize=(6, 2.5))
        sns.histplot(data=df, x="salary_usd", bins=40, kde=True, ax=ax)
        ax.set_xlabel("Salary (USD)")
        ax.set_title("Distribution of AI Salaries")
        st.pyplot(fig, use_container_width=False)
        st.caption(
            "Look for the peak and long right tail, most AI salaries "
            "fall between $50k and $150k, but a few executive roles "
            "push well beyond $400k."
        )

    # --- Salary by Location ---
    st.write("---")
    st.write("## Salary by Location")
    st.info(
        "Location plays a visible role in AI compensation. "
        "The charts below show the top 12 countries by median "
        "salary for both company headquarters and employee residence."
    )

    loc_left, loc_right = st.columns(2)
    with loc_left:
        st.write("#### By Company Location")
        top_comp = (df.groupby('company_location')['salary_usd']
                    .median().sort_values(ascending=False).head(12))
        fig, ax = plt.subplots(figsize=(5, 3.5))
        top_comp.plot.barh(ax=ax, color='steelblue')
        ax.set_xlabel('Median Salary (USD)')
        ax.set_ylabel('')
        ax.invert_yaxis()
        plt.tight_layout()
        st.pyplot(fig, use_container_width=False)
        st.caption(
            "Countries are ranked by median salary: Compare bar "
            "lengths to see which company locations pay the most."
        )

    with loc_right:
        st.write("#### By Employee Residence")
        top_res = (df.groupby('employee_residence')['salary_usd']
                   .median().sort_values(ascending=False).head(12))
        fig, ax = plt.subplots(figsize=(5, 3.5))
        top_res.plot.barh(ax=ax, color='darkorange')
        ax.set_xlabel('Median Salary (USD)')
        ax.set_ylabel('')
        ax.invert_yaxis()
        plt.tight_layout()
        st.pyplot(fig, use_container_width=False)
        st.caption(
            "Compare with the company location chart: Differences "
            "highlight where employees live vs. where companies are based."
        )

    # --- Correlation Study ---
    st.write("---")
    st.write("## Correlation Study")
    st.info(
        "A combination of **Pearson** (for numerical features) and "
        "**Spearman** (for categorical features) correlation analyses "
        "was used to identify which job attributes are most strongly "
        "associated with salary. The key features that emerged are:"
    )

    st.write(
        "* **Experience Level** — strongest categorical predictor\n"
        "* **Years of Experience** — strongest numerical predictor\n"
        "* **Company Size** — secondary but significant predictor\n"
        "* **Remote Ratio** — no meaningful correlation with salary"
    )

    tab_yrs, tab_exp, tab_size, tab_remote = st.tabs([
        "Years of Experience", "Experience Level",
        "Company Size", "Remote Ratio"
    ])

    with tab_yrs:
        from scipy import stats
        r_pearson, _ = stats.pearsonr(
            df['years_experience'], df['salary_usd']
        )
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
        st.caption(
            "The red line shows the trend: A steeper slope means "
            "each extra year of experience adds more salary."
        )

    with tab_exp:
        fig, ax = plt.subplots(figsize=(4, 3))
        exp_order = ["EN", "MI", "SE", "EX"]
        exp_labels = {
            "EN": "Entry", "MI": "Mid",
            "SE": "Senior", "EX": "Executive"
        }
        corr_df = df.copy()
        corr_df["experience_label"] = corr_df["experience_level"].map(
            exp_labels
        )
        sns.boxplot(data=corr_df, x="experience_label", y="salary_usd",
                    order=[exp_labels[o] for o in exp_order], ax=ax)
        ax.set_xlabel("Experience Level")
        ax.set_ylabel("Salary (USD)")
        ax.set_title("Salary by Experience Level")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=False)
        st.caption(
            "Compare the box heights and medians: A clear staircase "
            "from Entry to Executive confirms experience drives salary."
        )

    with tab_size:
        fig, ax = plt.subplots(figsize=(4, 3))
        corr_size_labels = {"S": "Small", "M": "Medium", "L": "Large"}
        corr_df = df.copy()
        corr_df["size_label"] = corr_df["company_size"].map(corr_size_labels)
        sns.boxplot(data=corr_df, x="size_label", y="salary_usd",
                    order=["Small", "Medium", "Large"], ax=ax)
        ax.set_xlabel("Company Size")
        ax.set_ylabel("Salary (USD)")
        ax.set_title("Salary by Company Size")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=False)
        st.caption(
            "Look for differences in median lines: Larger companies "
            "tend to pay more, but the overlap is substantial."
        )

    with tab_remote:
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.boxplot(data=df, x='remote_ratio', y='salary_usd',
                    order=[0, 50, 100], ax=ax)
        ax.set_title('Salary by Remote Ratio')
        ax.set_xlabel('Remote Ratio (0=On-site, 50=Hybrid, 100=Remote)')
        ax.set_ylabel('Salary (USD)')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=False)
        st.caption(
            "Nearly identical boxes confirm that remote, hybrid, and "
            "on-site roles pay about the same in AI."
        )

    # --- Parallel Categories ---
    st.write("---")
    st.write("## Parallel Categories Plot")

    salary_map = [-np.inf, 60000, 100000, 140000, np.inf]
    disc = ArbitraryDiscretiser(binning_dict={'salary_usd': salary_map})

    df_parallel = disc.fit_transform(
        df[['salary_usd', 'experience_level', 'remote_ratio',
            'company_size']].copy()
    )

    # numeric column used for coloring
    df_parallel['salary_band'] = df_parallel['salary_usd']

    # readable labels for display
    labels_map = {0: '<$60k', 1: '$60k-$100k', 2: '$100k-$140k', 3: '>$140k'}
    df_parallel['salary_usd'] = df_parallel['salary_usd'].replace(labels_map)

    fig = px.parallel_categories(
        df_parallel,
        dimensions=['salary_usd', 'experience_level', 'remote_ratio',
                    'company_size'],
        color='salary_band',
        color_continuous_scale=[
            [0, '#2166ac'],
            [0.33, '#67a9cf'],
            [0.66, '#ef8a62'],
            [1, '#b2182b'],
        ],
        width=950,
        height=500,
        title="Parallel Categories: "
              "Salary x Experience x Remote x Company Size"
    )
    fig.update_coloraxes(
        colorbar_title_text="Salary Band",
        colorbar_tickvals=[0, 1, 2, 3],
        colorbar_ticktext=['<$60k', '$60k-$100k', '$100k-$140k', '>$140k'],
    )

    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Hover over ribbons to trace how salary bands connect to "
        "experience, remote ratio, and company size."
    )

    st.info(
        "**How to read this plot:** Each vertical axis represents a "
        "feature (salary band, experience level, remote ratio, company "
        "size). The coloured ribbons connect the categories that each "
        "row in the dataset belongs to, making it easy to spot patterns "
        "at a glance.\n\n"
        "**Key takeaways:**\n"
        "* Senior (SE) and Executive (EX) roles are heavily concentrated "
        "in the higher salary bands (>$140k), while Entry-level (EN) "
        "roles cluster in the lower bands.\n"
        "* Remote ratio ribbons spread fairly evenly across all salary "
        "bands, confirming that work arrangement does not drive salary.\n"
        "* Large (L) companies contribute more to the higher salary "
        "bands than small (S) companies, consistent with the "
        "correlation findings above."
    )
