import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px


def page_cluster_body():

    version = "v1"
    cluster_silhouette = plt.imread(
        f"outputs/ml_pipeline/cluster_analysis/{version}/clusters_silhouette.png")
    features_to_cluster = plt.imread(
        f"outputs/ml_pipeline/cluster_analysis/{version}/features_define_cluster.png")
    cluster_profile = pd.read_csv(
        f"outputs/ml_pipeline/cluster_analysis/{version}/clusters_profile.csv")
    cluster_features = (
        pd.read_csv(
            f"outputs/ml_pipeline/cluster_analysis/{version}/TrainSet.csv")
        .columns
        .to_list()
    )

    # dataframe for cluster_distribution_per_variable()
    df_train = pd.read_csv(
        f"outputs/ml_pipeline/cluster_analysis/{version}/TrainSet.csv")
    df_salary_band = _load_salary_band()
    df_salary_vs_clusters = df_salary_band.copy()
    df_salary_vs_clusters["Clusters"] = _assign_cluster_labels(df_train)

    st.write("### ML Pipeline: Cluster Analysis")
    st.info(
        "* We refitted the cluster pipeline using fewer variables, and it "
        "delivered equivalent performance to the pipeline fitted using all "
        "variables.\n"
        "* The pipeline average silhouette score is **0.53**, indicating "
        "**weak cluster structure** with significant overlap between groups."
    )
    st.write("---")

    st.write("#### Cluster ML Pipeline steps")
    st.write(
        "The pipeline uses the following steps:\n"
        "1. **OrdinalMappingEncoder** — encodes experience_level, "
        "education_required, company_size\n"
        "2. **OneHotEncoder** — encodes employment_type\n"
        "3. **FrequencyEncoder** — encodes job_title, company_location, "
        "employee_residence, industry\n"
        "4. **SmartCorrelatedSelection** — removes correlated features "
        "(Spearman ρ > 0.8)\n"
        "5. **StandardScaler** — scales all features\n"
        "6. **PCA** — reduces to 6 components\n"
        "7. **KMeans** — 3 clusters"
    )

    st.write("#### The features the model was trained with")
    st.write(cluster_features)

    st.write("#### Clusters Silhouette Plot")
    st.image(cluster_silhouette)
    st.info(
        "The silhouette plot measures how similar each data point is to its own "
        "cluster compared to neighbouring clusters. Values range from -1 (wrong "
        "cluster) to +1 (well-matched). The average silhouette score of **0.53** "
        "indicates moderate cluster structure — the clusters capture some real "
        "differences in the data, but there is notable overlap between segments. "
        "This is expected given the continuous nature of salary and location data."
    )

    cluster_distribution_per_variable(
        df=df_salary_vs_clusters, target="SalaryBand")

    st.write("#### Most important features to define a cluster")
    st.image(features_to_cluster)
    st.info(
        "This chart shows which features have the greatest influence on cluster "
        "assignment. **Company location** and **employee residence** are the "
        "dominant drivers, confirming that the clusters primarily segment "
        "professionals by geographic market rather than by job role or seniority."
    )

    st.write("#### Cluster Profile")
    statement = (
        "* Historically, **Cluster 0** contains professionals from smaller "
        "markets (Romania, Argentina, Vietnam) with a **balanced salary "
        "distribution** across Low, Mid and High bands.\n"
        "* **Cluster 1** groups professionals from European markets "
        "(Germany, Netherlands, Austria) with a slight lean towards "
        "**lower salaries**.\n"
        "* **Cluster 2** groups professionals from markets like Ireland, "
        "Canada, and France with a slight lean towards **mid-to-high "
        "salaries**.\n"
        "* **One potential action:** when evaluating a job offer, compare "
        "the offered salary against the typical band for your cluster. "
        "If the offer falls in the Low band for a Cluster 2 profile, "
        "there may be room for negotiation."
    )
    st.info(statement)

    statement = (
        "* The cluster profile interpretation allowed us to label the "
        "clusters as follows:\n"
        "* **Cluster 0** — Emerging-market AI professionals with balanced "
        "salary spread.\n"
        "* **Cluster 1** — European-market AI professionals, slightly "
        "lower-paid.\n"
        "* **Cluster 2** — Established-market AI professionals, slightly "
        "higher-paid."
    )
    st.success(statement)

    # hide index in st.table()
    cluster_profile.index = [" "] * len(cluster_profile)
    st.table(cluster_profile)

    st.write("---")
    st.write("#### Limitations & Next Steps")
    st.warning(
        "The silhouette score of **0.53** indicates weak cluster separation. "
        "The AI salary dataset is **continuously distributed** — salary is "
        "driven by a smooth gradient of experience, location, and company "
        "size rather than discrete segments.\n\n"
        "**Alternative approaches** (residual clustering, UMAP + HDBSCAN, "
        "K-Prototypes) were explored. The most actionable result was "
        "**residual clustering**, which classifies roles as Overpaid, Fair, "
        "or Underpaid relative to model expectations. This is integrated "
        "into the **Predict Salary** page as market positioning."
    )


def _load_salary_band():
    """Derive SalaryBand from the raw dataset."""
    import numpy as np
    df = pd.concat([
        pd.read_csv("outputs/datasets/cleaned/TrainSet.csv"),
        pd.read_csv("outputs/datasets/cleaned/TestSet.csv"),
    ], ignore_index=True)

    q33, q67 = df["salary_usd"].quantile([0.33, 0.67])
    df["SalaryBand"] = pd.cut(
        df["salary_usd"],
        bins=[-np.inf, q33, q67, np.inf],
        labels=["Low", "Mid", "High"],
    ).astype("object")
    return df[["SalaryBand"]]


def _assign_cluster_labels(df_train):
    """Re-fit KMeans on the saved TrainSet to get cluster labels."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_train)
    km = KMeans(n_clusters=3, random_state=0)
    return km.fit_predict(X_scaled)


def cluster_distribution_per_variable(df, target):

    df_bar_plot = (df
                   .groupby(["Clusters", target])
                   .size()
                   .reset_index(name="Count"))
    df_bar_plot.columns = ["Clusters", target, "Count"]
    df_bar_plot[target] = df_bar_plot[target].astype("object")

    st.write(f"#### Clusters distribution across {target} levels")
    fig = px.bar(df_bar_plot, x="Clusters", y="Count",
                 color=target, width=800, height=350)
    fig.update_layout(xaxis=dict(tickmode="array",
                      tickvals=df["Clusters"].unique()))
    st.plotly_chart(fig)

    df_relative = (
        df
        .groupby(["Clusters", target])
        .size()
        .unstack(fill_value=0)
        .apply(lambda x: 100 * x / x.sum(), axis=1)
        .stack()
        .reset_index(name="Relative Percentage (%)")
        .sort_values(by=["Clusters", target])
    )
    df_relative.columns = ["Clusters", target, "Relative Percentage (%)"]

    st.write(f"#### Relative Percentage (%) of {target} in each cluster")
    fig = px.line(df_relative, x="Clusters", y="Relative Percentage (%)",
                  color=target, width=800, height=350)
    fig.update_layout(xaxis=dict(tickmode="array",
                      tickvals=df["Clusters"].unique()))
    fig.update_traces(mode="markers+lines")
    st.plotly_chart(fig)
