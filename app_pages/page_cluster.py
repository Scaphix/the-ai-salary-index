import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from src.data_management import load_pkl_file


def page_cluster_body():

    version = "v1"
    path = f"outputs/ml_pipeline/cluster_analysis/{version}"

    pipeline_cluster = load_pkl_file(f"{path}/pipeline_cluster.pkl")
    cluster_silhouette = plt.imread(f"{path}/clusters_silhouette.png")
    features_to_cluster = plt.imread(f"{path}/features_define_cluster.png")
    cluster_profile = pd.read_csv(f"{path}/clusters_profile.csv")

    df_train = pd.read_csv(f"{path}/TrainSet.csv")
    cluster_features = df_train.columns.to_list()

    # Use the saved pipeline's labels instead of re-fitting
    df_salary_band = _load_salary_band()
    df_salary_vs_clusters = df_salary_band.copy()
    df_salary_vs_clusters["Clusters"] = pipeline_cluster["model"].labels_

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
        "4. **StandardScaler** — scales all features\n"
        "5. **PCA** — keeps 95% of variance (auto-selects components)\n"
        "6. **KMeans** — 3 clusters"
    )

    st.write("#### The features the model was trained with")
    st.write(cluster_features)

    st.write("#### Clusters Silhouette Plot")
    st.image(cluster_silhouette)
    st.info(
        "The silhouette plot measures how similar each "
        "data point is to its own cluster compared to "
        "neighbouring clusters. Values range from -1 "
        "(wrong cluster) to +1 (well-matched). The "
        "average silhouette score of **0.53** indicates "
        "moderate cluster structure — the clusters "
        "capture some real differences in the data, but "
        "there is notable overlap between segments. "
        "This is expected given the continuous nature "
        "of salary and location data."
    )

    cluster_distribution_per_variable(
        df=df_salary_vs_clusters, target="SalaryBand")

    st.write("#### Most important features to define a cluster")
    st.image(features_to_cluster)
    st.info(
        "This chart shows which features have the "
        "greatest influence on cluster assignment. "
        "**Company location** and **employee "
        "residence** are the dominant drivers, "
        "confirming that the clusters primarily "
        "segment professionals by geographic market "
        "rather than by job role or seniority."
    )

    st.write("#### Cluster Profile")
    statement = (
        "* **Cluster 0 (The India Segment — ~4%):** India-centric roles. "
        "Salary band is 93% Low. Geography remains the strongest "
        "penalty — when employee and company are both in India, salaries "
        "are almost always low regardless of experience level.\n"
        "* **Cluster 1 (Senior Professionals — ~48%):** 7-15 years "
        "experience, SE/EX level. Distributed across developed markets. "
        "Salary is 66% High, 30% Mid. Experience drives these "
        "professionals into premium pay.\n"
        "* **Cluster 2 (Junior/Entry — ~48%):** 1-3 years experience, "
        "MI/EN level. Also in developed markets. Salary is 57% Low, "
        "40% Mid. Early-career professionals earning less, as expected.\n"
        "* **One potential action:** when evaluating a job offer, compare "
        "the offered salary against the typical band for your cluster. "
        "If the offer falls in the Low band for a Cluster 1 profile, "
        "there may be room for negotiation."
    )
    st.info(statement)

    st.success(
        "**Key insight:** PCA revealed that experience "
        "(PC1 = 48% of variance) is the dominant axis, while geography "
        "(PC2-PC3) creates the secondary separation."
    )

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
