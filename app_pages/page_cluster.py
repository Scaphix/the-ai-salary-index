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

    st.write("# ML Cluster Model Performance")
    st.info(
        "**Business Requirement 3**: The client wants to segment AI job "
        "postings into meaningful market clusters based on shared "
        "attributes, to reveal distinct salary segments and labour "
        "market patterns.\n\n"
        "**Model**: KMeans Clustering with PCA. PCA is applied "
        "beforehand to reduce dimensionality and balance feature "
        "contributions.\n\n"
        "* The pipeline average silhouette score is **0.16**, indicating "
        "**modest cluster structure**. While the score is modest, the "
        "resulting 4-cluster solution produces highly interpretable "
        "market segments with clear salary differentiation."
    )
    st.write("---")

    st.write("## Cluster ML Pipeline steps")
    st.write(
        "The pipeline uses the following steps:\n"
        "1. **OrdinalMappingEncoder** encodes experience_level, "
        "education_required, company_size\n"
        "2. **CountFrequencyEncoder** (count) encodes job_title, "
        "company_location, employee_residence, industry\n"
        "3. **StandardScaler** scales all features\n"
        "4. **PCA:** keeps 99% of variance (auto-selects components)\n"
        "5. **KMeans:** 4 clusters (init='random', n_init=10, "
        "max_iter=300)"
    )

    st.write("## The features the model was trained with")
    st.write(", ".join(cluster_features))

    st.write("---")

    st.write("## Model Performance")

    st.write("### Classifier Performance")
    st.write(
        "A GradientBoostingClassifier was trained to predict "
        "cluster labels from the original features, validating "
        "that clusters are **explainable** by the input variables."
    )

    clf_metrics = pd.DataFrame({
        "Cluster": ["0 — India Market", "1 — Junior Pipeline",
                     "2 — Senior Premium", "3 — Emerging Markets",
                     "Overall Accuracy"],
        "Train Precision": ["1.00", "1.00", "1.00", "1.00", ""],
        "Train Recall": ["1.00", "1.00", "1.00", "1.00", ""],
        "Train F1": ["1.00", "1.00", "1.00", "1.00", "**1.00**"],
        "Test Precision": ["1.00", "1.00", "1.00", "1.00", ""],
        "Test Recall": ["1.00", "1.00", "1.00", "1.00", ""],
        "Test F1": ["1.00", "1.00", "1.00", "1.00", "**1.00**"],
    })
    clf_metrics.index = [" "] * len(clf_metrics)
    st.table(clf_metrics)

    st.success(
        "The classifier achieves **100% accuracy** on both "
        "train (11,760 samples) and test (2,941 samples) sets. "
        "This confirms the clusters are fully determined by the "
        "input features — the PCA + KMeans groupings are stable "
        "and reproducible."
    )

    st.write("### Cluster Size Distribution")
    cluster_sizes = pd.DataFrame({
        "Cluster": [0, 1, 2, 3],
        "Label": ["India Market", "Junior Pipeline",
                  "Senior Premium", "Emerging Markets"],
        "Train Samples": [455, 2086, 4597, 4622],
        "Test Samples": [128, 512, 1128, 1173],
        "Share (%)": ["4%", "18%", "39%", "39%"],
    })
    cluster_sizes.index = [" "] * len(cluster_sizes)
    st.table(cluster_sizes)

    st.write("### Clusters Silhouette Plot")
    st.image(cluster_silhouette)
    st.info(
        "The silhouette plot measures how similar each "
        "data point is to its own cluster compared to "
        "neighbouring clusters. Values range from -1 "
        "(wrong cluster) to +1 (well-matched). The "
        "average silhouette score of **0.16** indicates "
        "modest cluster structure. While the score is modest, "
        "the 4-cluster solution produces highly interpretable "
        "market segments with clear salary differentiation. "
        "Interpretability was prioritised alongside the "
        "quantitative metric."
    )

    cluster_distribution_per_variable(
        df=df_salary_vs_clusters, target="SalaryBand")

    st.write("## Feature Importance")
    st.image(features_to_cluster)
    st.info(
        "This chart shows which features have the "
        "greatest influence on cluster assignment. "
        "**employee_residence** is the strongest cluster-defining "
        "feature (importance = 0.46), followed by "
        "**experience_level** (0.24), **years_experience** (0.22), "
        "and **company_location** (0.07) confirming that geography "
        "and experience are the dominant axes of AI salary segmentation."
    )

    st.write("## Cluster Profile")
    statement = (
        "* **Cluster 0 : India Market (~4%):** India-centric roles "
        "(employee residence: India 74%, company location: India 100%). "
        "Mixed experience levels (MI 31%, EX 30%, SE 29%). "
        "Salary is 93% Low, 7% Mid. Despite having senior professionals, "
        "geography is the dominant salary factor for this group.\n"
        "* **Cluster 1 : Junior/Entry Developed Markets (~18%):** "
        "1–3 years experience, MI 51% / EN 49%. Located in developed "
        "markets (Ireland, Switzerland, France, Canada). "
        "Salary is 57% Low, 41% Mid, 2% High. Early-career "
        "professionals earning less, as expected.\n"
        "* **Cluster 2 : Senior Professionals (~39%):** "
        "7–15 years experience, SE 50% / EX 50%. Located across "
        "developed markets (China, Singapore, Germany). "
        "Salary is 66% High, 30% Mid, 4% Low. Experience drives "
        "these professionals into premium pay.\n"
        "* **Cluster 3 : Mid-career Emerging Markets (~39%):** "
        "2–9 years experience, mixed levels (MI 26%, SE 25%, EN 25%). "
        "Located in emerging markets (Romania, Vietnam, Indonesia). "
        "Salary is nearly uniform: Mid 35%, High 34%, Low 31%. "
        "Experience still plays a role within this geographic group.\n"
        "* **One potential action:** when evaluating a job offer, compare "
        "the offered salary against the typical band for your cluster. "
        "If the offer falls in the Low band for a Cluster 2 profile, "
        "there may be room for negotiation."
    )
    st.info(statement)

    st.success(
        "**Key insight:** PCA Component 0 captures geographic variation "
        "(vertical columns correspond to different employee residence "
        "groups), while PCA Component 1 captures experience and seniority "
        "(senior profiles at the top, junior profiles at the bottom)."
    )

    # hide index in st.table()
    cluster_profile.index = [" "] * len(cluster_profile)
    st.table(cluster_profile)

    st.write("---")
    st.write("## Limitations & Next Steps")
    st.warning(
        "The silhouette score of **0.16** indicates modest cluster separation. "
        "The AI salary dataset is **continuously distributed**, salary is "
        "driven by a smooth gradient of experience, location, and company "
        "size rather than discrete segments.\n\n"
        "Hyperparameter tuning (192 combinations) improved the score from "
        "0.145 to 0.159 and surfaced a 4th cluster (India market) that was "
        "hidden at k=3.\n\n"
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
    st.caption(
        "Compare bar heights across clusters: taller bars in a "
        "salary band show where that segment is concentrated."
    )

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
    st.caption(
        "Lines crossing mean the salary-band mix shifts between "
        "clusters: look for which band dominates each cluster."
    )
