# Notebook 06 : Clustering Journey Summary

This document records the iterative development of the clustering model used in Business Requirement 3 (Market Segmentation). It traces every major decision, pipeline redesign, and debugging step from the initial setup through to the final 4-cluster solution, providing transparency into why certain approaches were adopted or discarded.

> **Source notebook:** `jupyter_notebooks/06 - ModelingEvaluation - Cluster.ipynb`

---

## Phase 1 : Initial Setup

- Created the notebook following the course material structure.
- Built a full KMeans clustering pipeline using: `OrdinalEncoder` → `OneHotEncoder` → `CountFrequencyEncoder` → `SmartCorrelatedSelection` → `StandardScaler` → `PCA(n_components=6)` → `KMeans(k=3)`.
- Included PCA analysis, elbow/silhouette evaluation, a GradientBoosting classifier to identify the most important cluster-defining features, cluster profiling, and SalaryBand distribution analysis.

## Phase 2 : Expanding Cluster Selection Methods & Bug Fix 

- Added 5 different cluster selection methods (elbow, silhouette, gap statistic, etc.) to rigorously evaluate optimal k.
- **Encountered BUG #1:** `IndexError: index 0 is out of bounds for axis 0 with size 0` in the `Clusters_IndividualDescription` function. The root cause was assigning a scalar string to a column of an empty DataFrame. The numeric branch didn't create a row, so when the object branch tried `.values[0]`, it failed. Fixed by wrapping all column assignments in a list `[value]`.
- Documented the bug and fix in `DEBUGGING.md`.

## Phase 3 : Initial Conclusion: No Natural Clusters

- After evaluating all 5 methods (GMM, DBSCAN, UMAP, Gower distance, and different encoding strategies), concluded that **no strong natural cluster structure existed** in the data. The PCA scatter plot showed one large overlapping blob with centroids too close together.
- Saved initial pipeline artifacts (TrainSet.csv, clusters_profile.csv, silhouette plot, feature importance chart).

## Phase 4 : Pipeline Redesign

Made three key changes based on the Phase 3 findings:

1. **Dropped `employment_type`** — after one-hot encoding, the rare PT flag dominated PCA variance and created artificial clusters based on contract type rather than meaningful market segments.
2. **Removed `SmartCorrelatedSelection`** — PCA handles correlation naturally by compressing correlated features into shared components, making explicit correlation removal redundant.
3. **Switched PCA to `n_components=0.99`** (variance-based) instead of a fixed number of components — auto-selects components explaining 99% of variance.
4. **Reduced to 3 clusters** initially as part of the simplification.

## Phase 5 : Hyperparameter Tuning & k=4

- Tuned 6 hyperparameters: `n_components`, `n_clusters`, `init`, `n_init`, `max_iter`, `encoding_method`.
- **k=4 emerged as optimal** (silhouette score 0.1585, +0.0132 over default), surfacing a distinct India labor market segment that was hidden at k=3.
- Best hyperparameters: `n_components=0.99`, `n_clusters=4`, `init='random'`, `n_init=10`, `max_iter=300`, `encoding_method='count'`.
- Saved the final pipeline (`pipeline_cluster.pkl`) and updated artifacts.
- While the silhouette score remained modest (0.16), the 4-cluster solution produced highly interpretable market segments with clear salary differentiation.

## Phase 6 : Dashboard Integration

- Created `app_pages/page_cluster.py` to display cluster results in the Streamlit dashboard.
- Refined the page layout, added `st.caption` to help users read the plots.
- Fixed PEP 8 validation errors across the codebase.

## Phase 7 : Analysis & Interpretation

- Wrote cluster profile descriptions and the final study summary.
- Added the `cluster_distribution_per_variable()` function to visualize absolute and relative salary distributions across clusters.
- Corrected cluster numbering mismatches between the data table output and the written analysis (Clusters 1, 2, 3 had been swapped in the narrative).

### Final Cluster Mapping

| Cluster | Label | Experience | Salary |
|---------|-------|------------|--------|
| 0 | India Market (~4%) | Mixed (MI/EX/SE), 3–11 yrs | Low (93%) |
| 1 | Junior Pipeline (~18%) | MI/EN, 1–3 yrs | Low (57%), Mid (41%) |
| 2 | Senior Premium (~39%) | SE/EX, 7–15 yrs | High (66%), Mid (30%) |
| 3 | Emerging Market Mix (~39%) | Mixed, 2–9 yrs | Even split (35/34/31%) |

### Key Findings

- **Employee residence** is the strongest cluster-defining feature (importance = 0.46)
- **Experience level** (0.24) and **years_experience** (0.22) are the next most important
- India forms its own distinct labor market where geography overrides experience as a salary determinant
- Job title, industry, education, and company size have negligible influence on cluster formation

### Final Pipeline

`OrdinalMapping` → `FrequencyEncoding` → `StandardScaler` → `PCA(0.99)` → `KMeans(k=4)`

[BACK TO README.md](Readme.md).