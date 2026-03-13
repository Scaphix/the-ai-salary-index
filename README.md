# [The AI Salary Index](https://tasi-the-ai-salary-index-361138766018.herokuapp.com/page_cluster_body)

## Overview

Pricing AI/ML talent is hard — experience, location, and company size all pull salary in different directions. **The AI Salary Index** uses 15,000+ global job listings to give recruiters three things:

1. **Salary Prediction** — A Gradient Boosting model (R² = 0.86) predicts salary from a candidate's profile.
2. **Market Segmentation** — PCA + K-Means clustering groups roles into three segments split by experience and geography, revealing where pay premiums and penalties lie.
3. **Data-driven insights** — Correlation analysis and hypothesis testing identify which attributes actually drive compensation.


## Dataset Content

The dataset is sourced from Kaggle:
[Global AI Job Market & Salary Trends 2025](https://www.kaggle.com/datasets/bismasajjad/global-ai-job-market-and-salary-trends-2025)
(dataset: `ai_job_dataset1.csv`).

It contains **15,000+ synthetic AI job listings** from 50+ countries, representing the global AI/ML labour market as of 2025. Each row is one job posting.

| Variable | Meaning | Units |
|---|---|---|
| `job_id` | Unique posting identifier | — |
| `job_title` | Job title (e.g., "ML Engineer") | Text |
| `job_category` | Broad category (e.g., "Data Science") | Categorical |
| `salary_usd` | Annual compensation | USD |
| `salary_currency` | Local currency of salary | ISO code |
| `salary_local` | Salary in local currency | Currency units |
| `experience_level` | EN / MI / SE / EX (Entry → Executive) | Ordinal |
| `employment_type` | FT / PT / CT / FL (Full-time → Freelance) | Categorical |
| `company_size` | S / M / L (Small / Medium / Large) | Ordinal |
| `company_location` | Company HQ country code | ISO 2-letter |
| `employee_residence` | Employee home country code | ISO 2-letter |
| `remote_ratio` | 0 = on-site, 50 = hybrid, 100 = remote | Ordinal (0/50/100) |
| `years_experience` | Years of experience required | Integer |
| `education_required` | Minimum education level | Ordinal text |
| `benefits_score` | Composite benefits rating | 0–10 float |
| `job_description_length` | Character count of description | Integer |
| `required_skills` | Comma-separated skill tags | Text list |
| `posting_date` | Date listing was posted | Date |
| `application_deadline` | Application close date | Date |


## Business Requirements

From a business perspective, this project helps recruiting firms optimise compensation strategy, accelerate hiring decisions, and reduce recruitment costs through data-driven salary intelligence.

---

**Business Requirement 1 — Data Visualisation & Correlation Study**
- Identify which attributes correlate most closely with AI/ML salary levels.

**Business Requirement 2 — Salary Prediction**
- Predict salary for a given candidate profile using a regression model.

**Business Requirement 3 — Market Segmentation**
- Group AI roles into market-based clusters to reveal distinct salary segments and labour market patterns.

## Hypotheses and how to validate them?

To better understand the factors influencing salary levels in the AI/ML job market, four key hypotheses were formulated based on domain knowledge and the available data. Each hypothesis focuses on a variable expected to impact salary prediction.

| Hypothesis | Rationale | Validation |
|------------|------------|-------------|
| **H1:** Higher `experience_level` is associated with higher `salary_usd` | Candidates with more experience (Entry → Executive) should command higher compensation | Visualize salary distribution by experience level, conduct ANOVA or regression analysis to confirm positive correlation |
| **H2:** Remote roles (`remote_ratio` = 100) have different salary expectations than hybrid/on-site roles | Geographic flexibility may impact salary—either premium for remote or discount based on cost-of-living | Visualize salary distribution by remote_ratio categories, conduct hypothesis test for salary difference |
| **H3:** `years_experience` required is positively correlated with `salary_usd` | Roles demanding more years of experience should offer higher compensation to attract experienced talent | Scatter plot with regression line, calculate correlation coefficient to confirm strength of relationship |
| **H4:** Job category (`job_category`) influences salary levels | Different AI/ML domains (e.g., Data Science vs. ML Engineering) may have distinct salary markets | Analyze mean salary by job_category, perform Kruskal-Wallis test for differences across categories |

These hypotheses will be tested through exploratory data analysis and statistical testing to identify whether the respective features are influential predictors of salary in the AI/ML market.


### **Visual Flow**
```
Salary Study (Explore AI/ML compensation data)
    ↓
Hypotheses (What drives salary?)
    ↓
Exploratory Data Analysis (Validate with visualisations & statistics)
    ↓
Feature Selection (Include only validated features)
    ↓
BR1: Correlation Study → BR2: Salary Prediction → BR3: Market Segmentation
```

## Mapping Business Requirements to Data Visualisations and ML Tasks

**BR1 — Data Visualisation & Correlation Study**
- Salary distribution plots by experience level, location, and remote ratio
- Pearson/Spearman correlation analysis and hypothesis testing (H1–H4)
- Output: Identify which attributes drive AI/ML compensation

**BR2 — Salary Prediction (Regression)**
- GradientBoostingRegressor trained on candidate profile features (experience, location, company size)
- Feature importance assessment and model evaluation (R², MAE, RMSE)
- Output: Predicted salary with confidence range

**BR3 — Market Segmentation (Clustering)**
- PCA for dimensionality reduction + K-Means (k=3) to segment roles
- Cluster profiling by experience level, geography, and salary band
- Output: Three market segments with distinct compensation patterns

---

## ML Business Cases

### **Business Case 2: Salary Prediction (Regression)**

| | |
|---|---|
| **ML Task** | Predict annual salary (USD) from candidate profile attributes |
| **Model Type** | Supervised regression (GradientBoostingRegressor) |
| **Ideal Outcome** | Accurate salary recommendations that reduce under-bidding and over-compensation |
| **Success Metrics** | R² ≥ 0.75 on train & test; RMSE ≤ 15% of mean salary; train–test R² gap ≤ 0.05 |
| **Failure Conditions** | R² < 0.65 on test; RMSE > 20% of mean salary; predictions off by >20% for >30% of placements |
| **Output** | Predicted salary (USD), confidence interval, top features driving the prediction |
| **Target Variable** | `salary_usd` |
| **Features** | `experience_level`, `company_size`, `company_location`, `employee_residence` |
| **Data** | ~15,000 job postings from Kaggle (`ai_job_dataset1.csv`) |

---

### **Business Case 3: Market Segmentation (Clustering)**

| | |
|---|---|
| **ML Task** | Group AI roles into market segments based on profile features (excluding salary) |
| **Model Type** | Unsupervised clustering (PCA + K-Means, k=3) |
| **Ideal Outcome** | Distinct segments that reveal experience- and geography-based compensation patterns |
| **Success Metrics** | Silhouette score ≥ 0.45; 3–6 interpretable clusters; min 500 samples per cluster |
| **Failure Conditions** | Silhouette < 0.35; clusters lack meaningful profile or salary differentiation |
| **Output** | Cluster ID per role, cluster profiles (experience, geography, salary band distribution) |
| **Features** | All profile features except `salary_usd`, `required_skills`, `company_name` |
| **Data** | Same dataset, train + test combined (14,701 rows) |


## Epics & User Stories

### Epic 1 — Data Insights (BR1)

- **US 1.1** — As a recruiter, I can view a project summary page to understand the dataset, terminology, and business requirements.
  - **Viz Task**: Data profiling in a Streamlit information page.

- **US 1.2** — As a recruiter, I can view an interactive salary study to understand how salary varies by experience level, company size, and location.
  - **Viz Task**: Box plots, histograms with KDE, parallel categories plot (Plotly).

- **US 1.3** — As a recruiter, I can review validated hypotheses to trust which factors genuinely influence AI salaries.
  - **Statistical Task**: Kruskal-Wallis H-test, Spearman and Pearson correlation with supporting visualisations.

### Epic 2 — Salary Prediction (BR2)

- **US 2.1** — As a recruiter, I can input a candidate's profile (experience level, company location, employee residence, company size) and get an instant predicted salary in USD.
  - **ML Task**: GradientBoostingRegressor pipeline served through a Streamlit form.

- **US 2.2** — As a data practitioner, I can view model performance metrics (R², MAE, RMSE) and feature importance to evaluate the model's accuracy and identify which variables drive predictions.
  - **Viz Task**: Train vs test evaluation metrics and feature importance bar chart.

### Epic 3 — Market Segmentation (BR3)

- **US 3.1** — As a recruiter, I can see which market segment a role belongs to so that I can understand its compensation context.
  - **ML Task**: PCA + K-Means clustering (k=3) assigning roles to experience/geography-based segments.

- **US 3.2** — As a data practitioner, I can understand how clusters are derived and their limitations.
  - **Viz Task**: Silhouette plot, cluster profiles, and distribution charts with methodology explanation.

### User Stories to Notebook & Dashboard Page Mapping

| Epic | User Story | Notebook | Dashboard Page |
|---|---|---|---|
| 1 | US 1.1 | 01-DataCollection | Quick Project Summary |
| 1 | US 1.2 | 02-JobMarketStudy | AI Salary Study |
| 1 | US 1.3 | 02-JobMarketStudy | Project Hypothesis |
| 2 | US 2.1 | 05-ModelingEvaluation-PredictSalary | Predict Salary |
| 2 | US 2.2 | 05-ModelingEvaluation-PredictSalary | Predict Salary |
| 3 | US 3.1 | 06-ModelingEvaluation-Cluster | Cluster Analysis |
| 3 | US 3.2 | 06-ModelingEvaluation-Cluster | Cluster Analysis |


## Dashboard Design

The dashboard is developed in Streamlit and designed to guide the user from business understanding to actionable insights and model-based predictions.
It consists of five main pages, each mapped to specific business requirements.

### Page 1: Quick Project Summary
**Purpose**: Provide a clear overview of the project and orient users.

**Sections**:
- General information about the AI job market and why data-driven salary decisions matter
- Project terms and jargon (experience levels, remote ratio, company size definitions)
- Dataset overview (15,000+ AI job listings from 50+ countries)
- Business requirements (BR1: Salary Prediction, BR2: Market Segmentation)

### Page 2: AI Salary Study
**Purpose**: Address **Business Requirement 1** (Data Insights). This page helps recruiters understand what drives AI salaries. It focuses on identifying key job attributes most correlated with salary and provides visual and statistical insights.

**Sections**:
- Dataset inspection: checkbox to display shape, column types, and first rows of the dataset
- KPI metrics: total jobs analysed, median salary, top-paying country, most common experience level
- Salary distribution analysis: histogram showing the right-skewed salary distribution with median vs mean comparison
- Salary by location: bar charts of top 12 countries by median salary for both company location and employee residence
- Correlation study: Pearson and Spearman heatmaps with analysis of experience level, years of experience, company size, and remote ratio
- Parallel categories plot: interactive Plotly visualisation showing salary flow across experience level, remote ratio, and company size

### Page 3: Project Hypothesis and Validation
**Purpose**: Present the four project hypotheses and their validation outcomes. 

**Sections**:
- **H1** — Experience level is the dominant salary driver: verdict (Confirmed), boxplot, statistical evidence
- **H2** — Remote roles have different salary expectations: verdict (Rejected), boxplot, statistical evidence
- **H3** — Years of experience positively correlates with salary: verdict (Confirmed), scatter plot with Pearson r
- **H4** — Company size is a significant salary driver: verdict (Confirmed), boxplot, statistical evidence
- Each hypothesis includes a plain-English conclusion and checkbox to reveal the supporting visualisation

### Page 4: Predict Salary
**Purpose**: Address **Business Requirement 2** (Salary Prediction). Allows users to input a candidate profile and receive a predicted annual salary with contextual insights.

**Sections**:
- Pipeline explanation: two-stage pipeline overview (data cleaning/feature engineering + regression model)
- Model performance metrics: R² and MAE for both train and test sets, with success/failure statement
- Feature importance: bar chart showing the top predictive features (experience_level > 50%)
- Live prediction interface: dropdown widgets for experience level, company size, company location, employee residence
- Prediction output: predicted salary with salary tier context and actionable takeaways

### Page 5: Cluster Analysis
**Purpose**: Address **Business Requirement 3** (Market Segmentation). Shows cluster analysis performance and interpretation, segmenting AI professionals by geographic market.

**Sections**:
- ML pipeline overview: encoding, feature selection, scaling, PCA, KMeans steps
- Model performance: silhouette score (0.53) with honest assessment of cluster separation
- Silhouette plot: visualisation of cluster cohesion
- Cluster distribution: interactive bar and line charts showing cluster breakdown across salary levels
- Features defining clusters: bar chart of most important features for cluster assignment
- Cluster profiles: table with per-cluster statistics and plain-English interpretation of each segment (Emerging-market, European-market, Established-market professionals)
- Limitations & next steps: honest note on weak cluster separation and alternative approaches considered

## Technologies Used

- **Python 3.13.2**
- **Pandas** — data manipulation and analysis
- **NumPy** — numerical computation
- **Matplotlib / Seaborn** — static data visualisation
- **Plotly** — interactive data visualisation
- **Scikit-Learn** — ML pipelines, preprocessing, model training, evaluation
- **XGBoost** — gradient boosting (evaluated during model selection)
- **Feature Engine** — feature engineering transformations
- **SciPy** — statistical tests for hypothesis validation
- **Streamlit** — interactive web dashboard
- **Joblib** — pipeline serialisation
- **Heroku** — cloud deployment platform
- **Git / GitHub** — version control

## Testing

### Manual Testing

#### User Story Testing
* Dashboard was manually tested using user stories as a basis for determining success.
* Jupyter notebooks were reliant on consecutive functions being successful so manual testing against user stories was deemed irrelevant.

---

*As a recruiter, I can view a project summary page so that I can quickly understand the dataset, terminology, and business requirements before exploring the data (US 1.1).*

| Feature | Action | Expected Result | Actual Result |
| --- | --- | --- | --- |
| Project Summary Page | Navigate to Quick Project Summary | Page displays general information, terms & jargon, dataset overview, and business requirements | Functions as intended |
| Business Requirements | Read the success callout | Three business requirements (BR1: salary correlations, BR2: salary prediction, BR3: market segments) are clearly listed | Functions as intended |
| README Link | Click the project README link | External link opens in a new tab, directing to the GitHub repository | Functions as intended |

---

*As a recruiter, I can view an interactive study of salary data so that I can understand how salary varies by experience level, company size, and location (US 1.2).*

| Feature | Action | Expected Result | Actual Result |
| --- | --- | --- | --- |
| Salary Study Page | Navigate to AI Salary Study | Page loads with business requirement statement, KPI metrics, and all sections visible | Functions as intended |
| Dataset Inspection | Tick "Inspect Dataset" checkbox | Table shows number of rows (15,000+), columns, and first 10 rows | Functions as intended |
| KPI Metrics | View metric cards | Four KPI cards display: total jobs, median salary, top country, and most common experience level | Functions as intended |
| Salary Distribution | Tick "Show Distribution of AI Salaries" checkbox | Histogram with KDE appears, showing right-skewed distribution with median vs mean comparison | Functions as intended |
| Salary by Location | View location section | Two horizontal bar charts display (company location and employee residence), each showing top 12 countries by median salary | Functions as intended |
| Correlation Study | Click through tabs (Years of Experience, Experience Level, Company Size, Remote Ratio) | Each tab displays the corresponding plot (scatter, boxplot) with introductory text | Functions as intended |
| Parallel Categories Plot | Scroll to parallel categories section | Interactive Plotly parallel categories plot renders with salary band colouring | Functions as intended |

---

*As a recruiter, I can review validated project hypotheses so that I can trust which factors genuinely influence AI salaries and which do not (US 1.3).*

| Feature | Action | Expected Result | Actual Result |
| --- | --- | --- | --- |
| Hypothesis Page | Navigate to Project Hypothesis | Page loads with introductory text and all four hypotheses displayed with Statement, Validation method, and Verdict | Functions as intended |
| H1: Experience Level | Read verdict and tick "Show Salary by Experience Level" | Verdict shows Confirmed (green). Checkbox reveals boxplot of salary by experience level | Functions as intended |
| H2: Remote Ratio | Read verdict and tick "Show Salary by Remote Ratio" | Verdict shows Rejected (red). Checkbox reveals boxplot confirming near-identical medians | Functions as intended |
| H3: Years of Experience | Read verdict and tick "Show Salary vs Years of Experience" | Verdict shows Confirmed (green). Checkbox reveals scatter plot with regression line and Pearson r value | Functions as intended |
| H4: Company Size | Read verdict and tick "Show Salary by Company Size" | Verdict shows Confirmed (green). Checkbox reveals boxplot with specific median values ($95k, $105k, $122k) | Functions as intended |
| Conclusion | Scroll to conclusion section | Summary states 3/4 hypotheses confirmed. Implication for modelling is explained | Functions as intended |

---

*As a recruiter, I can input a candidate's profile into an interactive form so that I can get an instant predicted salary (Business Requirement 2).*

| Feature | Action | Expected Result | Actual Result |
| --- | --- | --- | --- |
| Predict Salary Page | Navigate to Predict Salary | Page loads with business requirement statement, pipeline explanation, and model performance metrics | Functions as intended |
| Pipeline Explanation | Read "How the Model Works" section | Two-stage pipeline is described: data cleaning/feature engineering and regression model | Functions as intended |
| Model Performance | View metric cards | Train R² (0.878), Test R² (0.863), Train MAE ($15,208), Test MAE ($16,124) are displayed with success statement | Functions as intended |
| Feature Importance | Tick "Show Feature Importance Plot" checkbox | Bar chart image loads showing experience_level as the dominant feature | Functions as intended |
| Live Prediction Widgets | Select values from all four dropdowns | All dropdowns render with valid options. Experience level shows human-readable labels (Entry-level, Mid-level, etc.) | Functions as intended |
| Run Prediction | Click "Predict Salary" button | Predicted salary appears in USD with tier classification (top tier / above average / mid-range / lower end) | Functions as intended |
| Actionable Insights | Read "Key Takeaways" section | Tailored tips are shown based on the selected profile. Warning about ±$16k margin of error is displayed | Functions as intended |

---

*As a technical user, I can view the cluster analysis to understand how market segments are derived and interpret the methodology (Business Requirement 3).*

| Feature | Action | Expected Result | Actual Result |
| --- | --- | --- | --- |
| Cluster Analysis Page | Navigate to Cluster Analysis | Page loads with pipeline description, silhouette score, and all sections visible | Functions as intended |
| Pipeline Steps | Read pipeline description | Seven-step pipeline is listed (OrdinalMappingEncoder → OneHotEncoder → FrequencyEncoder → SmartCorrelatedSelection → StandardScaler → PCA → KMeans) | Functions as intended |
| Silhouette Plot | View silhouette image | Silhouette plot renders with text explaining what silhouette scores mean and interpreting the 0.53 score | Functions as intended |
| Cluster Distribution | View bar and line charts | Interactive Plotly bar chart shows cluster counts across salary bands. Line chart shows relative percentages per cluster | Functions as intended |
| Features Defining Clusters | View feature importance image | Bar chart renders showing company_location and employee_residence as the dominant drivers | Functions as intended |
| Cluster Profiles | View profile table and interpretation | Table displays per-cluster statistics. Text describes each cluster (Emerging-market, European-market, Established-market) | Functions as intended |
| Limitations | Read limitations section | Warning explains weak cluster separation (0.53 silhouette) and lists alternative approaches explored | Functions as intended |

---
