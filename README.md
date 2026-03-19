# [TASI(The AI Salary Index)](https://tasi-the-ai-salary-index-361138766018.herokuapp.com/page_cluster_body)

## Overview

**TASI** (The AI Salary Index) is a tool aimed to help recruiters in their search for candidates. 

Setting the right salary for AI/ML talent is challenging: experience, location, company size, and remote preferences all pull salary in different directions, making it difficult to benchmark compensation with confidence.

**TASI** solves this. Built on 15,000+ global AI job listings, it gives recruiters three things:

1. **Salary Prediction:**  A Gradient Boosting model (R² = 0.86) predicts salary from a candidate's profile.
2. **Market Segmentation:**  PCA + K-Means clustering groups roles into three segments split by experience and geography, revealing where pay premiums and penalties lie.
3. **Data-driven insights:**  Correlation analysis and hypothesis testing identify which attributes actually drive compensation.


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

**Business Requirement 1 : Data Visualisation & Correlation Study**
- Identify which attributes correlate most closely with AI/ML salary levels.
- *See `02-JobMarketStudy.ipynb` for exploratory analysis, correlation study, and hypothesis validation. Dashboard page: AI Salary Study.*

**Business Requirement 2 : Salary Prediction**
- Predict salary for a given candidate profile using a regression model.
- *See `05-ModelingEvaluation-PredictSalary.ipynb` for regression model training, hyperparameter optimisation, and evaluation. Dashboard page: Predict Salary & ML Regression Performance.*

**Business Requirement 3 : Market Segmentation**
- Group AI roles into market-based clusters to reveal distinct salary segments and labour market patterns.
- *See `06-ModelingEvaluation-Cluster.ipynb` for PCA + K-Means pipeline and cluster profiling. Dashboard page: ML Cluster Performance.*

## Hypotheses and how to validate them?

To better understand the factors influencing salary levels in the AI/ML job market, four key hypotheses were formulated based on domain knowledge and the available data. Each hypothesis focuses on a variable expected to impact salary prediction.

| Hypothesis | Rationale | Validation |
|------------|------------|-------------|
| **H1:** `experience_level` is the Dominant Salary Driver | Candidates with more experience (Entry → Executive) should command higher compensation | Visualize salary distribution by experience level, conduct ANOVA or regression analysis to confirm positive correlation |
| **H2:** Remote roles (`remote_ratio` = 100) have different salary expectations than hybrid/on-site roles |Remote roles may pay differently than on-site ones, some companies offer a premium for flexibility, while others adjust salaries based on the employee's local cost of living. | Spearman correlation between `remote_ratio` and `salary_usd`, plus a Kruskal-Wallis test across the three remote-ratio groups (on-site, hybrid, remote). |
| **H3:** `years_experience` required is positively correlated with `salary_usd` | Roles demanding more years of experience should offer higher compensation to attract experienced talent | Scatter plot with regression line, calculate correlation coefficient to confirm strength of relationship |
| **H4:** Company size (`company_size`) is a significant salary driver | Large companies pay significantly higher salaries than small and medium-sized companies, reflecting greater revenue, bigger budgets, and competitive hiring pressure | Kruskal-Wallis H-test across the three company size groups (S, M, L), compare mean and median salary per group, boxplot comparison |

These hypotheses are validated in `02-JobMarketStudy.ipynb` through exploratory data analysis and statistical testing (Spearman correlation, Kruskal-Wallis H-test). Results are summarised on the **Project Hypothesis** dashboard page.


### **Visual Flow**

### Visual Flow (CRISP-DM)

```

Business Understanding (Define recruiting goals & business requirements)
    ↓
Data Understanding (BR1: Salary correlation study & validate hypotheses )
    ↓
Data Preparation (Clean, encode, feature engineer & select validated features)
    ↓
Modeling (BR2: Salary Prediction — BR3: Market Segmentation)
    ↓
Evaluation (Assess model performance)
    ↓
Deployment (Streamlit dashboard on Heroku)

```

## Mapping Business Requirements to Data Visualisations and ML Tasks

**BR1 : Data Visualisation & Correlation Study**
- Salary distribution plots by experience level, location, and remote ratio
- Pearson/Spearman correlation analysis and hypothesis testing (H1–H4)
- Output: Identify which attributes drive AI/ML compensation

**BR2 : Salary Prediction (Regression)**
- GradientBoostingRegressor trained on candidate profile features (experience, location, company size)
- Feature importance assessment and model evaluation (R², MAE, RMSE)
- Output: Predicted salary with confidence range

Example predictions from the dashboard:

| Experience | Company Size | Location | Residence | Predicted Salary | Cluster |
|------------|-------------|----------|-----------|-----------------|---------|
| Senior (SE) | Small (S) | United States | United States | ~$138,000 USD | 2 - Senior Professionals |
| Mid-level (MI) | Medium (M) | Germany | Germany | ~$95,000 USD | 1 - Junior/Entry |
| Entry-level (EN) | Large (L) | India | India | ~$52,000 USD | 0 - India Segment |

*Predictions carry a margin of error of approximately ±$16,000 (model test MAE).*

**BR3 : Market Segmentation (Clustering)**
- PCA for dimensionality reduction + K-Means (k=4) to segment roles
- Cluster profiling by experience level, geography, and salary band
- Output: Four market segments with distinct compensation patterns

---

## ML Business Cases

### **Business Case 2: Salary Prediction (Regression)**

| | |
|---|---|
| **ML Task** | Predict annual salary (USD) from candidate profile attributes |
| **Model Type** | Supervised regression (GradientBoostingRegressor) |
| **Ideal Outcome** | Accurate salary recommendations that reduce under-bidding and over-compensation |
| **Success Metrics** | R² ≥ 0.75 on train & test; RMSE ≤ 15% of mean salary; train–test R² gap ≤ 0.05 |
| **Output** | Predicted salary (USD), confidence interval, top features driving the prediction |
| **Target Variable** | `salary_usd` |
| **Features** | `experience_level`, `company_size`, `company_location`, `employee_residence` |
| **Data** | ~15,000 job postings from Kaggle (`ai_job_dataset1.csv`) |

---

### **Business Case 3: Market Segmentation (Clustering)**

| | |
|---|---|
| **ML Task** | Group AI roles into market segments based on profile features (excluding salary) |
| **Model Type** | Unsupervised clustering (PCA + K-Means, k=4) |
| **Ideal Outcome** | Distinct segments that reveal experience and geography-based compensation patterns |
| **Success Metrics** | Silhouette score ≥ 0.15; 3–6 interpretable clusters; min 500 samples per cluster |
| **Output** | Cluster ID per role, cluster profiles (experience, geography, salary band distribution) |
| **Features** | All profile features except `salary_usd`, `required_skills`, `company_name` |
| **Data** | Same dataset, train + test combined (14,701 rows) |


## Epics & User Stories

### Epic 1 : Data Insights (BR1)

- **US 1.1** : As a recruiter, I can view a project summary page so that I can quickly understand the dataset, terminology, and business requirements before exploring the data.
  - **Viz Task**: Data profiling in a Streamlit information page.

- **US 1.2** : As a recruiter, I can view an interactive study of salary data so that I can understand how salary varies by experience level, company size, and location.
  - **Viz Task**: Box plots, histograms with KDE, parallel categories plot (Plotly).

- **US 1.3** : As a recruiter, I can review validated project hypotheses so that I can trust which factors genuinely influence AI salaries and which do not.
  - **Statistical Task**: Kruskal-Wallis H-test, Spearman and Pearson correlation with supporting visualisations.

### Epic 2 : Salary Prediction (BR2)

- **US 2.1** : As a recruiter, I can input a candidate's profile into an interactive form so that I can get an instant predicted salary in USD.
  - **ML Task**: GradientBoostingRegressor pipeline served through a Streamlit form.

- **US 2.2** : As a data practitioner, I can view model performance metrics (R², MAE, RMSE) and feature importance to evaluate the model's accuracy and identify which variables drive predictions.
  - **Viz Task**: Train vs test evaluation metrics and feature importance bar chart.

### Epic 3 : Market Segmentation (BR3)

- **US 3.1** : As a recruiter, I can see which market segment a role belongs to so that I can understand its compensation context.
  - **ML Task**: PCA + K-Means clustering (k=4) assigning roles to experience/geography-based segments.

- **US 3.2** : As a data practitioner, I can understand how clusters are derived and their limitations.
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

- Provide a clear overview of the project and orient users.
- Describe business requirements.

### Page 2: AI Salary Study
 
- Address **Business Requirement 1** (Data Visualisation & Correlation Study).
- Explore salary distributions by experience level, location, and company size.
- Present Pearson and Spearman correlation analysis.

### Page 3: Project Hypothesis and Validation
 
- Present the four project hypotheses and their validation outcomes.
- Provide statistical evidence and supporting visualisations for each hypothesis.

### Page 4: Predict Salary

- Address **Business Requirement 2 & 3** (Salary Prediction & segmentation).
- Allow users to input a candidate profile and receive a predicted salary.

### Page 5: ML Regression Performance

- Address **Business Requirement 2** (Salary Prediction ML pipeline).
- Show ML pipeline steps.
- Display model performance metrics and feature importance.

### Page 6: ML Cluster Performance

- Address **Business Requirement 3** (Market Segmentation ML pipeline).
- Show Cluster ML pipeline steps.
- Show cluster model performance and interpretation.
- Profile four market segments by geography and experience level.

## Technologies Used

- **Python 3.13.2**
- **Pandas**  data manipulation and analysis
- **NumPy**  numerical computation
- **Matplotlib / Seaborn**  static data visualisation
- **Plotly**  interactive data visualisation
- **Scikit-Learn**  ML pipelines, preprocessing, model training, evaluation
- **XGBoost**  gradient boosting (evaluated during model selection)
- **Feature Engine**  feature engineering transformations
- **SciPy**  statistical tests for hypothesis validation
- **Streamlit**  interactive web dashboard
- **Joblib**  pipeline serialisation
- **Heroku**  cloud deployment platform
- **Git / GitHub**  version control

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

---

*As a recruiter, I can view an interactive study of salary data so that I can understand how salary varies by experience level, company size, and location (US 1.2).*

| Feature | Action | Expected Result | Actual Result |
| --- | --- | --- | --- |
| Salary Study Page | Navigate to AI Salary Study | Page loads with business requirement statement, KPI metrics, and all sections visible | Functions as intended |
| Dataset Inspection | Tick "Inspect Dataset" checkbox | Table shows number of rows (15,000+), columns, and first 10 rows | Functions as intended |
| KPI Metrics | View metric cards | Four KPI cards display: total jobs, median salary, top country, and most common experience level | Functions as intended |
| Salary Distribution | Tick "Show Distribution of AI Salaries" checkbox | Histogram with KDE appears with explanatory caption, showing right-skewed distribution with median vs mean comparison | Functions as intended |
| Salary by Location | View location section | Two horizontal bar charts display with explanatory captions (company location and employee residence), each showing top 12 countries by median salary | Functions as intended |
| Correlation Study | Click through tabs (Years of Experience, Experience Level, Company Size, Remote Ratio) | Each tab displays the corresponding plot (scatter, boxplot) with explanatory caption and introductory text | Functions as intended |
| Parallel Categories Plot | Scroll to parallel categories section | Interactive Plotly parallel categories plot renders with salary band colouring and explanatory caption | Functions as intended |

---

*As a recruiter, I can review validated project hypotheses so that I can trust which factors genuinely influence AI salaries and which do not (US 1.3).*

| Feature | Action | Expected Result | Actual Result |
| --- | --- | --- | --- |
| Hypothesis Page | Navigate to Project Hypothesis | Page loads with introductory text and all four hypotheses displayed with Statement, Validation method, and Verdict | Functions as intended |
| H1–H4: Hypothesis Testing | Read each verdict and tick its "Show Salary by …" checkbox | Each hypothesis displays a color-coded verdict (Confirmed/Rejected) and reveals an interactive chart (boxplot or scatter plot) with explanatory caption | Functions as intended |
| Conclusion | Scroll to conclusion section | Summary states 3/4 hypotheses confirmed. Implication for modelling is explained | Functions as intended |

---

*As a recruiter, I can input a candidate's profile into an interactive form so that I can get an instant predicted salary (Business Requirement 2).*

| Feature | Action | Expected Result | Actual Result |
| --- | --- | --- | --- |
| Predict Salary Page | Navigate to Predict Salary | Page loads with business requirement statements, model explanation, and Live Prediction Widgets | Functions as intended |
| Pipeline Explanation | Read "How the Model Works" section | The outputs of the predictions are described | Functions as intended |
| Live Prediction Widgets | Select values from all four dropdowns | All dropdowns render with valid options. Experience level shows human-readable labels (Entry-level, Mid-level, etc.) | Functions as intended |
| Run Prediction | Click "Predict Salary" button | Predicted salary appears in USD with predicted market segment ( Cluster) | Functions as intended |
| Actionable Insights | Read "Key Takeaways" section | Tailored tips are shown based on the selected profile. Warning about ±$16k margin of error is displayed | Functions as intended |

---

*As a data practitioner, I can view model performance metrics (R², MAE, RMSE) and feature importance to evaluate the model's accuracy and identify which variables drive predictions (Business Requirement 2).*

| Feature | Action | Expected Result | Actual Result |
| --- | --- | --- | --- |
| ML Regression Performance Page | Navigate to ML Regression Performance | Page loads with business requirement statement, pipeline steps, and model performance sections | Functions as intended |
| Pipeline Steps | Read "ML Pipeline Steps" section | Two-stage pipeline is described: Stage 1 (data cleaning & feature engineering) and Stage 2 (GradientBoostingRegressor with 300 estimators, max depth 3, learning rate 0.2) | Functions as intended |
| Feature Importance | Tick "Show Feature Importance Plot" checkbox | Bar chart image loads with explanatory caption showing experience_level as the dominant feature (>50% predictive power), followed by company_location, employee_residence, and company_size | Functions as intended |
| Model Performance Metrics | View Train and Test metric cards | Train R² (0.878), Test R² (0.863), Train MAE ($15,208), Test MAE ($16,124), Train RMSE and Test RMSE are displayed in two-column layout | Functions as intended |
| Residual Analysis | Tick "Show residual plots" checkbox | Two scatter plots render (Train and Test) showing residuals vs predicted salary with red dashed zero line and explanatory caption | Functions as intended |
| Predicted vs Actual | Tick "Show predicted vs actual plots" checkbox | Two scatter plots render (Train and Test) with diagonal reference line for perfect prediction and explanatory caption | Functions as intended |
| Business Insights | Read "Business Insights" section | Recruiter-facing takeaways are shown (average prediction error, feature importance summary). Limitations section warns about 4-feature model and ±$16k margin of error | Functions as intended |

---

*As a recruiter, I can see which market segment a role belongs to so that I can understand its compensation context (Business Requirement 3).*

| Feature | Action | Expected Result | Actual Result |
| --- | --- | --- | --- |
| ML Cluster performance | Navigate to ML Cluster Performance | Page loads with business requirement statement, pipeline steps, and cluster model performance sections | Functions as intended |
| Pipeline Steps | Read "Cluster ML Pipeline steps" section | Five-step pipeline is listed (OrdinalMappingEncoder → CountFrequencyEncoder → StandardScaler → PCA 99% variance → KMeans k=4) | Functions as intended |
| Cluster Size Distribution | View cluster size table | Table shows 4 clusters (India Market, Junior Pipeline, Senior Premium, Emerging Markets) with train/test samples and share percentages | Functions as intended |
| Salary Band Distribution | View bar and line charts | Interactive Plotly bar chart shows cluster counts across salary bands with explanatory caption. Line chart shows relative percentages per cluster with explanatory caption | Functions as intended |
| Feature Importance | View feature importance image | Bar chart renders with explanatory caption showing employee_residence (0.46) and experience_level (0.24) as the dominant cluster-defining features | Functions as intended |
| Cluster Profiles | View profile section and table | Detailed profiles for 4 clusters (India Market, Junior/Entry, Senior Professionals, Mid-career Emerging Markets) with actionable salary negotiation tip | Functions as intended |

---

*As a data practitioner, I can understand how clusters are derived and their limitations (Business Requirement 3).*

| Feature | Action | Expected Result | Actual Result |
| --- | --- | --- | --- |
| Silhouette Plot | View silhouette image | Silhouette plot renders with explanatory caption and explanation of score range (-1 to +1) and interpretation of the 0.16 average score | Functions as intended |
| PCA Insight | Read key insight box | Explanation that PCA Component 0 captures geographic variation and Component 1 captures experience/seniority | Functions as intended |
| Cluster Profile Table | View per-cluster statistics table | Table displays detailed per-cluster statistics from clusters_profile.csv | Functions as intended |
| Limitations & Next Steps | Read limitations section | Warning explains modest silhouette score (0.16), continuous distribution nature, hyperparameter tuning (192 combinations), and alternative approaches explored (UMAP + HDBSCAN, K-Prototypes) | Functions as intended |

---
### Validation
All code in the Notebooks, `app_pages` and `src` directories was validated as conforming to PEP8 standards.

### Automated Unit Tests
No automated unit tests have been carried out at this time.


## Unfixed Bugs
* At the time of writing, there are no unfixed bugs within the project.


## Deployment

### Heroku

The project was deployed to Heroku using the following steps:

1. Within your working directory, ensure there is a `setup.sh` file containing the following:
```
mkdir -p ~/.streamlit/
echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
" > ~/.streamlit/config.toml
```
2. Within your working directory, ensure there is a `runtime.txt` file containing a [Heroku-20](https://devcenter.heroku.com/articles/python-support#supported-runtimes) stack supported version of Python.
```
python-3.13.2
```
3. Within your working directory, ensure there is a `Procfile` containing the following:
```
web: sh setup.sh && streamlit run app.py
```
4. Ensure your `requirements.txt` file contains all the packages necessary to run the Streamlit dashboard.
5. Update your `.gitignore` and `.slugignore` files with any files/directories that you do not want uploading to GitHub or are unnecessary for deployment.
6. Log in to [Heroku](https://id.heroku.com/login) or create an account if you do not already have one.
7. Click the **New** button on the dashboard and from the dropdown menu select "Create new app".
8. Enter a suitable app name and select your region, then click the **Create app** button.
9. Once the app has been created, navigate to the Deploy tab.
10. At the Deploy tab, in the Deployment method section select **GitHub**.
11. Enter your repository name and click **Search**. Once it is found, click **Connect**.
12. Navigate to the bottom of the Deploy page to the Manual deploy section and select **main** from the branch dropdown menu.
13. Click the **Deploy Branch** button to begin deployment.
14. The deployment process should happen smoothly if all deployment files are fully functional. Click the button **Open App** at the top of the page to access your App.
15. If the build fails, check the build log carefully to troubleshoot what went wrong.


## Forking and Cloning

If you wish to fork or clone this repository, please follow the instructions below:

### Forking
1. In the top right of the main repository page, click the **Fork** button.
2. Under **Owner**, select the desired owner from the dropdown menu.
3. **OPTIONAL:** Change the default name of the repository in order to distinguish it.
4. **OPTIONAL:** In the **Description** field, enter a description for the forked repository.
5. Ensure the 'Copy the main branch only' checkbox is selected.
6. Click the **Create fork** button.

### Cloning
1. On the main repository page, click the **Code** button.
2. Copy the HTTPS URL from the resulting dropdown menu.
3. In your IDE terminal, navigate to the directory you want the cloned repository to be created.
4. In your IDE terminal, type `git clone` and paste the copied URL.
5. Hit Enter to create the cloned repository.

### Installing Requirements
In order to ensure all the correct dependencies are installed in your local environment, run the following command in the terminal:

    pip install -r requirements.txt


## Credits

### Content

#### Jupyter Notebooks
* The code was adapted from the Code Institute "Churnometer" walkthrough project.


#### Streamlit Dashboard
* The multi-page navigation structure was adapted from the Code Institute "Data Analysis & Machine Learning Toolkit" Streamlit lessons.

#### Dataset
* The dataset used in this project is sourced from Kaggle: [Global AI Job Market & Salary Trends 2025](https://www.kaggle.com/datasets/bismasajjad/global-ai-job-market-and-salary-trends-2025) by Bisma Sajjad.


## Acknowledgements
* Thanks to my mentor for their support and guidance on the execution of the project.
* Thanks to the Code Institute for providing the walkthrough projects and learning materials that formed the foundation of this work.

[Back to top](#the-ai-salary-index)
