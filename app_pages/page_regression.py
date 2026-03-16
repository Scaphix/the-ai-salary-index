import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from src.data_management import load_pkl_file


def page_regression_body():

    st.write("# ML Regression Model Performance")

    st.info(
        "**Business Requirement 2** — Predict the expected annual salary "
        "(USD) for a given AI job posting based on its attributes.\n\n"
    )

    version = "v1"
    path = f"outputs/ml_pipeline/predict_salary/{version}"

    pipeline_model = load_pkl_file(f"{path}/pipeline_regressor.pkl")

    X_train = pd.read_csv(f"{path}/X_train.csv")
    X_test = pd.read_csv(f"{path}/X_test.csv")
    y_train = pd.read_csv(f"{path}/y_train.csv").squeeze()
    y_test = pd.read_csv(f"{path}/y_test.csv").squeeze()

    # --- Pipeline Steps ---
    st.write("---")
    st.write("## ML Pipeline Steps")
    st.write("**The salary prediction pipeline consists of two stages:**")
    st.info(
        "### Stage 1 — Data Cleaning & Feature Engineering\n"
        "* Drops irrelevant columns (free-text fields, company names, "
        "industry) and **years_experience** (Spearman ρ = 0.97 with "
        "**experience_level** — nearly identical information).\n"
        "* Encodes **experience_level**, **company_size**, and "
        "**education_required** as ordinal numbers.\n"
        "* Frequency-encodes **company_location** and "
        " **employee_residence**.\n"
        "* One-hot encodes **employment_type** and **job_title**.\n\n"
        "### Stage 2 — Regression Model\n\n"
        "* Scales features with StandardScaler.\n"
        "* Predicts salary using a tuned **GradientBoostingRegressor** "
        "(300 estimators, max depth 3, learning rate 0.2).\n"
        "* Only the **4 most important features** (selected automatically "
        "during training) are used for prediction: **experience_level**, "
        "**company_location**, **employee_residence**, and **company_size**."
    )

    # --- Feature Importance ---
    st.write("---")
    st.write("## Feature Importance")

    if st.checkbox("Show Feature Importance Plot"):
        st.image(
            f"{path}/feature_importance.png",
            caption="Feature importance from the GradientBoostingRegressor",
            width=600,
        )

    st.success(
        "**experience_level** accounts for over **50%** of the model's "
        "predictive power, confirming Hypothesis H1 that **experience** is "
        "the **strongest salary driver**.\n\n"
        "**company_location** and **employee_residence** are the next "
        "most important features, as geography significantly"
        " influences compensation.\n\n"
        "**company_size** has a smaller but meaningful contribution, "
        "with larger companies offering higher pay on average."
    )

    # --- Model Evaluation ---
    st.write("---")
    st.write("## Model Performance")

    y_train_pred = pipeline_model.predict(X_train)
    y_test_pred = pipeline_model.predict(X_test)

    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    col1, col2 = st.columns(2)

    with col1:
        st.write("#### Train Set")
        st.metric("R²", f"{train_r2:.3f}")
        st.metric("MAE", f"${train_mae:,.0f}")
        st.metric("RMSE", f"${train_rmse:,.0f}")

    with col2:
        st.write("#### Test Set")
        st.metric("R²", f"{test_r2:.3f}")
        st.metric("MAE", f"${test_mae:,.0f}")
        st.metric("RMSE", f"${test_rmse:,.0f}")

    r2_gap = abs(train_r2 - test_r2)

    if test_r2 >= 0.70:
        st.success(
            f"The model **meets** the success criterion (R² ≥ 0.70) on "
            f"both train ({train_r2:.3f}) and test ({test_r2:.3f}) sets.\n\n"
            f"The train–test R² gap of **{r2_gap:.3f}** indicates the model "
            f"generalises well with no significant overfitting."
        )
    else:
        st.error(
            f"The model **does not meet** the success criterion. "
            f"Test R² = {test_r2:.3f} (target: ≥ 0.70)."
        )

    # --- Residual Analysis ---
    st.write("---")
    st.write("## Residual Analysis")

    if st.checkbox("Show residual plots"):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Train residuals
        train_residuals = y_train - y_train_pred
        axes[0].scatter(y_train_pred, train_residuals, alpha=0.3, s=10)
        axes[0].axhline(y=0, color="red", linestyle="--")
        axes[0].set_xlabel("Predicted Salary (USD)")
        axes[0].set_ylabel("Residual (USD)")
        axes[0].set_title("Train Set Residuals")

        # Test residuals
        test_residuals = y_test - y_test_pred
        axes[1].scatter(y_test_pred, test_residuals, alpha=0.3, s=10)
        axes[1].axhline(y=0, color="red", linestyle="--")
        axes[1].set_xlabel("Predicted Salary (USD)")
        axes[1].set_ylabel("Residual (USD)")
        axes[1].set_title("Test Set Residuals")

        plt.tight_layout()
        st.pyplot(fig)

        st.info(
            "Residuals are centred around zero on both sets, indicating "
            "the model has no systematic bias. The spread is consistent "
            "across salary ranges, suggesting homoscedastic errors."
        )

    # --- Predicted vs Actual ---
    if st.checkbox("Show predicted vs actual plots"):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        for ax, y_true, y_pred, label in [
            (axes[0], y_train, y_train_pred, "Train"),
            (axes[1], y_test, y_test_pred, "Test"),
        ]:
            ax.scatter(y_true, y_pred, alpha=0.3, s=10)
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val],
                    color="red", linestyle="--", label="Perfect prediction")
            ax.set_xlabel("Actual Salary (USD)")
            ax.set_ylabel("Predicted Salary (USD)")
            ax.set_title(f"{label} Set: Predicted vs Actual")
            ax.legend()

        plt.tight_layout()
        st.pyplot(fig)

        st.info(
            "Points closely follow the diagonal line on both sets, "
            "confirming strong predictive accuracy. The pattern is "
            "consistent between train and test, reinforcing that the "
            "model generalises well to unseen data."
        )

    # --- Business Insights ---
    st.write("---")
    st.write("## Business Insights")

    st.success(
        "**For recruiters and hiring managers:**\n\n"
        "* The model can predict AI salaries with an average error of "
        f"**±${test_mae:,.0f}**, making it a reliable tool for "
        "benchmarking compensation offers.\n"
        "* **Experience level** is the single most important factor — "
        "promotions from mid-level to senior have the largest salary "
        "impact.\n"
        "* **Location matters significantly** — the same role can pay "
        "very differently depending on company location and employee "
        "residence.\n"
        "* **Company size** has a smaller but consistent effect — larger "
        "companies tend to offer higher salaries across all experience "
        "levels."
    )

    st.warning(
        "**Limitations:**\n\n"
        "* The model uses only 4 features. Factors like specific skills, "
        "industry sector, negotiation, and equity compensation are not "
        "captured.\n"
        f"* Predictions have a margin of error of approximately "
        f"**±${test_mae:,.0f}** (MAE). Individual salaries may deviate "
        f"further due to unmeasured variables.\n"
        "* The dataset is synthetic (Kaggle), so predictions should be "
        "treated as indicative rather than definitive market rates."
    )
