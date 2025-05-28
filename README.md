# Ames Housing Analysis Project

This project analyzes the Ames Housing dataset with a focus on different roles in data science, including Project Management, Data Engineering, Data Acquisition, Data Stewardship, Stakeholder Liaison, Data Science, and Machine Learning Engineering.

## Project Structure

```
ames_housing/
├── README.md                 # Project overview and setup instructions
├── requirements.txt          # Project dependencies
├── data/                     # All data files
│   ├── raw/                  # Original, unmodified data
│   │   ├── housing_raw.csv
│   │   └── schools_raw.csv
│   ├── processed/           # Cleaned and processed data
│   │   ├── housing_cleaned.csv
│   │   ├── schools_combined.csv
│   │   ├── schools_metrics.csv
│   │   ├── neighborhood_schools.csv
│   │   ├── nominal_column_mappings.json # Mappings used by models
│   │   ├── random_forest_model.joblib   # Trained Random Forest model
│   │   ├── rf_feature_names.joblib      # Feature names for RF model
│   │   ├── final_deep_learning_model.h5 # Trained Deep Learning model
│   │   ├── dl_numerical_scaler.joblib   # Scaler for DL model numerical features
│   │   └── model_comparison_results.joblib # Serialized model comparison metrics
│   └── sensitive/           # Protected data and access logs
│       ├── data_access_log.db
│       └── ames_sensitive_data.csv
├── src/                     # Source code
│   ├── regression_analysis_visualizations.py # Script for generating report visualizations
│   ├── etl/                 # Data engineering scripts
│   │   ├── etl_main.py
│   │   └── etl_documentation.md
│   ├── acquisition/         # Data acquisition scripts
│   │   ├── schools_download.py
│   │   └── schools_combine.py
│   ├── modeling/            # Modeling scripts and utilities
│   │   ├── __init__.py      # Package initializer
│   │   ├── models/
│   │   │   ├── deep_learning/
│   │   │   │   ├── deep_learning_baseline.py
│   │   │   │   ├── deep_learning_hyper_tune.py
│   │   │   │   ├── deep_learning_tuned_evaluation.py
│   │   │   │   └── prepare_deep_learning_artifacts.py
│   │   │   └── traditional/
│   │   │       └── random_forest_baseline.py
│   │   ├── analysis/        # Analysis files (if any)
│   │   └── utils/
│   │       ├── __init__.py
│   │       ├── data_loader.py
│   │       └── validation.py
│   ├── governance/          # Data stewardship scripts
│   │   ├── data_access_guide.md
│   │   ├── steward_guide.md
│   │   ├── access_control.py
│   │   └── access_monitor.py
│   └── app/                 # Gradio application source files
│       ├── rf_gradio_app.py # Source for the RF Gradio app
│       └── gradio_app.py    # Source for the DL Gradio app
├── docs/                    # Documentation
│   ├── analysis_summary.md  # Detailed summary of roles, processes, and learnings
│   ├── model_comparison_report.md
│   ├── model_comparison_report.pdf
│   ├── gradio_app_guide.md  # Guide for using the Gradio application(s)
│   ├── project_roadmap.md
│   ├── data_dictionary.md
│   ├── data_schema.json
│   ├── future_acquisition_plan.md
│   ├── stakeholder_liaison.md
│   ├── project_roles_journey_summary_v1_detailed.md   # Bulleted, detailed version of the project roles journey summary
│   ├── project_roles_journey_summary_v2_narrative.md  # First narrative version of the project roles journey summary
│   ├── project_roles_journey_summary_v3_concise.md    # Second narrative concise version of the project roles 
│   ├── Kristin_Henderson_homework1_summary.pdf        # Final concise, one-page summary for assignment submission
│   └── analysis_summary.md  # Detailed summary of roles, processes, and learnings
├── figures/                 # Generated charts and figures
├── keras_tuner_dir/         # Output from KerasTuner hyperparameter searches
├── spaces/                  # Hugging Face Space applications
│   ├── random-forest-predictor/
│   │   ├── app.py
│   │   ├── requirements.txt
│   │   ├── README.md
│   │   ├── .gitignore
│   │   ├── .gitattributes
│   │   └── data/
│   │       └── processed/
│   │           ├── random_forest_model.joblib
│   │           ├── rf_feature_names.joblib
│   │           └── nominal_column_mappings.json
│   └── deep-learning-predictor/
└── tests/                   # Test files (if any)
```

## Role-Based Organization

The project is organized around different data science roles, with detailed descriptions and learnings for each available in [`docs/analysis_summary.md`](docs/analysis_summary.md). Key areas and primary file locations include:

1.  **Project Management & Documentation** (`docs/`)
    - Core project planning, summaries, and data descriptions.
2.  **Data Engineering** (`src/etl/`, `docs/data_dictionary.md`, `docs/data_schema.json`)
    - Scripts for data cleaning, transformation, and ETL documentation.
3.  **Data Acquisition** (`src/acquisition/`, `docs/future_acquisition_plan.md`)
    - Scripts for acquiring external datasets and plans for future data.
4.  **Data Stewardship & Governance** (`src/governance/`)
    - Scripts and guides for data access control, security, and monitoring.
5.  **Stakeholder Liaison** (`docs/stakeholder_liaison.md`)
    - Documentation and communication strategies for stakeholder engagement.
6.  **Data Science / Modeling** (`src/modeling/models/`, `src/modeling/utils/`)
    - Scripts for model development (baselines, advanced), utilities, and evaluation.
7.  **Machine Learning Engineering** (`src/modeling/models/deep_learning/`, `requirements.txt`)
    - Implementation of deep learning models, hyperparameter tuning, and dependency management.
8.  **Visualization Expert** (`src/regression_analysis_visualizations.py`, `figures/`, `docs/model_comparison_report.md`)
    - Automated creation of model evaluation plots, feature importance, and visual summaries for reports and stakeholder communication.
9.  **Explainability Engineer** (`src/regression_analysis_visualizations.py`, `figures/shap_summary.png`, `docs/shap_explanation.md`)
    - Interprets and communicates model predictions using SHAP values and explainability tools.

## Modeling Overview

Predictive modeling for house prices involved:
- A **Random Forest baseline** (`src/modeling/models/random_forest_baseline.py`) which demonstrated strong initial performance.
- An iterative **Deep Learning approach** (`src/modeling/models/deep_learning/`) starting with a basic neural network and evolving to a more complex architecture using Embedding layers for nominal features. 
- **Hyperparameter tuning** for the deep learning model was performed using KerasTuner (`src/modeling/models/deep_learning/deep_learning_hyper_tune.py` and `keras_tuner_dir/`).
- **Robust evaluation** of the tuned deep learning model was conducted using K-fold cross-validation (`src/modeling/models/deep_learning/deep_learning_tuned_evaluation.py`).

For detailed model performance and development journey, please refer to the [Analysis Summary](docs/analysis_summary.md).

## Setup Instructions

1.  Create and activate virtual environment:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3.  Run the ETL pipeline (to generate `data/processed/housing_cleaned.csv` needed for modeling):
    ```bash
    python src/etl/etl_main.py
    ```

4.  To run modeling scripts (examples):
    ```bash
    # Run Random Forest baseline
    python src/modeling/models/random_forest_baseline.py

    # Run K-fold evaluation of the tuned Deep Learning model (ensure hyperparameters are set in the script)
    python src/modeling/models/deep_learning/deep_learning_tuned_evaluation.py
    ```

## Data Access

Sensitive data access is controlled through the governance scripts in `src/governance/`. All access events are logged in `data/sensitive/data_access_log.db`.

## Contributing

Please refer to the documentation in the `docs/` directory, especially [`docs/analysis_summary.md`](docs/analysis_summary.md), for detailed information about each component of the project. 