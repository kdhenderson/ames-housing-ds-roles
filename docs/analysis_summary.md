# Ames Housing Analysis Summary

## Roles and Reference Files

### 1. Project Manager
- **Reference File:** [project_roadmap.md](project_roadmap.md)
- **Description:** Outlines objectives, stakeholders, success metrics, and a detailed project roadmap for managing the Ames Housing data analysis project.

### Reflections & Lessons Learned

One key insight from this analysis is the importance of identifying stakeholders before defining project objectives, success metrics, and deliverables. Understanding who the stakeholders are:
- Guides the setting of relevant and actionable objectives
- Clarifies what metrics truly define success for the project
- Shapes the format and focus of deliverables to ensure they are useful and impactful
- Improves communication and alignment throughout the project
- Reduces the risk of rework by ensuring outputs meet stakeholder needs from the start

This user-centered approach is essential for effective project management and maximizing the value of data science initiatives.

### 2. Stakeholder Liaison

- **Reference File:** [stakeholder_liaison.md](stakeholder_liaison.md)
- **Description:** Acts as a bridge between business/subject matter experts and technical teams, ensuring alignment, clear communication, and that deliverables meet stakeholder needs.

#### Reflections & Lessons Learned

- This role stood out as having a huge breadth of required skills: communication, technical insight, organization, and documentation.
- Thinking through a specific example scenario—mapping out the types of meetings and conversations—highlighted just how central communication is to this role.
- The presence of a liaison is most valuable in projects with greater size, technical complexity, or a diverse set of stakeholders. In smaller or less complex projects, the PM or technical leads may fill this function.
- The liaison is not only key in clarifying objectives and managing expectations, but also in facilitating user education, training, and ongoing evaluation of tools and deliverables.
- This role supports a continuous feedback loop, helping to ensure the project remains aligned with stakeholder needs and adapts to changes.
- By maintaining clear documentation and supporting change management, the liaison helps mitigate risks and ensures project transparency.

*The process of thinking through concrete examples and the types of communication involved was especially helpful in understanding the full value and scope of this role.*

### 3. Data Engineer

- **Reference Files:**
    - [src/etl/etl_main.py](../src/etl/etl_main.py) (Python ETL script)
    - [src/etl/etl_documentation.md](../src/etl/etl_documentation.md) (ETL process documentation)
    - [data_dictionary.md](data_dictionary.md) (Data dictionary)
    - [data_schema.json](data_schema.json) (JSON schema for the dataset)
    - [data/processed/housing_cleaned.csv](../data/processed/housing_cleaned.csv) (Cleaned dataset)

#### Reflections & Lessons Learned

One of the main challenges in the data engineering process was the encoding of categorical features. Initially, all categorical variables were label encoded, but this approach did not distinguish between ordinal and nominal features. I had to request a more nuanced analysis to identify which features were ordinal and which were nominal. As a result, ordinal features were label encoded to preserve their order and minimize dimensionality, while nominal features were one-hot encoded to accurately represent their categories.

Another challenge arose when a new engineered feature, `TotalSF` (the sum of several area-related features), was introduced. The feature was initially added after the encoding step, which led to confusion and required additional debugging. This issue was not immediately apparent until I examined the cleaned dataset. Adding debugging print statements helped clarify the data flow and allowed for the correct sequencing of feature engineering and encoding.

There was also some back-and-forth in the prompt engineering process, as the AI sometimes generated scripts and documentation for the entire modeling pipeline rather than focusing solely on the data engineering steps. Additionally, I had to specifically request the creation of a data dictionary and a formal schema, which are essential for downstream modeling and collaboration but were not included by default.

These challenges highlight the importance of clear communication, iterative refinement, and thorough documentation in the data engineering role to ensure a robust and transparent pipeline for subsequent analysis and modeling.

### 4. Data Acquisition Specialist

**Reference Files:**
- [src/acquisition/schools_download.py](../src/acquisition/schools_download.py) (script to download and filter school data)
- [data/raw/schools_raw.csv](../data/raw/schools_raw.csv) (Ames, IA school listing)
- [data/processed/neighborhood_schools.csv](../data/processed/neighborhood_schools.csv) (mapping of neighborhoods to schools)
- [data/processed/schools_metrics.csv](../data/processed/schools_metrics.csv) (school ratings and metrics)
- [src/acquisition/schools_combine.py](../src/acquisition/schools_combine.py) (script to combine mapping, addresses, and metrics)
- [data/processed/schools_combined.csv](../data/processed/schools_combined.csv) (final joinable school data by neighborhood)
- [future_acquisition_plan.md](future_acquisition_plan.md) (plan and lessons learned for future data acquisition)

#### Value Added
- Sought to enhance the Ames Housing dataset with school and crime data, as these are known to influence property values and are of interest to stakeholders.
- School data, in particular, can provide interpretable features for modeling and analysis, especially when joined by neighborhood.

#### Challenges
- The Ames dataset does not include property addresses, latitude/longitude, or school district assignments, making direct matching to schools impossible.
- We approximated school assignments by manually matching neighborhoods to schools using Google Maps and public information, since no direct mapping or scrapeable data source was available.
- Attempts to scrape school metrics from GreatSchools failed due to JavaScript rendering and anti-bot measures; metrics were manually collected instead.
- Crime data was identified as valuable, but reliable neighborhood-level data was not readily available for direct integration.

#### Outcome
- Produced a listing of schools and their metrics for each neighborhood, formatted for easy joining to the cleaned Ames Housing data by neighborhood.
- Documented the process and challenges in `future_acquisition_plan.md` to guide future efforts, including recommendations for browser automation or official data sources for scalable data acquisition.

### 5. Data Steward / Governance Officer

**Reference Files:**
- [src/governance/access_control.py](../src/governance/access_control.py) (approved script for accessing sensitive data, with logging and user whitelist)
- [src/governance/access_monitor.py](../src/governance/access_monitor.py) (script for reviewing and monitoring the access log)
- [data/sensitive/data_access_log.db](../data/sensitive/data_access_log.db) (SQLite database logging all access events)
- [src/governance/data_access_guide.md](../src/governance/data_access_guide.md) (user guide for data access)
- [src/governance/steward_guide.md](../src/governance/steward_guide.md) (technical guide for data stewards)

**Reference File (in parent directory):**
- [data/sensitive/ames_sensitive_data.csv](../data/sensitive/ames_sensitive_data.csv) (sample sensitive data file; for demonstration purposes only, not real data)

#### Reflections & Lessons Learned
- This role is responsible for overseeing data quality, compliance, documentation, and access control, ensuring that sensitive data is handled appropriately and securely.
- The Data Steward is differentiated from the Data Engineer (who cleans and processes data) by focusing on governance, documentation, and policy. The Risk Analyst and Ethics/Bias Auditor roles are more focused on the model (e.g., model risk, fairness, and transparency) than on the data itself, though there is some overlap in ensuring responsible data use.
- Although the current dataset does not contain sensitive information and is stored in a simple CSV format, a system was implemented to separate potentially sensitive data, restrict access via a user whitelist and script, and log all access events for auditing.
- The approach is scalable: access can be layered by creating different files or database views for different user groups, and the logging system can be adapted for use with databases as the project grows.
- A plan was established for incorporating sensitive data into the analysis, with clear policies outlined for handling each field responsibly.

### 6. Data Scientist
- **Reference Files:**
    - `src/modeling/models/random_forest_baseline.py`
    - `src/modeling/models/deep_learning/deep_learning_baseline.py`
    - `src/modeling/models/deep_learning/deep_learning_hyper_tune.py`
    - `src/modeling/models/deep_learning/deep_learning_tuned_evaluation.py`
    - `src/etl/etl_main.py` (due to its direct impact on feature availability for modeling)
    - `src/modeling/utils/data_loader.py` (due to its role in preparing data for models)
- **Description:** Responsible for the end-to-end modeling process, including selecting appropriate algorithms, developing baseline models, iteratively improving complex models, designing and interpreting robust evaluation strategies, and deriving insights from model performance to guide project direction.
- **Focus & Journey:**
    - **Baseline Establishment:** Our initial focus was on establishing a strong baseline using `random_forest_baseline.py`. This worked very effectively, yielding an RMSE of ~$24.5k and R² of ~0.90, confirming the predictiveness of the engineered features from `src/etl/etl_main.py`.
    - **Initial Deep Learning:** We then developed an initial `deep_learning_baseline.py`. A key challenge here was achieving good performance. Early iterations performed poorly until the introduction of log-transforming the target variable (`SalePrice`) and, crucially, implementing **Embedding layers** for nominal features. This required significant changes to `src/etl/etl_main.py` (to output integer-coded nominals) and `src/modeling/utils/data_loader.py` (to handle the new input structure for numerical and multiple nominal feature lists). This change was a major breakthrough, bringing the deep learning model's performance (RMSE ~$28.5k, R² ~0.87) close to the Random Forest.
    - **Hyperparameter Tuning Strategy:** The next step involved setting up systematic hyperparameter tuning using KerasTuner in `deep_learning_hyper_tune.py`. This was initially problematic due to a `use_bias=False` setting in Dense layers and an incorrect method for retrieving the best trained model, both of which led to extremely poor reported metrics. Debugging these issues was a critical focus. Once resolved, the tuner found effective hyperparameters.
    - **Robust Evaluation:** A major emphasis was placed on moving beyond single train-test splits to K-fold cross-validation. This was implemented in `deep_learning_tuned_evaluation.py` to provide a more reliable estimate of the tuned model's performance (Average RMSE ~$35.2k, R² ~0.77). This highlighted the difference between performance on a single validation set versus a more generalized K-fold average.
    - **What Worked Well:** The Random Forest baseline was highly effective. Embedding layers dramatically improved the deep learning model. K-fold CV provided valuable, realistic insights.
    - **Challenges & Resolutions:**
        - Getting the deep learning model to perform competitively with Random Forest required significant architectural changes (embedding layers) and corresponding ETL modifications.
        - Debugging the KerasTuner implementation (bias terms, model retrieval) was essential for obtaining meaningful tuning results.
        - Understanding and interpreting the variance in K-fold results compared to a single validation split.

### 7. Machine Learning Engineer
- **Reference Files:**
    - `src/modeling/models/deep_learning/deep_learning_baseline.py`
    - `src/modeling/models/deep_learning/deep_learning_hyper_tune.py`
    - `src/modeling/models/deep_learning/deep_learning_tuned_evaluation.py`
    - `src/modeling/models/traditional/random_forest_baseline.py`
    - `data/processed/final_deep_learning_model.h5`
    - `data/processed/random_forest_model.joblib`
    - `data/processed/rf_feature_names.joblib`
    - `data/processed/nominal_column_mappings.json`
    - `data/processed/dl_numerical_scaler.joblib`
    - `requirements.txt`
- **Description:** Responsible for the technical implementation, training, saving, and operationalization of machine learning models. Ensures all model artifacts and preprocessing objects are saved and accessible for deployment and app integration.
- **Focus & Journey:**
    - Managed deep learning framework setup, including resolving environment and package compatibility issues (standardizing on Python 3.11, `numpy==1.24.3`, `tensorflow==2.13.0`, `keras-tuner==1.4.6`).
    - Built deep learning models using the Keras Functional API to support multi-input data (numerical features and multiple nominal features with embedding layers).
    - Developed and debugged automated hyperparameter tuning with KerasTuner, ensuring the best trained model was correctly retrieved and evaluated.
    - Scripted reproducible K-fold evaluation for deep learning models, abstracting model creation logic for robust assessment.
    - Trained both deep learning and random forest models on the full dataset, saving all relevant files for deployment and app use.
    - Used random forest feature importance to select the most relevant features for the app UI, setting others to mean values.
    - Noted that the deep learning model's predictions were consistently low, likely due to missing data for many features or input encoding/scaling issues, while the random forest model produced more realistic and interpretable results.
    - Saving all relevant files (model, scaler, feature names, mappings) was essential for reproducibility and smooth handoff to the front-end team.
    - Switched from Gemini 2.5 to Auto for the modeling and app portion: Gemini was verbose and independent but sometimes hard to follow, while Auto provided a better balance of autonomy and collaboration, especially for complex or interactive tasks.

### 8. Front-End Engineer
- **Reference Files:**
    - `src/app/gradio_app.py` (Deep Learning app)
    - `src/app/rf_gradio_app.py` (Random Forest app)
    - `docs/gradio_app_guide.md` (User guide)
    - `data/processed/nominal_column_mappings.json` (for dropdowns)
- **Description:** Designs and implements the user-facing Gradio web applications for model inference. Focuses on usability, interpretability, and a clean user experience.
- **Focus & Journey:**
    - **App Development:** Built two Gradio apps, one for each model, allowing for direct comparison of predictions and user experience.
    - **UI/UX Improvements:**
        - Mapped full names to nominal values in dropdowns (e.g., "Northridge (noridge)") for interpretability.
        - Used the same friendly format for kitchen quality and neighborhood.
        - Removed decimals from default values for a cleaner look.
        - Ensured categorical dropdowns use human-friendly labels.
    - **Documentation:** Authored a user guide (`docs/gradio_app_guide.md`) for non-technical stakeholders, explaining how to use the app and interpret results.
    - **Lessons Learned:**
        - Building two apps highlighted the impact of model choice on user experience and interpretability.
        - Small UI changes (like label formatting and default value rounding) can significantly improve usability and stakeholder trust.
        - Collaboration with the ML Engineer was essential to ensure the app inputs matched the model's expectations and that all necessary files were available.

### 9. Visualization Expert / Data Storyteller
- **Reference Files:**
    - `src/regression_analysis_visualizations.py`
    - `figures/`
    - `docs/model_comparison_report.md`
    - `docs/model_comparison_report.pdf`
    - `spaces/random-forest-predictor/app.py`
- **Description:**  
  Designed and implemented visualizations to communicate model results and insights, and supported deployment of the Gradio app for stakeholder access.
- **Focus & Journey:**
    - Automated metric calculation and plot generation with a consistent, accessible style.
    - Integrated figures into the model comparison report for clear model comparison.
    - Supported Gradio app deployment to Hugging Face Spaces, ensuring visual and user experience consistency.
    - Refined plots and content based on feedback and deployment needs.
- **Lessons Learned:**
    - Deploying to Hugging Face required troubleshooting dependencies and file paths.
    - Collaboration with other roles was essential for a smooth workflow.
    - Understanding metrics like MAPE (Mean Absolute Percentage Error) is important: MAPE expresses the average prediction error as a percentage, making it intuitive for stakeholders. Including brief metric explanations in reports helps both the team and readers. For example, if the MAPE is 10%, it means that, on average, the model's predictions are off by 10% from the actual house prices.

### 10. Explainability Engineer
- **Reference Files:**
    - `src/regression_analysis_visualizations.py` (SHAP value computation and plotting)
    - `figures/shap_summary.png`
    - `docs/model_comparison_report.md`
    - `docs/model_comparison_report.pdf`
    - `docs/shap_explanation.md`
- **Description:**  
  Interprets and communicates model predictions using explainability tools, focusing on SHAP (SHapley Additive exPlanations) to provide transparency and insight into feature impacts.
- **Focus & Journey:**
    - Computed SHAP values for the Random Forest model to quantify how each feature contributed to individual and overall predictions.
    - Generated and interpreted SHAP summary plots, integrating them into the model comparison report.
- **Lessons Learned:**
    - SHAP values assign each feature a contribution to a specific prediction, showing how much each feature increases or decreases the predicted value compared to the average prediction. For example, a positive SHAP value for 'Overall Quality' means that a higher quality score increased the predicted house price for that instance.
    - Interpreting SHAP plots helps identify which features are most influential and how they impact predictions across the dataset.
    - In our Random Forest model, SHAP analysis showed that 'Overall Quality' and 'Total Square Footage' had the strongest positive impacts on price predictions, while age-related features had more complex, non-linear effects.

*Add additional roles and reference files as the project expands.* 