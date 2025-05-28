# Project Roles: Journey, Learnings, and Accomplishments

This document summarizes the key challenges, learnings, skill development, and accomplishments for each of the 10 defined roles within the Ames Housing Analysis project. It draws insights from the detailed `docs/analysis_summary.md` and the overall project progression.

## 1. Project Manager

-   **Challenges Encountered & Solutions:**
    -   Ensuring alignment between project objectives and stakeholder needs from the outset. Addressed by emphasizing early stakeholder identification.
-   **Significant Learning & Skill Development:**
    -   Deepened understanding of user-centered project management: identifying stakeholders *before* defining objectives, success metrics, and deliverables is crucial for impact and efficiency.
    -   Reinforced the value of a clear project roadmap for guiding complex data analysis projects.
-   **Key Accomplishments/Contributions:**
    -   Developed the initial `project_roadmap.md`, outlining objectives, stakeholders, success metrics, and a detailed project plan.

## 2. Stakeholder Liaison

-   **Challenges Encountered & Solutions:**
    -   The role demands a very broad skill set, including strong communication, technical understanding, organizational abilities, and thorough documentation.
-   **Significant Learning & Skill Development:**
    -   Recognized the centrality of communication in bridging the gap between technical teams and business/subject matter experts.
    -   Understood that the liaison's value is magnified in larger, more technically complex projects with diverse stakeholders.
    -   Learned that this role is key not just for initial alignment but also for ongoing user education, training, and facilitating a continuous feedback loop.
-   **Key Accomplishments/Contributions:**
    -   Developed `stakeholder_liaison.md`, outlining communication strategies and the pivotal role in ensuring deliverables meet stakeholder needs.

## 3. Data Engineer

-   **Challenges Encountered & Solutions:**
    -   Distinguishing between and correctly encoding ordinal vs. nominal categorical features; resolved by deeper analysis and implementing appropriate encoding strategies (label for ordinal, one-hot for nominal).
    -   Incorrect sequencing of new engineered features (e.g., `TotalSF` added after encoding); addressed by debugging data flow and reordering steps.
    -   Adapting the ETL pipeline (`src/etl/etl_main.py`) based on evolving modeling requirements from the Data Scientist role, such as the need for integer-coded nominal features for deep learning embedding layers. This required iterative adjustments to data output formats.
    -   AI prompt engineering for focused tasks; required iterative refinement to get specific data engineering outputs rather than full pipeline scripts.
    -   Ensuring creation of essential documentation like data dictionaries and schemas; addressed by specific requests.
-   **Significant Learning & Skill Development:**
    -   Importance of clear communication, iterative refinement, and robust documentation for building a transparent and reliable data pipeline.
    -   Practical experience in debugging data flows and understanding the impact of transformation sequences.
    -   Reinforced understanding of the iterative nature of ETL development, where modeling needs often drive further data preparation refinements.
-   **Key Accomplishments/Contributions:**
    -   Developed `src/etl/etl_main.py` for data cleaning and transformation.
    -   Authored `src/etl/etl_documentation.md`, `data_dictionary.md`, and `data_schema.json`.
    -   Produced the `data/processed/housing_cleaned.csv` dataset, foundational for all modeling efforts.

## 4. Data Acquisition Specialist

-   **Challenges Encountered & Solutions:**
    -   Lack of direct property identifiers (addresses, lat/lon) in the Ames dataset, making automated school matching difficult; approximated by manual neighborhood-to-school mapping.
    -   Failed attempts to scrape school metrics due to JavaScript rendering and anti-bot measures; resorted to manual data collection.
    -   Difficulty finding reliable, neighborhood-level crime data for direct integration.
-   **Significant Learning & Skill Development:**
    -   Recognized the need for robust data sources (APIs, official data) or advanced scraping techniques (e.g., browser automation) for scalable data acquisition.
    -   Importance of thoroughly documenting data acquisition challenges and processes to inform future efforts.
-   **Key Accomplishments/Contributions:**
    -   Acquired and processed school data, creating `data/processed/neighborhood_schools.csv` and `data/processed/schools_metrics.csv`.
    -   Produced `data/processed/schools_combined.csv` for potential joining with housing data.
    -   Authored `future_acquisition_plan.md`, detailing the process, challenges, and recommendations.

## 5. Data Steward / Governance Officer

-   **Challenges Encountered & Solutions:**
    -   The project's primary dataset was a single CSV file without inherently sensitive information. This simplicity meant that a governance system was implemented for *potential* or future sensitive data, rather than managing complex database views or existing granular access permissions.
-   **Significant Learning & Skill Development:**
    -   Clarified the distinction between Data Steward (governance, policy, documentation) and Data Engineer (processing) or model-focused risk roles.
    -   Developed an understanding of how to design scalable access control, logging, and monitoring systems, even for simple data storage.
    -   Importance of establishing clear policies for handling sensitive data fields.
-   **Key Accomplishments/Contributions:**
    -   Implemented `src/governance/access_control.py` and `src/governance/access_monitor.py`.
    -   Set up `data/sensitive/data_access_log.db` for logging access.
    -   Created `src/governance/data_access_guide.md` and `src/governance/steward_guide.md`.
    -   Established a plan for incorporating sensitive data responsibly.

## 6. Data Scientist

-   **Challenges Encountered & Solutions:**
    -   Initial poor performance of the deep learning model; resolved by log-transforming the target variable and implementing Embedding layers for nominal features. This was a significant breakthrough that required close collaboration with the Data Engineer to modify `src/etl/etl_main.py` (to output integer-coded nominals) and `src/modeling/utils/data_loader.py` (to handle the new input structure).
    -   Debugging KerasTuner implementation (e.g., `use_bias=False` issue, incorrect model retrieval method) which led to misleading initial tuning results.
-   **Significant Learning & Skill Development:**
    -   Confirmed the effectiveness of Random Forest as a strong baseline.
    -   Learned the dramatic positive impact of Embedding layers on deep learning model performance for datasets with categorical features.
    -   Gained practical experience in systematic hyperparameter tuning and robust K-fold cross-validation.
    -   Understood the importance of interpreting variance in K-fold results versus performance on a single validation split.
-   **Key Accomplishments/Contributions:**
    -   Developed `random_forest_baseline.py` establishing a strong performance benchmark.
    -   Iteratively developed and improved deep learning models (`deep_learning_baseline.py`, `deep_learning_hyper_tune.py`, `deep_learning_tuned_evaluation.py`).
    -   Spearheaded crucial modifications to `src/etl/etl_main.py` and `src/modeling/utils/data_loader.py` to support advanced deep learning architectures.

## 7. Machine Learning Engineer

-   **Challenges Encountered & Solutions:**
    -   Setting up a consistent deep learning framework: This involved resolving environment and package compatibility issues (e.g., Python version, numpy, TensorFlow), including managing the transition from a Conda environment to a project-local `venv` and restructuring project files accordingly.
    -   Iterative model debugging influencing data loading: For instance, while debugging the deep learning model, it was necessary to adapt data loading mechanisms initially designed for the Random Forest model to ensure universal compatibility or to isolate model-specific issues, highlighting cross-model development interdependencies.
    -   Deep learning model predictions were initially consistently low, potentially due to how missing data was handled for numerous features or input encoding/scaling issues.
-   **Significant Learning & Skill Development:**
    -   Initial exposure to the Keras Functional API for building multi-input deep learning models.
    -   Gained familiarity with model serialization and common file formats (e.g., `.h5`, `.joblib`) for saving and loading trained models.
    -   Hands-on experience debugging automated hyperparameter tuning processes and adapting data loading strategies for different model architectures and debugging needs.
    -   Critical importance of saving all model artifacts (model files, scalers, feature names, mappings) for reproducibility, deployment, and app integration.
    -   Observed differences in AI interaction styles (Gemini vs. Auto) and their suitability for different tasks.
-   **Key Accomplishments/Contributions:**
    -   Successfully trained and saved both Random Forest (`random_forest_model.joblib`) and Deep Learning (`final_deep_learning_model.h5`) models.
    -   Saved all associated preprocessing objects and artifacts (`rf_feature_names.joblib`, `nominal_column_mappings.json`, `dl_numerical_scaler.joblib`).
    -   Ensured `requirements.txt` reflected necessary dependencies for model execution.

## 8. Front-End Engineer

-   **Challenges Encountered & Solutions:**
    -   Ensuring the Gradio app inputs correctly mapped to the expectations of the trained models, requiring close collaboration with the ML Engineer.
    -   Ongoing consideration of which features are most relevant and user-friendly for an interactive prediction app, recognizing that app design is an iterative process involving potential additions of visualizations and enhancements to overall usefulness.
-   **Significant Learning & Skill Development:**
    -   Understanding how model choice (e.g., Random Forest vs. Deep Learning) can impact user experience and the design of interactive prediction tools.
    -   Recognized that small UI/UX improvements (like human-readable labels for dropdowns, rounding default values) significantly enhance usability and stakeholder trust.
-   **Key Accomplishments/Contributions:**
    -   Developed two Gradio applications: `src/app/rf_gradio_app.py` (for Random Forest) and `src/app/gradio_app.py` (for Deep Learning).
    -   Implemented UI/UX enhancements for better interpretability and a cleaner user experience.
    -   Authored `docs/gradio_app_guide.md` for non-technical users.

## 9. Visualization Expert / Data Storyteller

-   **Challenges Encountered & Solutions:**
    -   Deploying the Gradio app (developed by Front-End Engineer) to Hugging Face Spaces involved troubleshooting dependency issues and file path configurations.
    -   Making complex evaluation metrics like MAPE intuitive for stakeholders; addressed by adding clear, example-based explanations.
-   **Significant Learning & Skill Development:**
    -   Practical experience in troubleshooting deployment issues for web applications in cloud environments like Hugging Face.
    -   Reinforced the importance of consistent styling and clear labeling in visualizations for effective communication.
    -   Better understanding of how to explain statistical metrics (e.g., MAPE) in an accessible way.
    -   Appreciated that a key goal of data storytelling is to produce clear, tangible reports (like `docs/model_comparison_report.md`) that document findings, created artifacts, and their current utility for stakeholders, recognizing this often requires iterative refinement.
-   **Key Accomplishments/Contributions:**
    -   Developed `src/regression_analysis_visualizations.py` to automate model evaluation plots and visual summaries.
    -   Generated numerous figures (stored in `figures/`) and integrated them into `docs/model_comparison_report.md`.
    -   Supported the successful deployment of the Random Forest Gradio app to Hugging Face Spaces.

## 10. Explainability Engineer

-   **Challenges Encountered & Solutions:**
    -   Communicating the concepts behind SHAP values and their interpretation to a non-technical audience. Addressed by creating a dedicated explanation document with analogies.
-   **Significant Learning & Skill Development:**
    -   Gained initial experience in using and interpreting SHAP values to understand model behavior, particularly for tree-based models.
    -   Learned the value of combining visual SHAP plots with narrative explanations for maximum clarity.
    -   Recognized how SHAP analysis can reveal non-linear feature impacts and interaction effects not evident from standard feature importance.
-   **Key Accomplishments/Contributions:**
    -   Performed SHAP analysis for the Random Forest model.
    -   Generated `figures/shap_summary.png` and integrated it into the model comparison report.
    -   Authored `docs/shap_explanation.md` to make SHAP concepts accessible.