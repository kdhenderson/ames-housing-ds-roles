# Ames Housing Modeling

This directory contains the scripts and utilities for developing, training, tuning, and evaluating predictive models for the Ames Housing dataset. It includes both traditional machine learning approaches (Random Forest) and deep learning models.

## Project Structure within `src/modeling/`

```
src/modeling/
├── README.md            # This overview document
├── models/              # Model implementation code
│   ├── deep_learning/   # Deep learning model implementations
│   │   ├── deep_learning_baseline.py
│   │   ├── deep_learning_hyper_tune.py
│   │   └── deep_learning_tuned_evaluation.py
│   └── random_forest_baseline.py # Random Forest baseline model
├── utils/               # Utility functions for modeling
│   ├── data_loader.py   # Script for loading and preparing data for models
│   └── validation.py    # Script containing validation helper functions (e.g., custom metrics)
├── analysis/            # Placeholder for scripts related to model analysis and visualization
└── __init__.py          # Makes src/modeling a Python package
```

*(Note: `keras_tuner_dir/` which stores KerasTuner trial outputs is located in the project root directory.)*

## 1. Overview of Modeling Approaches

The project explores two primary modeling approaches to predict house sale prices:
1.  A **Random Forest Regressor** serving as a robust baseline model.
2.  A **Deep Learning model** utilizing neural networks, with a focus on leveraging embeddings for categorical features.

The goal is to compare their performance and understand the benefits and trade-offs of each approach for this dataset.

## 2. Data Loading and Preprocessing

The data journey from raw files to model input involves two main stages: initial ETL processing and model-specific loading/preprocessing.

### 2.1. Initial ETL Processing (`src/etl/etl_main.py`)

*   **Input:** Raw housing data from `data/raw/housing_raw.csv`.
*   **Key Operations:**
    *   Removal of identifier columns (e.g., "Order", "PID").
    *   Handling of missing values:
        *   Numeric features: Imputed with the mean.
        *   Categorical features: Imputed with the mode.
    *   Feature Engineering: Creation of `TotalSF` by summing basement, first, and second-floor square footage.
    *   Categorical Variable Encoding:
        *   **Ordinal Variables:** Label encoded based on predefined order (e.g., 'Exter Qual' mapped from 'Po' to 'Ex').
        *   **Nominal Variables:** Integer encoded (e.g., 'MS Zoning' categories mapped to 0, 1, 2...). This prepares them for further processing by the model loader, especially for embedding layers in the neural network.
*   **Output:** A cleaned dataset saved as `data/processed/housing_cleaned.csv`. This file serves as the primary input for the modeling phase.

### 2.2. Model-Specific Data Loading (`src/modeling/utils/data_loader.py`)

The `load_housing_data` function in this script is responsible for loading `housing_cleaned.csv` and preparing features and targets for the models.

*   **Input:** `data/processed/housing_cleaned.csv`.
*   **Feature Categorization:**
    *   **`NOMINAL_VARS_FOR_EMBEDDING`**: A predefined list of nominal features (e.g., 'Neighborhood', 'Garage Type') that are intended for use with embedding layers in the Deep Learning model. These are already integer-encoded by `etl_main.py`. `load_housing_data` separates these into a list of arrays, one for each nominal feature, and calculates their cardinalities.
    *   **`ORDINAL_VARS_AS_NUMERIC`**: A predefined list of ordinal features (e.g., 'Exter Qual', 'Kitchen Qual') that were label-encoded by `etl_main.py` into numerical representations. These are treated as standard numerical features.
    *   **Other Numerical Features**: Any column not 'SalePrice' and not in the nominal or ordinal lists (e.g., 'Lot Area', 'Year Built', 'Garage Cars', and the engineered 'TotalSF') is treated as a standard numerical feature.
*   **Output Structure:** The function returns:
    *   `numerical_ordinal_features`: A NumPy array containing all features treated as numeric (this includes actual numerical features and the label-encoded ordinal features).
    *   `nominal_features_list`: A list of NumPy arrays, where each array is a column vector for a single integer-encoded nominal feature. This structure is ready for input into a neural network with multiple embedding layers.
    *   `target`: A NumPy array of 'SalePrice'.
    *   Headers and cardinalities for the nominal features.

### 2.3. Feature Handling by Model Type:

*   **Deep Learning Model:**
    *   Utilizes the `numerical_ordinal_features` directly as one input.
    *   Takes each array from `nominal_features_list` as a separate input, feeding them into distinct Embedding layers. The cardinalities provided by `data_loader.py` are crucial for defining the `input_dim` of these embedding layers.
    *   This approach allows the model to learn dense vector representations (embeddings) for each categorical feature, capturing complex relationships.

*   **Random Forest Model:**
    *   Typically, Random Forest models in scikit-learn can handle integer-encoded categorical features directly alongside numerical features.
    *   The current setup from `data_loader.py` provides numerical/ordinal features in one block and nominal features as a list of separate arrays. For a standard scikit-learn Random Forest, these would need to be recombined into a single 2D feature matrix. The `random_forest_baseline.py` script handles this recombination.
    *   The integer encoding performed by `etl_main.py` for nominal variables is generally suitable for tree-based models, as they can make splits based on these integer codes. No explicit one-hot encoding is performed by the `data_loader` for the Random Forest.

## 3. Model Descriptions & Key Scripts

This section details the implemented models and points to their respective scripts.

### 3.1. Random Forest Baseline (`src/modeling/models/random_forest_baseline.py`)
*   **Description:** This model serves as a strong baseline to compare against the Deep Learning approach. It uses the `RandomForestRegressor` from scikit-learn. Features are a combination of numerical, label-encoded ordinal, and integer-encoded nominal variables, all treated as a single feature matrix.
*   **Key Script:** `models/random_forest_baseline.py`

### 3.2. Deep Learning Models (`src/modeling/models/deep_learning/`)
*   **Description:** These models are built using Keras/TensorFlow. They employ a functional API structure to handle multiple inputs: one for numerical/ordinal features and separate inputs for each nominal feature (for embedding layers). The outputs of the embedding layers are concatenated with the numerical features and then passed through Dense layers for prediction.
*   **Key Scripts:**
    *   `models/deep_learning/deep_learning_baseline.py`: Implements an initial deep learning model with K-fold cross-validation.
    *   `models/deep_learning/deep_learning_hyper_tune.py`: Handles hyperparameter tuning using KerasTuner.
    *   `models/deep_learning/deep_learning_tuned_evaluation.py`: Performs K-fold cross-validation of the tuned deep learning model.

## 4. Hyperparameter Tuning

*   **Deep Learning Model:** KerasTuner is used for hyperparameter optimization as detailed in `models/deep_learning/deep_learning_hyper_tune.py`. Tuned hyperparameters include learning rate, number of units in dense layers, embedding dimensions, and dropout rates. Results are stored in `keras_tuner_dir/` (project root).
*   **Random Forest Model:** Hyperparameter tuning (e.g., `GridSearchCV`, `RandomizedSearchCV`) is not explicitly implemented in the current baseline script but can be added.

## 5. Validation Strategy (`src/modeling/utils/validation.py`)

*   **Metrics:** Key performance metrics include Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and Mean Absolute Percentage Error (MAPE), available in `utils/validation.py`.
*   **Techniques:**
    *   **Train-Test Split:** Standard splitting is performed by `src/modeling/utils/data_loader.py`'s `train_test_split` function.
    *   **K-Fold Cross-Validation:** Implemented in scripts like `deep_learning_baseline.py` and `deep_learning_tuned_evaluation.py`, using helper functions from `utils/validation.py` for robust performance estimation.

## 6. Final Model Training, Artifacts, and Lessons Learned

- **Model Training & Saving:** Both the deep learning and random forest models were trained on the full dataset. All relevant model files and preprocessing artifacts were saved in `data/processed/` for deployment and app integration:
    - Deep Learning model: `final_deep_learning_model.h5`
    - Random Forest model: `random_forest_model.joblib`
    - Feature order for RF: `rf_feature_names.joblib`
    - Nominal/categorical mappings: `nominal_column_mappings.json`
    - Numerical scaler for DL: `dl_numerical_scaler.joblib`
- **Feature Selection:** Variable importance from the random forest model was used to select the most relevant features for the app UI, with the rest set to mean values from the training data.
- **Model Performance:** The deep learning model's predictions were consistently low, likely due to missing data for many features or input encoding/scaling issues. The random forest model produced more realistic and interpretable results and was preferred for deployment.
- **Reproducibility & Handoff:** Saving all relevant files (model, scaler, feature names, mappings) was essential for reproducibility and smooth handoff to the front-end team for app development.

## Implementation Phases (Conceptual)

### Phase 1: Baseline & Initial Models (Largely Completed)
- Implement Random Forest baseline.
- Set up initial deep learning architecture with embedding layers.
- Establish K-fold cross-validation framework.

### Phase 2: Model Enhancement & Tuning (In Progress/Completed)
- Perform hyperparameter tuning for deep learning models (using KerasTuner).
- Conduct robust statistical validation of tuned models.

### Phase 3: Finalization & Analysis (Future Work)
- Comprehensive comparison of final model performances.
- In-depth error analysis and feature importance studies (e.g., using `analysis/` scripts).
- Final documentation of model findings and insights.

## Role Responsibilities (Context: Modeling Tasks)

-   **Data Scientist (Modeler):** Feature analysis impact, model selection, interpretation, performance metrics, deriving business insights from models.
-   **Advanced Statistician (Conceptual):** Guiding statistical validation, uncertainty quantification, model diagnostics.
-   **Machine Learning Engineer:** Model implementation (esp. deep learning), pipeline development (tuning, evaluation scripts), managing ML framework dependencies, ensuring reproducibility.

## Getting Started

Ensure `requirements.txt` (in the project root) are installed in your virtual environment.
The ETL pipeline (`src/etl/etl_main.py`) must be run first to generate `data/processed/housing_cleaned.csv`.

1.  **Run the Random Forest baseline:**
    ```bash
    python src/modeling/models/random_forest_baseline.py
    ```

2.  **Run the initial Deep Learning baseline (K-fold CV):**
    ```bash
    python src/modeling/models/deep_learning/deep_learning_baseline.py
    ```

3.  **Run Deep Learning Hyperparameter Tuning (KerasTuner):**
    ```bash
    python src/modeling/models/deep_learning/deep_learning_hyper_tune.py
    ```
    *(This will generate/update `keras_tuner_dir/` in the project root.)*

4.  **Run K-fold Evaluation of the Tuned Deep Learning Model:**
    *(Ensure best hyperparameters are set within the script first, based on KerasTuner output.)*
    ```bash
    python src/modeling/models/deep_learning/deep_learning_tuned_evaluation.py
    ```

## Data Sources
-   Primary processed housing data: `data/processed/housing_cleaned.csv` (generated by `src/etl/etl_main.py`)

## Further Documentation
For a higher-level overview of all project roles and their contributions, including detailed narratives of the modeling journey, see `docs/analysis_summary.md` in the main `docs/` directory. This current README focuses specifically on the `src/modeling/` scope.

---
*This README should be updated as the modeling process evolves.* 