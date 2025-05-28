"""
Baseline Random Forest model for Ames Housing price prediction.
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
import sys
import os
import joblib

# Get the directory containing the 'src' folder (which is the project root)
# For a script in src/modeling/models/traditional/script.py:
# os.path.dirname(__file__) is src/modeling/models/traditional
# '..' -> src/modeling/models
# '..' -> src/modeling
# '..' -> src
# '..' -> ames_housing (project root)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR) # Insert src at the beginning of the path

from modeling.utils.data_loader import load_housing_data
from modeling.utils.validation import perform_kfold_cv

def create_rf_model(random_state: int = 42) -> RandomForestRegressor:
    """
    Create a baseline Random Forest Regressor model.
    
    Args:
        random_state: Seed for reproducibility.
        
    Returns:
        RandomForestRegressor model instance.
    """
    model = RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1) # Use all available cores
    return model

def main():
    print("Loading and preparing data for Random Forest...")
    (
        numerical_ordinal_features_np, 
        nominal_features_list_np, 
        target_np, 
        numerical_ordinal_headers,
        nominal_headers_ordered,
        _ # nominal_cardinalities not directly used by RF
    ) = load_housing_data(
        'data/processed/housing_cleaned.csv' # Path relative to project root
    )

    # Combine numerical/ordinal features with integer-coded nominal features
    # The nominal_features_list_np contains arrays of shape (n_samples, 1)
    # We need to hstack them with numerical_ordinal_features_np

    if nominal_features_list_np: # If there are nominal features
        # Ensure all nominal feature arrays are 2D before hstack
        squeezed_nominal_features = [nf.reshape(-1, 1) if nf.ndim == 1 else nf for nf in nominal_features_list_np]
        all_nominal_features_np = np.hstack(squeezed_nominal_features)
        features_combined_np = np.hstack((numerical_ordinal_features_np, all_nominal_features_np))
        combined_feature_names = numerical_ordinal_headers + nominal_headers_ordered
    else:
        features_combined_np = numerical_ordinal_features_np
        combined_feature_names = numerical_ordinal_headers

    # For Random Forest, target is not log-transformed as per original script intention
    target_rf = target_np 
    
    print(f"Combined features shape for RF: {features_combined_np.shape}")
    print(f"Target shape for RF: {target_rf.shape}")
    print(f"Number of combined feature names: {len(combined_feature_names)}")

    # Create model builder function
    def model_builder():
        return create_rf_model(random_state=42)
    
    # Perform k-fold cross-validation
    # perform_kfold_cv will use StandardScaler as currently implemented
    print("\nStarting Random Forest K-fold Cross-Validation...")
    fold_metrics, avg_metrics = perform_kfold_cv(
        features=features_combined_np, # Use the combined features
        target=target_rf, 
        model_builder=model_builder,
        n_splits=5,
        random_state=42,
        target_is_log_transformed=False
    )
    
    print("\nAverage Performance Metrics for Random Forest:")
    print(f"RMSE: ${avg_metrics['rmse']:,.2f} ± ${avg_metrics['rmse_std']:,.2f}")
    print(f"R² Score: {avg_metrics['r2']:.4f} ± {avg_metrics['r2_std']:.4f}")
    print(f"MAPE: {avg_metrics['mape']:.2f}% ± {avg_metrics['mape_std']:.2f}%")

    # --- Feature Importances ---
    print("\nCalculating feature importances for Random Forest...")
    # Train a model on the full dataset to get stable importances
    full_data_model = create_rf_model(random_state=42)
    full_data_model.fit(features_combined_np, target_rf)

    importances = full_data_model.feature_importances_
    feature_importance_pairs = list(zip(combined_feature_names, importances))

    # Sort features by importance
    feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)

    print("\nTop 20 most important features for Random Forest:")
    for i, (feature, importance) in enumerate(feature_importance_pairs[:20]):
        print(f"{i+1}. {feature}: {importance:.4f}")

    # Save the model and feature names
    print("\nSaving the Random Forest model and feature names...")
    model_save_path = os.path.join(PROJECT_ROOT, 'data', 'processed', 'random_forest_model.joblib')
    feature_names_save_path = os.path.join(PROJECT_ROOT, 'data', 'processed', 'rf_feature_names.joblib')
    
    # Save the model
    joblib.dump(full_data_model, model_save_path)
    # Save the feature names
    joblib.dump(combined_feature_names, feature_names_save_path)
    
    print(f"Model saved to: {model_save_path}")
    print(f"Feature names saved to: {feature_names_save_path}")

if __name__ == "__main__":
    main() 