"""
Validation utilities for model evaluation.
"""

import numpy as np
import tensorflow as tf
from typing import Tuple, List, Dict, Callable
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

def perform_kfold_cv(
    features: np.ndarray,
    target: np.ndarray,
    model_builder: Callable,
    n_splits: int = 5,
    random_state: int = 42,
    target_is_log_transformed: bool = False
) -> Tuple[List[Dict[str, float]], Dict[str, float]]:
    """
    Perform k-fold cross-validation.
    
    Args:
        features: Feature matrix
        target: Target vector
        model_builder: Function that returns a new model instance
        n_splits: Number of folds
        random_state: Random seed for reproducibility
        target_is_log_transformed: Flag indicating if the target is log-transformed
        
    Returns:
        List of metrics for each fold
        Dictionary of average metrics across folds
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_metrics = []
    
    print(f"\nPerforming {n_splits}-fold cross-validation...")
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(features), 1):
        print(f"\nFold {fold}/{n_splits}")
        
        # Split data
        X_train, X_val = features[train_idx], features[val_idx]
        y_train, y_val = target[train_idx], target[val_idx]
        
        # DEBUG: Check for NaNs or Infs before scaling
        if np.any(np.isnan(X_train)) or np.any(np.isinf(X_train)):
            print("WARNING: NaNs or Infs found in X_train before scaling!")
        if np.any(np.isnan(y_train)) or np.any(np.isinf(y_train)):
            print("WARNING: NaNs or Infs found in y_train!")
        
        # Scale features using StandardScaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Get new model instance
        model = model_builder()
        
        # Train and evaluate
        if hasattr(model, 'fit'):  # scikit-learn style
            model.fit(X_train_scaled, y_train)
            predictions = model.predict(X_val_scaled)
        else:  # Assuming tensorflow model
            history = model.fit(
                X_train_scaled, y_train,
                epochs=200,
                batch_size=32,
                validation_data=(X_val_scaled, y_val),
                verbose=0,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(
                        monitor='val_loss',
                        patience=15,
                        restore_best_weights=True
                    )
                ]
            )
            predictions = model.predict(X_val_scaled, verbose=0)
            if len(predictions.shape) > 1:
                predictions = predictions.flatten()
        
        if target_is_log_transformed:
            # DEBUG: Check predictions before inverse transform
            if np.any(np.isnan(predictions)) or np.any(np.isinf(predictions)):
                print(f"WARNING: Fold {fold} - NaNs or Infs found in log-scale predictions BEFORE expm1!")
                print(f"Sample predictions (log-scale): {predictions[:5]}")
            predictions_final = np.expm1(predictions)
            y_val_final = np.expm1(y_val)
        else:
            predictions_final = predictions
            y_val_final = y_val

        # Calculate metrics on the final scale (original or log, depending on what y_val_final is)
        # For consistent interpretation (RMSE in dollars, etc.), y_val_final should be original scale.
        mse = np.mean((y_val_final - predictions_final) ** 2)
        rmse = np.sqrt(mse)
        
        # Calculate R² score on the final scale
        ss_res = np.sum((y_val_final - predictions_final) ** 2)
        ss_tot = np.sum((y_val_final - np.mean(y_val_final)) ** 2)
        # Add a check for ss_tot to avoid division by zero if y_val_final has no variance
        if ss_tot == 0:
            r2 = 0.0 if ss_res == 0 else -np.inf # Or handle as appropriate, e.g. R2 = 0 if res and tot are 0
        else:
            r2 = 1 - (ss_res / ss_tot)
        
        # Calculate MAPE on the final scale, handle potential division by zero
        epsilon = 1e-8  # A small number to avoid division by zero
        # Ensure y_val_safe is based on y_val_final to match the scale of predictions_final
        y_val_safe = np.where(y_val_final == 0, epsilon, y_val_final)
        # Check for NaNs in inputs to mape calculation that might arise from expm1 overflow if not handled
        if np.any(np.isnan(y_val_final)) or np.any(np.isnan(predictions_final)):
            mape = np.nan # Can't compute mape if there are NaNs
            print(f"WARNING: Fold {fold} - NaNs detected in y_val_final or predictions_final before MAPE calc. MAPE set to NaN.")
        else:
            mape_values = np.abs((y_val_final - predictions_final) / y_val_safe)
            mape = np.mean(mape_values) * 100
        
        metrics = {
            'rmse': rmse,
            'r2': r2,
            'mape': mape
        }
        
        fold_metrics.append(metrics)
        print(f"Fold {fold} Metrics:")
        print(f"RMSE: ${metrics['rmse']:,.2f}")
        print(f"R² Score: {metrics['r2']:.4f}")
        print(f"MAPE: {metrics['mape']:.2f}%")
    
    # Calculate average metrics
    avg_metrics = {
        metric: np.mean([fold[metric] for fold in fold_metrics])
        for metric in fold_metrics[0].keys()
    }
    
    # Calculate standard deviations
    std_metrics = {
        f"{metric}_std": np.std([fold[metric] for fold in fold_metrics])
        for metric in fold_metrics[0].keys()
    }
    
    avg_metrics.update(std_metrics)
    
    return fold_metrics, avg_metrics 