"""
Evaluation of the Deep Learning model using best hyperparameters from KerasTuner
with K-fold cross-validation.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
# import keras_tuner as kt # No longer needed
import sys
import os
from sklearn.model_selection import KFold # train_test_split might still be useful for an initial holdout
from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestRegressor # Not needed for this script
# from sklearn.feature_selection import SelectFromModel # Not needed for this script
from sklearn.metrics import mean_squared_error, r2_score

# Get the directory containing the 'src' folder (which is the project root)
# For a script in src/modeling/models/deep_learning/script.py:
# os.path.dirname(__file__) is src/modeling/models/deep_learning
# '..' -> src/modeling/models
# '..' -> src/modeling
# '..' -> src
# '..' -> ames_housing (project root)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR) # Insert src at the beginning of the path

from modeling.utils.data_loader import load_housing_data
# We are removing the perform_kfold_cv import as logic is now internal for multi-input
# from modeling.utils.validation import perform_kfold_cv 

# Custom MAPE, adjusted to handle actuals close to zero and log-transformed predictions
def mean_absolute_percentage_error(y_true, y_pred_log, target_is_log_transformed):
    if target_is_log_transformed:
        y_true_original = np.expm1(y_true) # y_true is log(actual + 1)
        y_pred_original = np.expm1(y_pred_log) # y_pred is log(prediction + 1)
    else: # Should not happen with current setup, but good for robustness
        y_true_original = y_true
        y_pred_original = y_pred_log

    # Filter out cases where true value is zero or very close to zero to avoid division by zero or extreme MAPE values.
    # This threshold can be adjusted.
    mask = np.abs(y_true_original) > 1e-8 
    if not np.any(mask):
        return np.nan # Or some other indicator of an issue / all-zero true values

    y_true_filtered = y_true_original[mask]
    y_pred_filtered = y_pred_original[mask]
    
    return np.mean(np.abs((y_true_filtered - y_pred_filtered) / y_true_filtered)) * 100

def create_model_with_fixed_hps(numerical_input_shape, 
                                nominal_input_specs,
                                learning_rate_val,
                                num_dense_layers_val,
                                units_l_vals, # List or dict: e.g., [128, 64, 32]
                                dropout_l_vals # List or dict: e.g., [0.4, 0.3, 0.0]
                                ) -> tf.keras.Model:
    """
    Create a deep learning model with fixed hyperparameters.
    
    Args:
        numerical_input_shape: Shape of the numerical input.
        nominal_input_specs: List of tuples for nominal features (name, cardinality, emb_dim).
        learning_rate_val: The learning rate.
        num_dense_layers_val: Number of dense layers.
        units_l_vals: List of unit counts for each dense layer.
        dropout_l_vals: List of dropout rates for each dense layer.
    Returns:
        Compiled Keras functional API model.
    """
    numerical_input = layers.Input(shape=numerical_input_shape, name='numerical_input')
    current_numerical_branch = numerical_input
    
    embedding_outputs = []
    nominal_keras_inputs = []

    for name, cardinality, base_emb_dim in nominal_input_specs:
        sanitized_name = name.replace(' ', '_').replace('-', '_neg_').replace('/', '_slash_').replace('.', '_dot_')
        emb_dim = base_emb_dim 
        
        nominal_input_layer = layers.Input(shape=(1,), name=f'{sanitized_name}_input')
        nominal_keras_inputs.append(nominal_input_layer)
        embedding_layer = layers.Embedding(input_dim=cardinality,
                                           output_dim=emb_dim,
                                           name=f'{sanitized_name}_embedding')(nominal_input_layer)
        flattened_embedding = layers.Flatten(name=f'{sanitized_name}_flatten')(embedding_layer)
        embedding_outputs.append(flattened_embedding)
        
    if embedding_outputs:
        concatenated_features = layers.concatenate([current_numerical_branch] + embedding_outputs)
    else:
        concatenated_features = current_numerical_branch
    
    x = concatenated_features
    # Ensure lengths of units_l_vals and dropout_l_vals match num_dense_layers_val
    if len(units_l_vals) != num_dense_layers_val or len(dropout_l_vals) != num_dense_layers_val:
        raise ValueError("Length of units_l_vals and dropout_l_vals must match num_dense_layers_val")

    for i in range(num_dense_layers_val):
        x = layers.Dense(units=units_l_vals[i])(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        # Only add dropout if rate is > 0, as Dropout layer with 0 rate can sometimes cause issues or is just redundant.
        if dropout_l_vals[i] > 0.0: 
            x = layers.Dropout(rate=dropout_l_vals[i])(x)

    output_layer = layers.Dense(1, name='output')(x)
    all_inputs = [numerical_input] + nominal_keras_inputs
    model = models.Model(inputs=all_inputs, outputs=output_layer)
    
    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate_val),
        loss='mse',
        metrics=['mae']
    )
    return model

def main():
    print("Loading data for K-fold cross-validation with tuned hyperparameters...")
    (numerical_ordinal_features_all, 
     nominal_features_list_all, 
     target_all, 
     _, 
     nominal_headers, 
     nominal_cardinalities) = load_housing_data('data/processed/housing_cleaned.csv')

    print("Log-transforming target variable.")
    target_log_all = np.log1p(target_all)

    nominal_input_specs_static = []
    for header in nominal_headers:
        cardinality = nominal_cardinalities[header]
        emb_dim = min(50, cardinality // 2 if cardinality > 1 else 1)
        if emb_dim == 0: emb_dim = 1
        nominal_input_specs_static.append((header, cardinality, emb_dim))

    # --- BEGIN USER INPUT REQUIRED ---
    # Please replace these with the actual best hyperparameters from your KerasTuner run.
    # Example:
    # best_hyperparameters = {
    #     'learning_rate': 0.001,
    #     'num_dense_layers': 3,
    #     'units_l1': 128, 'units_l2': 64, 'units_l3': 32, 
    #     'dropout_l1': 0.4, 'dropout_l2': 0.3, 'dropout_l3': 0.0 
    # }
    # Make sure the number of 'units_lx' and 'dropout_lx' matches 'num_dense_layers'.

    # Placeholder - replace with your actual values
    best_hyperparameters = {
        'learning_rate': 0.01,
        'num_dense_layers': 2,
        'units_l1': 96, 'units_l2': 160, # num_dense_layers is 2, so only l1 and l2 are used.
        'dropout_l1': 0.2, 'dropout_l2': 0.2 # num_dense_layers is 2, so only l1 and l2 are used.
    }
    # --- END USER INPUT REQUIRED ---

    # Extract hyperparameter values for the create_model_with_fixed_hps function
    lr = best_hyperparameters['learning_rate']
    num_layers = best_hyperparameters['num_dense_layers']
    
    units_values = []
    dropout_values = []
    for i in range(1, num_layers + 1):
        units_values.append(best_hyperparameters[f'units_l{i}'])
        # Dropout might not be defined for all layers if num_dense_layers was tuned to be less than max.
        # Or it might be defined but we only use up to num_layers.
        # KerasTuner usually names them consistently, e.g., dropout_l1, dropout_l2 even if num_dense_layers is 1.
        # We will assume that if num_dense_layers is N, then units_l1..N and dropout_l1..N exist.
        dropout_values.append(best_hyperparameters.get(f'dropout_l{i}', 0.0)) # Default to 0.0 if not found, though it should be there.


    print("\n--- Starting K-Fold Cross-Validation (5 Folds) ---")
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    fold_rmses = []
    fold_r2s = []
    fold_mapes = []

    fold_count = 0
    for train_index, val_index in kf.split(numerical_ordinal_features_all):
        fold_count += 1
        print(f"\n--- Fold {fold_count}/{n_splits} ---")

        # Split data for this fold
        X_train_num_ord_fold = numerical_ordinal_features_all[train_index]
        X_val_num_ord_fold = numerical_ordinal_features_all[val_index]
        
        y_train_log_fold = target_log_all[train_index]
        y_val_log_fold = target_log_all[val_index]
        
        X_train_nom_list_fold = [nom_feat_array[train_index] for nom_feat_array in nominal_features_list_all]
        X_val_nom_list_fold = [nom_feat_array[val_index] for nom_feat_array in nominal_features_list_all]

        # Scale numerical/ordinal features (fit on this fold's training data)
        scaler_fold = StandardScaler()
        X_train_num_ord_scaled_fold = scaler_fold.fit_transform(X_train_num_ord_fold)
        X_val_num_ord_scaled_fold = scaler_fold.transform(X_val_num_ord_fold)

        keras_train_inputs_fold = [X_train_num_ord_scaled_fold] + X_train_nom_list_fold
        keras_val_inputs_fold = [X_val_num_ord_scaled_fold] + X_val_nom_list_fold
        
        # Create and compile a new model for this fold
        print("Creating and compiling model for fold...")
        model_fold = create_model_with_fixed_hps(
            numerical_input_shape=(X_train_num_ord_scaled_fold.shape[1],),
            nominal_input_specs=nominal_input_specs_static,
            learning_rate_val=lr,
            num_dense_layers_val=num_layers,
            units_l_vals=units_values,
            dropout_l_vals=dropout_values
        )
        # model_fold.summary() # Optional: print model summary for each fold

        print("Training model for fold...")
        # Consider adding EarlyStopping here as well
        early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        history = model_fold.fit(
            keras_train_inputs_fold,
            y_train_log_fold,
            epochs=100, # Max epochs, early stopping will likely trigger
            batch_size=32, # You might want to make this configurable or use a default from best_hps if tuned
            validation_data=(keras_val_inputs_fold, y_val_log_fold),
            callbacks=[early_stopping_cb],
            verbose=1 # Set to 0 for less output during K-fold, or 1 to see epoch progress
        )
        
        print("Evaluating model for fold...")
        y_pred_log_fold = model_fold.predict(keras_val_inputs_fold).flatten()
        
        # Inverse transform for metrics
        y_val_original_fold = np.expm1(y_val_log_fold)
        y_pred_original_fold = np.expm1(y_pred_log_fold)

        rmse_fold = np.sqrt(mean_squared_error(y_val_original_fold, y_pred_original_fold))
        r2_fold = r2_score(y_val_original_fold, y_pred_original_fold)
        mape_fold = mean_absolute_percentage_error(y_val_log_fold, y_pred_log_fold, target_is_log_transformed=True)

        fold_rmses.append(rmse_fold)
        fold_r2s.append(r2_fold)
        fold_mapes.append(mape_fold)

        print(f"  Fold {fold_count} RMSE: ${rmse_fold:,.2f}")
        print(f"  Fold {fold_count} R²: {r2_fold:.4f}")
        print(f"  Fold {fold_count} MAPE: {mape_fold:.2f}%")

    # --- Aggregate and print results ---
    print("\n--- K-Fold Cross-Validation Results ---")
    print(f"Average RMSE over {n_splits} folds: ${np.mean(fold_rmses):,.2f} (Std: ${np.std(fold_rmses):,.2f})")
    print(f"Average R² over {n_splits} folds: {np.mean(fold_r2s):.4f} (Std: {np.std(fold_r2s):.4f})")
    print(f"Average MAPE over {n_splits} folds: {np.mean(fold_mapes):.2f}% (Std: {np.std(fold_mapes):.2f}%)")

    print("\nScript finished.")

if __name__ == "__main__":
    main() 