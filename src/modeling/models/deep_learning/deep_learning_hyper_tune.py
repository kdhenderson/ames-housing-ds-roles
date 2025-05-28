"""
Hyperparameter tuning for the Baseline Deep Learning model for Ames Housing price prediction.
Uses KerasTuner.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import keras_tuner as kt # Import KerasTuner
import sys
import os
# KFold is not directly used in main for tuner, but useful for final eval if desired
from sklearn.model_selection import KFold, train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
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

def create_model_for_tuning(hp, numerical_input_shape_static, nominal_input_specs_static) -> tf.keras.Model:
    """
    Create a deep learning model with hyperparameters defined by KerasTuner.
    
    Args:
        hp: KerasTuner HyperParameters object.
        numerical_input_shape_static: Static shape of the numerical input.
        nominal_input_specs_static: Static list of tuples for nominal features 
                                     (name, cardinality, base_embedding_dim_heuristic).
                                     Actual embedding dim can be tuned if desired.
    Returns:
        Compiled Keras functional API model.
    """
    numerical_input = layers.Input(shape=numerical_input_shape_static, name='numerical_input')
    current_numerical_branch = numerical_input
    
    embedding_outputs = []
    nominal_keras_inputs = []

    # Option to tune embedding dimension factor
    # emb_dim_factor = hp.Float('embedding_dim_factor', min_value=0.5, max_value=1.5, step=0.25, default=1.0)

    for name, cardinality, base_emb_dim in nominal_input_specs_static:
        sanitized_name = name.replace(' ', '_').replace('-', '_neg_').replace('/', '_slash_').replace('.', '_dot_')
        # Tunable embedding dimension per feature or use a factor (keeping it simple for now)
        # emb_dim = max(1, int(base_emb_dim * emb_dim_factor))
        emb_dim = base_emb_dim # Using pre-calculated heuristic for now
        
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
    
    # Tunable Dense Layers
    x = concatenated_features
    for i in range(hp.Int('num_dense_layers', 1, 3, default=3)): # Tune number of dense layers (1 to 3)
        x = layers.Dense(
            units=hp.Int(f'units_l{i+1}', min_value=32, max_value=256, step=32, default=128 if i==0 else (64 if i==1 else 32))
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(
            rate=hp.Float(f'dropout_l{i+1}', min_value=0.1, max_value=0.5, step=0.1, default=0.4 if i==0 else (0.3 if i==1 else 0.0))
        )(x)

    output_layer = layers.Dense(1, name='output')(x)
    all_inputs = [numerical_input] + nominal_keras_inputs
    model = models.Model(inputs=all_inputs, outputs=output_layer)
    
    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(
            learning_rate=hp.Choice('learning_rate', values=[0.01, 0.005, 0.001, 0.0005], default=0.001)
        ),
        loss='mse',
        metrics=['mae']
    )
    return model

def main():
    print("Loading data for hyperparameter tuning...")
    (numerical_ordinal_features_all, 
     nominal_features_list_all, 
     target_all, 
     _, 
     nominal_headers, 
     nominal_cardinalities) = load_housing_data('data/processed/housing_cleaned.csv')

    print("Log-transforming target variable.")
    target_log_all = np.log1p(target_all)

    # Prepare static nominal_input_specs for the model builder
    nominal_input_specs_static = []
    for header in nominal_headers:
        cardinality = nominal_cardinalities[header]
        emb_dim = min(50, cardinality // 2 if cardinality > 1 else 1)
        if emb_dim == 0: emb_dim = 1
        nominal_input_specs_static.append((header, cardinality, emb_dim))
        # print(f"  Static Embedding spec for '{header}': Cardinality={cardinality}, Embedding Dim={emb_dim}")

    # Split data into training and validation for the tuner
    # The tuner needs its own validation set separate from a final test set.
    # We'll use 80% for tuner training, 20% for tuner validation.
    
    # Stratify by target is tricky for regression. Simple random split for now.
    # Need to split all parts of the input data consistently.
    num_samples = numerical_ordinal_features_all.shape[0]
    indices = np.arange(num_samples)
    
    train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=42)

    X_train_num_ord = numerical_ordinal_features_all[train_indices]
    X_val_num_ord = numerical_ordinal_features_all[val_indices]
    
    y_train_log = target_log_all[train_indices]
    y_val_log = target_log_all[val_indices]
    
    X_train_nom_list = [nom_feat_array[train_indices] for nom_feat_array in nominal_features_list_all]
    X_val_nom_list = [nom_feat_array[val_indices] for nom_feat_array in nominal_features_list_all]

    # Scale numerical/ordinal features (fit on training part, transform both)
    scaler = StandardScaler()
    X_train_num_ord_scaled = scaler.fit_transform(X_train_num_ord)
    X_val_num_ord_scaled = scaler.transform(X_val_num_ord)

    keras_train_inputs_for_tuner = [X_train_num_ord_scaled] + X_train_nom_list
    keras_val_inputs_for_tuner = [X_val_num_ord_scaled] + X_val_nom_list

    # Define a partial function for the model builder to pass static args
    # KerasTuner expects the model-building function to take only one argument: hp
    model_builder = lambda hp: create_model_for_tuning(
        hp,
        numerical_input_shape_static=(X_train_num_ord_scaled.shape[1],),
        nominal_input_specs_static=nominal_input_specs_static
    )

    tuner = kt.RandomSearch(
        hypermodel=model_builder,
        objective='val_loss',
        max_trials=15, # Number of hyperparameter combinations to try
        executions_per_trial=1, # Number of models to train per combination
        directory='keras_tuner_dir',
        project_name='ames_dl_tuning_v1',
        overwrite=True
    )

    print("Starting hyperparameter search...")
    tuner.search(
        keras_train_inputs_for_tuner,
        y_train_log,
        epochs=100, # Max epochs for each trial, early stopping will trigger
        validation_data=(keras_val_inputs_for_tuner, y_val_log),
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)] # Early stopping for each trial
    )

    print("Hyperparameter search complete.")
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    print("Best hyperparameters found:")
    for hp_name, hp_value in best_hps.values.items():
        print(f"  {hp_name}: {hp_value}")

    # Build the best model with the found hyperparameters
    best_models = tuner.get_best_models(num_models=1)
    best_model = best_models[0]
    best_model.summary()

    print("\n--- Evaluating the best model found by KerasTuner (on the tuner's validation set) ---")
    # Normally, you'd evaluate on a separate, held-out test set or using full K-fold CV.
    # For now, we'll re-evaluate on the tuner's validation split to get a quick idea.
    # Note: The model is already trained with these HPs during the search to get val_loss.
    # We are just predicting here.
    
    y_pred_log_best = best_model.predict(keras_val_inputs_for_tuner).flatten()
    y_pred_original_best = np.expm1(y_pred_log_best)
    y_val_original_for_eval = np.expm1(y_val_log)

    rmse_best = np.sqrt(mean_squared_error(y_val_original_for_eval, y_pred_original_best))
    r2_best = r2_score(y_val_original_for_eval, y_pred_original_best)
    mape_best = mean_absolute_percentage_error(y_val_log, y_pred_log_best, target_is_log_transformed=True)

    print("Performance of the best model on the tuner validation set:")
    print(f"  Validation Loss (from tuner objective): {best_hps.get('val_loss') if 'val_loss' in best_hps.values else tuner.oracle.get_best_trials(1)[0].score:.4f}") # Get actual score if available
    print(f"  RMSE: ${rmse_best:,.2f}")
    print(f"  R² Score: {r2_best:.4f}")
    print(f"  MAPE: {mape_best:.2f}%")
    
    print("\nTo perform a more robust evaluation, retrain this best model using K-fold cross-validation.")

if __name__ == "__main__":
    main() 