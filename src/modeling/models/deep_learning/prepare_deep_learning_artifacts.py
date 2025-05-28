"""
Script to train the final deep learning model on the entire dataset using 
the best hyperparameters and save the model, scaler, and nominal mappings.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import sys
import os
import joblib # For saving the scaler
from sklearn.preprocessing import StandardScaler

# Ensure the project root and src directory are in the Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from modeling.utils.data_loader import load_housing_data

# Define artifact paths
MODEL_ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
FINAL_MODEL_PATH = os.path.join(MODEL_ARTIFACTS_DIR, 'final_deep_learning_model.h5')
SCALER_PATH = os.path.join(MODEL_ARTIFACTS_DIR, 'dl_numerical_scaler.joblib')
NOMINAL_MAPS_PATH = os.path.join(MODEL_ARTIFACTS_DIR, 'nominal_column_mappings.json') # Already created by etl_main.py

def create_model_with_fixed_hps(numerical_input_shape, 
                                nominal_input_specs,
                                learning_rate_val,
                                num_dense_layers_val,
                                units_l_vals, 
                                dropout_l_vals
                                ) -> tf.keras.Model:
    """
    Create a deep learning model with fixed hyperparameters.
    (Copied from deep_learning_tuned_evaluation.py)
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
    if len(units_l_vals) != num_dense_layers_val or len(dropout_l_vals) != num_dense_layers_val:
        raise ValueError("Length of units_l_vals and dropout_l_vals must match num_dense_layers_val")

    for i in range(num_dense_layers_val):
        x = layers.Dense(units=units_l_vals[i])(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
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
    print("--- Preparing Deep Learning Model Artifacts ---")

    # Create directory for artifacts if it doesn't exist
    if not os.path.exists(MODEL_ARTIFACTS_DIR):
        os.makedirs(MODEL_ARTIFACTS_DIR)
        print(f"Created directory: {MODEL_ARTIFACTS_DIR}")

    print("Loading full dataset...")
    (numerical_ordinal_features_all, 
     nominal_features_list_all, 
     target_all, 
     _, 
     nominal_headers, 
     nominal_cardinalities) = load_housing_data(os.path.join(PROJECT_ROOT, 'data', 'processed', 'housing_cleaned.csv'))

    print("Log-transforming target variable...")
    target_log_all = np.log1p(target_all)

    # 1. Fit and save the StandardScaler
    print("Fitting StandardScaler on all numerical/ordinal training data...")
    scaler = StandardScaler()
    numerical_ordinal_features_scaled = scaler.fit_transform(numerical_ordinal_features_all)
    joblib.dump(scaler, SCALER_PATH)
    print(f"StandardScaler saved to: {SCALER_PATH}")

    # Prepare nominal input specifications for the model
    nominal_input_specs_static = []
    for header in nominal_headers:
        cardinality = nominal_cardinalities[header]
        # Consistent embedding dimension calculation
        emb_dim = min(50, cardinality // 2 if cardinality > 1 else 1) 
        if emb_dim == 0: emb_dim = 1 
        nominal_input_specs_static.append((header, cardinality, emb_dim))

    # 2. Define best hyperparameters (as confirmed by user)
    best_hyperparameters = {
        'learning_rate': 0.01,
        'num_dense_layers': 2,
        'units_l1': 96, 
        'units_l2': 160,
        'dropout_l1': 0.2, 
        'dropout_l2': 0.2
    }

    lr = best_hyperparameters['learning_rate']
    num_layers = best_hyperparameters['num_dense_layers']
    units_values = [best_hyperparameters[f'units_l{i+1}'] for i in range(num_layers)]
    dropout_values = [best_hyperparameters[f'dropout_l{i+1}'] for i in range(num_layers)]

    # Prepare Keras model inputs
    keras_train_inputs = [numerical_ordinal_features_scaled] + nominal_features_list_all
    
    # 3. Create and train the model on the full dataset
    print("Creating final model with best hyperparameters...")
    final_model = create_model_with_fixed_hps(
        numerical_input_shape=(numerical_ordinal_features_scaled.shape[1],),
        nominal_input_specs=nominal_input_specs_static,
        learning_rate_val=lr,
        num_dense_layers_val=num_layers,
        units_l_vals=units_values,
        dropout_l_vals=dropout_values
    )
    # final_model.summary() # Optional: print model summary

    print("Training final model on the entire dataset...")
    # Using EarlyStopping on 'loss' since there's no validation split here.
    # The goal is to train until convergence on the full training set.
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
    
    final_model.fit(
        keras_train_inputs,
        target_log_all,
        epochs=150, # Increased epochs, EarlyStopping will manage.
        batch_size=32,
        callbacks=[early_stopping_cb],
        verbose=1
    )

    # 4. Save the trained Keras model
    final_model.save(FINAL_MODEL_PATH)
    print(f"Trained Keras model saved to: {FINAL_MODEL_PATH}")

    print("\n--- Artifact Summary ---")
    print(f"1. Nominal Feature Mappings (JSON): {NOMINAL_MAPS_PATH} (Generated by etl_main.py)")
    print(f"2. Numerical Feature Scaler (StandardScaler): {SCALER_PATH}")
    print(f"3. Trained Keras Deep Learning Model (.h5): {FINAL_MODEL_PATH}")
    print("\nArtifact preparation complete.")

if __name__ == '__main__':
    main() 