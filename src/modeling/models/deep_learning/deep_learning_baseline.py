"""
Baseline Deep Learning model for Ames Housing price prediction.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import sys
import os
from sklearn.model_selection import KFold
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

def create_model(numerical_input_shape: tuple,
                 nominal_input_specs: list) -> tf.keras.Model:
    """
    Create a deep learning model with embedding layers for nominal features.
    
    Args:
        numerical_input_shape: Shape of the numerical input (e.g., (num_numerical_features,))
        nominal_input_specs: A list of tuples, where each tuple contains:
                             (feature_name, cardinality, embedding_dim)
                             for a nominal feature.
        
    Returns:
        Compiled Keras functional API model
    """
    # Numerical input branch
    numerical_input = layers.Input(shape=numerical_input_shape, name='numerical_input')
    current_numerical_branch = numerical_input # Start with the input itself
    
    # Embedding layers for nominal features
    embedding_outputs = []
    nominal_keras_inputs = []

    for name, cardinality, emb_dim in nominal_input_specs:
        # Sanitize the name for TensorFlow layer naming (replace spaces with underscores, remove other problematic chars)
        sanitized_name = name.replace(' ', '_').replace('-', '_neg_').replace('/', '_slash_').replace('.', '_dot_')
        
        nominal_input_layer = layers.Input(shape=(1,), name=f'{sanitized_name}_input') 
        nominal_keras_inputs.append(nominal_input_layer)
        embedding_layer = layers.Embedding(input_dim=cardinality, 
                                           output_dim=emb_dim, 
                                           name=f'{sanitized_name}_embedding')(nominal_input_layer)
        flattened_embedding = layers.Flatten(name=f'{sanitized_name}_flatten')(embedding_layer)
        embedding_outputs.append(flattened_embedding)
        
    # Concatenate numerical features and embedding outputs
    if embedding_outputs: # If there are embedding layers
        concatenated_features = layers.concatenate([current_numerical_branch] + embedding_outputs)
    else: # Only numerical features
        concatenated_features = current_numerical_branch
    
    # Dense layers for combined features
    x = layers.Dense(128, use_bias=False)(concatenated_features) # Reverted to 128
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.4)(x) # Reverted to 0.4
    
    x = layers.Dense(64, use_bias=False)(x) # Reverted to 64
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.3)(x) # Reverted to 0.3

    x = layers.Dense(32, use_bias=False)(x) # Reverted to 32
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    # No dropout after the last hidden layer before output
    
    output_layer = layers.Dense(1, name='output')(x)  # Output layer for regression
    
    all_inputs = [numerical_input] + nominal_keras_inputs
    model = models.Model(inputs=all_inputs, outputs=output_layer)
    
    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model

def main():
    # Load data using the new data loader structure
    print("Loading data with new structure...")
    (numerical_ordinal_features, 
     nominal_features_list, 
     target, 
     numerical_ordinal_headers, 
     nominal_headers, 
     nominal_cardinalities) = load_housing_data('data/processed/housing_cleaned.csv')
    
    print(f"Loaded {numerical_ordinal_features.shape[0]} samples.")
    print(f"Numerical/Ordinal features: {numerical_ordinal_features.shape[1]} ({len(numerical_ordinal_headers)} headers)")
    print(f"Nominal features for embedding: {len(nominal_features_list)} ({len(nominal_headers)} headers)")
    # for i, nom_header in enumerate(nominal_headers):
    #     print(f"  - {nom_header}: shape {nominal_features_list[i].shape}, cardinality {nominal_cardinalities[nom_header]}")

    # Log transform the target variable (SalePrice)
    print("Log-transforming target variable (SalePrice).")
    target_log = np.log1p(target)
    
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_metrics_list = []
    
    print(f"Starting {n_splits}-fold cross-validation with embeddings (no feature selection in this step)...")

    # Prepare nominal_input_specs for model creation (once, outside the loop)
    nominal_input_specs = []
    for header in nominal_headers:
        cardinality = nominal_cardinalities[header]
        emb_dim = min(50, cardinality // 2 if cardinality > 1 else 1) # Ensure emb_dim is at least 1
        if emb_dim == 0: emb_dim = 1 # Safety net if cardinality was 1, then //2 gives 0
        # The 'name' in nominal_input_specs is the original header name, 
        # sanitization happens inside create_model
        nominal_input_specs.append((header, cardinality, emb_dim))
        print(f"  Embedding for '{header}': Cardinality={cardinality}, Embedding Dim={emb_dim}")


    for fold_idx, (train_index, val_index) in enumerate(kf.split(numerical_ordinal_features, target_log)):
        print(f"--- Fold {fold_idx + 1}/{n_splits} ---")
        
        # Split numerical/ordinal features
        X_train_num_ord, X_val_num_ord = numerical_ordinal_features[train_index], numerical_ordinal_features[val_index]
        
        # Split nominal features (list of arrays)
        X_train_nom_list_fold = [nom_feat_array[train_index] for nom_feat_array in nominal_features_list]
        X_val_nom_list_fold = [nom_feat_array[val_index] for nom_feat_array in nominal_features_list]
        
        y_train_log, y_val_log = target_log[train_index], target_log[val_index]
        
        # Scale numerical/ordinal features
        scaler = StandardScaler()
        X_train_num_ord_scaled = scaler.fit_transform(X_train_num_ord)
        X_val_num_ord_scaled = scaler.transform(X_val_num_ord)
        
        # Prepare model inputs for Keras: a list containing the scaled numerical block first,
        # then each of the nominal feature arrays for this fold.
        keras_train_inputs = [X_train_num_ord_scaled] + X_train_nom_list_fold
        keras_val_inputs = [X_val_num_ord_scaled] + X_val_nom_list_fold

        # Create and train DL model
        # Pass the shape of the scaled numerical/ordinal features for the numerical_input_shape
        dl_model = create_model(numerical_input_shape=(X_train_num_ord_scaled.shape[1],),
                                nominal_input_specs=nominal_input_specs)
        
        if fold_idx == 0: # Print model summary only for the first fold
            dl_model.summary()

        print(f"Training DL model with embeddings on Fold {fold_idx+1}...")
        history = dl_model.fit(
            keras_train_inputs,
            y_train_log,
            epochs=150, # Increased epochs slightly
            batch_size=32,
            validation_data=(keras_val_inputs, y_val_log),
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)],
            verbose=0 
        )
        
        final_train_loss = history.history['loss'][-1]
        final_train_mae = history.history['mae'][-1]
        val_loss_at_best_epoch = min(history.history.get('val_loss', [np.nan]))
        print(f"DL model training complete. Final train loss: {final_train_loss:.4f}, Final train MAE: {final_train_mae:.4f} (log-target). Best val_loss: {val_loss_at_best_epoch:.4f}")

        # Evaluate
        y_pred_log = dl_model.predict(keras_val_inputs).flatten()
        
        y_pred_original = np.expm1(y_pred_log)
        y_val_original = np.expm1(y_val_log)
        
        rmse = np.sqrt(mean_squared_error(y_val_original, y_pred_original))
        r2 = r2_score(y_val_original, y_pred_original)
        mape = mean_absolute_percentage_error(y_val_log, y_pred_log, target_is_log_transformed=True)

        print(f"Fold {fold_idx + 1} Metrics (original scale): RMSE=${rmse:,.2f}, R2={r2:.4f}, MAPE={mape:.2f}%")
        fold_metrics_list.append({
            'rmse': rmse, 
            'r2': r2, 
            'mape': mape,
            'train_loss': final_train_loss, 
            'train_mae': final_train_mae,
            'val_loss': val_loss_at_best_epoch
        })

    # Calculate average metrics
    avg_metrics = {}
    std_metrics = {}
    for key in fold_metrics_list[0].keys():
        values = [m[key] for m in fold_metrics_list if m[key] is not None and not np.isnan(m[key])]
        if values:
            avg_metrics[key] = np.mean(values)
            std_metrics[key] = np.std(values)
        else:
            avg_metrics[key] = np.nan
            std_metrics[key] = np.nan
    
    print("Average Performance Metrics (with embeddings):")
    print(f"Train Loss (log): {avg_metrics.get('train_loss', np.nan):.4f} ± {std_metrics.get('train_loss', np.nan):.4f}")
    print(f"Train MAE (log): {avg_metrics.get('train_mae', np.nan):.4f} ± {std_metrics.get('train_mae', np.nan):.4f}")
    print(f"Validation Loss (log): {avg_metrics.get('val_loss', np.nan):.4f} ± {std_metrics.get('val_loss', np.nan):.4f}")
    print(f"RMSE: ${avg_metrics.get('rmse', np.nan):,.2f} ± ${std_metrics.get('rmse', np.nan):,.2f}")
    print(f"R² Score: {avg_metrics.get('r2', np.nan):.4f} ± {std_metrics.get('r2', np.nan):.4f}")
    print(f"MAPE: {avg_metrics.get('mape', np.nan):.2f}% ± {std_metrics.get('mape', np.nan):.2f}%")

if __name__ == "__main__":
    main() 