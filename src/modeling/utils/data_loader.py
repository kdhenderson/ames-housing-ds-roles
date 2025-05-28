"""
Data loading and preprocessing utilities for the Ames Housing dataset.
Uses numpy for data handling instead of pandas.
"""

import numpy as np
import csv
from typing import Tuple, Dict, List

# These lists should be consistent with how src/etl/etl_main.py processes variables.
# NOMINAL_VARS are integer-encoded by etl_main.py for embedding layers.
NOMINAL_VARS_FOR_EMBEDDING = [
    'MS Zoning', 'Street', 'Alley', 'Land Contour', 'Lot Config',
    'Neighborhood', 'Condition 1', 'Condition 2', 'Bldg Type', 'House Style',
    'Roof Style', 'Roof Matl', 'Exterior 1st', 'Exterior 2nd', 'Mas Vnr Type',
    'Foundation', 'Heating', 'Central Air', 'Electrical', 'Misc Feature',
    'Sale Type', 'Sale Condition', 'Garage Type'
]

# ORDINAL_VARS are numerically encoded by etl_main.py based on defined order
# and will be treated as standard numerical features.
ORDINAL_VARS_AS_NUMERIC = [
    'Exter Qual', 'Exter Cond', 'Bsmt Qual', 'Bsmt Cond', 'Bsmt Exposure',
    'BsmtFin Type 1', 'BsmtFin Type 2', 'Heating QC', 'Kitchen Qual', 'Fireplace Qu',
    'Garage Finish', 'Garage Qual', 'Garage Cond', 'Pool QC', 'Fence', 'Functional',
    'Land Slope', 'Lot Shape', 'Paved Drive', 'Utilities'
]


def load_housing_data(filepath: str) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray, List[str], List[str], Dict[str, int]]:
    """
    Load the preprocessed housing data from CSV, separating features for a model
    that uses embeddings for nominal variables.
    
    Args:
        filepath: Path to the preprocessed CSV file (e.g., housing_cleaned.csv)
        
    Returns:
        numerical_ordinal_features: numpy array of numerical and ordinal feature values.
        nominal_features_list: A list of numpy arrays. Each array contains integer codes
                               for a single nominal feature, ready for an embedding layer.
        target: numpy array of target values (SalePrice).
        numerical_ordinal_headers: list of headers for the numerical_ordinal_features.
        nominal_headers: list of headers for the nominal_features_list (maintaining order).
        nominal_cardinalities: A dictionary mapping nominal feature names to their cardinality (number of unique values).
    """
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        all_headers = next(reader)
        
        try:
            target_idx = all_headers.index('SalePrice')
        except ValueError:
            raise ValueError("Target column 'SalePrice' not found in the CSV headers.")
            
        # Prepare lists to hold raw data before converting to numpy arrays
        raw_numerical_ordinal_data = []
        raw_nominal_data_columns = {header: [] for header in NOMINAL_VARS_FOR_EMBEDDING if header in all_headers}
        raw_target_data = []
        
        # Segregate headers
        numerical_ordinal_headers = []
        nominal_headers_ordered = [] # To maintain the order of nominal features as they appear

        for header in all_headers:
            if header == 'SalePrice':
                continue
            if header in NOMINAL_VARS_FOR_EMBEDDING:
                if header not in nominal_headers_ordered: # Ensure we only add them once and in order of appearance
                    nominal_headers_ordered.append(header)
            elif header in ORDINAL_VARS_AS_NUMERIC:
                numerical_ordinal_headers.append(header)
            else: # Assumed to be other numerical features
                numerical_ordinal_headers.append(header)

        # Check if all expected nominal vars are present
        for nominal_var in NOMINAL_VARS_FOR_EMBEDDING:
            if nominal_var not in all_headers:
                print(f"Warning: Expected nominal variable '{nominal_var}' for embedding not found in CSV headers.")


        for row_idx, row_str_values in enumerate(reader):
            if len(row_str_values) != len(all_headers):
                print(f"Warning: Row {row_idx+1} has {len(row_str_values)} values, expected {len(all_headers)}. Skipping row.")
                continue

            raw_target_data.append(float(row_str_values[target_idx]))
            
            current_row_numerical_ordinal = []
            
            temp_row_nominal_values = {} # Store nominal values for this row by header

            for i, val_str in enumerate(row_str_values):
                if i == target_idx:
                    continue
                
                header = all_headers[i]
                
                try:
                    if header in NOMINAL_VARS_FOR_EMBEDDING:
                        # These are integer-encoded by ETL for embeddings
                        temp_row_nominal_values[header] = int(float(val_str)) # ETL saves as float string "1.0"
                    elif header in numerical_ordinal_headers : # Includes ordinals and other numerics
                        current_row_numerical_ordinal.append(float(val_str))
                    # If a header is not in either, it's skipped (e.g. if it wasn't in the predefined lists)
                except ValueError:
                    print(f"Warning: Could not convert value '{val_str}' for header '{header}' in row {row_idx+1}. Using 0.")
                    if header in NOMINAL_VARS_FOR_EMBEDDING:
                         temp_row_nominal_values[header] = 0 # Default for problematic nominal
                    elif header in numerical_ordinal_headers:
                         current_row_numerical_ordinal.append(0.0) # Default for problematic numeric/ordinal
            
            raw_numerical_ordinal_data.append(current_row_numerical_ordinal)
            for nom_header in nominal_headers_ordered: # Ensure order
                 if nom_header in temp_row_nominal_values:
                    raw_nominal_data_columns[nom_header].append(temp_row_nominal_values[nom_header])
                 else:
                    # This case should ideally not be hit if header checks are done correctly
                    # and all nominal vars are in CSV
                    print(f"Warning: Nominal header {nom_header} missing from parsed row {row_idx+1}, appending 0.")
                    raw_nominal_data_columns[nom_header].append(0)


    numerical_ordinal_features_np = np.array(raw_numerical_ordinal_data, dtype=np.float32)
    target_np = np.array(raw_target_data, dtype=np.float32)
    
    nominal_features_list_np = []
    nominal_cardinalities = {}

    for header in nominal_headers_ordered:
        if header in raw_nominal_data_columns and raw_nominal_data_columns[header]:
            col_data = np.array(raw_nominal_data_columns[header], dtype=np.int32)
            nominal_features_list_np.append(col_data.reshape(-1, 1)) # Each nominal feature as a column vector
            # Calculate cardinality: max_val + 1 (assumes integer codes are 0-indexed)
            nominal_cardinalities[header] = np.max(col_data) + 1
        else:
            # If a nominal column ended up empty or wasn't in headers, add a placeholder
            # This helps prevent errors in model construction if a nominal var is unexpectedly missing
            print(f"Warning: Nominal feature '{header}' data is missing or empty. Creating a placeholder column of zeros.")
            placeholder_col = np.zeros((numerical_ordinal_features_np.shape[0], 1), dtype=np.int32)
            nominal_features_list_np.append(placeholder_col)
            nominal_cardinalities[header] = 1 # Cardinality of 1 for a placeholder

    return (numerical_ordinal_features_np, 
            nominal_features_list_np, 
            target_np, 
            numerical_ordinal_headers,
            nominal_headers_ordered, # Return the actual order of nominal features processed
            nominal_cardinalities)

def load_school_data(filepath: str) -> Tuple[np.ndarray, List[str]]:
    """
    Load the school data from CSV.
    
    Args:
        filepath: Path to the school data CSV
        
    Returns:
        features: numpy array of school features
        feature_names: list of feature names
    """
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)
        
        # Initialize list to store data
        data = []
        
        for row in reader:
            # Convert string values to float, handle missing values
            row_data = []
            for val in row:
                try:
                    row_data.append(float(val) if val else 0.0)
                except ValueError:
                    # For non-numeric columns, create a simple label encoding
                    # In practice, you might want more sophisticated encoding
                    row_data.append(hash(val) % 100)  # Simple hash-based encoding
            data.append(row_data)
            
    return np.array(data, dtype=np.float32), headers

def train_test_split(features: np.ndarray, 
                    target: np.ndarray, 
                    # Adjusted to accept multiple feature inputs for splitting
                    nominal_features_list: List[np.ndarray] = None,
                    test_size: float = 0.2,
                    random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray], np.ndarray, np.ndarray]:
    """
    Split data into training and testing sets.
    Can handle a primary feature matrix and an optional list of nominal feature arrays.
    
    Args:
        features: Main numerical feature matrix
        target: Target vector
        nominal_features_list: Optional list of nominal feature arrays.
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        
    Returns:
        X_train_num, X_test_num, 
        X_train_nom_list (or None), X_test_nom_list (or None),
        y_train, y_test
    """
    np.random.seed(random_state)
    indices = np.random.permutation(len(features))
    test_count = int(test_size * len(features)) # Corrected variable name
    
    test_idx = indices[:test_count]
    train_idx = indices[test_count:]
    
    X_train_num, X_test_num = features[train_idx], features[test_idx]
    y_train, y_test = target[train_idx], target[test_idx]
    
    X_train_nom_list, X_test_nom_list = None, None
    if nominal_features_list:
        X_train_nom_list = [nom_feat_array[train_idx] for nom_feat_array in nominal_features_list]
        X_test_nom_list = [nom_feat_array[test_idx] for nom_feat_array in nominal_features_list]
        
    return (X_train_num, X_test_num,
            X_train_nom_list, X_test_nom_list,
            y_train, y_test) 