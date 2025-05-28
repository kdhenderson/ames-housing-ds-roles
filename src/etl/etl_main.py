"""
Ames Housing ETL Pipeline

This script implements an ETL (Extract, Transform, Load) pipeline for the Ames Housing dataset.
It processes the raw data through the following steps:
1. Extract: Load data from CSV file
2. Transform: 
   - Handle missing values
   - Encode categorical variables (label encoding for ordinal, one-hot for nominal)
   - Perform feature engineering
3. Load: Save processed data to a new CSV file

The script uses only numpy and standard libraries (no pandas) to demonstrate
core data processing concepts and maintain minimal dependencies.

Author: [Your Name]
Date: [Current Date]
"""

import csv
import numpy as np
from collections import Counter, defaultdict
import json

# File paths for input and output
INPUT_FILE = 'data/raw/housing_raw.csv'
OUTPUT_FILE = 'data/processed/housing_cleaned.csv'
NOMINAL_MAPS_FILE = 'data/processed/nominal_column_mappings.json'

# Define categorical variables based on Ames Housing data dictionary
# Ordinal variables have a meaningful order that should be preserved
ORDINAL_VARS = [
    'Exter Qual', 'Exter Cond', 'Bsmt Qual', 'Bsmt Cond', 'Bsmt Exposure',
    'BsmtFin Type 1', 'BsmtFin Type 2', 'Heating QC', 'Kitchen Qual', 'Fireplace Qu',
    'Garage Finish', 'Garage Qual', 'Garage Cond', 'Pool QC', 'Fence', 'Functional',
    'Land Slope', 'Lot Shape', 'Paved Drive', 'Utilities'
]

# Nominal variables have no inherent order and require one-hot encoding
NOMINAL_VARS = [
    'MS Zoning', 'Street', 'Alley', 'Land Contour', 'Lot Config',
    'Neighborhood', 'Condition 1', 'Condition 2', 'Bldg Type', 'House Style',
    'Roof Style', 'Roof Matl', 'Exterior 1st', 'Exterior 2nd', 'Mas Vnr Type',
    'Foundation', 'Heating', 'Central Air', 'Electrical', 'Misc Feature',
    'Sale Type', 'Sale Condition', 'Garage Type'
]

# Define a mapping for each nominal variable to store its string-to-integer encoding
# This can be populated during the encoding process and potentially saved if needed elsewhere,
# but for now, the ETL just produces integer-encoded columns.
NOMINAL_MAPS = defaultdict(dict)

# Define the order for ordinal variables to ensure consistent encoding
ORDINAL_ORDER = {
    'Exter Qual': ['po', 'fa', 'ta', 'gd', 'ex'],
    'Exter Cond': ['po', 'fa', 'ta', 'gd', 'ex'],
    'Bsmt Qual': ['na', 'po', 'fa', 'ta', 'gd', 'ex'],
    'Bsmt Cond': ['na', 'po', 'fa', 'ta', 'gd', 'ex'],
    'Bsmt Exposure': ['na', 'no', 'mn', 'av', 'gd'],
    'BsmtFin Type 1': ['na', 'unf', 'lwq', 'rec', 'blq', 'alq', 'glq'],
    'BsmtFin Type 2': ['na', 'unf', 'lwq', 'rec', 'blq', 'alq', 'glq'],
    'Heating QC': ['po', 'fa', 'ta', 'gd', 'ex'],
    'Kitchen Qual': ['po', 'fa', 'ta', 'gd', 'ex'],
    'Fireplace Qu': ['na', 'po', 'fa', 'ta', 'gd', 'ex'],
    'Garage Finish': ['na', 'unf', 'rfn', 'fin'],
    'Garage Qual': ['na', 'po', 'fa', 'ta', 'gd', 'ex'],
    'Garage Cond': ['na', 'po', 'fa', 'ta', 'gd', 'ex'],
    'Pool QC': ['na', 'fa', 'ta', 'gd', 'ex'],
    'Fence': ['na', 'mnww', 'gdwo', 'mnprv', 'gdprv'],
    'Functional': ['sal', 'sev', 'maj2', 'maj1', 'mod', 'min2', 'min1', 'typ'],
    'Land Slope': ['sev', 'mod', 'gtl'],
    'Lot Shape': ['ir3', 'ir2', 'ir1', 'reg'],
    'Paved Drive': ['n', 'p', 'y'],
    'Utilities': ['elo', 'nosewa', 'nosewr', 'allpub']
}

def read_csv(filename):
    """
    Read data from a CSV file and return headers and data as separate lists.
    Strips whitespace from header names and data values for consistency.
    """
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        headers = next(reader)
        # Strip whitespace from header names
        headers = [h.strip() for h in headers]
        # Read data and strip whitespace from each cell
        data = []
        for row in reader:
            data.append([cell.strip() for cell in row])
    return headers, data

def get_column_index(headers, column_name):
    """
    Get the index of a column in the data array, ignoring leading/trailing whitespace.
    """
    # Strip whitespace from both headers and the search name
    headers_stripped = [h.strip() for h in headers]
    return headers_stripped.index(column_name.strip())

def handle_missing_values(data, headers):
    """
    Handle missing values in the dataset.
    For numeric columns: fill with mean
    For categorical columns: fill with mode
    
    Args:
        data (list): List of lists containing the data
        headers (list): List of column names
        
    Returns:
        list: Data with missing values filled
    """
    # Convert data to numpy array for easier manipulation
    data_array = np.array(data)
    
    for i, header in enumerate(headers):
        # Skip the target variable (SalePrice) and ID columns
        if header in ['Id', 'SalePrice']:
            continue
            
        # Get column data (already stripped by read_csv)
        column = data_array[:, i]
        
        # Try to convert to numeric
        try:
            # Attempt to convert, ensure empty strings become NaN for nanmean
            numeric_data = np.array([float(x) if x and x.strip() else np.nan for x in column], dtype=float)
            if np.all(np.isnan(numeric_data)): # All were empty or non-convertible
                 # Fallback for columns that are not truly numeric but might have numbers
                 # For example, a categorical column that happens to have only '1', '2' as strings
                 # This will be handled by the categorical logic below
                 raise ValueError("Column not purely numeric, treat as categorical for missing handling")

            mean_val = np.nanmean(numeric_data)
            # Replace original NaNs (from empty strings) with the mean
            # Need to be careful if mean_val is also nan (e.g. all values were nan)
            if not np.isnan(mean_val):
                 data_array[:, i] = np.where(np.isnan(numeric_data), mean_val, numeric_data).astype(str)
            else: # if mean is nan (all values were nan/empty), fill with a placeholder like '0' or specific string
                 data_array[:, i] = np.where(np.isnan(numeric_data), '0', numeric_data).astype(str)

        except ValueError:
            # For categorical columns, fill with mode (after cleaning)
            # Clean values (lowercase, strip) before finding mode
            cleaned_column = [x.strip().lower() for x in column if x and x.strip()]
            if cleaned_column:
                mode_val = Counter(cleaned_column).most_common(1)[0][0]
                # Apply mode to original column where values were empty strings
                for row_idx in range(data_array.shape[0]):
                    if not data_array[row_idx, i] or not data_array[row_idx, i].strip():
                        data_array[row_idx, i] = mode_val
            # If column was empty or only had empty strings, mode_val might not be set
            # In such cases, we might fill with a specific placeholder like 'unknown'
            else:
                for row_idx in range(data_array.shape[0]):
                    if not data_array[row_idx, i] or not data_array[row_idx, i].strip():
                        # Ensure 'unknown_mode' is treated as a string, not a float by mistake later
                        data_array[row_idx, i] = 'unknown_mode'


    return data_array.tolist()

def encode_categorical_variables(data, headers):
    """
    Encode categorical variables using appropriate methods:
    - Label encoding for ordinal variables
    - Integer encoding for nominal variables (for use with embedding layers)
    
    Args:
        data (list): List of lists containing the data
        headers (list): List of column names
        
    Returns:
        tuple: (encoded_data, new_headers) where encoded_data is the processed data
               and new_headers is the updated list of column names
    """
    # Convert data to numpy array for easier manipulation
    data_array = np.array(data)
    new_headers = headers.copy() # Start with a copy of original headers
    
    # Process ordinal variables with label encoding
    for var in ORDINAL_VARS:
        if var in headers:
            col_idx = get_column_index(headers, var)
            order = ORDINAL_ORDER[var] 
            mapping = {cat.strip().lower(): idx for idx, cat in enumerate(order)}
            
            encoded_col = []
            for val in data_array[:, col_idx]:
                cleaned_val = val.strip().lower()
                encoded_col.append(str(mapping.get(cleaned_val, 0))) 
            data_array[:, col_idx] = encoded_col
    
    # Process nominal variables with integer encoding
    global NOMINAL_MAPS # Use the global NOMINAL_MAPS to store mappings
    NOMINAL_MAPS.clear() # Clear any previous maps

    for var in NOMINAL_VARS:
        if var in new_headers: # Check if header exists in current new_headers
            col_idx = get_column_index(new_headers, var) # Get index from new_headers
            
            # Get unique categories (cleaned: lowercase, stripped)
            # Make sure to handle potential empty strings or "NA" strings if they are meant to be distinct
            column_values = data_array[:, col_idx]
            # Clean values, map 'na' (if it's a common placeholder) or empty strings to a specific category like 'unknown'
            # or handle them based on how missing values were imputed.
            # For simplicity, we'll use 'unknown_nominal' for values that become empty after stripping.
            cleaned_values = []
            for val_str in column_values:
                cleaned_val = str(val_str).strip().lower()
                if not cleaned_val: # If empty after strip
                    cleaned_val = 'unknown_nominal_val' # assign a placeholder
                cleaned_values.append(cleaned_val)

            unique_categories = sorted(list(set(cleaned_values)))
            
            # Create mapping from category to integer
            # Start integers from 0 for each variable
            mapping = {cat: idx for idx, cat in enumerate(unique_categories)}
            NOMINAL_MAPS[var] = mapping # Store the map

            # Apply mapping to data
            encoded_col = [str(mapping.get(val, mapping.get('unknown_nominal_val', 0))) for val in cleaned_values] # Default to 'unknown_nominal_val's code
            data_array[:, col_idx] = encoded_col
            # The header for this column remains 'var', e.g., 'MSZoning'. No new headers are added.

    # No headers are removed or added for nominal vars in this integer encoding scheme,
    # so new_headers should already be correct (unless a nominal var was missing entirely).
    # We return new_data as the modified data_array and the original new_headers.
    return data_array.tolist(), new_headers

def perform_feature_engineering(data, headers):
    """
    Create new features through combination of existing ones.
    Currently implements:
    - TotalSF: Sum of basement, first, and second floor areas
    Handles missing columns and prints detailed errors if not found.
    """
    data_array = np.array(data)
    new_headers = headers.copy()
    try:
        # Use stripped header names for matching
        header_map = {h.strip(): i for i, h in enumerate(headers)}
        required_cols = ['Total Bsmt SF', '1st Flr SF', '2nd Flr SF']
        missing = [col for col in required_cols if col not in header_map]
        if missing:
            print(f"Warning: Could not create TotalSF feature. Missing columns: {missing}")
            return data, headers
        basement_idx = header_map['Total Bsmt SF']
        first_floor_idx = header_map['1st Flr SF']
        second_floor_idx = header_map['2nd Flr SF']
        basement = np.array([float(x) if x != '' else 0 for x in data_array[:, basement_idx]])
        first_floor = np.array([float(x) if x != '' else 0 for x in data_array[:, first_floor_idx]])
        second_floor = np.array([float(x) if x != '' else 0 for x in data_array[:, second_floor_idx]])
        total_sf = basement + first_floor + second_floor
        new_data = np.column_stack((data_array, total_sf))
        new_headers.append('TotalSF')
        return new_data.tolist(), new_headers
    except Exception as e:
        print(f"Warning: Could not create TotalSF feature due to error: {e}")
        return data, headers

def save_to_csv(data, headers, filename):
    """
    Save processed data to a CSV file.
    
    Args:
        data (list): List of lists containing the data
        headers (list): List of column names
        filename (str): Path to save the CSV file
    """
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(data)

def remove_column_if_exists(data, headers, column_name_to_remove):
    """
    Removes a specified column from the data and headers if it exists.
    """
    new_headers = list(headers) # Make a copy
    new_data_list = [list(row) for row in data] # Convert data to list of lists

    try:
        col_idx_to_remove = get_column_index(new_headers, column_name_to_remove)
        new_headers.pop(col_idx_to_remove)
        for row in new_data_list:
            row.pop(col_idx_to_remove)
        print(f"Info: Column '{column_name_to_remove}' removed successfully.")
    except ValueError:
        print(f"Info: Column '{column_name_to_remove}' not found. No changes made.")
    
    return new_data_list, new_headers

def main():
    """
    Main function to execute the ETL pipeline.
    """
    try:
        # Extract
        print("Reading data...")
        headers, data = read_csv(INPUT_FILE)
        
        # Explicitly remove "Order" and "PID" columns early if they exist
        # This is done before missing value handling or any other processing
        data_np = np.array(data)
        headers_list = list(headers)

        for col_to_remove in ["Order", "PID"]:
            if col_to_remove in headers_list:
                print(f"Removing column: {col_to_remove}")
                col_idx = headers_list.index(col_to_remove)
                data_np = np.delete(data_np, col_idx, axis=1)
                headers_list.pop(col_idx)
            else:
                print(f"Column {col_to_remove} not found, cannot remove.")

        data = data_np.tolist()
        headers = headers_list
        
        # Transform
        print("Handling missing values...")
        data = handle_missing_values(data, headers)
        
        # Perform feature engineering BEFORE encoding categorical variables
        print("Performing feature engineering...")
        data, headers = perform_feature_engineering(data, headers)
        
        print("Encoding categorical variables...")
        data, headers = encode_categorical_variables(data, headers)
        
        # Save NOMINAL_MAPS to a JSON file
        print(f"Saving nominal variable mappings to {NOMINAL_MAPS_FILE}...")
        try:
            # Convert defaultdict to a regular dict for JSON serialization
            regular_nominal_maps = dict(NOMINAL_MAPS)
            with open(NOMINAL_MAPS_FILE, 'w') as f:
                json.dump(regular_nominal_maps, f, indent=4)
            print(f"Nominal mappings saved successfully to {NOMINAL_MAPS_FILE}")
        except Exception as e:
            print(f"Error saving nominal mappings: {str(e)}")

        # Load
        print("Saving processed data...")
        save_to_csv(data, headers, OUTPUT_FILE)
        
        print(f"ETL pipeline completed successfully. Output saved to {OUTPUT_FILE}")
        
    except Exception as e:
        print(f"Error in ETL pipeline: {str(e)}")

if __name__ == "__main__":
    main() 