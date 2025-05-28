# ETL Pipeline Documentation

## Input/Output Files
- Input: `data/raw/housing_raw.csv`
- Output: `data/processed/housing_cleaned.csv`

## Overview
This document describes the usage and functionality of `src/etl/etl_main.py`, a Python script for performing ETL (Extract, Transform, Load) operations on the Ames Housing dataset. The script standardizes data cleaning and feature engineering to prepare the data for analysis and modeling.

## Requirements
- Python 3.x
- numpy (install with `pip install numpy`)
- Standard Python libraries: csv, collections

## Usage
1. Place `etl_main.py` in the `src/etl/` directory.
2. Run the script from the project root directory:
   ```bash
   python src/etl/etl_main.py
   ```
3. The cleaned and transformed data will be saved as `data/processed/housing_cleaned.csv`.

## ETL Steps
### Extract
- Loads the raw data from `data/raw/housing_raw.csv` using Python's built-in `csv` module
- Handles UTF-8 encoding for proper character support
- Strips whitespace from header names for consistency

### Transform
#### Initial Data Cleaning
- The "Order" column (an observation index from the raw data) is explicitly removed as it is not a predictive feature.

#### Missing Values
- Numeric columns: Missing values are filled with the column mean
- Categorical columns: Missing values are filled with the column mode (most frequent value). String values are normalized (lowercased, whitespace stripped) before mode calculation.
- Special handling for ID and target (SalePrice) columns

#### Categorical Encoding
- **General**: All string category values from the raw data are normalized (converted to lowercase and stripped of leading/trailing whitespace) before any mapping or encoding logic is applied. This ensures consistent handling of categorical data.
- **Ordinal Variables**: Label encoding preserving meaningful order
  - Includes quality ratings (Ex, Gd, TA, Fa, Po)
  - Includes condition ratings
  - Includes basement, garage, and other categorical features with inherent order
  - Unmapped or unknown ordinal values (after cleaning and normalization) are defaulted to a consistent numerical value (typically 0).
- **Nominal Variables**: One-hot encoding
  - Includes zoning, street type, neighborhood
  - Includes building and house styles
  - Includes exterior and foundation types
  - "Garage Type" is now explicitly included and processed via one-hot encoding.
  - Creates binary columns for each category

#### Feature Engineering
- **TotalSF**: Sum of:
  - Total Basement Square Feet
  - First Floor Square Feet
  - Second Floor Square Feet
- Handles missing values in component features
- Provides detailed error reporting for missing columns

### Load
- Writes the cleaned and transformed data to `data/processed/housing_cleaned.csv`
- Preserves all original columns plus new engineered features
- Maintains data types and encoding consistency

## Data Dictionary
The script includes predefined mappings for:
- Ordinal variables and their value orders (note: category strings in mappings are now lowercase and stripped)
- Nominal variables for one-hot encoding
- Quality and condition ratings
- Special feature categories

## Error Handling
- Detailed error reporting for missing columns
- Graceful handling of missing values
- Validation of data types and conversions
- UTF-8 encoding support for special characters

## Customization
- To modify the feature engineering step, edit the `perform_feature_engineering` function
- To change categorical encoding, update the `ORDINAL_VARS`, `NOMINAL_VARS`, and `ORDINAL_ORDER` dictionaries. Ensure consistency with lowercase, stripped string categories in `ORDINAL_ORDER`.
- To modify missing value handling, update the `handle_missing_values` function
- The script is modular and can be extended for additional transformations

## Contact / Support
For questions or suggestions, please contact the data engineering team or the project maintainer. 