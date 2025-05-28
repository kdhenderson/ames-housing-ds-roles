"""
Gradio application for Ames Housing Price Prediction using the Deep Learning model.
"""
import gradio as gr
import numpy as np
import tensorflow as tf
import joblib
import json
import os

# --- Configuration: Paths to Artifacts ---
FINAL_MODEL_PATH = 'final_deep_learning_model.h5'
SCALER_PATH = 'dl_numerical_scaler.joblib'
NOMINAL_MAPS_PATH = 'nominal_column_mappings.json'

# --- Load Artifacts ---
try:
    model = tf.keras.models.load_model(FINAL_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    with open(NOMINAL_MAPS_PATH, 'r') as f:
        nominal_column_mappings = json.load(f)
    print("Model, scaler, and nominal mappings loaded successfully.")
except Exception as e:
    print(f"Error loading artifacts: {e}")
    model, scaler, nominal_column_mappings = None, None, None

# --- Feature Definitions ---
EXPECTED_NUMERICAL_ORDINAL_HEADERS = [
    'MS SubClass', 'Lot Frontage', 'Lot Area', 'Lot Shape', 'Utilities', 
    'Land Slope', 'Overall Qual', 'Overall Cond', 'Year Built', 'Year Remod/Add', 
    'Mas Vnr Area', 'Exter Qual', 'Exter Cond', 'Bsmt Qual', 'Bsmt Cond', 
    'Bsmt Exposure', 'BsmtFin Type 1', 'BsmtFin SF 1', 'BsmtFin Type 2', 'BsmtFin SF 2', 
    'Bsmt Unf SF', 'Total Bsmt SF', 'Heating QC', '1st Flr SF', '2nd Flr SF', 
    'Low Qual Fin SF', 'Gr Liv Area', 'Bsmt Full Bath', 'Bsmt Half Bath', 'Full Bath', 
    'Half Bath', 'Bedroom AbvGr', 'Kitchen AbvGr', 'Kitchen Qual', 'TotRms AbvGrd', 
    'Functional', 'Fireplaces', 'Fireplace Qu', 'Garage Yr Blt', 'Garage Finish', 
    'Garage Cars', 'Garage Area', 'Garage Qual', 'Garage Cond', 'Paved Drive', 
    'Wood Deck SF', 'Open Porch SF', 'Enclosed Porch', '3Ssn Porch', 'Screen Porch', 
    'Pool Area', 'Pool QC', 'Fence', 'Misc Val', 'Mo Sold', 'Yr Sold', 'TotalSF'
]

DEFAULT_NUMERICAL_ORDINAL_MEANS = {
    'MS SubClass': 57.3874, 'Lot Frontage': 69.2246, 'Lot Area': 10147.9218, 
    'Lot Shape': 2.5976, 'Utilities': 2.9986, 'Land Slope': 1.9464, 
    'Overall Qual': 6.0949, 'Overall Cond': 5.5631, 'Year Built': 1971.3563, 
    'Year Remod/Add': 1984.2666, 'Mas Vnr Area': 101.8968, 'Exter Qual': 2.3986, 
    'Exter Cond': 2.0853, 'Bsmt Qual': 3.4799, 'Bsmt Cond': 2.9239, 
    'Bsmt Exposure': 1.6307, 'BsmtFin Type 1': 3.5509, 'BsmtFin SF 1': 442.6296, 
    'BsmtFin Type 2': 1.2758, 'BsmtFin SF 2': 49.7224, 'Bsmt Unf SF': 559.2625, 
    'Total Bsmt SF': 1051.6145, 'Heating QC': 3.1495, '1st Flr SF': 1159.5577, 
    '2nd Flr SF': 335.4560, 'Low Qual Fin SF': 4.6768, 'Gr Liv Area': 1499.6904, 
    'Bsmt Full Bath': 0.4314, 'Bsmt Half Bath': 0.0611, 'Full Bath': 1.5666, 
    'Half Bath': 0.3795, 'Bedroom AbvGr': 2.8543, 'Kitchen AbvGr': 1.0444, 
    'Kitchen Qual': 2.5113, 'TotRms AbvGrd': 6.4430, 'Functional': 6.8440, 
    'Fireplaces': 0.5993, 'Fireplace Qu': 1.7703, 'Garage Yr Blt': 1978.1323, 
    'Garage Finish': 1.7205, 'Garage Cars': 1.7668, 'Garage Area': 472.8197, 
    'Garage Qual': 2.8027, 'Garage Cond': 2.8106, 'Paved Drive': 1.8314, 
    'Wood Deck SF': 93.7519, 'Open Porch SF': 47.5334, 'Enclosed Porch': 23.0116, 
    '3Ssn Porch': 2.5925, 'Screen Porch': 16.0020, 'Pool Area': 2.2433, 
    'Pool QC': 0.0123, 'Fence': 0.5795, 'Misc Val': 50.6352, 
    'Mo Sold': 6.2160, 'Yr Sold': 2007.7904, 'TotalSF': 2546.6282
}

EXPECTED_NOMINAL_HEADERS = [
    'MS Zoning', 'Street', 'Alley', 'Land Contour', 'Lot Config', 'Neighborhood', 
    'Condition 1', 'Condition 2', 'Bldg Type', 'House Style', 'Roof Style', 
    'Roof Matl', 'Exterior 1st', 'Exterior 2nd', 'Mas Vnr Type', 'Foundation', 
    'Heating', 'Central Air', 'Electrical', 'Garage Type', 'Misc Feature', 
    'Sale Type', 'Sale Condition'
]

# Specific Mappings for UI and Model
KITCHEN_QUAL_MAP_UI_TO_CODE = {
    "Poor (Po)": "Po",
    "Fair (Fa)": "Fa",
    "Typical/Average (TA)": "TA",
    "Good (Gd)": "Gd",
    "Excellent (Ex)": "Ex"
}
KITCHEN_QUAL_CODE_TO_MODEL_VALUE = {
    'Po': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4
}

# Features exposed in Gradio UI
UI_FEATURES_NUMERICAL_ORDINAL = {
    'Overall Qual': {'min': 1, 'max': 10, 'step': 1, 'label': 'Overall Quality (1-10)'},
    '1st Flr SF': {'label': '1st Floor Square Feet'},
    '2nd Flr SF': {'label': '2nd Floor Square Feet (can be 0)'},
    'Total Bsmt SF': {'label': 'Total Basement Square Feet (can be 0)'},
    'Year Built': {'label': 'Year Built (e.g., 1995)', 'min': 1800, 'max': 2024},
    'Gr Liv Area': {'label': 'Above Grade Living Area (sq. ft.)'},
    'Lot Area': {'label': 'Lot Area (sq. ft.)'},
    'Kitchen Qual': {'label': 'Kitchen Quality', 'choices': list(KITCHEN_QUAL_MAP_UI_TO_CODE.keys())},
    'Garage Cars': {'label': 'Garage Capacity (Number of Cars)', 'min': 0, 'max': 5, 'step': 1}
}
UI_FEATURES_NOMINAL = {
    'Neighborhood': {'label': 'Neighborhood'}
}

# Neighborhood mappings
neighborhood_code_to_full = {
    "blmngtn": "Bloomington Heights",
    "blueste": "Bluestem",
    "brdale": "Briardale",
    "brkside": "Brookside",
    "clearcr": "Clear Creek",
    "collgcr": "College Creek",
    "crawfor": "Crawford",
    "edwards": "Edwards",
    "gilbert": "Gilbert",
    "greens": "Greens",
    "grnhill": "Green Hills",
    "idotrr": "Iowa DOT and Rail Road",
    "landmrk": "Landmark",
    "meadowv": "Meadow Village",
    "mitchel": "Mitchell",
    "names": "North Ames",
    "noridge": "Northridge",
    "npkvill": "Northpark Villa",
    "nridght": "Northridge Heights",
    "nwames": "Northwest Ames",
    "oldtown": "Old Town",
    "swisu": "South & West of Iowa State University",
    "sawyer": "Sawyer",
    "sawyerw": "Sawyer West",
    "somerst": "Somerset",
    "stonebr": "Stone Brook",
    "timber": "Timberland",
    "veenker": "Veenker"
}
full_to_code = {v: k for k, v in neighborhood_code_to_full.items()}

def predict_ames_price(
    overall_qual, first_flr_sf, second_flr_sf, total_bsmt_sf, 
    year_built, gr_liv_area, lot_area, kitchen_qual_label_str, garage_cars,
    neighborhood_str
    ):
    if not model or not scaler or not nominal_column_mappings:
        return "Error: Model artifacts not loaded. Check console for details."

    try:
        # 1. Calculate derived features
        calculated_total_sf = float(first_flr_sf) + float(second_flr_sf) + float(total_bsmt_sf)

        # 2. Prepare Numerical Features array
        num_ord_input_array = np.zeros(len(EXPECTED_NUMERICAL_ORDINAL_HEADERS))
        
        # Create a dictionary of the UI inputs for easier lookup
        kitchen_short_code = KITCHEN_QUAL_MAP_UI_TO_CODE.get(kitchen_qual_label_str, "TA")
        kitchen_model_value = KITCHEN_QUAL_CODE_TO_MODEL_VALUE.get(kitchen_short_code, 2)

        ui_inputs = {
            'Overall Qual': float(overall_qual),
            '1st Flr SF': float(first_flr_sf),
            '2nd Flr SF': float(second_flr_sf),
            'Total Bsmt SF': float(total_bsmt_sf),
            'Year Built': float(year_built),
            'Gr Liv Area': float(gr_liv_area),
            'Lot Area': float(lot_area),
            'Kitchen Qual': float(kitchen_model_value),
            'Garage Cars': float(garage_cars),
            'TotalSF': calculated_total_sf
        }

        for i, header_name in enumerate(EXPECTED_NUMERICAL_ORDINAL_HEADERS):
            if header_name in ui_inputs:
                num_ord_input_array[i] = ui_inputs[header_name]
            else:
                num_ord_input_array[i] = DEFAULT_NUMERICAL_ORDINAL_MEANS[header_name]
        
        # Scale numerical features
        numerical_features_scaled = scaler.transform(num_ord_input_array.reshape(1, -1))

        # 3. Prepare Nominal Features list
        nominal_features_input_list = []
        # User-provided nominal inputs
        ui_nominal_inputs = {
            'Neighborhood': neighborhood_str
        }

        if ' (' in neighborhood_str and neighborhood_str.endswith(')'):
            neighborhood_code = neighborhood_str.split(' (')[-1][:-1]
        else:
            neighborhood_code = full_to_code.get(neighborhood_str, neighborhood_str)

        for header_name in EXPECTED_NOMINAL_HEADERS:
            int_code = 0  # Default
            if header_name in ui_nominal_inputs:
                user_str_value = ui_nominal_inputs[header_name]
                if header_name == 'Neighborhood':
                    if ' (' in user_str_value and user_str_value.endswith(')'):
                        user_str_value = user_str_value.split(' (')[-1][:-1]
                if header_name in nominal_column_mappings and user_str_value in nominal_column_mappings[header_name]:
                    int_code = nominal_column_mappings[header_name][user_str_value]
            nominal_features_input_list.append(int_code)

        # 4. Combine features and make prediction
        nominal_features_array = np.array(nominal_features_input_list).reshape(1, -1)
        combined_features = np.hstack([numerical_features_scaled, nominal_features_array])
        
        predicted_price = model.predict(combined_features)[0][0]
        
        # Clip predictions to reasonable range
        predicted_price = np.clip(predicted_price, 34900, 755000)
        
        return f"${predicted_price:,.2f}"

    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        return f"Error during prediction: {str(e)}"

def error_fn():
    return "Error: Model artifacts not loaded. Please check the console for details."

# Create the Gradio interface
with gr.Blocks(title="Ames Housing Price Predictor (Deep Learning)", theme=gr.themes.Default()) as demo:
    gr.Markdown("""
    # 🏠 Ames Housing Price Predictor (Deep Learning)
    
    This tool uses a Deep Learning model to predict house prices in Ames, Iowa. Enter the details of a property below to get an estimated price.
    The model has been trained on historical data and achieves:
    - RMSE: ~$32,900 ± $1,100 (18.2% of mean home value)
    - R² Score: 0.827 ± 0.016
    - MAPE: 11.53% ± 0.26%
    
    *Note: For more accurate predictions, consider using our Random Forest model instead.*
    """)
    
    with gr.Row():
        with gr.Column():
            # Basic Property Details
            gr.Markdown("### Basic Property Details")
            overall_qual = gr.Slider(**UI_FEATURES_NUMERICAL_ORDINAL['Overall Qual'])
            year_built = gr.Number(**UI_FEATURES_NUMERICAL_ORDINAL['Year Built'])
            
            # Square Footage
            gr.Markdown("### Square Footage")
            first_flr_sf = gr.Number(**UI_FEATURES_NUMERICAL_ORDINAL['1st Flr SF'])
            second_flr_sf = gr.Number(**UI_FEATURES_NUMERICAL_ORDINAL['2nd Flr SF'])
            total_bsmt_sf = gr.Number(**UI_FEATURES_NUMERICAL_ORDINAL['Total Bsmt SF'])
            gr_liv_area = gr.Number(**UI_FEATURES_NUMERICAL_ORDINAL['Gr Liv Area'])
            
            # Additional Features
            gr.Markdown("### Additional Features")
            lot_area = gr.Number(**UI_FEATURES_NUMERICAL_ORDINAL['Lot Area'])
            kitchen_qual = gr.Dropdown(**UI_FEATURES_NUMERICAL_ORDINAL['Kitchen Qual'])
            garage_cars = gr.Slider(**UI_FEATURES_NUMERICAL_ORDINAL['Garage Cars'])
            
            # Location
            gr.Markdown("### Location")
            neighborhood_choices = [f"{full} ({code})" for code, full in neighborhood_code_to_full.items()]
            neighborhood = gr.Dropdown(choices=sorted(neighborhood_choices), label=UI_FEATURES_NOMINAL['Neighborhood']['label'])

        with gr.Column():
            # Output
            gr.Markdown("### Predicted Price")
            output = gr.Textbox(label="Estimated Price")
            predict_btn = gr.Button("Predict Price", variant="primary")
            
            gr.Markdown("""
            ### Understanding the Inputs
            
            - **Overall Quality**: Rate from 1-10, where 10 is excellent
            - **Square Footage**: Enter actual measurements or estimates
            - **Year Built**: Use the actual construction year
            - **Kitchen Quality**: Choose the closest match
            - **Garage**: Number of cars it can hold
            - **Neighborhood**: Select from available Ames neighborhoods
            
            *Note: All predictions are estimates. Please consult with real estate professionals for accurate valuations.*
            """)

    predict_btn.click(
        fn=predict_ames_price if model else error_fn,
        inputs=[
            overall_qual, first_flr_sf, second_flr_sf, total_bsmt_sf,
            year_built, gr_liv_area, lot_area, kitchen_qual, garage_cars,
            neighborhood
        ],
        outputs=output
    )

if __name__ == "__main__":
    demo.launch() 