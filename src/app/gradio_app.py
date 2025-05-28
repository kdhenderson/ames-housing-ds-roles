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
# Assuming the script is in src/app/ and artifacts are in data/processed/
# Adjust an additional '..' to navigate from src/app/ to the project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
MODEL_ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
FINAL_MODEL_PATH = os.path.join(MODEL_ARTIFACTS_DIR, 'final_deep_learning_model.h5')
SCALER_PATH = os.path.join(MODEL_ARTIFACTS_DIR, 'dl_numerical_scaler.joblib')
NOMINAL_MAPS_PATH = os.path.join(MODEL_ARTIFACTS_DIR, 'nominal_column_mappings.json')

# --- Load Artifacts ---
try:
    model = tf.keras.models.load_model(FINAL_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    with open(NOMINAL_MAPS_PATH, 'r') as f:
        nominal_column_mappings = json.load(f)
    print("Model, scaler, and nominal mappings loaded successfully.")
except Exception as e:
    print(f"Error loading artifacts: {e}")
    # Placeholder for Gradio to still launch with an error message
    model, scaler, nominal_column_mappings = None, None, None

# --- Feature Definitions (from temp_get_headers.py output) ---
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

# Features exposed in Gradio UI (as per user's refined list)
UI_FEATURES_NUMERICAL_ORDINAL = {
    'Overall Qual': {'min': 1, 'max': 10, 'step': 1, 'label': 'Overall Quality (1-10)'},
    '1st Flr SF': {'label': '1st Floor Square Feet'},
    '2nd Flr SF': {'label': '2nd Floor Square Feet (can be 0)'},
    'Total Bsmt SF': {'label': 'Total Basement Square Feet (can be 0)'}, # Used to calculate TotalSF
    'Year Built': {'label': 'Year Built (e.g., 1995)', 'min': 1800, 'max': 2024},
    'Gr Liv Area': {'label': 'Above Grade Living Area (sq. ft.)'},
    'Lot Area': {'label': 'Lot Area (sq. ft.)'},
    'Kitchen Qual': {'label': 'Kitchen Quality', 'choices': list(KITCHEN_QUAL_MAP_UI_TO_CODE.keys())},
    'Garage Cars': {'label': 'Garage Capacity (Number of Cars)', 'min': 0, 'max': 5, 'step': 1}
}
UI_FEATURES_NOMINAL = {
    'Neighborhood': {'label': 'Neighborhood'} # Choices will be populated from mappings
}

# Add this mapping at the top after imports
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

# --- Prediction Function ---
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
        kitchen_short_code = KITCHEN_QUAL_MAP_UI_TO_CODE.get(kitchen_qual_label_str, "TA") # Default to TA if not found
        kitchen_model_value = KITCHEN_QUAL_CODE_TO_MODEL_VALUE.get(kitchen_short_code, 2) # Default to 2 (TA) for model

        ui_inputs = {
            'Overall Qual': float(overall_qual),
            '1st Flr SF': float(first_flr_sf),
            '2nd Flr SF': float(second_flr_sf),
            'Total Bsmt SF': float(total_bsmt_sf), # This specific feature is used for TotalSF calculation
            'Year Built': float(year_built),
            'Gr Liv Area': float(gr_liv_area),
            'Lot Area': float(lot_area),
            'Kitchen Qual': float(kitchen_model_value),
            'Garage Cars': float(garage_cars),
            'TotalSF': calculated_total_sf # Add the calculated TotalSF
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
            # Add other nominal UI inputs here if any in the future
        }

        if ' (' in neighborhood_str and neighborhood_str.endswith(')'):
            neighborhood_code = neighborhood_str.split(' (')[-1][:-1]
        else:
            neighborhood_code = full_to_code.get(neighborhood_str, neighborhood_str)

        for header_name in EXPECTED_NOMINAL_HEADERS:
            int_code = 0 # Default
            if header_name in ui_nominal_inputs:
                user_str_value = ui_nominal_inputs[header_name]
                if header_name == 'Neighborhood':
                    if ' (' in user_str_value and user_str_value.endswith(')'):
                        user_str_value = user_str_value.split(' (')[-1][:-1]
                if header_name in nominal_column_mappings and user_str_value in nominal_column_mappings[header_name]:
                    int_code = nominal_column_mappings[header_name][user_str_value]
                # else: int_code remains 0 (or some other default for unknown category if model trained for it)
            elif header_name in nominal_column_mappings and nominal_column_mappings[header_name]: # Non-UI nominal, use first mapped value as default
                first_key = list(nominal_column_mappings[header_name].keys())[0]
                int_code = nominal_column_mappings[header_name][first_key]
            
            nominal_features_input_list.append(np.array([[int_code]], dtype=np.int32))

        # 4. Make Prediction
        model_inputs = [numerical_features_scaled] + nominal_features_input_list
        predicted_log_price = model.predict(model_inputs)
        
        # 5. Inverse Transform Prediction
        predicted_price = np.expm1(predicted_log_price[0][0])
        
        return f"${predicted_price:,.2f}"

    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        return f"Error during prediction: {str(e)}"

# --- Define Gradio Inputs ---
gradio_inputs = []
# Numerical/Ordinal Inputs from UI_FEATURES_NUMERICAL_ORDINAL
gradio_inputs.append(gr.Slider(minimum=UI_FEATURES_NUMERICAL_ORDINAL['Overall Qual']['min'], 
                                maximum=UI_FEATURES_NUMERICAL_ORDINAL['Overall Qual']['max'], 
                                step=UI_FEATURES_NUMERICAL_ORDINAL['Overall Qual']['step'],
                                label=UI_FEATURES_NUMERICAL_ORDINAL['Overall Qual']['label'],
                                value=int(round(DEFAULT_NUMERICAL_ORDINAL_MEANS['Overall Qual']))))
gradio_inputs.append(gr.Number(label=UI_FEATURES_NUMERICAL_ORDINAL['1st Flr SF']['label'], value=DEFAULT_NUMERICAL_ORDINAL_MEANS['1st Flr SF'], precision=0))
gradio_inputs.append(gr.Number(label=UI_FEATURES_NUMERICAL_ORDINAL['2nd Flr SF']['label'], value=DEFAULT_NUMERICAL_ORDINAL_MEANS['2nd Flr SF'], precision=0))
gradio_inputs.append(gr.Number(label=UI_FEATURES_NUMERICAL_ORDINAL['Total Bsmt SF']['label'], value=DEFAULT_NUMERICAL_ORDINAL_MEANS['Total Bsmt SF'], precision=0))
gradio_inputs.append(gr.Number(label=UI_FEATURES_NUMERICAL_ORDINAL['Year Built']['label'], 
                                minimum=UI_FEATURES_NUMERICAL_ORDINAL['Year Built']['min'],
                                maximum=UI_FEATURES_NUMERICAL_ORDINAL['Year Built']['max'],
                                value=int(round(DEFAULT_NUMERICAL_ORDINAL_MEANS['Year Built'])), precision=0))
gradio_inputs.append(gr.Number(label=UI_FEATURES_NUMERICAL_ORDINAL['Gr Liv Area']['label'], value=DEFAULT_NUMERICAL_ORDINAL_MEANS['Gr Liv Area'], precision=0))
gradio_inputs.append(gr.Number(label=UI_FEATURES_NUMERICAL_ORDINAL['Lot Area']['label'], value=DEFAULT_NUMERICAL_ORDINAL_MEANS['Lot Area'], precision=0))

# Kitchen Quality (Ordinal handled as dropdown using string keys from its mapping)
kitchen_qual_choices = list(KITCHEN_QUAL_MAP_UI_TO_CODE.keys())
gradio_inputs.append(gr.Dropdown(choices=kitchen_qual_choices, 
                                 label=UI_FEATURES_NUMERICAL_ORDINAL['Kitchen Qual']['label'],
                                 value="Typical/Average (TA)"))

gradio_inputs.append(gr.Slider(minimum=UI_FEATURES_NUMERICAL_ORDINAL['Garage Cars']['min'], 
                                maximum=UI_FEATURES_NUMERICAL_ORDINAL['Garage Cars']['max'], 
                                step=UI_FEATURES_NUMERICAL_ORDINAL['Garage Cars']['step'],
                                label=UI_FEATURES_NUMERICAL_ORDINAL['Garage Cars']['label'],
                                value=int(round(DEFAULT_NUMERICAL_ORDINAL_MEANS['Garage Cars']))))


# Nominal Inputs from UI_FEATURES_NOMINAL
# Neighborhood
neighborhood_choices = ["Pick a Neighborhood"] # Placeholder if loading fails
default_neighborhood_str = "NAmes" # A common default
if nominal_column_mappings and 'Neighborhood' in nominal_column_mappings and nominal_column_mappings['Neighborhood']:
    neighborhood_choices = [f"{neighborhood_code_to_full.get(code.lower(), code)} ({code.lower()})" for code in sorted(nominal_column_mappings['Neighborhood'].keys())]
    if default_neighborhood_str not in neighborhood_choices and neighborhood_choices:
        default_neighborhood_str = neighborhood_choices[0]
elif not (nominal_column_mappings and 'Neighborhood' in nominal_column_mappings and nominal_column_mappings['Neighborhood']): 
    print("Warning: Neighborhood choices could not be loaded from mappings.")

gradio_inputs.append(gr.Dropdown(choices=neighborhood_choices, 
                                 label=UI_FEATURES_NOMINAL['Neighborhood']['label'],
                                 value=default_neighborhood_str))


# --- Define Gradio Output ---
gradio_output = gr.Textbox(label="Predicted Sale Price")

# --- Create and Launch the Interface ---
# Check if artifacts loaded correctly before defining the interface
if model and scaler and nominal_column_mappings:
    interface = gr.Interface(
        fn=predict_ames_price,
        inputs=gradio_inputs,
        outputs=gradio_output,
        title="Ames Housing Price Predictor (Deep Learning)",
        description=("Enter house details to predict sale price. "
                     "Features not listed are set to average values from training data."),
        flagging_mode='never'
    )
else: # Fallback interface if artifacts failed to load
    def error_fn():
        return "Critical Error: Model artifacts could not be loaded. Please check the application logs."
    interface = gr.Interface(fn=error_fn, inputs=[], outputs=gr.Textbox(label="Error"), title="Ames Housing Price Predictor - ERROR")

if __name__ == "__main__":
    print("Attempting to launch Gradio app...")
    interface.launch() 