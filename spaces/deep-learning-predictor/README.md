---
title: Ames Housing Price Predictor (Deep Learning)
emoji: 🏠
colorFrom: purple
colorTo: indigo
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
license: mit
---

# Ames Housing Price Predictor (Deep Learning)

This Gradio app provides an interactive interface for predicting house prices in Ames, Iowa using a Deep Learning model. The model has been trained on the Ames Housing dataset and achieves the following performance metrics:

- RMSE: ~$32,900 ± $1,100 (18.2% of mean home value)
- R² Score: 0.827 ± 0.016
- MAPE: 11.53% ± 0.26%

*Note: For more accurate predictions, consider using our [Random Forest model](https://huggingface.co/spaces/ames-housing/random-forest-predictor) instead.*

## Using the App

1. Enter property details in the left column:
   - Overall Quality (1-10)
   - Year Built
   - Square Footage (1st Floor, 2nd Floor, Basement)
   - Living Area
   - Lot Area
   - Kitchen Quality
   - Garage Capacity
   - Neighborhood

2. Click "Predict Price" to get an estimated value.

## Model Details

The Deep Learning model is a neural network trained on historical housing data from Ames, Iowa. It processes both numerical features (like square footage) and categorical features (like neighborhood and quality ratings).

### Model Architecture
- Input Layer: Handles both numerical and categorical features
- Hidden Layers: Multiple dense layers with ReLU activation
- Output Layer: Single neuron for price prediction
- Features scaled using StandardScaler
- Categorical features encoded using integer encoding

## Important Notes

- All predictions are estimates and should be validated by professional appraisers
- The model is based on historical data and local market conditions
- Some features use default values when not specified
- Predictions work best for properties similar to those in the training data
- The model tends to underpredict high-value properties

## Technical Implementation

The app is built using:
- Gradio 4.0.0+ for the web interface
- TensorFlow 2.13.0+ for the Deep Learning model
- numpy for numerical operations
- joblib for model loading

## License

This project is licensed under the MIT License. 