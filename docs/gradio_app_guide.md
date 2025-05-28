# Ames Housing Price Predictor App: User Guide

## Purpose
Predict the sale price of a house in Ames, Iowa, using a machine learning model trained on historical data. The app is designed for ease of use by non-technical users and stakeholders.

## How to Use the App

1. **Enter Property Details**
   - Fill in the fields for property features such as:
     - Overall Quality (1-10)
     - 1st Floor Square Feet
     - 2nd Floor Square Feet (can be 0)
     - Total Basement Square Feet (can be 0)
     - Year Built
     - Above Grade Living Area (sq. ft.)
     - Lot Area (sq. ft.)
     - Kitchen Quality (choose from dropdown)
     - Garage Capacity (Number of Cars)
     - Neighborhood (choose from dropdown, shown as Full Name (code))

2. **Submit for Prediction**
   - Click the "Submit" or "Predict" button.
   - The predicted sale price will appear on the right side of the app.

3. **Interpreting Results**
   - The predicted price is based on the information you provided and average values for other features.
   - For best results, enter as much accurate information as possible.

## Notes
- **Neighborhood and Kitchen Quality**: These are shown as friendly names with their codes for clarity (e.g., "Northridge (noridge)").
- **Defaults**: If you leave a field blank, the app uses the average value from the training data.
- **Model**: The app uses a machine learning model trained on real Ames housing data. The Random Forest model is recommended for more realistic predictions.
- **Privacy**: No personal data is stored; predictions are made in real time.

## Troubleshooting
- If you see an error message, ensure all required fields are filled and try again.
- For technical issues, contact the project team. 