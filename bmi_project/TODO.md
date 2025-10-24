# TODO List for BMI Prediction App Enhancements

## 1. Modify Flask Backend (app.py)
- [x] Add BMI category classification function (Underweight, Normal, Overweight, Obese with subcategories).
- [x] Update `/predict_numerical` endpoint to return BMI value and category.
- [x] Update `/predict_image` endpoint to return BMI value and category.
- [x] Add `/calorie_calculator` endpoint for BMR and daily calorie needs.
- [x] Fix JSON serialization issue for image predictions.

## 2. Update Streamlit Frontend (bmi_app.py)
- [x] Modify numerical prediction section to display BMI category alongside value.
- [x] Modify image prediction section to display BMI category alongside value.
- [x] Add a new section for Calorie Calculator:
  - [x] Add inputs: gender, height, weight, age, activity level.
  - [x] Calculate BMR using Mifflin-St Jeor equation.
  - [x] Apply activity factor to get daily calorie needs.
  - [x] Display results.

## 3. Testing
- [x] Run Flask app and test endpoints.
- [x] Run Streamlit app and test all features.
- [x] Ensure BMI categories are accurate.
- [x] Verify calorie calculator formulas.
- [x] Fix image prediction feature mismatch issue (pad with zeros for scaler, use only 1280 features for model).

## 4. Final Checks
- [x] Code review for any errors.
- [x] Update documentation if needed.
