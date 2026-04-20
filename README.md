# F12026_Prediction

### Project Overview
This project uses machine learning (Gradient Boosting Regressor), data from the FastF1 API, and historical F1 race results to predict race outcomes for the 2026 Formula 1 season. 

Side Note: Due to the new regulation changes (new power units, no DRS, aerodynamics, etc.), every historical data is deemed unreliable. The model prediction accuracy will only improve after every race as new session data arrives. 


### How It Works
1. Data Collection: We pull historical data and 2026 qualifying data from FastF1 API 
2. Preprocessing & Feature Engineering: convert lap times (from <code>timedelta</code> to plain seconds <code>.total_seconds</code>), normalise driver names, flag safety cars, etc.
3. Model Training: Gradient Boosting Regressor
4. Model Prediction: The model predicts race times for 2026 GP and ranks drivers
5. Model Evaluation: Mean Absolute Error (MAE) to evaluate how well the model predicts race times. Lower MAE values indicate more accurate predictions.
6. Model Visualisation

# Required Python Libraries
* <code>fastF1</code>
* <code>numpy</code>
* <code>pandas</code>
* <code>scikit-learn</code>
* <code>matplotlib</code>

### File Organisation
For every race, there will be a new .ipynb script labelled to the race on the calendar, e.g., F12026AusGP, F12026SHGP, etc. 
