# ✈️ Flight Fare Prediction

Predicting flight ticket prices using machine learning, based on airline, route, timing, and stop count — helping customers compare fares and plan their journey.

**🔗 Live App:** [flight-fare-prediction-rakesh30-05.streamlit.app](https://flight-fare-prediction-23jbqgkaxpadwmghhbevko.streamlit.app/)

---

## Overview

This project predicts flight fares using historical flight data and compares four regression models to find the best-performing one. The final model is deployed as an interactive Streamlit web app where users can input flight details and get a predicted fare, or compare fares across different days of the month.

## Dataset

- Indian domestic flight data
- Features: Airline, Source, Destination, Route, Date of Journey, Departure/Arrival Time, Duration, Total Stops, Additional Info
- Target: Price

## Approach

1. **Data Cleaning** — removed nulls and duplicates
2. **EDA** — analyzed price trends across airline, route, month, departure hour, and stops
3. **Feature Engineering** — extracted day/month/hour/minute from date-time columns, converted duration text to minutes, encoded categorical features
4. **Modeling** — compared four models with increasing complexity:
   - Linear Regression (baseline)
   - Decision Tree
   - Random Forest (tuned with RandomizedSearchCV)
   - XGBoost
5. **Evaluation** — R², MAE, RMSE, and cross-validation used to check generalization and avoid overfitting

## Results

| Model | R² Score |
|---|---|
| Linear Regression | ~0.61 (underfits — price isn't linear) |
| Decision Tree | ~0.72 (overfits on its own) |
| **Random Forest (tuned)** | **0.84** ✅ Best model |
| XGBoost | ~0.82 |

**Final model: Random Forest Regressor**, selected for the best balance of accuracy and generalization on this dataset.

## Tech Stack

Python, Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, Seaborn, Streamlit, Joblib

## How to Run Locally

```bash
git clone https://github.com/rakesh30-05/flight-fare-prediction.git
cd flight-fare-prediction
pip install -r requirements.txt
streamlit run app.py
```

## Project Structure

```
flight-fare-prediction/
├── app.py                              # Streamlit web app
├── predict.py                          # Standalone prediction script
├── Flight_Fare_Prediction_FIXED.ipynb  # Full analysis & model notebook
├── Flight_Fare.xlsx                    # Dataset
├── flight_fare_model.pkl               # Saved trained model
├── flight_fare_model_columns.pkl       # Saved feature columns
└── requirements.txt
```

## Limitations

- Trained on historical/static data — doesn't reflect real-time pricing or demand surges
- One-hot encoding means new/unseen airlines or routes aren't handled automatically
- Not validated for production use — built as a learning/portfolio project

## Author

D N Rakesh — [LinkedIn](www.linkedin.com/in/dnrakesh8008) | [GitHub](https://github.com/rakesh30-05)