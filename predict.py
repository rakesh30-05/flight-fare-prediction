"""
predict.py
-----------
Loads the trained flight fare model and predicts the price for a NEW flight
that a customer wants to book — so they can compare options and plan their
journey (e.g. "is it cheaper to fly next Tuesday or next Friday?").

Usage:
    python predict.py

Requires (saved earlier from the notebook):
    flight_fare_model.pkl
    flight_fare_model_columns.pkl
"""

import joblib
import pandas as pd

MODEL_PATH = "flight_fare_model.pkl"
COLUMNS_PATH = "flight_fare_model_columns.pkl"


def load_model():
    model = joblib.load(MODEL_PATH)
    model_columns = joblib.load(COLUMNS_PATH)
    return model, model_columns


def build_input_row(model_columns, airline, source, destination,
                     journey_day, journey_month,
                     dep_hour, dep_min,
                     arrival_hour, arrival_min,
                     duration_minutes, total_stops):
    """
    Builds a single-row DataFrame matching the exact columns the model
    was trained on (including one-hot encoded Airline/Source/Destination),
    filling 0 for any dummy column that doesn't match this flight.
    """
    row = {col: 0 for col in model_columns}

    row["Journey_Day"] = journey_day
    row["Journey_Month"] = journey_month
    row["Dep_Hour"] = dep_hour
    row["Dep_Min"] = dep_min
    row["Arrival_Hour"] = arrival_hour
    row["Arrival_Min"] = arrival_min
    row["Duration_Minutes"] = duration_minutes
    row["Total_Stops"] = total_stops

    airline_col = f"Airline_{airline}"
    source_col = f"Source_{source}"
    dest_col = f"Destination_{destination}"

    for col in (airline_col, source_col, dest_col):
        if col in row:
            row[col] = 1
        # If the category wasn't seen during training (e.g. a new airline),
        # it's silently dropped to 0 across all dummy columns for that
        # feature — the model falls back to the "baseline" category.
        # This is a known limitation; see note at the bottom of this file.

    return pd.DataFrame([row], columns=model_columns)


def predict_fare(airline, source, destination,
                  journey_day, journey_month,
                  dep_hour, dep_min,
                  arrival_hour, arrival_min,
                  duration_minutes, total_stops):
    model, model_columns = load_model()
    input_row = build_input_row(
        model_columns, airline, source, destination,
        journey_day, journey_month,
        dep_hour, dep_min, arrival_hour, arrival_min,
        duration_minutes, total_stops
    )
    predicted_price = model.predict(input_row)[0]
    return round(float(predicted_price), 2)


def compare_dates(base_kwargs, day_options):
    """
    Helper for customers: predicts fare across multiple candidate journey
    days (same flight, different dates) so they can pick the cheapest one.
    """
    results = []
    for day in day_options:
        kwargs = dict(base_kwargs)
        kwargs["journey_day"] = day
        price = predict_fare(**kwargs)
        results.append((day, price))
    return sorted(results, key=lambda x: x[1])


if __name__ == "__main__":
    # Example: customer wants to fly Delhi -> Cochin on IndiGo,
    # and wants to know which day in the month is cheapest.
    base_flight = dict(
        airline="IndiGo",
        source="Delhi",
        destination="Cochin",
        journey_month=5,
        dep_hour=9, dep_min=0,
        arrival_hour=14, arrival_min=30,
        duration_minutes=330,
        total_stops=1,
    )

    single_price = predict_fare(journey_day=15, **base_flight)
    print(f"Predicted fare for May 15th: Rs. {single_price}")

    print("\nComparing fares across the month to help the customer plan:")
    cheapest_first = compare_dates(base_flight, day_options=[5, 10, 15, 20, 25])
    for day, price in cheapest_first:
        print(f"  Day {day}: Rs. {price}")

    best_day, best_price = cheapest_first[0]
    print(f"\nRecommendation: Booking on day {best_day} is predicted to be cheapest at Rs. {best_price}")
