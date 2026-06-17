import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Flight Fare Predictor", page_icon="✈️", layout="centered")

st.title("✈️ Flight Fare Prediction")
st.write("Enter your flight details below to get a predicted fare and plan your journey.")

# ---- Load model ----
@st.cache_resource
def load_model():
    model = joblib.load("flight_fare_model.pkl")
    model_columns = joblib.load("flight_fare_model_columns.pkl")
    return model, model_columns

model, model_columns = load_model()

# ---- Extract available categories from the trained columns ----
def extract_options(prefix):
    options = [col.replace(prefix, "") for col in model_columns if col.startswith(prefix)]
    return sorted(options)

airlines = extract_options("Airline_")
sources = extract_options("Source_")
destinations = extract_options("Destination_")

# ---- Input form ----
with st.form("flight_form"):
    col1, col2 = st.columns(2)

    with col1:
        airline = st.selectbox("Airline", airlines)
        source = st.selectbox("Source", sources)
        destination = st.selectbox("Destination", destinations)
        total_stops = st.selectbox("Total Stops", [0, 1, 2, 3, 4])

    with col2:
        journey_day = st.number_input("Journey Day", min_value=1, max_value=31, value=15)
        journey_month = st.number_input("Journey Month", min_value=1, max_value=12, value=5)
        dep_time = st.time_input("Departure Time")
        arrival_time = st.time_input("Arrival Time")

    duration_minutes = st.slider("Flight Duration (minutes)", min_value=30, max_value=1500, value=180)

    submitted = st.form_submit_button("Predict Fare")

# ---- Prediction ----
if submitted:
    row = {col: 0 for col in model_columns}

    row["Journey_Day"] = journey_day
    row["Journey_Month"] = journey_month
    row["Dep_Hour"] = dep_time.hour
    row["Dep_Min"] = dep_time.minute
    row["Arrival_Hour"] = arrival_time.hour
    row["Arrival_Min"] = arrival_time.minute
    row["Duration_Minutes"] = duration_minutes
    row["Total_Stops"] = total_stops

    for col in (f"Airline_{airline}", f"Source_{source}", f"Destination_{destination}"):
        if col in row:
            row[col] = 1

    input_df = pd.DataFrame([row], columns=model_columns)
    predicted_price = model.predict(input_df)[0]

    st.success(f"### Predicted Fare: ₹{round(float(predicted_price), 2)}")
    st.caption(
        "This is a model estimate based on historical data, not a live airline quote. "
        "Use it to compare options, not as a guaranteed price."
    )

# ---- Compare across days ----
st.divider()
st.subheader("📅 Compare Fares Across the Month")
st.write("Keep all other details the same and see which day might be cheapest.")

if st.button("Compare Day 5, 10, 15, 20, 25 of this month"):
    results = []
    for day in [5, 10, 15, 20, 25]:
        row = {col: 0 for col in model_columns}
        row["Journey_Day"] = day
        row["Journey_Month"] = journey_month
        row["Dep_Hour"] = dep_time.hour
        row["Dep_Min"] = dep_time.minute
        row["Arrival_Hour"] = arrival_time.hour
        row["Arrival_Min"] = arrival_time.minute
        row["Duration_Minutes"] = duration_minutes
        row["Total_Stops"] = total_stops
        for col in (f"Airline_{airline}", f"Source_{source}", f"Destination_{destination}"):
            if col in row:
                row[col] = 1
        input_df = pd.DataFrame([row], columns=model_columns)
        price = round(float(model.predict(input_df)[0]), 2)
        results.append({"Day": day, "Predicted Fare (₹)": price})

    results_df = pd.DataFrame(results).sort_values("Predicted Fare (₹)")
    st.dataframe(results_df, use_container_width=True, hide_index=True)
    cheapest = results_df.iloc[0]
    st.info(f"Cheapest option: Day {int(cheapest['Day'])} at ₹{cheapest['Predicted Fare (₹)']}")

st.divider()
st.caption("Model: Random Forest Regressor | R² ≈ 0.84 on test data | Built as a portfolio project")
