import streamlit as st
import numpy as np
import datetime
import pickle
from tensorflow.keras.models import load_model

# ===============================
# LOAD MODEL AND SCALER
# ===============================

model = load_model("models/lstm_model.keras")
scaler = pickle.load(open("models/scaler.pkl", "rb"))

LOOKBACK = 24
FEATURES = 16

# ===============================
# PAGE TITLE
# ===============================

st.title("EV Charging Load Forecasting (LSTM)")
st.write("Predict next hour EV charging demand")

# ===============================
# USER INPUT FEATURES (7)
# ===============================

st.subheader("Charging Station Parameters")

num_arrivals = st.number_input("Number of EV Arrivals", value=5)

stay_min = st.number_input("Average Stay Time (minutes)", value=60.0)

pmax = st.number_input("Maximum Charging Power (W)", value=11000.0)

preq = st.number_input("Requested Charging Power (W)", value=9000.0)

soc_arrival = st.slider("SOC Arrival (%)", 0, 100, 30)

soc_departure = st.slider("SOC Departure (%)", 0, 100, 80)

battery_capacity = st.number_input("Battery Capacity (Wh)", value=60000.0)

# ===============================
# AUTO GENERATED TIME FEATURES
# ===============================

now = datetime.datetime.now()

hour_of_day = now.hour
day_of_week = now.weekday()
month = now.month
weekend = 1 if day_of_week >= 5 else 0

# ===============================
# ESTIMATED SYSTEM FEATURES
# ===============================

controlled_sessions = int(num_arrivals * 0.3)

lag_1 = 10
lag_24 = 9
rolling_mean = (lag_1 + lag_24) / 2

current_load = lag_1

# ===============================
# BUILD FEATURE VECTOR
# ===============================

input_row = [
    current_load,
    stay_min,
    pmax,
    preq,
    controlled_sessions,
    soc_arrival,
    soc_departure,
    battery_capacity,
    num_arrivals,
    hour_of_day,
    day_of_week,
    month,
    weekend,
    lag_1,
    lag_24,
    rolling_mean
]

sequence = [input_row] * LOOKBACK

# ===============================
# PREDICTION
# ===============================

if st.button("Predict Next Hour Load"):

    seq = np.array(sequence)
    seq = seq.reshape(1, LOOKBACK, FEATURES)

    scaled = scaler.transform(seq.reshape(-1, FEATURES))
    scaled = scaled.reshape(1, LOOKBACK, FEATURES)

    pred_scaled = model.predict(scaled)

    dummy = np.zeros((1, FEATURES))
    dummy[0, 0] = pred_scaled[0, 0]

    prediction = scaler.inverse_transform(dummy)[0, 0]

    st.success(f"Predicted EV Charging Load: {prediction:.2f} kWh")


import os

port = int(os.environ.get("PORT", 8501))

if __name__ == "__main__":
    import streamlit.web.cli as stcli
    import sys
    sys.argv = ["streamlit", "run", "app.py", "--server.port", str(port), "--server.address", "0.0.0.0"]
    stcli.main()