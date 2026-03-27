import numpy as np
import pickle
from tensorflow.keras.models import load_model

# Load trained model and scaler

model = load_model("models/lstm_model.keras")
scaler = pickle.load(open("models/scaler.pkl", "rb"))

LOOKBACK = 24
FEATURES = 16

def predict_load(input_sequence):


    input_sequence = np.array(input_sequence)

    # reshape to LSTM format
    input_sequence = input_sequence.reshape(1, LOOKBACK, FEATURES)

    # scale data
    scaled = scaler.transform(input_sequence.reshape(-1, FEATURES))
    scaled = scaled.reshape(1, LOOKBACK, FEATURES)

    # prediction
    pred_scaled = model.predict(scaled)

    # inverse scaling
    dummy = np.zeros((1, FEATURES))
    dummy[0,0] = pred_scaled[0,0]

    prediction = scaler.inverse_transform(dummy)[0,0]

    return prediction

