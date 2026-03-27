# ==========================================

# IMPORT LIBRARIES

# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ==========================================

# LOAD DATA

# ==========================================

file_path = "data/Session_data.xlsx"
df = pd.read_excel(file_path)

df.columns = df.columns.str.strip()

df.rename(columns={
'Controlled session (0=False, 1=True)': 'Controlled_session'
}, inplace=True)

df.columns = (
df.columns
.str.replace(" ", "_")
.str.replace("(", "", regex=False)
.str.replace(")", "", regex=False)
)

# ==========================================

# DATETIME & ENERGY CONVERSION

# ==========================================

df['Arrival'] = pd.to_datetime(df['Arrival'])
df['Departure'] = pd.to_datetime(df['Departure'])

df['Energy_kWh'] = df['Energy_Wh'] / 1000

# ==========================================

# RECONSTRUCT HOURLY LOAD

# ==========================================

hourly_index = pd.date_range(
start=df['Arrival'].min().floor('h'),
end=df['Departure'].max().ceil('h'),
freq='h'
)

hourly_load = pd.Series(0.0, index=hourly_index)

for _, row in df.iterrows():


    duration_hours = (row['Departure'] - row['Arrival']).total_seconds() / 3600

    if duration_hours > 0:

        power_per_hour = row['Energy_kWh'] / duration_hours

        session_hours = pd.date_range(
            start=row['Arrival'].floor('h'),
            end=row['Departure'].ceil('h'),
            freq='h'
        )

        for hour in session_hours:
            if hour in hourly_load.index:
                hourly_load[hour] += power_per_hour


data = hourly_load.to_frame(name='Load')

# ==========================================

# AGGREGATE SESSION FEATURES

# ==========================================

df['Hour'] = df['Arrival'].dt.floor('h')

hourly_features = df.groupby('Hour').agg({


'Stay_min': 'mean',
'Pmax_W': 'mean',
'Preq_max_W': 'mean',
'Controlled_session': 'sum',
'SOC_arrival': 'mean',
'SOC_departure': 'mean',
'Energy_capacity_Wh': 'mean',
'Session': 'count'


})

hourly_features.rename(columns={'Session': 'Num_arrivals'}, inplace=True)

data = data.merge(hourly_features, left_index=True, right_index=True, how='left')
data.fillna(0, inplace=True)

# ==========================================

# TIME + LAG FEATURES

# ==========================================

data['Hour_of_day'] = data.index.hour
data['Day_of_week'] = data.index.dayofweek
data['Month'] = data.index.month
data['Weekend'] = data['Day_of_week'].isin([5,6]).astype(int)

data['Lag_1'] = data['Load'].shift(1)
data['Lag_24'] = data['Load'].shift(24)
data['Rolling_mean_24'] = data['Load'].rolling(24).mean()

data.dropna(inplace=True)

# ==========================================

# SCALE DATA

# ==========================================

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# ==========================================

# CREATE LSTM SEQUENCES

# ==========================================

def create_sequences(dataset, lookback=24):


    X, y = [], []

    for i in range(len(dataset) - lookback):

        X.append(dataset[i:i+lookback])
        y.append(dataset[i+lookback, 0])

    return np.array(X), np.array(y)


lookback = 24
X_seq, y_seq = create_sequences(scaled_data, lookback)

# ==========================================

# TRAIN TEST SPLIT

# ==========================================

split = int(len(X_seq) * 0.8)

X_train = X_seq[:split]
X_test = X_seq[split:]

y_train = y_seq[:split]
y_test = y_seq[split:]

# ==========================================

# BUILD LSTM MODEL

# ==========================================

model = Sequential()

model.add(LSTM(64, return_sequences=True,
input_shape=(lookback, X_seq.shape[2])))

model.add(Dropout(0.2))

model.add(LSTM(32))

model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

early_stop = EarlyStopping(
monitor='val_loss',
patience=5,
restore_best_weights=True
)

# ==========================================

# TRAIN MODEL

# ==========================================

history = model.fit(
X_train,
y_train,
epochs=50,
batch_size=32,
validation_split=0.1,
callbacks=[early_stop],
verbose=1
)

# ==========================================

# PREDICTIONS

# ==========================================

y_pred_scaled = model.predict(X_test)

dummy_pred = np.zeros((len(y_pred_scaled), scaled_data.shape[1]))
dummy_pred[:,0] = y_pred_scaled.flatten()
y_pred = scaler.inverse_transform(dummy_pred)[:,0]

dummy_actual = np.zeros((len(y_test), scaled_data.shape[1]))
dummy_actual[:,0] = y_test.flatten()
y_actual = scaler.inverse_transform(dummy_actual)[:,0]

# ==========================================

# MODEL EVALUATION

# ==========================================

mae = mean_absolute_error(y_actual, y_pred)
rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
r2 = r2_score(y_actual, y_pred)

print("\nLSTM MODEL PERFORMANCE")
print("----------------------------")
print("MAE :", mae)
print("RMSE:", rmse)
print("R2  :", r2)

# ==========================================

# SAVE MODEL

# ==========================================

model.save("models/lstm_model.keras")
pickle.dump(scaler, open("models/scaler.pkl","wb"))

print("Model saved successfully!")

# ==========================================

# PLOTS

# ==========================================

# Training Loss

plt.figure()
plt.plot(history.history['loss'], label="Train Loss")
plt.plot(history.history['val_loss'], label="Validation Loss")
plt.title("Training Loss Curve")
plt.legend()
plt.grid(True)
plt.savefig("models/training_loss.png")

# Actual vs Predicted

plt.figure()
plt.plot(y_actual, label="Actual")
plt.plot(y_pred, label="Predicted")
plt.title("Actual vs Predicted Load")
plt.legend()
plt.grid(True)
plt.savefig("models/actual_vs_predicted.png")

# Residuals

residuals = y_actual - y_pred

plt.figure()
plt.plot(residuals)
plt.title("Residual Errors")
plt.grid(True)
plt.savefig("models/residual_plot.png")

# Residual Distribution

plt.figure()
plt.hist(residuals, bins=30)
plt.title("Residual Distribution")
plt.grid(True)
plt.savefig("models/residual_histogram.png")

# Scatter Plot

plt.figure()
plt.scatter(y_actual, y_pred)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Predicted vs Actual")
plt.grid(True)
plt.savefig("models/scatter_plot.png")

# Rolling Trend

rolling_actual = pd.Series(y_actual).rolling(24).mean()
rolling_pred = pd.Series(y_pred).rolling(24).mean()

plt.figure()
plt.plot(rolling_actual, label="Actual Trend")
plt.plot(rolling_pred, label="Predicted Trend")
plt.legend()
plt.title("24 Hour Rolling Trend")
plt.grid(True)
plt.savefig("models/trend_comparison.png")

print("All plots saved successfully.")
