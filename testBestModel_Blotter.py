import pandas as pd
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Define functions to calculate technical indicators
def compute_macd(price, slow=26, fast=12, signal=9):
    ema_fast = price.ewm(span=fast, adjust=False).mean()
    ema_slow = price.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line, macd - signal_line

def compute_rsi(price, window=14):
    delta = price.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_obv(price, volume):
    sign = np.sign(price.diff())
    return (sign * volume).fillna(0).cumsum()

# Load the data
data = pd.read_csv('xnas-itch-20230703.tbbo.csv')
# data = pd.read_csv('dummy_stock_data.csv')
data['price'] = data['price'] / 1e9  # Normalize the price

# Apply the technical indicator functions
data['MACD'], data['MACDSignal'], data['MACDHist'] = compute_macd(data['price'])
data['RSI'] = compute_rsi(data['price'])
data['OBV'] = compute_obv(data['price'], data['size'])
data.dropna(inplace=True)

# Normalize features
scaler = MinMaxScaler()
features = ['price', 'MACD', 'MACDSignal', 'MACDHist', 'RSI', 'OBV']
data[features] = scaler.fit_transform(data[features])

# Prepare data for testing
def create_sequences(data, features, time_steps=10):
    X = []
    for i in range(len(data) - time_steps):
        X.append(data[features].iloc[i:(i + time_steps)].values)
    return np.array(X)

X_test = create_sequences(data, features)

# Load the model and make predictions
model = keras.models.load_model('transformerBlotter_Kfold3.keras')
predictions = model.predict(X_test)

# Determine trade actions
def determine_actions(predictions, threshold=0.02):
    actions = []
    for i in range(1, len(predictions)):
        if predictions[i] > predictions[i-1] * (1 + threshold):
            actions.append('Buy')
        elif predictions[i] < predictions[i-1] * (1 - threshold):
            actions.append('Sell')
        else:
            actions.append('Hold')
    return actions

actions = determine_actions(predictions.flatten())

# Creating a blotter to record transactions
blotter = pd.DataFrame({
    'Time Step': range(len(actions)),
    'Action': actions,
    'Predicted Price': predictions.flatten()[1:],  # Adjusted to match actions length
    'Actual Price': data['price'].iloc[-len(actions):].values
})

# Visualization of predictions and actual prices with trade signals
plt.figure(figsize=(12, 6))
plt.plot(data['price'].iloc[-len(predictions):], 'b', label='Actual Prices')
plt.plot(predictions.flatten(), 'r--', label='Predicted Prices', alpha=0.7)
for i, action in enumerate(actions):
    if action == 'Buy':
        plt.plot(len(predictions) - len(actions) + i, data['price'].iloc[-len(actions) + i], 'g^')
    elif action == 'Sell':
        plt.plot(len(predictions) - len(actions) + i, data['price'].iloc[-len(actions) + i], 'rv')
plt.title('Actual vs Predicted Prices with Trade Signals')
plt.legend()
plt.show()

# Print the trading blotter
print(blotter)

# # Optionally, plot the blotter as a table in a plot for better visual integration
# fig, ax = plt.subplots(figsize=(12, 8))
# ax.axis('tight')
# ax.axis('off')
# the_table = ax.table(cellText=blotter.values, colLabels=blotter.columns, loc='center')
# plt.show()

# Quantitative evaluation metrics
mae = mean_absolute_error(data['price'].iloc[-len(predictions):], predictions)
mse = mean_squared_error(data['price'].iloc[-len(predictions):], predictions)
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)

# Error distribution
errors = data['price'].iloc[-len(predictions):] - predictions.flatten()
plt.figure(figsize=(10, 6))
plt.hist(errors, bins=50, alpha=0.7, color='blue')
plt.title('Distribution of Prediction Errors')
plt.xlabel('Prediction Error')
plt.ylabel('Frequency')
plt.show()

