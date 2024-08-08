import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler

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

def predict(file):
    model = keras.models.load_model('transformerHyperP-adjusted.keras')
    data = pd.read_csv(file)
    data['price'] = data['price'] / 1e9

    data['MACD'], data['MACDSignal'], data['MACDHist'] = compute_macd(data['price'])
    data['RSI'] = compute_rsi(data['price'])
    data['OBV'] = compute_obv(data['price'], data['size'])
    data.dropna(inplace=True)

    features = ['price', 'MACD', 'MACDSignal', 'MACDHist', 'RSI', 'OBV']
    scaler = MinMaxScaler()
    data[features] = scaler.fit_transform(data[features])

    X = []
    time_steps = 10
    for i in range(len(data) - time_steps):
        X.append(data[features].iloc[i:(i + time_steps)].values)
    X_test = np.array(X)

    predictions = model.predict(X_test)

    actions = []
    threshold = 0.05
    for i in range(1, len(predictions)):
        if predictions[i] > predictions[i - 1] * (1 + threshold):
            actions.append(f"Buy at Step {i+10}")
        elif predictions[i] < predictions[i - 1] * (1 - threshold):
            actions.append(f"Sell at Step {i+10}")
        else:
            actions.append(f"Hold at Step {i+10}")

    action_table = "<table><tr><th>Time Step</th><th>Action</th></tr>"
    for action in actions:
        action_table += f"<tr><td>{action.split(' ')[-1]}</td><td>{' '.join(action.split(' ')[:-2])}</td></tr>"
    action_table += "</table>"

    plt.figure(figsize=(10, 6))
    plt.plot(data['price'].iloc[-len(predictions):], label='Actual Prices')
    plt.plot(predictions.flatten(), label='Predicted Prices', alpha=0.7)
    plt.title('Actual vs Predicted Prices')
    plt.legend()
    plt.xlabel('Time Steps')
    plt.ylabel('Normalized Price')
    plt.grid(True)
    plt.savefig('predictions.png')
    plt.close()

    return action_table, 'predictions.png'

app = gr.Interface(
    fn=predict,
    inputs=gr.File(label="Upload CSV File"),
    outputs=[gr.HTML(label="Recommended Actions"), gr.Image(label="Prediction Plot")],
    title="Stock Price Prediction",
    description="Upload your CSV file with stock prices to predict future prices and actions."
)

app.launch()
