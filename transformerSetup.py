import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Load the dataset
data = pd.read_csv('xnas-itch-20230703.tbbo.csv')
data['price'] = data['price'] / 1e9  # Normalize price to a more manageable scale

# Define functions to calculate technical indicators
def compute_macd(price, slow=26, fast=12, signal=9):
    """
    Calculate the Moving Average Convergence Divergence (MACD) indicator.
    
    Parameters:
        price (pd.Series): The price time series.
        slow (int): The period for the slow exponential moving average.
        fast (int): The period for the fast exponential moving average.
        signal (int): The period for the signal line.
    
    Returns:
        tuple: Tuple containing the MACD line, signal line, and MACD histogram.
    """
    ema_fast = price.ewm(span=fast, adjust=False).mean()
    ema_slow = price.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line, macd - signal_line

def compute_rsi(price, window=14):
    """
    Calculate the Relative Strength Index (RSI).
    
    Parameters:
        price (pd.Series): The price time series.
        window (int): The moving average window size.
    
    Returns:
        pd.Series: The RSI values.
    """
    delta = price.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_obv(price, volume):
    """
    Calculate the On-Balance Volume (OBV).
    
    Parameters:
        price (pd.Series): The price time series.
        volume (pd.Series): The trading volume time series.
    
    Returns:
        pd.Series: The OBV values.
    """
    sign = np.sign(price.diff())
    return (sign * volume).fillna(0).cumsum()

# Calculate technical indicators
data['MACD'], data['Signal'], data['MACD_Hist'] = compute_macd(data['price'])
data['RSI'] = compute_rsi(data['price'])
data['OBV'] = compute_obv(data['price'], data['size'])

# Drop NA values created by indicators
data.dropna(inplace=True)

# Normalize features
scaler = MinMaxScaler()
features = ['price', 'MACD', 'Signal', 'MACD_Hist', 'RSI', 'OBV']
data[features] = scaler.fit_transform(data[features])

# Prepare data for training
def create_sequences(X, y, time_steps=10):
    """
    Create sequences from dataset for training time series models.
    
    Parameters:
        X (np.array): Input features.
        y (np.array): Target variable.
        time_steps (int): Number of time steps per sequence.
    
    Returns:
        tuple: Tuple containing input sequences and target variables.
    """
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

# Shift the target price for future prediction
future_period = 10
data['future_price'] = data['price'].shift(-future_period)
data.dropna(inplace=True)

X, y = create_sequences(data[features].values, data['future_price'].values)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the model
def build_model(input_shape):
    """
    Build a Transformer-based neural network model.
    
    Parameters:
        input_shape (tuple): Shape of the input data.
    
    Returns:
        model (keras.Model): Compiled neural network model.
    """
    inputs = keras.Input(shape=input_shape)
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    for _ in range(3):  # Add complexity with multiple layers
        x = layers.MultiHeadAttention(num_heads=4, key_dim=4)(x, x)
        x = layers.Dropout(0.2)(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        x = layers.Dense(128, activation='relu')(x)
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(1)(x)
    model = keras.Model(inputs, outputs)
    return model

model = build_model((X_train.shape[1], X_train.shape[2]))
model.compile(optimizer='adam', loss='mean_squared_error')

# Define callbacks for training
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
model_checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True)

# Fit the model
history = model.fit(X_train, y_train, epochs=50, batch_size=64, validation_split=0.1, callbacks=[early_stopping, reduce_lr, model_checkpoint])

# Load the best model
model = keras.models.load_model('best_model.keras')

# Predictions for testing
y_pred = model.predict(X_test)

# Visualization of predictions vs actual
plt.figure(figsize=(10, 6))
plt.plot(y_test, label='Actual Price')
plt.plot(y_pred, label='Predicted Price', alpha=0.7)
plt.title('Actual vs Predicted Prices')
plt.legend()
plt.show()

# Error Distribution
errors = y_test - y_pred.flatten()
plt.figure(figsize=(10, 6))
plt.hist(errors, bins=50, alpha=0.7, color='blue')
plt.title('Distribution of Prediction Errors')
plt.xlabel('Prediction Error')
plt.ylabel('Frequency')
plt.show()

# Learning Curves
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()