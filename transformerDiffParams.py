import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# Load and preprocess the data
data = pd.read_csv('xnas-itch-20230703.tbbo.csv')
data['price'] = data['price'] / 1e9  # Normalize price to a billionth for scaling reasons

# Define technical indicators
def compute_macd(price, slow=26, fast=12, signal=9):
    """
    Compute MACD (Moving Average Convergence Divergence) which is a trend-following momentum indicator.
    
    Parameters:
        price (pd.Series): Price data series.
        slow (int): Number of periods for slow moving average.
        fast (int): Number of periods for fast moving average.
        signal (int): Number of periods for the signal line.
        
    Returns:
        tuple: MACD line, signal line, MACD histogram.
    """
    ema_fast = price.ewm(span=fast, adjust=False).mean()
    ema_slow = price.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line, macd - signal_line

def compute_rsi(price, window=14):
    """
    Compute RSI (Relative Strength Index), a momentum oscillator measuring speed and change of price movements.
    
    Parameters:
        price (pd.Series): Price data series.
        window (int): Window for calculation.
        
    Returns:
        pd.Series: RSI values.
    """
    delta = price.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_obv(price, volume):
    """
    Compute OBV (On-Balance Volume), a cumulative indicator that uses volume and price to show how money may be flowing into or out of a stock.
    
    Parameters:
        price (pd.Series): Price data series.
        volume (pd.Series): Volume data series.
        
    Returns:
        pd.Series: OBV values.
    """
    sign = np.sign(price.diff())
    return (sign * volume).fillna(0).cumsum()

data['MACD'], data['Signal'], data['MACD_Hist'] = compute_macd(data['price'])
data['RSI'] = compute_rsi(data['price'])
data['OBV'] = compute_obv(data['price'], data['size'])
data.dropna(inplace=True)

# Normalize features
scaler = MinMaxScaler()
features = ['price', 'MACD', 'Signal', 'MACD_Hist', 'RSI', 'OBV']
data[features] = scaler.fit_transform(data[features])

# Prepare data for training
def create_sequences(X, y, time_steps=10):
    """
    Create sequences from dataset for model training.
    
    Parameters:
        X (np.array): Array of input features.
        y (np.array): Array of target values.
        time_steps (int): Number of steps in each sequence.
        
    Returns:
        tuple: Tuple containing sequence of inputs and sequence of outputs.
    """
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

data['future_price'] = data['price'].shift(-10)
data.dropna(inplace=True)

X, y = create_sequences(data[features].values, data['future_price'].values)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model building with adjustable parameters
def build_model(input_shape):
    """
    Build a transformer-based neural network model with customizable parameters.
    
    Parameters:
        input_shape (tuple): Shape of the input layer.
        
    Returns:
        model (keras.Model): Compiled Keras model.
    """
    inputs = keras.Input(shape=input_shape)
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    for _ in range(3):  # Enhance model complexity by adjusting layer numbers and dimensions
        x = layers.MultiHeadAttention(num_heads=4, key_dim=4)(x, x)
        x = layers.Dropout(0.3)(x) #adjusted the dropouts bit
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        x = layers.Dense(128, activation='relu')(x)
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(1)(x)
    model = keras.Model(inputs, outputs)
    return model

model = build_model((X_train.shape[1], X_train.shape[2]))
optimizer = Adam(learning_rate=0.001)  # added a GD optimizer
model.compile(optimizer=optimizer, loss='mean_squared_error')

# Callbacks for model training
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)
model_checkpoint = ModelCheckpoint('best_model_1.keras', save_best_only=True)

history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, callbacks=[early_stopping, reduce_lr, model_checkpoint])

# Load the best model
model = keras.models.load_model('best_model_1.keras')

# Predictions and evaluation
y_pred = model.predict(X_test)
plt.figure(figsize=(10, 6))
plt.plot(y_test, label='Actual Price')
plt.plot(y_pred, label='Predicted Price', alpha=0.7)
plt.title('Actual vs Predicted Prices')
plt.legend()
plt.show()

errors = y_test - y_pred.flatten()
plt.figure(figsize=(10, 6))
plt.hist(errors, bins=50, alpha=0.7, color='blue')
plt.title('Distribution of Prediction Errors')
plt.xlabel('Prediction Error')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
