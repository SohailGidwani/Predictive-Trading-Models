import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# Load and preprocess the data
data = pd.read_csv('xnas-itch-20230703.tbbo.csv')
data['price'] = data['price'] / 1e9  # Normalize price for numerical stability

# Define functions to calculate technical indicators
def compute_macd(price, slow=26, fast=12, signal=9):
    """
    Calculate MACD, a trend-following momentum indicator that shows the relationship between two moving averages of prices.
    Parameters:
        price (pd.Series): Price data series.
        slow (int): Periods for slow moving average.
        fast (int): Periods for fast moving average.
        signal (int): Periods for signal line.
    Returns:
        tuple: MACD line, signal line, and MACD histogram.
    """
    ema_fast = price.ewm(span=fast, adjust=False).mean()
    ema_slow = price.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line, macd - signal_line

def compute_rsi(price, window=14):
    """
    Compute RSI, which measures the speed and magnitude of recent price changes to evaluate overbought or oversold conditions.
    Parameters:
        price (pd.Series): Price data series.
        window (int): Look-back period for calculation.
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
    Calculate OBV, used to predict changes in stock price by measuring volume flow.
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
    Transform the data into sequences for model training.
    Parameters:
        X (np.array): Input features.
        y (np.array): Target variable.
        time_steps (int): Number of historical steps to use for predictions.
    Returns:
        tuple: Input and output sequences.
    """
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

data['future_price'] = data['price'].shift(-10)
data.dropna(inplace=True)

X, y = create_sequences(data[features].values, data['future_price'].values)

# K-fold Cross-Validation Setup
num_folds = 5
kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)

# Model building with adjustable parameters and L1, L2 regularization
def build_model(input_shape):
    """
    Build a Transformer-based model with regularization to prevent overfitting.
    Parameters:
        input_shape (tuple): The shape of the input data.
    Returns:
        model (keras.Model): Compiled neural network model.
    """
    inputs = keras.Input(shape=input_shape)
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    for _ in range(3):
        x = layers.MultiHeadAttention(num_heads=4, key_dim=4)(x, x)
        x = layers.Dropout(0.3)(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4))(x)
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(1)(x)
    model = keras.Model(inputs, outputs)
    return model

# Apply k-fold cross-validation to assess model robustness
fold_no = 1
for train, test in kfold.split(X, y):
    model = build_model((X.shape[1], X.shape[2]))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

    # Model training with callbacks for performance optimization
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)
    model_checkpoint = ModelCheckpoint(f'best_model_fold{fold_no}.keras', save_best_only=True)

    history = model.fit(X[train], y[train], epochs=50, batch_size=32, validation_data=(X[test], y[test]),
                        callbacks=[early_stopping, reduce_lr, model_checkpoint])

    print(f"Training fold {fold_no} completed.")
    fold_no += 1

# Load and predict with the best model from the first fold
model = keras.models.load_model('best_model_fold1.keras')
y_pred = model.predict(X[test])

# Visualization of the prediction results
plt.figure(figsize=(10, 6))
plt.plot(y[test], label='Actual Price')
plt.plot(y_pred, label='Predicted Price', alpha=0.7)
plt.title('Actual vs Predicted Prices')
plt.legend()
plt.show()

