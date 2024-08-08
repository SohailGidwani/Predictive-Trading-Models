import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load and preprocess the data
data = pd.read_csv('xnas-itch-20230703.tbbo.csv')
data['price'] = data['price'] / 1e9  # Normalize price for numerical stability

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

data['MACD'], data['Signal'], data['MACD_Hist'] = compute_macd(data['price'])
data['RSI'] = compute_rsi(data['price'])
data['OBV'] = compute_obv(data['price'], data['size'])
data.dropna(inplace=True)

scaler = MinMaxScaler()
features = ['price', 'MACD', 'Signal', 'MACD_Hist', 'RSI', 'OBV']
data[features] = scaler.fit_transform(data[features])

def create_sequences(X, y, time_steps=10):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

data['future_price'] = data['price'].shift(-10)
data.dropna(inplace=True)
X, y = create_sequences(data[features].values, data['future_price'].values)

num_folds = 5
kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)

blotter = pd.DataFrame(columns=['Time Step', 'Action', 'Trade Price', 'Profit/Loss'])
initial_investment = 10000
cash = initial_investment
shares_held = 0

def build_model(input_shape):
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

fold_no = 1
for train, test in kfold.split(X, y):
    model = build_model((X.shape[1], X.shape[2]))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)
    model_checkpoint = ModelCheckpoint(f'best_blotter_model_fold{fold_no}.keras', save_best_only=True)

    history = model.fit(X[train], y[train], epochs=50, batch_size=32, validation_data=(X[test], y[test]), callbacks=[early_stopping, reduce_lr, model_checkpoint])
    predictions = model.predict(X[test])

    max_index = len(data) - 1
    for i in range(len(predictions) - 11):  # Reduce range by 11 to ensure we don't go out of bounds
        index = test[i + 10] if test[i + 10] < max_index else max_index
        current_price = data['price'].iloc[index]
        if predictions[i + 1] > predictions[i] * 1.02:  # Buy
            shares_held = cash // current_price
            cash -= shares_held * current_price
            blotter = blotter._append({'Time Step': index, 'Action': 'Buy', 'Trade Price': current_price, 'Profit/Loss': 0}, ignore_index=True)
        elif predictions[i + 1] < predictions[i] * 0.98:  # Sell
            cash += shares_held * current_price
            profit_loss = (shares_held * current_price) - (shares_held * blotter.iloc[-1]['Trade Price']) if shares_held else 0
            blotter = blotter._append({'Time Step': index, 'Action': 'Sell', 'Trade Price': current_price, 'Profit/Loss': profit_loss}, ignore_index=True)
            shares_held = 0

    fold_no += 1

# Plot results and print the trading blotter
plt.figure(figsize=(12, 6))
plt.plot(data['price'], 'b', label='Actual Prices')
plt.plot(np.arange(X[test].shape[0]) + 10, predictions, 'r--', label='Predicted Prices', alpha=0.7)
for i, action in enumerate(blotter['Action']):
    if action == 'Buy':
        plt.plot(blotter['Time Step'].iloc[i], blotter['Trade Price'].iloc[i], 'g^')
    elif action == 'Sell':
        plt.plot(blotter['Time Step'].iloc[i], blotter['Trade Price'].iloc[i], 'rv')
plt.title('Actual vs Predicted Prices with Trade Signals')
plt.legend()
plt.show()

# Error distribution
errors = data['price'].iloc[-len(predictions):] - predictions.flatten()
plt.figure(figsize=(10, 6))
plt.hist(errors, bins=50, alpha=0.7, color='blue')
plt.title('Distribution of Prediction Errors')
plt.xlabel('Prediction Error')
plt.ylabel('Frequency')
plt.show()

# Quantitative evaluation metrics
mae = mean_absolute_error(data['price'].iloc[-len(predictions):], predictions)
mse = mean_squared_error(data['price'].iloc[-len(predictions):], predictions)
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)

# Print the trading blotter
print(blotter.head())

# Save the trading blotter to CSV
blotter.to_csv('trading_blotter.csv', index=False)
print("Blotter saved to CSV.")

# Output final results
final_value = cash + (shares_held * data['price'].iloc[-1])
total_profit = final_value - initial_investment
print(f"Total Profit: {total_profit}")
print(f"Final Account Value: {final_value}")

# Plot blotter table as an image
fig, ax = plt.subplots(figsize=(10, 4))
ax.axis('tight')
ax.axis('off')
tbl = ax.table(cellText=blotter.values, colLabels=blotter.columns, cellLoc='center', loc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1.2, 1.2)
plt.title('Trading Blotter')
plt.savefig('blotter_table.png')
plt.show()
