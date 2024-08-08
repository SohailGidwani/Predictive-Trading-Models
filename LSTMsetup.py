import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
import matplotlib.pyplot as plt

def create_enhanced_model(input_shape):
    """
    Create a LSTM model with specified input shape.
    
    Parameters:
        input_shape (tuple): The shape of the input data, e.g., (n_steps, n_features).
    
    Returns:
        model (Sequential): A compiled LSTM model ready for training.
    """
    model = Sequential([
        LSTM(100, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(100, return_sequences=True),
        Dropout(0.3),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def create_sequences(data, n_steps):
    """
    Converts a series of data into sequences used for training the LSTM model.
    
    Parameters:
        data (np.array): The input data.
        n_steps (int): The number of time steps per sequence.
    
    Returns:
        X (np.array): The input features for each sequence.
        y (np.array): The output (target) of each sequence.
    """
    X, y = [], []
    for i in range(n_steps, len(data)):
        X.append(data[i-n_steps:i])
        y.append(data[i])
    return np.array(X), np.array(y)

# Load the preprocessed data
data = pd.read_csv('preprocessed_data_complete.csv')

# Calculate indices for splits
total_entries = len(data)
train_end = int(total_entries * 0.70)
validation_end = int(total_entries * 0.85)

# Split the data
train_data = data[:train_end]
validation_data = data[train_end:validation_end]
test_data = data[validation_end:]

# Print sizes of splits
print("Training set size:", len(train_data))
print("Validation set size:", len(validation_data))
print("Test set size:", len(test_data))

# Ensure only numeric columns are used for features
numeric_features = data.select_dtypes(include=[np.number]).columns.tolist()
target = 'price' if 'price' in numeric_features else numeric_features[-1]
features = numeric_features
features.remove(target)


scaler = MinMaxScaler()
data[features] = scaler.fit_transform(data[features])

# Create sequences
n_steps = 10  # Number of timesteps per sequence
X, y = create_sequences(data[features].values, n_steps)
y = data[target].values[n_steps:]  # Adjust target indexing based on sequence creation

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Loads the model that we just made
model = load_model('trained_lstm_model.h5', custom_objects={'mse': MeanSquaredError()})

print("Model loaded successfully!")

# Predict and plot results
y_pred = model.predict(X_test)
plt.figure(figsize=(10, 5))
plt.plot(y_test, label='Actual')
plt.plot(y_pred, label='Predicted', alpha=0.7)
plt.title('Actual vs Predicted Prices')
plt.xlabel('Time Steps')
plt.ylabel('Price')
plt.legend()
plt.show()
