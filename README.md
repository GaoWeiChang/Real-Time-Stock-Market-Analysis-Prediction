# Real Time Stock Market Analysis & Prediction

This project implements a stock market analysis and prediction system using Long Short-Term Memory (LSTM) neural networks. The system analyzes historical stock data from major tech companies (Apple, Google, Microsoft, and Amazon) and predicts future stock prices.

## Data Source
This project uses Yahoo Finance as the primary data source through the `yfinance` library. This library allow us to fetched real-time historical stock data  from Yahoo Finance API. <br />
```
!pip install yfinance
``` 

## Features
- ### Analyze the stock
  - Moving Average calculations (10, 20, and 50 days)
  - Daily return analysis

- ### Predicting the stock
  - Using the LSTM neural networks to predict the stock price

## Project Structure
  ### 1. Data Collection
  - Fetches historical stock data using yfinance
```
import yfinance as yf

# Fetch stock data
stock_data = yf.download('AAPL', start='2023-01-01', end='2024-01-01')
```

  ### 2. Exploratory Data Analysis (EDA)
  - Statistical analysis of stock data
  - Visualization of closing prices
  - Volume analysis
  - Moving averages calculation
  - Daily returns calculation

  ### 3. Stock Price Prediction
  - LSTM model implementation
```
# Build model
model = tf.keras.Sequential([
    LSTM(50, input_shape=(sequence_length, 1), kernel_regularizer=l2(0.01)),
    Dense(25),
    Dense(1)
])

# Train model
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
```

## Model Architecture for Stock Prediction
The LSTM model architecture includes:
  - Input layer with 50 LSTM units
  - L2 regularization for preventing overfitting
  - Dense layer with 25 units
  - Output layer with 1 unit
  - Adam optimizer
  - Mean Squared Error loss function
```
# build NN model
model = tf.keras.Sequential([
    LSTM(50, input_shape=(X_train.shape[1], 1), kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    Dense(25),
    Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()
```

## Results
  - Tracking of stock price trends and predict the future price movements
  - Root Mean Square Error (RMSE) evaluation
  - Visual comparison of predicted vs actual price
