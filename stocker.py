import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from datetime import timedelta

# ==========================================
# 1. DATA COLLECTION
# ==========================================
def get_stock_data(ticker):
    print(f"\n[INFO] Fetching 5 years of data for {ticker}...")
    # Download data from Yahoo Finance
    data = yf.download(ticker, period="5y", interval="1d")
    
    # Check if data was found
    if data.empty:
        print(f"[ERROR] No data found for ticker '{ticker}'. Please check the symbol.")
        return None
    
    # Keep only the 'Close' column and drop NaN values
    df = data[['Close']].copy()
    df.dropna(inplace=True)
    return df

# ==========================================
# 2. FEATURE ENGINEERING
# ==========================================
def prepare_data(df, forecast_days=30):
    # Create the target variable (Prediction) shifted by 'forecast_days'
    # This aligns today's price with the price 30 days in the future
    df['Prediction'] = df['Close'].shift(-forecast_days)
    
    # X (Independent Variable) = Close Price (Convert to numpy array)
    # y (Target Variable) = Prediction
    
    # We remove the last 30 rows for training because they have no 'Prediction' (NaN)
    # However, we KEEP them for the final future forecast
    X = np.array(df.drop(['Prediction'], axis=1))
    
    # Create X_forecast: This contains the last 30 days of Close prices
    # We will use this to predict the ACTUAL future after training
    X_forecast = X[-forecast_days:] 
    
    # Remove the last 30 rows from X and y for training purposes
    X = X[:-forecast_days]
    y = np.array(df['Prediction'])[:-forecast_days]
    
    return X, y, X_forecast, df

# ==========================================
# 3. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # --- Input Ticker ---
    ticker = input("Enter Stock Ticker (e.g., AAPL, NVDA, RELIANCE.NS): ").upper()
    
    df = get_stock_data(ticker)
    
    if df is not None:
        # --- Prepare Data ---
        forecast_days = 30
        X, y, X_forecast, original_df = prepare_data(df, forecast_days)
        
        # --- Train/Test Split (80% Train, 20% Test) ---
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # --- Model Building (Linear Regression) ---
        print("[INFO] Training Linear Regression Model...")
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)
        
        # --- Evaluation ---
        # Test model accuracy on data it hasn't seen
        score = lr_model.score(X_test, y_test)
        predictions_test = lr_model.predict(X_test)
        
        mae = mean_absolute_error(y_test, predictions_test)
        r2 = r2_score(y_test, predictions_test)
        
        print(f"\n=== Model Evaluation ===")
        print(f"Model Confidence (R² Score): {r2:.2%}")
        print(f"Mean Absolute Error (MAE): {mae:.2f} (Avg price deviation)")
        
        # --- Forecasting the Future ---
        # Predict the next 30 days using the last 30 days of known data
        forecast_prediction = lr_model.predict(X_forecast)
        
        # --- Visualization ---
        print("\n[INFO] Generating Visualization...")
        
        plt.figure(figsize=(14, 7))
        plt.title(f"{ticker} - Stock Price Prediction (Linear Regression)", fontsize=16)
        
        # Plot Historical Close Prices
        plt.plot(original_df.index, original_df['Close'], label='Historical Close Price', color='blue', linewidth=1.5)
        
        # Determine Future Dates
        last_date = original_df.index[-1]
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=30)
        
        # Plot Predicted Future Prices
        plt.plot(future_dates, forecast_prediction, label='Next 30 Days Prediction', linestyle='--', color='red', linewidth=2)
        
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
        # --- Print Forecast Data ---
        print(f"\n=== Next 30 Days Forecast for {ticker} ===")
        print(f"{'Date':<15} | {'Predicted Price':<15}")
        print("-" * 35)
        for date, price in zip(future_dates, forecast_prediction):
            print(f"{date.strftime('%Y-%m-%d'):<15} | {price:.2f}")