# StockWise-Predictor

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg?style=for-the-badge)
![Made By](https://img.shields.io/badge/Made%20with%20%E2%9D%A4%EF%B8%8F%20by-Shreyaan-red?style=for-the-badge)

**StockWise-Predictor** is a Python-based machine learning prototype designed to analyze historical stock market data and forecast future price trends. Using **Simple Linear Regression**, it predicts stock closing prices for the next 30 days based on 5 years of historical performance.

---

## 🚀 Key Features

* **Automated Data Retrieval:** Fetches real-time 5-year historical data using `yfinance`.
* **Smart Forecasting:** Shifts time-series data to predict prices exactly 30 days into the future.
* **Visual Analytics:** Generates interactive charts comparing historical trends with future predictions.
* **Performance Metrics:** Calculates **R² Score** (Model Confidence) and **MAE** (Mean Absolute Error) to ensure accuracy.
* **Global Support:** Works with any ticker symbol (e.g., `AAPL` for Apple, `RELIANCE.NS` for Indian Markets).

---

## 🛠️ Technical Stack

![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)

---

## 📋 Installation & Usage

pip install pandas numpy yfinance scikit-learn matplotlib

python stocker.py

Enter a Ticker

When prompted, enter a valid stock symbol.

USA: AAPL, TSLA, GOOG

India: RELIANCE.NS, TCS.NS, INFY.NS
