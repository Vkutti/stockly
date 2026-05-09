<p align="center">
  <img src="stocklylogo.svg" alt="Stockly Logo" width="750">
</p>

# Stockly

Stockly is a machine learning–based web application that predicts future stock prices using historical market data. The app provides a simple interface where users can choose a stock and instantly receive a model-generated prediction.

## Features

* LSTM-based stock price forecasting
* Automated data preprocessing (scaling, sequence generation)

## How It Works
* Historical price data is collected for the selected stock.
* Data is scaled and converted into training sequences.
* A trained LSTM model predicts the next closing price.

## Tools Used

* Python, Flask
* TensorFlow, Keras
* NumPy, Pandas, Scikit-learn
* Railway
