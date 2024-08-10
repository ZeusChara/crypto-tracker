import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm
import matplotlib.dates as mdates

# Title of the app
st.title("Cryptocurrency Price Prediction Using ARIMA")

# Allow the user to select which cryptocurrency to analyze
crypto_options = ['Bitcoin', 'TeraWulf']
crypto_choice = st.selectbox('Choose a cryptocurrency', crypto_options)

# Upload the appropriate CSV file
uploaded_file = st.file_uploader(f"Choose a {crypto_choice} CSV file", type="csv")

if uploaded_file is not None:
    # Read the CSV file into a DataFrame
    df = pd.read_csv(uploaded_file)

    # Ensure the 'Date' column is in datetime format
    df['Date'] = pd.to_datetime(df['Date'])

    # Filter the last two years of data
    df_last_2_years = df[df['Date'] >= df['Date'].max() - pd.DateOffset(years=2)]

    # Set the Date column as the index and explicitly set the frequency
    df_last_2_years.set_index('Date', inplace=True)
    df_last_2_years = df_last_2_years.asfreq('D')

    # Use auto_arima to find the best ARIMA model
    auto_arima_model = pm.auto_arima(df_last_2_years['Close'], seasonal=False, trace=True)

    # Predict future values up to 2030
    future_periods = (2030 - df_last_2_years.index[-1].year) * 365  # Days from end of last data to 2030
    future_index = pd.date_range(df_last_2_years.index[-1] + pd.Timedelta(days=1), periods=future_periods, freq='D')

    # Generate future predictions
    future_predictions = auto_arima_model.predict(n_periods=future_periods)

    # Create a DataFrame for future predictions with a proper datetime index
    future_df = pd.DataFrame({'ARIMA': future_predictions}, index=future_index)

    # Combine historical and future data
    combined_df = pd.concat([df_last_2_years, future_df])

    # Plot historical data and predictions
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(df_last_2_years.index, df_last_2_years['Close'], label=f'{crypto_choice} Close Price (Last 2 Years)', color='blue')
    ax.plot(future_df.index, future_df['ARIMA'], label=f'Future ARIMA Prediction (to 2030)', color='red', linestyle='--')

    # Set the x-axis limits to cover the entire forecast period
    ax.set_xlim([df_last_2_years.index.min(), future_index.max()])

    # Set major and minor ticks for the x-axis
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_minor_formatter(mdates.DateFormatter('%m'))

    plt.xticks(rotation=45)
    ax.set_title(f'{crypto_choice} Price Prediction Using Last 2 Years of Data (2020 to 2030)')
    ax.legend()

    st.pyplot(fig)

    # Display the model summary
    st.subheader(f"{crypto_choice} ARIMA Model Summary")
    st.text(auto_arima_model.summary())
