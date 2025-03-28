import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

# Generate dates for 5 years of historical data plus 6 months of future dates
historical_dates = pd.date_range(start='2019-01-01', end='2023-12-31', freq='M')
future_dates = pd.date_range(start='2024-01-01', end='2024-06-30', freq='M')
all_dates = historical_dates.union(future_dates)
n_historical = len(historical_dates)
n_future = len(future_dates)

# Create the base dataframe with dates
df = pd.DataFrame({'Date': all_dates})

# Generate target variables with seasonal pattern and trend for historical data
seasonal = 10 * np.sin(2 * np.pi * np.arange(n_historical) / 12)
trend = np.linspace(0, 25, n_historical)
noise = np.random.normal(0, 5, n_historical)

# Standard lumber has base pattern
df.loc[:n_historical-1, 'Lumber'] = seasonal + trend + noise

# Intermodal has higher base value, stronger seasonality, and steeper trend
df.loc[:n_historical-1, 'Intermodal'] = (
    seasonal * 1.5 +  # Stronger seasonality
    trend * 1.3 +    # Steeper trend
    noise +          # Same noise pattern
    50              # Higher base value
)

# Generate 6 indicator variables (covariates) for historical data
for target, prefix in [('Lumber', 'Standard'), ('Intermodal', 'Fancy')]:
    df.loc[:n_historical-1, f'{prefix}_Indicator_1'] = df[target][:n_historical] * 0.7 + np.random.normal(0, 3, n_historical)
    df.loc[:n_historical-1, f'{prefix}_Indicator_2'] = df[target][:n_historical] * 0.5 + seasonal + np.random.normal(0, 2, n_historical)
    df.loc[:n_historical-1, f'{prefix}_Indicator_3'] = trend * 1.2 + np.random.normal(0, 4, n_historical)
    df.loc[:n_historical-1, f'{prefix}_Indicator_4'] = seasonal * 1.5 + np.random.normal(0, 3, n_historical)
    df.loc[:n_historical-1, f'{prefix}_Indicator_5'] = df[target][:n_historical].shift(1) + np.random.normal(0, 2, n_historical)
    df.loc[:n_historical-1, f'{prefix}_Indicator_6'] = df[target][:n_historical].rolling(3).mean() + np.random.normal(0, 2, n_historical)

# Generate forecasted values for the last 12 months of historical data and future dates
forecast_start_idx = n_historical - 12
forecast_end_idx = len(df)
n_forecast = forecast_end_idx - forecast_start_idx

# Generate forecasts for both types
for target, forecast_col in [('Lumber', 'Lumber_Forecast'), 
                           ('Intermodal', 'Intermodal_Forecast')]:
    df[forecast_col] = np.nan
    last_true_target = df[target].iloc[forecast_start_idx-1]
    
    # Adjust trend multiplier for intermodal
    trend_multiplier = 1.3 if target == 'Intermodal' else 1.0
    seasonal_multiplier = 1.5 if target == 'Intermodal' else 1.0
    
    forecast_trend = np.linspace(last_true_target, last_true_target + (10 * trend_multiplier), n_forecast)
    forecast_seasonal = 10 * seasonal_multiplier * np.sin(2 * np.pi * np.arange(n_forecast) / 12)
    forecast_noise = np.random.normal(0, 2, n_forecast)
    df.loc[forecast_start_idx:forecast_end_idx-1, forecast_col] = forecast_trend + forecast_seasonal + forecast_noise

    # Generate attribution scores for the forecasted period
    prefix = 'Standard' if target == 'Lumber' else 'Fancy'
    for i in range(1, 7):
        col_name = f'{prefix}_Attribution_Indicator_{i}'
        df[col_name] = np.nan
        # Generate random attribution scores that sum to 1 for each row
        attributions = np.random.dirichlet(alpha=[1]*6, size=n_forecast)
        df.loc[forecast_start_idx:forecast_end_idx-1, col_name] = attributions[:, i-1]

# Clean up the data
df = df.round(3)

# Display the first few rows and last few rows
print("\nFirst few rows:")
print(df.head())
print("\nLast few rows:")
print(df.tail())

# Save to CSV
df.to_csv('forecasting_dataset.csv', index=False)