import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

# Generate dates for 5 years of monthly data
dates = pd.date_range(start='2019-01-01', end='2023-12-31', freq='M')
n_samples = len(dates)

# Create the base dataframe with dates
df = pd.DataFrame({'Date': dates})

# Generate target variable with seasonal pattern and trend
seasonal = 10 * np.sin(2 * np.pi * np.arange(n_samples) / 12)
trend = np.linspace(0, 25, n_samples)
noise = np.random.normal(0, 5, n_samples)
df['Target'] = seasonal + trend + noise

# Generate 6 indicator variables (covariates)
df['Indicator_1'] = df['Target'] * 0.7 + np.random.normal(0, 3, n_samples)
df['Indicator_2'] = df['Target'] * 0.5 + seasonal + np.random.normal(0, 2, n_samples)
df['Indicator_3'] = trend * 1.2 + np.random.normal(0, 4, n_samples)
df['Indicator_4'] = seasonal * 1.5 + np.random.normal(0, 3, n_samples)
df['Indicator_5'] = df['Target'].shift(1) + np.random.normal(0, 2, n_samples)
df['Indicator_6'] = df['Target'].rolling(3).mean() + np.random.normal(0, 2, n_samples)

# Generate forecasted values for the last 12 months
df['Target_Forecast'] = np.nan
df.loc[df.index[-12:], 'Target_Forecast'] = df['Target'].iloc[-12:] + np.random.normal(0, 2, 12)

# Generate attribution scores for the last 12 months
for i in range(1, 7):
    col_name = f'Attribution_Indicator_{i}'
    df[col_name] = np.nan
    # Generate random attribution scores that sum to 1 for each row
    attributions = np.random.dirichlet(alpha=[1]*6, size=12)
    df.loc[df.index[-12:], col_name] = attributions[:, i-1]

# Clean up the data
df = df.round(3)

# Display the first few rows and last few rows
print("\nFirst few rows:")
print(df.head())
print("\nLast few rows:")
print(df.tail())

# Save to CSV
df.to_csv('forecasting_dataset.csv', index=False)