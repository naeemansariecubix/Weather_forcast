import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from prophet import Prophet
import pickle
import warnings

warnings.filterwarnings("ignore")

# MySQL connection parameters
user = 'root'
password = 'admin'
MYSQL_HOST = 'localhost'
database = 'weather_db'

# Create an engine for the MySQL connection
engine = create_engine(f'mysql+mysqlconnector://{user}:{password}@{MYSQL_HOST}/{database}')

# Load data from the database
query = """
    SELECT city_name, date, max_temperature, min_temperature, max_humidity, max_wind_speed
    FROM WeatherData
"""
df = pd.read_sql(query, con=engine)

# Data cleaning and processing
df.dropna(inplace=True)
df['max_temperature'] = pd.to_numeric(df['max_temperature'], errors='coerce')
df['min_temperature'] = pd.to_numeric(df['min_temperature'], errors='coerce')
df['max_humidity'] = pd.to_numeric(df['max_humidity'], errors='coerce')
df['max_wind_speed'] = pd.to_numeric(df['max_wind_speed'], errors='coerce')
df['date'] = pd.to_datetime(df['date'])
# Drop duplicate records based on the 'date' column
df.drop_duplicates(subset=['date'], inplace=True)

# Proceed with the rest of your code for sorting and model training
# Sort values
df.sort_values(['date'], inplace=True)

df['date'] = pd.to_datetime(df['date'])

# List of variables to forecast
variables = ['max_temperature', 'min_temperature', 'max_humidity', 'max_wind_speed']

for var in variables:
    # Prepare data for the current variable
    var_df = df[['date', var]].rename(columns={'date': 'ds', var: 'y'})
    
    # Initialize and train the model
    model = Prophet()
    model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    model.fit(var_df)
    
    # Save the model
    with open(f"{var}_model.pkl", 'wb') as file:
        pickle.dump(model, file)


###########################################################################################

