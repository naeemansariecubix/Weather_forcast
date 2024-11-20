########################## Weather Data forcasting model ##########################


import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import pickle
from prophet import Prophet
import warnings

warnings.filterwarnings("ignore")

# Function to process data and forecast
def get_weather_forecast(forecast_days=7):
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
    df.dropna(inplace=True)

    # Sort values
    df.sort_values(['date'], inplace=True)

    # Extract user input as the last row
    user_input = df.iloc[-1]

    # Extract the last row values for relevant columns
    last_row_values = user_input[['max_temperature', 'min_temperature', 'max_humidity', 'max_wind_speed']].to_dict()

    # Extract city_name from user input
    city_name = user_input['city_name']

    # Dictionary to store predictions
    predictions = {}

    # For each weather variable, load the corresponding model and make predictions
    for var, value in last_row_values.items():
        # Load the saved model for the variable
        with open(f"{var}_model.pkl", 'rb') as file:
            model = pickle.load(file)
        
        # Generate future dates for forecasting
        future = model.make_future_dataframe(periods=forecast_days)
        
        # Predict the future values
        forecast = model.predict(future)
        
        # Retrieve the last `forecast_days` predictions and rename 'yhat' to the variable name
        predictions[var] = forecast[['ds', 'yhat']].tail(forecast_days).rename(columns={'yhat': var})

    # Initialize the combined DataFrame with one of the forecasts
    combined_forecast_df = predictions['max_temperature']

    # Merge the remaining forecasts on the 'ds' column
    for var, forecast in predictions.items():
        if var != 'max_temperature':
            combined_forecast_df = combined_forecast_df.merge(forecast, on='ds')

    # Add the city_name to the combined DataFrame
    combined_forecast_df['city_name'] = city_name

    # Replace negative values with NaN in only the forecasted columns, then apply forward-fill
    forecast_columns = list(last_row_values.keys())
    combined_forecast_df[forecast_columns] = combined_forecast_df[forecast_columns].applymap(lambda x: x if x >= 0 else None).ffill()

    # Rearrange the DataFrame to match the requested format
    final_df = combined_forecast_df[['ds', 'city_name', 'max_temperature', 'min_temperature', 'max_humidity', 'max_wind_speed']]

    # Rename columns to match the requested format
    final_df = final_df.rename(columns={
        'ds': 'date',
        'max_temperature': 'temperature',
        'max_humidity': 'humidity',
        'max_wind_speed': 'wind_speed'
    })

    return final_df,df

# Example usage
if __name__ == "__main__":
    final_df = get_weather_forecast(forecast_days=7)
    # print("Final Forecast DataFrame:")
    # print(final_df)
    # print("\nOriginal DataFrame:")
    # print(df)






##############################################################################################

# import pandas as pd
# import numpy as np
# from sqlalchemy import create_engine
# from statsmodels.tsa.arima.model import ARIMA
# from prophet import Prophet
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense
# import pickle

# # MySQL connection parameters
# user = 'root'
# password = 'admin'
# MYSQL_HOST = 'localhost'
# database = 'weather_db'

# # Number of days to forecast
# forecast_days = 7

# # Create an engine for the MySQL connection
# engine = create_engine(f'mysql+mysqlconnector://{user}:{password}@{MYSQL_HOST}/{database}')

# # Define your SQL query
# query = """
#     SELECT city_name, date, max_temperature, min_temperature, max_humidity, max_wind_speed
#     FROM WeatherData
# """

# try:
#     # Load data from the database
#     df = pd.read_sql(query, con=engine)
    
#     # Data cleaning and processing
#     df.dropna(inplace=True)
#     df['max_temperature'] = pd.to_numeric(df['max_temperature'], errors='coerce')
#     df['min_temperature'] = pd.to_numeric(df['min_temperature'], errors='coerce')
#     df['max_humidity'] = pd.to_numeric(df['max_humidity'], errors='coerce')
#     df['max_wind_speed'] = pd.to_numeric(df['max_wind_speed'], errors='coerce')
#     df['date'] = pd.to_datetime(df['date'])
#     df.dropna(inplace=True)

#     # Initialize dictionaries to store forecasts
#     arima_forecasts = {}
#     prophet_forecasts = {}
#     lstm_forecasts = {}

#     # Forecast functions
#     def build_arima_model(data):
#         model = ARIMA(data['max_temperature'], order=(1, 1, 1))
#         model_fit = model.fit()
#         forecast = model_fit.get_forecast(steps=forecast_days)
#         return forecast.predicted_mean.tolist()

#     def build_prophet_model(data):
#         prophet_data = data[['date', 'max_temperature']].rename(columns={'date': 'ds', 'max_temperature': 'y'})
#         model = Prophet()
#         model.fit(prophet_data)
#         future = model.make_future_dataframe(periods=forecast_days)
#         forecast = model.predict(future)
#         return forecast[['ds', 'yhat']].tail(forecast_days).values.tolist()

#     def build_lstm_model(data):
#         features = data[['max_temperature', 'min_temperature', 'max_humidity', 'max_wind_speed']].values
#         features = features.reshape((features.shape[0], features.shape[1], 1))
        
#         model = Sequential()
#         model.add(LSTM(50, activation='relu', input_shape=(features.shape[1], features.shape[2])))
#         model.add(Dense(1))
#         model.compile(optimizer='adam', loss='mse')
#         model.fit(features, data['max_temperature'], epochs=200, batch_size=32, verbose=0)
        
#         forecast_values = []
#         last_known_data = features[-1].reshape((1, features.shape[1], 1))
#         for _ in range(forecast_days):
#             predicted_temp = model.predict(last_known_data)[0, 0]
#             forecast_values.append(predicted_temp)
#             last_known_data = np.roll(last_known_data, -1, axis=1)
#             last_known_data[0, -1, 0] = predicted_temp

#         return forecast_values

#     # Perform city-based forecasting
#     for city in df['city_name'].unique():
#         city_data = df[df['city_name'] == city].sort_values(by='date')
        
#         arima_forecasts[city] = build_arima_model(city_data)
#         prophet_forecasts[city] = build_prophet_model(city_data)
#         lstm_forecasts[city] = build_lstm_model(city_data)

#     # Output forecasts for each city
#     for city in df['city_name'].unique():
#         print(f"\nForecasts for {city}:\n")
#         print(f"ARIMA Forecast for {forecast_days} days:\n{arima_forecasts[city]}")
#         print(f"Prophet Forecast for {forecast_days} days:\n{prophet_forecasts[city]}")
#         print(f"LSTM Forecast for {forecast_days} days:\n{lstm_forecasts[city]}")

#     # Save models as needed (optional)
#     with open('arima_forecasts.pkl', 'wb') as f:
#         pickle.dump(arima_forecasts, f)

#     with open('prophet_forecasts.pkl', 'wb') as f:
#         pickle.dump(prophet_forecasts, f)

#     with open('lstm_forecasts.pkl', 'wb') as f:
#         pickle.dump(lstm_forecasts, f)

#     print("City-based forecasts saved successfully!")

# except Exception as e:
#     print(f"An error occurred: {e}")
#############################################################################################





# import pandas as pd
# import numpy as np
# from sqlalchemy import create_engine
# import pickle
# from prophet import Prophet
# from flask import Flask, request, jsonify
# import warnings

# warnings.filterwarnings("ignore")
# # MySQL connection parameters
# user = 'root'
# password = 'admin'
# MYSQL_HOST = 'localhost'
# database = 'weather_db'

# # Create an engine for the MySQL connection
# engine = create_engine(f'mysql+mysqlconnector://{user}:{password}@{MYSQL_HOST}/{database}')

# # Load data from the database
# query = """
#     SELECT city_name, date, max_temperature, min_temperature, max_humidity, max_wind_speed
#     FROM WeatherData
# """
# df = pd.read_sql(query, con=engine)

# # Data cleaning and processing
# df.dropna(inplace=True)
# df['max_temperature'] = pd.to_numeric(df['max_temperature'], errors='coerce')
# df['min_temperature'] = pd.to_numeric(df['min_temperature'], errors='coerce')
# df['max_humidity'] = pd.to_numeric(df['max_humidity'], errors='coerce')
# df['max_wind_speed'] = pd.to_numeric(df['max_wind_speed'], errors='coerce')
# df['date'] = pd.to_datetime(df['date'])
# df.dropna(inplace=True)



# # Sort values
# df.sort_values(['date'], inplace=True)

# # Sample DataFrame for user-provided last known values (example df)
# user_input = df.iloc[-1]

# # Extract the last row values for relevant columns
# last_row_values = user_input[['max_temperature', 'min_temperature', 'max_humidity', 'max_wind_speed']].to_dict()

# # Extract city_name from user input
# city_name = user_input['city_name']

# # Define the number of days to forecast
# forecast_days = 7

# # Dictionary to store predictions
# predictions = {}

# # For each weather variable, load the corresponding model and make predictions
# for var, value in last_row_values.items():
#     # Load the saved model for the variable
#     with open(f"{var}_model.pkl", 'rb') as file:
#         model = pickle.load(file)
    
#     # Generate future dates for forecasting
#     future = model.make_future_dataframe(periods=forecast_days)
    
#     # Predict the future values
#     forecast = model.predict(future)
#     # Retrieve the last `forecast_days` predictions and rename 'yhat' to the variable name
#     predictions[var] = forecast[['ds', 'yhat']].tail(forecast_days).rename(columns={'yhat': var})

# # Initialize the combined DataFrame with one of the forecasts
# combined_forecast_df = predictions['max_temperature']

# # Merge the remaining forecasts on the 'ds' column
# for var, forecast in predictions.items():
#     if var != 'max_temperature':
#         combined_forecast_df = combined_forecast_df.merge(forecast, on='ds')

# # Add the city_name to the combined DataFrame
# combined_forecast_df['city_name'] = city_name

# # Replace negative values with NaN in only the forecasted columns, then apply forward-fill
# forecast_columns = list(last_row_values.keys())
# combined_forecast_df[forecast_columns] = combined_forecast_df[forecast_columns].applymap(lambda x: x if x >= 0 else None).ffill()

# # Rearrange the DataFrame to match the requested format
# final_df = combined_forecast_df[['ds', 'city_name', 'max_temperature', 'min_temperature', 'max_humidity', 'max_wind_speed']]

# # Rename columns to match the requested format
# final_df = final_df.rename(columns={
#     'ds': 'date',
#     'max_temperature': 'temperature',
#     'max_humidity': 'humidity',
#     'max_wind_speed': 'wind_speed'
# })

# # Display the final DataFrame
# print(final_df)





##############################################################################################

# import pandas as pd
# import numpy as np
# import pickle
# import os
# from statsmodels.tsa.arima.model import ARIMA
# from prophet import Prophet
# from tensorflow.keras.models import Sequential, load_model
# from tensorflow.keras.layers import LSTM, Dense

# # Directory to save models
# model_dir = 'saved_models'
# os.makedirs(model_dir, exist_ok=True)

# # Function to save model to file
# def save_model(model, model_type, column):
#     if model_type == "LSTM":
#         model.save(os.path.join(model_dir, f'LSTM_{column}.h5'))
#     else:
#         with open(os.path.join(model_dir, f'{model_type}_{column}.pkl'), 'wb') as f:
#             pickle.dump(model, f)

# # Function to load model from file if exists
# def load_model_file(model_type, column):
#     model_path = os.path.join(model_dir, f'{model_type}_{column}.pkl' if model_type != "LSTM" else f'LSTM_{column}.h5')
#     if os.path.exists(model_path):
#         if model_type == "LSTM":
#             return load_model(model_path)
#         else:
#             with open(model_path, 'rb') as f:
#                 return pickle.load(f)
#     return None

# # Functions for ARIMA, Prophet, LSTM forecasting, with saving/loading logic
# def arima_forecast(df, column):
#     model = load_model_file("ARIMA", column) or ARIMA(df[column], order=(5, 1, 0)).fit()
#     if model and not load_model_file("ARIMA", column):
#         save_model(model, "ARIMA", column)
#     return model.forecast(steps=forecast_days)

# def prophet_forecast(df, column):
#     prophet_data = df[['Date', column]].rename(columns={'Date': 'ds', column: 'y'})
#     model = load_model_file("Prophet", column) or Prophet().fit(prophet_data)
#     if model and not load_model_file("Prophet", column):
#         save_model(model, "Prophet", column)
#     future = model.make_future_dataframe(periods=forecast_days)
#     forecast = model.predict(future)
#     return forecast[['ds', 'yhat']].tail(forecast_days)['yhat']

# def lstm_forecast(df, column):
#     look_back = 10
#     X, y = [], []
#     for i in range(len(df) - look_back):
#         X.append(df[column].values[i:i+look_back])
#         y.append(df[column].values[i+look_back])
#     X, y = np.array(X), np.array(y)
#     X = X.reshape((X.shape[0], X.shape[1], 1))

#     model = load_model_file("LSTM", column)
#     if model is None:
#         model = Sequential([LSTM(50, activation='relu', input_shape=(look_back, 1)), Dense(1)])
#         model.compile(optimizer='adam', loss='mse')
#         model.fit(X, y, epochs=50, verbose=0)
#         save_model(model, "LSTM", column)

#     predictions = []
#     input_seq = X[-1]
#     for _ in range(forecast_days):
#         pred = model.predict(input_seq.reshape(1, look_back, 1), verbose=0)
#         predictions.append(pred[0, 0])
#         input_seq = np.append(input_seq[1:], pred)
    
#     return predictions

# # Add new input data to DataFrame
# def add_new_city_data(df, city_name, max_temp, min_temp, max_hum, max_wind_speed):
#     today = pd.to_datetime("today").normalize()
#     new_data = pd.DataFrame({
#         'Date': [today],
#         'city_name': [city_name],
#         'max_temp': [max_temp],
#         'min_temp': [min_temp],
#         'max_hum': [max_hum],
#         'max_wind_speed': [max_wind_speed]
#     })
#     return pd.concat([df, new_data], ignore_index=True)

# # Input for city data and new parameters
# user_city = input("Enter the city name: ").strip().lower()
# max_temp = float(input("Enter max temperature: "))
# min_temp = float(input("Enter min temperature: "))
# max_hum = float(input("Enter max humidity: "))
# max_wind_speed = float(input("Enter max wind speed: "))

# # Assuming 'data' is the existing DataFrame with historical data
# # Load data from file or generate random placeholder data
# np.random.seed(42)
# dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
# cities = ['ahmadabad', 'surat', 'thane', 'pune']
# data = pd.DataFrame({
#     'city_name': np.random.choice(cities, size=len(dates) * len(cities)),
#     'Date': np.tile(dates, len(cities)),
#     'max_temp': np.random.uniform(20, 40, size=len(dates) * len(cities)),
#     'min_temp': np.random.uniform(10, 25, size=len(dates) * len(cities)),
#     'max_hum': np.random.uniform(50, 100, size=len(dates) * len(cities)),
#     'max_wind_speed': np.random.uniform(5, 20, size=len(dates) * len(cities))
# })
# data['Date'] = pd.to_datetime(data['Date'])
# data = data.groupby(['city_name', 'Date']).agg({
#     'max_temp': 'mean',
#     'min_temp': 'mean',
#     'max_hum': 'mean',
#     'max_wind_speed': 'mean'
# }).reset_index()

# # Update data with the new city data and parameters
# data = add_new_city_data(data, user_city, max_temp, min_temp, max_hum, max_wind_speed)
# data = data[data['city_name'] == user_city]  # Filter for the specific city for forecasting

# # Input for number of forecast days
# forecast_days = int(input('Enter the number of days to forecast: '))

# # Forecast for each metric
# city_forecast = {'Date': pd.date_range(data['Date'].max() + pd.Timedelta(days=1), periods=forecast_days)}
# forecast_columns = ['max_temp', 'min_temp', 'max_hum', 'max_wind_speed']
# for col in forecast_columns:
#     try:
#         forecast = arima_forecast(data, col)
#     except:
#         print(f"ARIMA failed for {col}, switching to Prophet.")
#         try:
#             forecast = prophet_forecast(data, col)
#         except:
#             print(f"Prophet also failed for {col}, switching to LSTM.")
#             forecast = lstm_forecast(data, col)
#     city_forecast[col] = forecast

# # Convert forecast dictionary to DataFrame and display
# city_forecast_df = pd.DataFrame(city_forecast)
# city_forecast_df['city_name'] = user_city

# print(f"\nForecast DataFrame for {user_city}:")
# print(city_forecast_df)

