from sqlalchemy import create_engine
from flask import Flask, jsonify, render_template,request, redirect, url_for
from Data_Dump import WeatherDataHandler
import logging
from model import get_weather_forecast
import pandas as pd

from flask import Flask, render_template, request
import logging

# Assuming you have the WeatherDataHandler and other functions imported

app = Flask(__name__)

@app.route('/')
def home():
    # This route will serve as the homepage, typically a welcome message or basic info
    return "Welcome to the Weather Forecasting App!"

@app.route('/view', methods=['GET', 'POST'])
def view():
    if request.method == 'POST':
        # Retrieve city name from the form
        city_name = request.form.get('city_name', 'ahmadabad')  # Default to 'ahmadabad' if no input
        
        # Log the input for debugging
        logging.info(f"Received city name: {city_name}")
        
        # Call the load_current_weather function with the city name
        weather_handler = WeatherDataHandler()
        try:
            weather_handler.load_current_weather(city_name)
            logging.info(f"Weather data for {city_name} processed successfully.")
        except Exception as e:
            logging.error(f"Error processing weather data for {city_name}: {e}")
            return f"An error occurred: {e}", 500
        
        # Redirect to a confirmation or the same page
        final_df, df = get_weather_forecast(forecast_days=7)
        df.rename(columns={'max_temperature': 'Temperature'}, inplace=True)
        df = df.iloc[-1].to_dict()
        print(df)
        print(final_df)
        is_data_available = not final_df.empty
        return render_template(
            'index.html',
            df_data=df,
            final_table_data=final_df if is_data_available else None,
            is_data_available=is_data_available
        )
    
    # For GET requests, render the input form
    return render_template('index.html', df_data=None, final_table_data=None, is_data_available=False)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)









