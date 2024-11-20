#################################### Final Code ##################################

import mysql.connector
import numpy as np
import pandas as pd
import logging
from geopy.geocoders import Nominatim
import requests
from datetime import date, timedelta
import subprocess
import sys
from sqlalchemy import create_engine

python_executable = sys.executable


# Configure logging
logging.basicConfig(
    filename='LOGGING_INFO.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class WeatherDataHandler:
    def __init__(self, host="localhost", user="root", password="admin", database="weather_db"):
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.connection = None
        logging.info("Initialized WeatherDataHandler")

    def create_initial_connection(self):
        """Create MySQL connection without specifying a database."""
        self.connection = mysql.connector.connect(
            host=self.host,
            user=self.user,
            password=self.password
        )
        return self.connection

    def create_connection(self):
        """Create MySQL connection with the database."""
        self.connection = mysql.connector.connect(
            host=self.host,
            user=self.user,
            password=self.password,
            database=self.database
        )
        return self.connection

    def create_database(self):
        """Create the database if it doesn't exist."""
        with self.create_initial_connection() as connection:
            cursor = connection.cursor()
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS {self.database}")
            logging.info(f"Database '{self.database}' created or already exists.")
            connection.commit()

    def create_table(self):
        """Create the weather table if it doesn't exist."""
        create_table_query = """
        CREATE TABLE IF NOT EXISTS WeatherData (
            id INT AUTO_INCREMENT PRIMARY KEY,
            city_name VARCHAR(100),
            date DATE,
            max_temperature FLOAT,
            min_temperature FLOAT,
            max_humidity INT,
            max_wind_speed FLOAT,
            UNIQUE KEY unique_weather (city_name, date)  -- Add a unique constraint
        );
        """
        with self.create_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(create_table_query)
            logging.info("WeatherData table created or already exists.")
            conn.commit()

    def city_exists(self, city_name):
        """Check if the city exists in the database."""
        with self.create_connection() as conn:
            cursor = conn.cursor()
            query = "SELECT COUNT(*) FROM WeatherData WHERE city_name = %s"
            cursor.execute(query, (city_name,))
            result = cursor.fetchone()
            return result[0] > 0  # Return True if city exists

    def insert_dataframe_to_mysql(self, df):
        """Insert the DataFrame into MySQL with NaN handling and count total inserted/updated rows."""
        with self.create_connection() as conn:
            cursor = conn.cursor()
            df = df.replace({np.nan: None})  # Replace NaN with None to handle missing data

            total_inserted_count = 0  # Initialize a counter for total rows inserted/updated

            for _, row in df.iterrows():
                insert_query = """
                INSERT INTO WeatherData (city_name, date, max_temperature, min_temperature, max_humidity, max_wind_speed)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    max_temperature = VALUES(max_temperature),
                    min_temperature = VALUES(min_temperature),
                    max_humidity = VALUES(max_humidity),
                    max_wind_speed = VALUES(max_wind_speed);
                """
                try:
                    cursor.execute(insert_query, (
                        row['City'], 
                        row['Date'], 
                        row['Max Temperature'], 
                        row['Min Temperature'], 
                        row['Max Humidity'], 
                        row['Max Wind Speed']
                    ))
                    inserted_count = cursor.rowcount                    
                    # Add the affected row count to the total count
                    total_inserted_count += inserted_count

                except mysql.connector.Error as err:
                    logging.error(f"Error inserting data into MySQL: {err}")

            conn.commit()  # Commit all changes after the loop

            # Return the final total count
            return total_inserted_count

    def load_current_weather(self,city_name):
        """Load current weather data and insert it into MySQL."""
        # Instead of directly instantiating WeatherDataFetcher, pass the city_name here
        #city_name = get_city_name()
        #city_name = input("Enter the city name: ")  # Take city name from user input or another source
        weather_fetcher = WeatherDataFetcher(city_name)  # Instantiate WeatherDataFetcher with city_name
        
        # Now, proceed with fetching current weather data as before
        #current_weather_data = weather_fetcher.get_current_weather(weather_fetcher.get_historical_weather())
        current_weather_data = weather_fetcher.get_current_weather(weather_fetcher.get_historical_weather())
        if current_weather_data:
            # Creating DataFrame for current weather data
            current_df = pd.DataFrame({
                "Date": [current_weather_data['Date']],  # Adjust the key based on your API response
                "City": [weather_fetcher.city_name],
                "Max Temperature": [current_weather_data['Current Temperature']],  
                "Min Temperature": [current_weather_data['Min Temperature']], 
                "Max Humidity": [current_weather_data['Max Humidity']],  
                "Max Wind Speed": [current_weather_data['Max Wind Speed']]  
            })

            self.create_connection()  # Connect to the database
            self.create_table()       # Create the table if it doesn't exist

            # Initialize the total_inserted_count variable
            total_inserted_count = 0

            # Check if the city already exists in the database
            if not self.city_exists(weather_fetcher.city_name):
                historical_data = weather_fetcher.get_historical_weather()
                if historical_data is not None:
                    historical_df = pd.DataFrame({
                        "Date": historical_data['daily']['time'],
                        "City": weather_fetcher.city_name,
                        "Max Temperature": historical_data['daily']['temperature_2m_max'], 
                        "Min Temperature": historical_data['daily']['temperature_2m_min'], 
                        "Max Humidity": historical_data['daily']['relative_humidity_2m_max'], 
                        "Max Wind Speed": historical_data['daily']['windspeed_10m_max'] 
                    })
                    
                    # Insert historical data into MySQL and get the count
                    total_inserted_count += self.insert_dataframe_to_mysql(historical_df)
                    logging.info(f"Historical weather data for {weather_fetcher.city_name} has been inserted into the database.")
                else:
                    logging.info(f"No historical data available for {weather_fetcher.city_name}.")
            else:
                logging.info(f"City '{weather_fetcher.city_name}' already exists in the database.")
                print(f"City {weather_fetcher.city_name} already exists in the database. Skipping historical data insertion.")
            
            # Insert current weather data into MySQL and add to total_inserted_count
            total_inserted_count += self.insert_dataframe_to_mysql(current_df)  
            logging.info(f"Current weather data for {weather_fetcher.city_name} has been inserted into the database.")
            self.connection.close()    # Close the connection

            # Based on total_inserted_count, decide which script(s) to run
            if total_inserted_count > 1:
                subprocess.run([python_executable, r"model_traning.py"], check=True)
                logging.info("Model training completed through subprocess.")
                
                subprocess.run([python_executable, r"model.py"], check=True)
                logging.info("Forecasting completed through subprocess.")
                
            elif total_inserted_count == 1:
                subprocess.run([python_executable, r"model.py"], check=True)
                logging.info("Forecasting completed through subprocess.")
            else:
                logging.info("No data inserted. Skipping model execution.")
        else:
            logging.info("Failed to fetch current weather data.")
   
#r"D:\practics_CICD Pipleline\project\weather\Scripts\python.exe"
class WeatherDataFetcher:
    
    def __init__(self, city_name):
        self.city_name = city_name  # Get city name from the passed argument
        self.latitude, self.longitude = self.get_coordinates_for_city()
        logging.info(f"Initialized WeatherDataFetcher for {self.city_name}")

    def get_coordinates_for_city(self):
        geolocator = Nominatim(user_agent="weather_app")
        location = geolocator.geocode(self.city_name)

        if location:
            logging.info(f"Coordinates for {self.city_name} found: {location.latitude}, {location.longitude}")
            return location.latitude, location.longitude
        else:
            logging.info(f"City '{self.city_name}' not found.")
            return None, None

    def clean_weather_data(self, weather_data):
        keys_to_remove = [
            "generationtime_ms", "utc_offset_seconds", "timezone",
            "timezone_abbreviation", "elevation", "daily_units"
        ]

        for key in keys_to_remove:
            if key in weather_data:
                del weather_data[key]
        return weather_data

    def get_current_weather(self, weather_data):
        daily_weather = weather_data['daily']
        
        dates = daily_weather['time']
        min_temperatures = daily_weather['temperature_2m_min']
        max_humidity = daily_weather['relative_humidity_2m_max']
        max_wind_speed = daily_weather['windspeed_10m_max']

        today = str(date.today())
        
        if today in dates:
            index = dates.index(today)

            current_weather_data = self.get_current_temperature()
            
            current_weather = {
                "Date": dates[index],
                "City": self.city_name,
                "Current Temperature": current_weather_data['temperature'],  # Adjust based on actual structure
                "Min Temperature": min_temperatures[index],  
                "Max Humidity": max_humidity[index],          
                "Max Wind Speed": max_wind_speed[index]       
            }
            
            return current_weather
        else:
            print("Today's weather data not found.")
            return None

    def get_current_temperature(self):
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            'latitude': self.latitude,
            'longitude': self.longitude,
            'current_weather': True,
            'timezone': 'auto'
        }
        
        response = requests.get(url, params=params)
        if response.status_code == 200:
            current_weather_data = response.json()
            return current_weather_data['current_weather']  # Adjust based on actual structure
        else:
            print("Failed to retrieve current weather data.")
            return None

    def get_historical_weather(self):
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            'latitude': self.latitude,
            'longitude': self.longitude,
            'daily': 'temperature_2m_max,temperature_2m_min,relative_humidity_2m_max,windspeed_10m_max',
            'timezone': 'auto',
            'start_date': (date.today() - timedelta(days=120)).strftime('%Y-%m-%d'),  # Get past week's data
            'end_date': date.today().strftime('%Y-%m-%d')
        }
        
        response = requests.get(url, params=params)
        if response.status_code == 200:
            return self.clean_weather_data(response.json())
        else:
            print("Failed to retrieve historical weather data.")
            return None

if __name__ == "__main__":
    weather_data_handler = WeatherDataHandler()
    weather_data_handler.create_database()  # Create the database
    weather_data_handler.create_table()      # Create the table
    #weather_data_handler.load_current_weather('Tonk')  # Load current weather data and insert it










##############################################################################################




# if __name__ == "__main__":
#     # Create an instance of WeatherDataHandler
#     weather_handler = WeatherDataHandler()

#     # Step 1: Create the database if it doesn't exist
#     weather_handler.create_database()  

#     # Step 2: Create the connection to the database
#     weather_handler.create_connection()  

#     # Step 3: Load current weather data (and historical data if necessary)
#     weather_handler.load_current_weather()  

#     # Step 4: Drop duplicates from the WeatherData table
#     weather_handler.drop_duplicates()  

#     # Step 5: Close the database connection
#     weather_handler.connection.close()  




##############################################################################################

