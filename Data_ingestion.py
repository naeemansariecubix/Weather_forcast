#################################### Final Code ##################################

# Data_ingestion.py

import pandas as pd
import requests
from datetime import date, timedelta
from geopy.geocoders import Nominatim
from Data_Dump import WeatherDataHandler  # Import the WeatherDataHandler class
import logging
import datetime

class WeatherDataFetcher:
    def __init__(self):
        self.city_name = input("Enter the city name: ")  # Get city name from user input
        self.latitude, self.longitude = self.get_coordinates_for_city()
        logging.info("Initialized WeatherDataFetcher")

    # Function to get the latitude and longitude for a city
    def get_coordinates_for_city(self):
        geolocator = Nominatim(user_agent="weather_app")
        location = geolocator.geocode(self.city_name)

        if location:
            logging.info(f"Coordinates for {self.city_name} found: {location.latitude}, {location.longitude}")
            return location.latitude, location.longitude
        else:
            logging.info(f"City '{self.city_name}' not found.")
            return None, None

    # Function to fetch minute-level temperature data for the current day
    def get_minute_temperature_data(self):
        url = "https://api.open-meteo.com/v1/forecast"  # Update to a valid endpoint that supports minute-level data
        params = {
            'latitude': self.latitude,
            'longitude': self.longitude,
            'minute': 'temperature_2m',  # Update parameter to match the actual API for minute temperature data
            'timezone': 'auto'
        }

        response = requests.get(url, params=params)
        if response.status_code == 200:
            weather_data = response.json()
            minute_data = weather_data.get("minute", [])  # Adjust key based on actual API response structure

            if minute_data:
                # Extract minute-level temperature data for today
                minute_temperature_data = []
                today = date.today()

                for entry in minute_data:
                    timestamp = entry["time"]
                    datetime_obj = datetime.fromisoformat(timestamp)

                    # Filter for data points that match today's date
                    if datetime_obj.date() == today:
                        minute_temperature_data.append({
                            "Time": datetime_obj.strftime("%H:%M"),
                            "Temperature": entry.get("temperature_2m")  # Adjust key based on API
                        })

                return minute_temperature_data
            else:
                print("No minute-level temperature data available for today.")
                return None
        else:
            print(f"Failed to retrieve minute-level temperature data: {response.status_code}")
            return None

    # Function to get 6 months of historical weather data
    def get_historical_weather(self):
        url = "https://api.open-meteo.com/v1/forecast"

        # Calculate start and end dates for 3 months
        end_date = date.today()
        start_date = end_date - timedelta(days=120)

        params = {
            'latitude': self.latitude,
            'longitude': self.longitude,
            'start_date': start_date.isoformat(),
            'end_date': end_date.isoformat(),
            'daily': ['temperature_2m_max', 'temperature_2m_min', 'windspeed_10m_max', 'relative_humidity_2m_max'],
            'timezone': 'auto'
        }

        response = requests.get(url, params=params)

        if response.status_code == 200:
            weather_data = response.json()
            weather_data = self.clean_weather_data(weather_data)
            return weather_data
        else:
            print(f"Failed to retrieve historical weather data for {start_date} to {end_date}")
            return None

    # Function to get all weather data (current and historical)
    def get_weather_data(self):
        historical_data = self.get_historical_weather()
        
        if historical_data:
            historical_df = pd.DataFrame({
                "Date": historical_data['daily']['time'],
                "City": self.city_name,
                "Current Temperature": [None] * len(historical_data['daily']['time']),  # Placeholder for current temp
                "Min Temperature": historical_data['daily']['temperature_2m_min'], 
                "Max Humidity": historical_data['daily']['relative_humidity_2m_max'], 
                "Max Wind Speed": historical_data['daily']['windspeed_10m_max'] 
            })
            
            current_weather_data = self.get_current_weather(historical_data)
            
            if current_weather_data:
                current_temp = current_weather_data.get('temperature')  # Get the current temperature from the API response
                current_df = pd.DataFrame([{
                    "Date": pd.to_datetime('today').normalize(),  # Assuming current date for current weather
                    "City": self.city_name,
                    "Current Temperature": current_temp,
                    "Min Temperature": None,  # Placeholder
                    "Max Humidity": None,  # Placeholder
                    "Max Wind Speed": None   # Placeholder
                }])
                
                # Combine historical and current data
                weather_df = pd.concat([historical_df, current_df], ignore_index=True)
                weather_df = weather_df.drop_duplicates(subset=["Date", "City"], keep='last')
                
                return weather_df
            else:
                print("Failed to get current weather data.")
        else:
            print(f"Could not fetch historical weather data for {self.city_name}.")
        
        return None


# Main function to fetch and store weather data
def main():
    # Create an instance of WeatherDataHandler
    weather_handler = WeatherDataHandler()
    weather_handler.create_database()  # Create the database if it doesn't exist
    weather_handler.load_current_weather()  # Load current weather data into MySQL

if __name__ == "__main__":
    main()




####################################################################################################
 # def drop_duplicates(self):
    #     """Drop duplicate entries from the WeatherData table while keeping one instance of each."""
    #     try:
    #         with self.create_connection() as conn:
    #             cursor = conn.cursor()
    #             delete_query = """
    #             DELETE wd1 FROM WeatherData wd1
    #             INNER JOIN WeatherData wd2 
    #             WHERE 
    #                 wd1.id > wd2.id AND  -- Keep the record with the smaller id
    #                 wd1.city_name = wd2.city_name AND 
    #                 wd1.date = wd2.date;  -- Define duplicates based on city_name and date
    #             """
    #             cursor.execute(delete_query)  # Execute the query
    #             conn.commit()  # Commit the transaction
    #             logging.info("Duplicate entries have been removed from the WeatherData table.")
    #     except mysql.connector.Error as err:
    #         logging.error(f"Error removing duplicates from WeatherData table: {err}")

    #weather_data_handler.drop_duplicates()   # Drop duplicate entries
####################################################################################################

# import requests
# from geopy.geocoders import Nominatim
# from datetime import date, timedelta
# import json
# import os
# from Data_Dump import WeatherDataHandler  # Replace 'your_module' with the name of your Python file containing the WeatherDataHandler class.

# # Function to get the coordinates (latitude and longitude) for a city
# def get_coordinates_for_city(city_name):
#     geolocator = Nominatim(user_agent="weather_app")
#     location = geolocator.geocode(city_name)
    
#     if location:
#         return location.latitude, location.longitude
#     else:
#         print(f"City '{city_name}' not found.")
#         return None, None

# # Function to clean up unnecessary fields from weather data
# def clean_weather_data(weather_data):
#     keys_to_remove = [
#         "generationtime_ms", "utc_offset_seconds", "timezone", 
#         "timezone_abbreviation", "elevation", "daily_units"
#     ]
    
#     for key in keys_to_remove:
#         if key in weather_data:
#             del weather_data[key]
#     return weather_data

# # Function to get the current day's weather data
# def get_current_weather(city_name, weather_data):
#     daily_weather = weather_data['daily']
    
#     # Extract daily weather data
#     dates = daily_weather['time']
#     max_temperatures = daily_weather['temperature_2m_max']
#     min_temperatures = daily_weather['temperature_2m_min']
#     max_humidity = daily_weather['relative_humidity_2m_max']
#     max_wind_speed = daily_weather['windspeed_10m_max']

#     # Get today's date
#     today = str(date.today())
    
#     # Find today's weather data
#     if today in dates:
#         index = dates.index(today)
#         current_weather = {
#             "Date": dates[index],
#             "City": city_name,
#             "Max Temperature": f"{max_temperatures[index]}°C",
#             "Min Temperature": f"{min_temperatures[index]}°C",
#             "Max Humidity": f"{max_humidity[index]}%",
#             "Max Wind Speed": f"{max_wind_speed[index]} km/h"
#         }
#         return current_weather
#     else:
#         print("Today's weather data not found.")
#         return None

# # Function to get 6 months of historical weather data
# def get_historical_weather(latitude, longitude):
#     url = "https://api.open-meteo.com/v1/forecast"
    
#     # Calculate start and end dates for 6 months
#     end_date = date.today()
#     start_date = end_date - timedelta(days=180)
    
#     params = {
#         'latitude': latitude,
#         'longitude': longitude,
#         'start_date': start_date.isoformat(),
#         'end_date': end_date.isoformat(),
#         'daily': ['temperature_2m_max', 'temperature_2m_min', 'windspeed_10m_max', 'relative_humidity_2m_max'],
#         'timezone': 'auto'
#     }
    
#     response = requests.get(url, params=params)
    
#     if response.status_code == 200:
#         weather_data = response.json()
#         weather_data = clean_weather_data(weather_data)  # Clean the data
#         return weather_data
#     else:
#         print(f"Failed to retrieve historical weather data for {start_date} to {end_date}")
#         return None

# # Function to check and store weather data in JSON
# def check_and_store_city_weather(city_name, weather_file='city_weather.json'):
#     # Load existing data from JSON
#     if os.path.exists(weather_file):
#         with open(weather_file, 'r') as file:
#             city_data = json.load(file)
#     else:
#         city_data = {}
    
#     # Check if city is already in the JSON file
#     if city_name in city_data:
#         print(f"Weather data for {city_name} is already available.")
#         return city_data[city_name]
    
#     # Get the coordinates of the city
#     latitude, longitude = get_coordinates_for_city(city_name)
    
#     if latitude and longitude:
#         print(f"Coordinates for {city_name}: Latitude = {latitude}, Longitude = {longitude}")
        
#         # Fetch 6 months of historical weather data
#         historical_data = get_historical_weather(latitude, longitude)
        
#         if historical_data:
#             # Update the data to include city name
#             historical_data['city_name'] = city_name
#             historical_data['latitude'] = latitude
#             historical_data['longitude'] = longitude
            
#             city_data[city_name] = historical_data
            
#             # Save the updated city weather data to the JSON file
#             with open(weather_file, 'w') as file:
#                 json.dump(city_data, file, indent=4)
                
#             print(f"Historical weather data for {city_name} stored in {weather_file}")
#         else:
#             print(f"Could not fetch historical weather data for {city_name}.")
#     else:
#         print(f"Unable to fetch coordinates for {city_name}.")
    
#     return city_data.get(city_name, None)

# # Main function to get current weather and update JSON
# def main():
#     city_name = input("Enter the city name:\n ")
    
#     # Check if city data exists, if not, fetch historical data
#     weather_data = check_and_store_city_weather(city_name)
    
#     if weather_data:
#         current_weather = get_current_weather(city_name, weather_data)
        
#         if current_weather:
#             print(f"Current Weather Data: {json.dumps(current_weather, indent=4)}")
            
#             # Save data to MySQL database
#             weather_handler = WeatherDataHandler()  # Create an instance of WeatherDataHandler
#             weather_handler.create_database()  # Ensure the database is created
#             weather_handler.load_data('city_weather.json')  # Load the weather data and save it to MySQL
#         else:
#             print("Failed to get current weather data.")
#     else:
#         print(f"Failed to get data for {city_name}.")

# if __name__ == "__main__":
#     main()


####################################################################################################


# import requests
# from geopy.geocoders import Nominatim
# from datetime import date, timedelta
# import json
# import os

# # Function to get the coordinates (latitude and longitude) for a city
# def get_coordinates_for_city(city_name):
#     geolocator = Nominatim(user_agent="weather_app")
#     location = geolocator.geocode(city_name)
    
#     if location:
#         return location.latitude, location.longitude
#     else:
#         print(f"City '{city_name}' not found.")
#         return None, None

# # Function to clean up unnecessary fields from weather data
# def clean_weather_data(weather_data):
#     keys_to_remove = [
#         "generationtime_ms", "utc_offset_seconds", "timezone", 
#         "timezone_abbreviation", "elevation", "daily_units"
#     ]
    
#     for key in keys_to_remove:
#         if key in weather_data:
#             del weather_data[key]
#     return weather_data

# # Function to get the current day's weather data
# def get_current_weather(city_name, weather_data):
#     daily_weather = weather_data['daily']
    
#     # Extract daily weather data
#     dates = daily_weather['time']
#     max_temperatures = daily_weather['temperature_2m_max']
#     min_temperatures = daily_weather['temperature_2m_min']
#     max_humidity = daily_weather['relative_humidity_2m_max']
#     max_wind_speed = daily_weather['windspeed_10m_max']

#     # Get today's date
#     today = str(date.today())
    
#     # Find today's weather data
#     if today in dates:
#         index = dates.index(today)
#         current_weather = {
#             "Date": dates[index],
#             "City": city_name,
#             "Max Temperature": f"{max_temperatures[index]}°C",
#             "Min Temperature": f"{min_temperatures[index]}°C",
#             "Max Humidity": f"{max_humidity[index]}%",
#             "Max Wind Speed": f"{max_wind_speed[index]} km/h"
#         }
#         return current_weather
#     else:
#         print("Today's weather data not found.")
#         return None

# # Function to get 6 months of historical weather data
# def get_historical_weather(latitude, longitude):
#     url = "https://api.open-meteo.com/v1/forecast"
    
#     # Calculate start and end dates for 6 months
#     end_date = date.today()
#     start_date = end_date - timedelta(days=180)
    
#     params = {
#         'latitude': latitude,
#         'longitude': longitude,
#         'start_date': start_date.isoformat(),
#         'end_date': end_date.isoformat(),
#         'daily': ['temperature_2m_max', 'temperature_2m_min', 'windspeed_10m_max', 'relative_humidity_2m_max'],
#         'timezone': 'auto'
#     }
    
#     response = requests.get(url, params=params)
    
#     if response.status_code == 200:
#         weather_data = response.json()
#         weather_data = clean_weather_data(weather_data)  # Clean the data
#         return weather_data
#     else:
#         print(f"Failed to retrieve historical weather data for {start_date} to {end_date}")
#         return None

# # Function to check and store weather data in JSON
# def check_and_store_city_weather(city_name, weather_file='city_weather.json'):
#     # Load existing data from JSON
#     if os.path.exists(weather_file):
#         with open(weather_file, 'r') as file:
#             city_data = json.load(file)
#     else:
#         city_data = {}
    
#     # Check if city is already in the JSON file
#     if city_name in city_data:
#         print(f"Weather data for {city_name} is already available.")
#         return city_data[city_name]
    
#     # Get the coordinates of the city
#     latitude, longitude = get_coordinates_for_city(city_name)
    
#     if latitude and longitude:
#         print(f"Coordinates for {city_name}: Latitude = {latitude}, Longitude = {longitude}")
        
#         # Fetch 6 months of historical weather data
#         historical_data = get_historical_weather(latitude, longitude)
        
#         if historical_data:
#             # Update the data to include city name
#             historical_data['city_name'] = city_name
#             historical_data['latitude'] = latitude
#             historical_data['longitude'] = longitude
            
#             city_data[city_name] = historical_data
            
#             # Save the updated city weather data to the JSON file
#             with open(weather_file, 'w') as file:
#                 json.dump(city_data, file, indent=4)
                
#             print(f"Historical weather data for {city_name} stored in {weather_file}")
#         else:
#             print(f"Could not fetch historical weather data for {city_name}.")
#     else:
#         print(f"Unable to fetch coordinates for {city_name}.")
    
#     return city_data.get(city_name, None)

# # Main function to get current weather and update JSON
# def main():
#     city_name = input("Enter the city name:\n ")
    
#     # Check if city data exists, if not, fetch historical data
#     weather_data = check_and_store_city_weather(city_name)
    
#     if weather_data:
#         current_weather = get_current_weather(city_name, weather_data)
        
#         if current_weather:
#             print(f"Current Weather Data: {json.dumps(current_weather, indent=4)}")
#         else:
#             print("Failed to get current weather data.")
#     else:
#         print(f"Failed to get data for {city_name}.")

# if __name__ == "__main__":
#     main()