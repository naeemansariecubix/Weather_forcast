# Use the Python 3.11 base image
FROM python:3.11

# Copy all files from the current directory to /app in the container
COPY . /app

# Set the working directory to /app
WORKDIR /app

# Install dependencies from requirements.txt
RUN pip install -r requirements.txt

# Expose the port specified in the $PORT environment variable
EXPOSE $PORT

# Run the application using Gunicorn
CMD gunicorn --workers=4 --bind 0.0.0.0:$PORT app:app
