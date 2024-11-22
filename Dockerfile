# Use official Python image from Docker Hub
FROM python:3.11

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Set the working directory inside the container
WORKDIR /app

# Copy all files from your app to the container's working directory
COPY . /app/

# Install Python dependencies
RUN pip install -r requirements.txt

# Expose the port that Flask will run on (8080)
EXPOSE 8080

# Set the default command to run your application using gunicorn (for production)
CMD ["gunicorn", "--workers=4", "--bind", "0.0.0.0:8080", "app:app"]
