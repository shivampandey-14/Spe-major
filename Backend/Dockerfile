# Use official Python base image
FROM python:3.9

# Set working directory inside the container
WORKDIR /app

# Copy all project files to the container's working directory
COPY . /app

# Copy dataset to the correct location
COPY Assignment-2_Data.csv /app/data/dataset.csv 

# Install all dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose Flask default port
EXPOSE 5001

# Set environment variable for dataset path (optional, if used in your code)
ENV DATA_PATH="/app/data/dataset.csv"

# Run the Flask backend (assuming it's in ml_project.py)
CMD ["python", "ml_project.py"]
