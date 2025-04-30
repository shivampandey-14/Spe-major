# Use official Python image
FROM python:3.9

# Set working directory
WORKDIR /app

# Copy project files (including dataset)
COPY . /app
COPY Assignment-2_Data.csv /app/data/dataset.csv 

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set environment variable for dataset location (optional)
ENV DATA_PATH="/app/data/dataset.csv"

# Expose Flask port (if using an API)
# EXPOSE 5000

# Run inference script
CMD ["python", "ml_project.py"]
