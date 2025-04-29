FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose frontend Flask port
# EXPOSE 5000

# Ensure Flask listens on all interfaces
CMD ["python", "app.py"]
