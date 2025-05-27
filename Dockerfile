# Use base image with Python
FROM python:3.9-slim

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential cmake libboost-all-dev \
    libatlas-base-dev libopenblas-dev liblapack-dev \
    libhdf5-dev libx11-dev libgtk-3-dev wget git ffmpeg \
    && apt-get clean

# Set working directory
WORKDIR /app

# Copy all files
COPY . /app

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose Django port
EXPOSE 8000

# Run using gunicorn
CMD ["gunicorn", "app:application", "--bind", "0.0.0.0:8000"]
