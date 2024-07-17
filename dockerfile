# Use a multi-architecture base image
FROM --platform=$BUILDPLATFORM python:3.11-slim

# Install required libraries
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install QEMU for multi-architecture support
RUN apt-get update && apt-get install -y qemu-user-static

# Set the working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose the port
EXPOSE 5000

# Command to run the Flask application with Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
