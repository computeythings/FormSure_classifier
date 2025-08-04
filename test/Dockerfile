# Development Dockerfile for local testing
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libgthread-2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install development dependencies
RUN pip install --no-cache-dir \
    pytest \
    black \
    flake8 \
    ipython \
    jupyter

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Set environment variables for development
ENV PYTHONPATH=/app
ENV DEBUG_MODE=true
ENV LOCAL_MODE=true

# Default command
CMD ["python", "server.py", "--host", "0.0.0.0", "--port", "8000"]