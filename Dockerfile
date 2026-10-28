# Production Dockerfile for Mobile Phone Model Classification Service
FROM python:3.10-slim

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=5000

# Install system dependencies for imaging libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python requirements (including TensorFlow & Keras)
COPY requirements.txt config/requirements-train.txt ./
RUN pip install --no-cache-dir -r requirements-train.txt

# Copy application source code and artifacts
COPY . .

# Expose default port
EXPOSE 5000

# Health check to ensure model engine is responsive
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:5000/api/health || exit 1

# Start production server
CMD ["python", "app.py"]
