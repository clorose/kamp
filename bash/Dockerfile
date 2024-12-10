FROM --platform=$TARGETPLATFORM python:3.11-slim

# Install system packages and basic tools
RUN apt-get update && apt-get install -y \
  build-essential \
  libgl1-mesa-glx \
  libglib2.0-0 \
  libpng-dev \
  libjpeg-dev \
  curl \
  git \
  vim \
  nano \
  && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Set working directory
WORKDIR /app

# Create necessary directories
RUN mkdir -p /app/data /app/runs /app/models /app/src \
  && chmod 777 /app/data /app/runs /app/models /app/src

# Copy and install requirements
COPY requirements.txt .
RUN uv pip install -r requirements.txt

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app