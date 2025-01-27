FROM python:3.9-slim

# Update and install required system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    g++ \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install coreferee dependencies
RUN pip install --upgrade pip setuptools wheel

# Install thinc and coreferee
RUN pip install thinc coreferee
