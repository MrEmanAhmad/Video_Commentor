FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    python3-opencv \
    tesseract-ocr \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libgtk-3-0 \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libcairo2 \
    libnss3 \
    libatk1.0-0 \
    libgbm1 \
    libasound2 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create needed directories
RUN mkdir -p /app/output /app/tmp /app/lib

# Expose the port the app runs on
EXPOSE 8080

# Command to run the application
CMD streamlit run streamlit_app.py --server.address=0.0.0.0 --server.port=${PORT:-8080} --server.enableCORS=false 