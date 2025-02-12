# Use Python 3.10 slim image as base
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app \
    PORT=8501 \
    RAILWAY_ENVIRONMENT=production \
    DEBIAN_FRONTEND=noninteractive

# Create a non-root user
RUN useradd -m -s /bin/bash app_user

# Add Chrome repository and install system dependencies
RUN apt-get update && \
    apt-get install -y wget gnupg2 && \
    wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add - && \
    echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    google-chrome-stable \
    chromium-driver \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    git \
    libmagic1 \
    libpython3-dev \
    build-essential \
    python3-dev \
    pkg-config \
    curl \
    unzip \
    xvfb \
    libxi6 \
    libgconf-2-4 \
    default-jdk \
    apt-transport-https \
    ca-certificates \
    # Video processing dependencies
    libavcodec-extra \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libatlas-base-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    # Additional dependencies for yt-dlp
    python3-pip \
    # Cleanup
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set up Chrome and ChromeDriver
ENV CHROME_BIN=/usr/bin/google-chrome \
    CHROME_PATH=/usr/bin/google-chrome \
    CHROMEDRIVER_PATH=/usr/bin/chromedriver \
    DISPLAY=:99

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies with optimizations
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    # Install latest yt-dlp directly
    pip install --no-cache-dir --upgrade yt-dlp && \
    # Clean up pip cache
    rm -rf /root/.cache/pip/* && \
    # Pre-compile Python files
    python -m compileall /app

# Create necessary directories with proper structure
RUN mkdir -p \
    /home/app_user/.streamlit \
    /home/app_user/.cache/yt-dlp \
    /home/app_user/.cache/youtube-dl \
    /home/app_user/.cache/selenium \
    /home/app_user/.config/chromium \
    /home/app_user/.config/google-chrome \
    /app/credentials \
    /app/analysis_temp \
    /app/sample_generated_videos \
    /app/framesAndLogo/Nature \
    /app/framesAndLogo/News \
    /app/framesAndLogo/Funny \
    /app/framesAndLogo/Infographic

# Copy Streamlit config
COPY .streamlit/config.toml /home/app_user/.streamlit/config.toml

# Copy the entire application
COPY . .

# Set proper permissions
RUN chown -R app_user:app_user \
    /app \
    /home/app_user/.streamlit \
    /home/app_user/.cache \
    /home/app_user/.config \
    && chmod -R 755 /app pipeline \
    && chmod -R 777 \
    /app/credentials \
    /app/analysis_temp \
    /app/sample_generated_videos \
    /app/framesAndLogo \
    /home/app_user/.config \
    /home/app_user/.streamlit \
    /home/app_user/.cache

# Switch to non-root user
USER app_user

# Set environment variables for the application
ENV HOME=/home/app_user \
    PYTHONPATH=${PYTHONPATH}:/app \
    SELENIUM_CACHE_PATH=/home/app_user/.cache/selenium \
    LC_ALL=C.UTF-8 \
    LANG=C.UTF-8 \
    # OpenCV optimizations
    OPENCV_FFMPEG_CAPTURE_OPTIONS="video_codec;h264_cuvid" \
    OPENCV_VIDEOIO_PRIORITY_BACKEND=2 \
    # FFmpeg optimizations
    FFREPORT=file=/app/analysis_temp/ffmpeg-%p-%t.log \
    # Chrome/Selenium settings
    SELENIUM_HEADLESS=true \
    PYTHONWARNINGS="ignore:Unverified HTTPS request"

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8501}/_stcore/health || exit 1

# Expose the port that will be used by Streamlit
EXPOSE ${PORT:-8501}

# Start Xvfb and run Streamlit with optimized settings
CMD Xvfb :99 -screen 0 1280x1024x24 -ac +extension GLX +render -noreset & \
    streamlit run \
    --server.port=${PORT:-8501} \
    --server.address=0.0.0.0 \
    --server.maxUploadSize=50 \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false \
    --server.maxMessageSize=200 \
    --browser.gatherUsageStats=false \
    --theme.base=dark \
    streamlit_app.py 