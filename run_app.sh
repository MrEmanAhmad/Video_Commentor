#!/bin/bash

echo "Starting Video Commentary Bot..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python is not installed or not in PATH"
    echo "Please install Python from https://www.python.org/downloads/"
    exit 1
fi

# Check if streamlit is installed
if ! python3 -c "import streamlit" &> /dev/null; then
    echo "Installing Streamlit..."
    pip3 install streamlit
fi

# Check if .env file exists and load environment variables
if [ -f .env ]; then
    echo "Loading environment variables from .env file..."
    export $(grep -v '^#' .env | xargs)
else
    echo "Warning: .env file not found. Make sure you have set your API keys."
fi

# Check for Google credentials file
if [ ! -f google_credentials.json ]; then
    echo "Warning: google_credentials.json not found"
    echo "Please add your Google Cloud credentials file to the current directory"
fi

# Set permissions for execution if needed
export GOOGLE_APPLICATION_CREDENTIALS="$(pwd)/google_credentials.json"

# Run the Streamlit app
echo "Starting Streamlit server..."
streamlit run streamlit_app.py 