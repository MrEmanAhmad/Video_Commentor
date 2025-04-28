@echo off
echo Starting Video Commentary Bot...

REM Check if Python is installed
where python >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Python is not installed or not in PATH
    echo Please install Python from https://www.python.org/downloads/
    pause
    exit /b 1
)

REM Check if streamlit is installed
python -c "import streamlit" >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Installing Streamlit...
    pip install streamlit
)

REM Check if .env file exists and load environment variables
if exist .env (
    echo Loading environment variables from .env file...
    for /F "tokens=*" %%A in (.env) do set %%A
) else (
    echo Warning: .env file not found. Make sure you have set your API keys.
)

REM Check for Google credentials file
if not exist google_credentials.json (
    echo Warning: google_credentials.json not found
    echo Please add your Google Cloud credentials file to the current directory
)

REM Run the Streamlit app
echo Starting Streamlit server...
streamlit run streamlit_app.py

pause 