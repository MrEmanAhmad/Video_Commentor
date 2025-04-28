#!/usr/bin/env python
"""
Script to prepare the environment for running the Video Commentary Bot on Railway.
This script writes Google credentials from environment variables to a file.
"""

import os
import json
import subprocess
import sys

def setup_google_credentials():
    """Set up Google credentials from environment variables."""
    # Check if the JSON credentials are provided as an environment variable
    google_creds_json = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS_JSON')
    
    if not google_creds_json:
        print("Warning: GOOGLE_APPLICATION_CREDENTIALS_JSON environment variable not found.")
        return
    
    # Try to parse the credentials
    try:
        if isinstance(google_creds_json, str):
            creds_dict = json.loads(google_creds_json)
        else:
            creds_dict = google_creds_json
            
        # Write the credentials to a file
        creds_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'google_credentials.json')
        
        with open(creds_file_path, 'w') as f:
            json.dump(creds_dict, f, indent=2)
        
        # Set the environment variable to point to this file
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = creds_file_path
        print(f"Google credentials written to {creds_file_path}")
        
    except Exception as e:
        print(f"Error setting up Google credentials: {str(e)}")

if __name__ == "__main__":
    # Set up credentials
    setup_google_credentials()
    
    # Start the Streamlit app
    cmd = ["streamlit", "run", "streamlit_app.py"]
    
    # If PORT is set in environment variables, use it
    port = os.environ.get('PORT')
    if port:
        cmd.extend(["--server.port", port])
    
    # Run the Streamlit app
    subprocess.run(cmd) 