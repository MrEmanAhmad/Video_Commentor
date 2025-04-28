#!/usr/bin/env python
"""
Script to prepare the environment for running the Video Commentary Bot on Railway.
This script writes Google credentials from environment variables to a file.
"""

import os
import json
import subprocess
import sys
import glob

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

def setup_library_paths():
    """Set up library paths for OpenCV and other dependencies."""
    print("Setting up library paths...")
    
    # Run the CV2 fix script to copy necessary libraries
    try:
        print("Running CV2 fix script...")
        import cv2_fix
        lib_dir = cv2_fix.main()
        
        # Make sure the lib directory is in LD_LIBRARY_PATH
        current_path = os.environ.get('LD_LIBRARY_PATH', '')
        if lib_dir not in current_path:
            if current_path:
                os.environ['LD_LIBRARY_PATH'] = f"{current_path}:{lib_dir}"
            else:
                os.environ['LD_LIBRARY_PATH'] = lib_dir
        
        print(f"Updated LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH')}")
    except Exception as e:
        print(f"Error running CV2 fix script: {str(e)}")
        
        # Fall back to the original library path setup
        # Common library paths in Nix
        lib_paths = [
            "/nix/store/*/lib",
            "/nix/var/nix/profiles/default/lib",
            "/usr/lib",
            "/usr/lib/x86_64-linux-gnu"
        ]
        
        # Expand glob patterns to find all matching directories
        expanded_paths = []
        for path in lib_paths:
            if '*' in path:
                expanded_paths.extend(glob.glob(path))
            else:
                if os.path.exists(path):
                    expanded_paths.append(path)
        
        # Set the LD_LIBRARY_PATH environment variable
        current_path = os.environ.get('LD_LIBRARY_PATH', '')
        new_path = ':'.join(expanded_paths)
        if current_path:
            full_path = f"{current_path}:{new_path}"
        else:
            full_path = new_path
        
        os.environ['LD_LIBRARY_PATH'] = full_path
        print(f"LD_LIBRARY_PATH set to: {full_path}")
        
        # Try to locate libGL.so.1
        found = False
        for path in expanded_paths:
            if os.path.exists(os.path.join(path, 'libGL.so.1')):
                print(f"Found libGL.so.1 in {path}")
                found = True
                break
        
        if not found:
            print("Warning: libGL.so.1 not found in any of the library paths")
            # Try to find it anywhere in /nix/store
            try:
                result = subprocess.run(["find", "/nix/store", "-name", "libGL.so.1"], 
                                       capture_output=True, text=True, check=False)
                if result.stdout:
                    locations = result.stdout.strip().split('\n')
                    print(f"Found libGL.so.1 in the following locations:")
                    for loc in locations:
                        print(f"  - {loc}")
                        # Add the directory containing libGL.so.1 to LD_LIBRARY_PATH
                        lib_dir = os.path.dirname(loc)
                        if lib_dir not in os.environ['LD_LIBRARY_PATH']:
                            os.environ['LD_LIBRARY_PATH'] = f"{os.environ['LD_LIBRARY_PATH']}:{lib_dir}"
                            print(f"Added {lib_dir} to LD_LIBRARY_PATH")
            except Exception as e:
                print(f"Error searching for libGL.so.1: {str(e)}")

if __name__ == "__main__":
    # Set up library paths first
    setup_library_paths()
    
    # Set up credentials
    setup_google_credentials()
    
    # Print some debug information
    print("\nEnvironment variables:")
    for name, value in sorted(os.environ.items()):
        if name.startswith('LD_') or name.startswith('PYTHON') or name.startswith('PATH'):
            print(f"  {name}={value}")
    
    # Start the Streamlit app
    cmd = ["streamlit", "run", "streamlit_app.py"]
    
    # If PORT is set in environment variables, use it
    port = os.environ.get('PORT')
    if port:
        cmd.extend(["--server.port", port])
    
    print(f"\nStarting Streamlit with command: {' '.join(cmd)}")
    
    # Run the Streamlit app
    subprocess.run(cmd) 