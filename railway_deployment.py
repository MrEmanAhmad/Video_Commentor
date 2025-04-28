#!/usr/bin/env python
"""
Simplified Railway deployment script for Video Commentary Bot.
This script handles all environment setup required for Railway deployment.
"""

import os
import sys
import subprocess
import glob
import shutil
import platform
import json
from pathlib import Path

def check_command_exists(command):
    """Check if a command exists in the system path."""
    return shutil.which(command) is not None

def run_command(command, silent=False):
    """Run a shell command and return the output."""
    try:
        result = subprocess.run(
            command, 
            check=False, 
            stdout=subprocess.PIPE if silent else None,
            stderr=subprocess.PIPE if silent else None,
            text=True,
            shell=isinstance(command, str)
        )
        return result.returncode == 0, result.stdout if silent else ""
    except Exception as e:
        print(f"Error running command {command}: {e}")
        return False, ""

def install_system_dependencies():
    """Install critical system dependencies without sudo (for Railway)."""
    if not check_command_exists("apt-get"):
        print("apt-get not found, skipping apt dependency installation")
        return False
        
    # Critical dependencies needed for OpenCV and other libraries
    dependencies = [
        "libglib2.0-0",  # Includes libgthread
        "libgtk-3-0",
        "ffmpeg",
        "python3-opencv",
        "libgl1-mesa-glx",
        "libsm6",
        "libxext6",
        "tesseract-ocr",
        "libnss3",         # Required for Chromium
        "libxcomposite1",  # Required for Chromium
        "libxi6",          # Required for Chromium
        "libxrandr2",      # Required for Chromium
        "libxtst6",        # Required for Chromium
        "libcairo2",       # Required for GTK
        "libgirepository1.0-1", # Required for GTK
        "chromium",        # Browser for YouTube cookies extraction
        "chromium-driver"  # WebDriver for Selenium
    ]
    
    print("Installing system dependencies for Railway...")
    try:
        # Update package lists
        print("Running: apt-get update")
        run_command("apt-get update -y")
        
        # Install packages
        for package in dependencies:
            print(f"Installing {package}...")
            run_command(f"apt-get install -y {package}")
            
        return True
    except Exception as e:
        print(f"Error installing packages: {e}")
        return False

def setup_library_paths():
    """Set up library paths for OpenCV and other dependencies."""
    print("Setting up library paths...")
    
    # Create a directory for the libraries if it doesn't exist
    lib_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lib')
    os.makedirs(lib_dir, exist_ok=True)
    
    # Add this directory to LD_LIBRARY_PATH
    current_path = os.environ.get('LD_LIBRARY_PATH', '')
    if lib_dir not in current_path:
        if current_path:
            os.environ['LD_LIBRARY_PATH'] = f"{current_path}:{lib_dir}"
        else:
            os.environ['LD_LIBRARY_PATH'] = lib_dir
    
    # Print current platform for debugging
    print(f"Running on platform: {platform.system()} - {platform.platform()}")
    
    # Skip on Windows (not using Railway)
    if platform.system() == "Windows":
        print("Windows detected, skipping library fixes")
        return lib_dir
    
    # Critical libraries to find
    critical_libraries = [
        'libgthread-2.0.so*',
        'libglib-2.0.so*',
        'libGL.so*',
    ]
    
    # System paths to search for libraries
    system_paths = [
        "/usr/lib",
        "/usr/lib/x86_64-linux-gnu",
        "/lib",
        "/lib64",
        "/opt/conda/lib",  # Railway might use conda
    ]
    
    # Find and copy critical libraries
    for lib_pattern in critical_libraries:
        try:
            lib_paths = []
            # Search in system paths
            for path in system_paths:
                if os.path.exists(path):
                    system_find_cmd = ["find", path, "-name", lib_pattern]
                    try:
                        sys_result = subprocess.run(system_find_cmd, capture_output=True, text=True, check=False)
                        if sys_result.stdout:
                            system_libs = sys_result.stdout.strip().split('\n')
                            print(f"Found {len(system_libs)} matching libraries in {path}")
                            lib_paths.extend([p for p in system_libs if p])
                    except Exception as e:
                        print(f"Error searching in {path}: {str(e)}")
            
            # Copy libraries to our lib directory
            for src_path in lib_paths:
                if src_path and os.path.isfile(src_path):
                    filename = os.path.basename(src_path)
                    dest_path = os.path.join(lib_dir, filename)
                    
                    # Copy the file if it doesn't exist or if it's different
                    if not os.path.exists(dest_path) or os.path.getsize(src_path) != os.path.getsize(dest_path):
                        print(f"Copying {src_path} to {dest_path}")
                        shutil.copy2(src_path, dest_path)
                        
                        # Create symlinks for versioned libraries
                        if '.' in filename and not filename.endswith('.a') and not filename.endswith('.la'):
                            base_name = filename.split('.so.')[0] + '.so'
                            symlink_path = os.path.join(lib_dir, base_name)
                            if not os.path.exists(symlink_path) or os.path.islink(symlink_path):
                                if os.path.exists(symlink_path):
                                    os.remove(symlink_path)
                                print(f"Creating symlink {symlink_path} -> {filename}")
                                os.symlink(filename, symlink_path)
            
            # Special check for libgthread-2.0.so.0
            if lib_pattern == 'libgthread-2.0.so*' and not glob.glob(os.path.join(lib_dir, 'libgthread-2.0.so*')):
                print("libgthread-2.0.so.0 not found - trying to install libglib2.0-0")
                run_command("apt-get install -y libglib2.0-0")
                
        except Exception as e:
            print(f"Error processing library pattern '{lib_pattern}': {str(e)}")
    
    # Print the contents of the lib directory
    print("\nContents of the lib directory:")
    for filename in sorted(os.listdir(lib_dir)):
        file_path = os.path.join(lib_dir, filename)
        if os.path.islink(file_path):
            target = os.readlink(file_path)
            print(f"  {filename} -> {target}")
        else:
            print(f"  {filename}")
    
    return lib_dir

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

def check_python_dependencies():
    """Check and install required Python dependencies."""
    try:
        print("Checking Python dependencies...")
        
        # Try importing key modules
        import_success = True
        try:
            import streamlit
            import openai
            import dashscope
            import google.cloud.texttospeech
            import cv2
            import yt_dlp
            import numpy
            import PIL
        except ImportError as e:
            import_success = False
            print(f"Import error: {e}")
            
        # If imports failed, try reinstalling dependencies
        if not import_success:
            print("Some dependencies are missing, installing from requirements.txt...")
            run_command("pip install -r requirements.txt")
            
        return True
    except Exception as e:
        print(f"Error checking Python dependencies: {e}")
        return False

def main():
    """Main function to set up all dependencies and start the Streamlit app."""
    print(f"Setting up Railway deployment for platform: {platform.platform()}")
    
    # Print useful environment information
    print("\nCurrent environment:")
    print(f"  Python: {sys.version}")
    print(f"  Working directory: {os.getcwd()}")
    print(f"  LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', 'Not set')}")
    
    # Install system dependencies
    install_system_dependencies()
    
    # Set up libraries
    lib_dir = setup_library_paths()
    
    # Check Python dependencies
    check_python_dependencies()
    
    # Set up Google credentials
    setup_google_credentials()
    
    # Print environment variables
    print("\nEnvironment variables:")
    for name, value in sorted(os.environ.items()):
        if name.startswith('LD_') or name.startswith('PYTHON') or name.startswith('PATH'):
            print(f"  {name}={value}")
    
    # Add current directory to Python path
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    # Start the Streamlit app
    cmd = ["streamlit", "run", "streamlit_app.py"]
    
    # If PORT is set in environment variables, use it
    port = os.environ.get('PORT')
    if port:
        cmd.extend(["--server.port", port])
    
    print(f"\nStarting Streamlit with command: {' '.join(cmd)}")
    
    # Run the Streamlit app
    subprocess.run(cmd)

if __name__ == "__main__":
    main() 