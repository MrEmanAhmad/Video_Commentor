#!/usr/bin/env python3
"""
Railway deployment script for Video Commentary Bot
This script sets up the necessary environment and dependencies before starting the Streamlit app
"""

import os
import subprocess
import sys
import glob
import logging
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger("railway_deployment")

def check_command_exists(command):
    """Check if a command exists in the PATH"""
    try:
        subprocess.run(
            ["which", command], 
            check=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE
        )
        return True
    except subprocess.CalledProcessError:
        return False

def run_command(command, shell=False):
    """Run a shell command and return the output"""
    logger.info(f"Running command: {command}")
    result = subprocess.run(
        command,
        shell=shell,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    if result.returncode != 0:
        logger.warning(f"Command failed with code {result.returncode}")
        logger.warning(f"STDERR: {result.stderr}")
    else:
        logger.info(f"Command completed successfully")
    
    return result

def setup_opencvlib():
    """Set up OpenCV library paths"""
    logger.info("Setting up OpenCV library paths...")
    
    if os.name == 'nt':  # Windows
        logger.info("Windows detected, skipping OpenCV library path setup")
        return True
    
    # Find OpenCV library locations
    opencv_paths = []
    
    # Check standard locations
    standard_paths = [
        "/usr/lib",
        "/usr/local/lib",
        "/lib",
        "/nix/store",
        "/opt/conda/lib"
    ]
    
    # Find installed OpenCV libraries
    for base_path in standard_paths:
        if os.path.exists(base_path):
            # Look for libopencv in the base path
            opencv_libs = glob.glob(f"{base_path}/*opencv*")
            opencv_paths.extend(opencv_libs)
            
            # Look for OpenCV libraries in subdirectories
            if base_path == "/nix/store":
                # For Nix, we need to find all potential paths that contain OpenCV libraries
                opencv_dirs = subprocess.run(
                    f"find {base_path} -path '*opencv*' -type d", 
                    shell=True, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE,
                    text=True
                ).stdout.strip().split('\n')
                
                # Find lib directories within these paths
                for dir in opencv_dirs:
                    if dir:
                        lib_paths = glob.glob(f"{dir}/lib")
                        opencv_paths.extend(lib_paths)
    
    # Add the paths to LD_LIBRARY_PATH
    if opencv_paths:
        logger.info(f"Found OpenCV library paths: {opencv_paths}")
        current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
        
        # Ensure each path is only added once
        new_paths = []
        for path in opencv_paths:
            if path and path not in current_ld_path:
                new_paths.append(path)
        
        if new_paths:
            new_ld_path = ':'.join(new_paths)
            if current_ld_path:
                os.environ['LD_LIBRARY_PATH'] = f"{current_ld_path}:{new_ld_path}"
            else:
                os.environ['LD_LIBRARY_PATH'] = new_ld_path
            
            logger.info(f"Updated LD_LIBRARY_PATH: {os.environ['LD_LIBRARY_PATH']}")
            return True
        else:
            logger.info("No new OpenCV library paths to add")
            return True
    else:
        logger.warning("Could not find OpenCV library paths")
        return False

def check_dependencies():
    """Check if required system dependencies are installed"""
    logger.info("Checking system dependencies...")
    
    # List of dependencies to check
    dependencies = ['ffmpeg', 'python3', 'tesseract']
    
    missing_deps = []
    for dep in dependencies:
        if not check_command_exists(dep):
            missing_deps.append(dep)
    
    if missing_deps:
        logger.warning(f"Missing dependencies: {', '.join(missing_deps)}")
        return False
    else:
        logger.info("All required dependencies are installed")
        return True

def setup_environment():
    """Set up environment variables and paths"""
    logger.info("Setting up environment...")
    
    # Set Railway-specific environment variables
    if 'PORT' not in os.environ:
        os.environ['PORT'] = '8080'
        logger.info("Set default PORT to 8080")
    
    # Set up LD_LIBRARY_PATH with common library locations in Railway/Nixpacks
    lib_paths = [
        "/nix/store/*/lib",
        "/usr/lib",
        "/usr/lib/x86_64-linux-gnu",
        "/lib/x86_64-linux-gnu",
        "/nix/var/nix/profiles/default/lib"
    ]
    
    # Expand glob patterns
    expanded_paths = []
    for path in lib_paths:
        if '*' in path:
            expanded = glob.glob(path)
            expanded_paths.extend(expanded)
        elif os.path.exists(path):
            expanded_paths.append(path)
    
    if expanded_paths:
        current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
        new_ld_path = ':'.join(expanded_paths)
        
        if current_ld_path:
            os.environ['LD_LIBRARY_PATH'] = f"{current_ld_path}:{new_ld_path}"
        else:
            os.environ['LD_LIBRARY_PATH'] = new_ld_path
        
        logger.info(f"Set LD_LIBRARY_PATH to: {os.environ['LD_LIBRARY_PATH']}")
    
    return True

def start_streamlit():
    """Start the Streamlit application"""
    logger.info("Starting Streamlit application...")
    
    port = os.environ.get('PORT', '8080')
    
    # Define Streamlit command
    command = [
        "streamlit", "run", "streamlit_app.py",
        "--server.address=0.0.0.0",
        f"--server.port={port}",
        "--server.enableCORS=false",
        "--server.enableWebsocketCompression=false"
    ]
    
    # Log the command we're about to run
    logger.info(f"Running Streamlit with command: {' '.join(command)}")
    
    # Execute command with process replacement
    os.execvp(command[0], command)

def main():
    """Main entry point"""
    logger.info("Starting Railway deployment script")
    
    # Wait a moment for system stabilization
    time.sleep(1)
    
    # Print the environment for debugging
    logger.info("Environment variables:")
    for key, value in os.environ.items():
        logger.info(f"  {key}={value}")
    
    # Setup steps
    steps = [
        ("Checking dependencies", check_dependencies),
        ("Setting up environment", setup_environment),
        ("Setting up OpenCV libraries", setup_opencvlib),
    ]
    
    # Run setup steps
    for step_name, step_func in steps:
        logger.info(f"Starting step: {step_name}")
        try:
            result = step_func()
            if not result:
                logger.warning(f"Step '{step_name}' did not complete successfully")
            else:
                logger.info(f"Step '{step_name}' completed successfully")
        except Exception as e:
            logger.error(f"Error in step '{step_name}': {str(e)}")
    
    # Start Streamlit
    try:
        start_streamlit()
    except Exception as e:
        logger.error(f"Failed to start Streamlit: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 