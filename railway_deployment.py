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
import threading
import http.server
import socketserver
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger("railway_deployment")

# Global variable to track if application is healthy
APP_HEALTHY = False

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

def find_libraries(pattern, base_paths):
    """Find libraries matching the pattern in the given base paths"""
    matched_paths = []
    
    for base_path in base_paths:
        if not os.path.exists(base_path):
            continue
            
        if "*" in base_path:
            # Handle glob patterns in the base path
            expanded_bases = glob.glob(base_path)
            for expanded_base in expanded_bases:
                if os.path.exists(expanded_base):
                    # Look for the pattern in this expanded base path
                    matches = glob.glob(os.path.join(expanded_base, pattern))
                    matched_paths.extend(matches)
        else:
            # Direct path without glob pattern
            matches = glob.glob(os.path.join(base_path, pattern))
            matched_paths.extend(matches)
    
    return matched_paths

def setup_opencvlib():
    """Set up OpenCV library paths"""
    logger.info("Setting up OpenCV library paths...")
    
    if os.name == 'nt':  # Windows
        logger.info("Windows detected, skipping OpenCV library path setup")
        return True
    
    # Standard library paths to check
    standard_paths = [
        "/usr/lib",
        "/usr/local/lib",
        "/lib",
        "/nix/store/*/lib",
        "/usr/lib/x86_64-linux-gnu",
        "/lib/x86_64-linux-gnu",
        "/opt/conda/lib",
        "/nix/var/nix/profiles/default/lib"
    ]
    
    # Critical libraries to ensure are in the path
    critical_libs = [
        "libGL.so*", 
        "libgthread-2.0.so*", 
        "libopencv*.so*",
        "libgtk*.so*",
        "libglib*.so*"
    ]
    
    # Find all relevant libraries
    library_paths = set()
    
    for lib_pattern in critical_libs:
        found_libs = find_libraries(lib_pattern, standard_paths)
        for lib in found_libs:
            # Add the directory containing the library
            lib_dir = os.path.dirname(lib)
            if lib_dir:
                library_paths.add(lib_dir)
    
    # Specific check for OpenCV libraries in Nix store
    try:
        nix_opencv_dirs = run_command(
            "find /nix/store -path '*opencv*' -type d", 
            shell=True
        ).stdout.strip().split('\n')
        
        for dir_path in nix_opencv_dirs:
            if dir_path and "lib" in dir_path:
                library_paths.add(dir_path)
    except Exception as e:
        logger.warning(f"Error searching for OpenCV in Nix store: {str(e)}")
    
    # Update the LD_LIBRARY_PATH environment variable
    if library_paths:
        current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
        new_paths = list(library_paths)
        new_ld_path = ':'.join(new_paths)
        
        if current_ld_path:
            os.environ['LD_LIBRARY_PATH'] = f"{current_ld_path}:{new_ld_path}"
        else:
            os.environ['LD_LIBRARY_PATH'] = new_ld_path
        
        logger.info(f"Updated LD_LIBRARY_PATH: {os.environ['LD_LIBRARY_PATH']}")
        return True
    else:
        logger.warning("No library paths found to add to LD_LIBRARY_PATH")
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
    
    # Ensure Python path includes the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.append(current_dir)
        logger.info(f"Added {current_dir} to Python path")
    
    # Create necessary directories
    for dir_name in ['output', 'tmp', 'lib']:
        dir_path = os.path.join(current_dir, dir_name)
        os.makedirs(dir_path, exist_ok=True)
        logger.info(f"Created directory: {dir_path}")
    
    return True

class HealthcheckHandler(http.server.SimpleHTTPRequestHandler):
    """Simple healthcheck handler that responds to /healthz"""
    
    def do_GET(self):
        if self.path == '/healthz':
            self.send_response(200 if APP_HEALTHY else 503)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            status = "healthy" if APP_HEALTHY else "not healthy"
            self.wfile.write(f"Status: {status}".encode())
        else:
            # Redirect all other requests to Streamlit
            self.send_response(302)
            host = self.headers.get('Host', 'localhost:8080')
            self.send_header('Location', f'http://{host}/')
            self.end_headers()

def run_healthcheck_server():
    """Run a simple HTTP server for healthchecks"""
    try:
        # Use a port different from Streamlit
        healthcheck_port = int(os.environ.get('HEALTHCHECK_PORT', '8081'))
        handler = HealthcheckHandler
        # Reduce logging noise
        handler.log_message = lambda *args: None
        
        with socketserver.TCPServer(("", healthcheck_port), handler) as httpd:
            logger.info(f"Healthcheck server started on port {healthcheck_port}")
            httpd.serve_forever()
    except Exception as e:
        logger.error(f"Healthcheck server error: {str(e)}")

def verify_streamlit_running(port):
    """Check if Streamlit is running and accessible"""
    max_attempts = 20
    delay = 3  # seconds
    
    logger.info(f"Verifying Streamlit is running on port {port}...")
    
    for attempt in range(1, max_attempts + 1):
        try:
            import requests
            r = requests.get(f"http://localhost:{port}", timeout=5)
            if r.status_code < 400:
                logger.info(f"Streamlit is running (status code: {r.status_code})")
                return True
            else:
                logger.warning(f"Attempt {attempt}/{max_attempts}: Streamlit returned status code {r.status_code}")
        except Exception as e:
            logger.warning(f"Attempt {attempt}/{max_attempts}: Cannot connect to Streamlit - {str(e)}")
        
        if attempt < max_attempts:
            logger.info(f"Waiting {delay} seconds before next check...")
            time.sleep(delay)
    
    logger.error("Streamlit verification failed after maximum attempts")
    return False

def start_streamlit():
    """Start the Streamlit application"""
    global APP_HEALTHY
    
    logger.info("Starting Streamlit application...")
    
    port = os.environ.get('PORT', '8080')
    
    # Start a healthcheck server in a separate thread
    healthcheck_thread = threading.Thread(target=run_healthcheck_server, daemon=True)
    healthcheck_thread.start()
    
    # Define Streamlit command
    command = [
        "streamlit", "run", "streamlit_app.py",
        "--server.address=0.0.0.0",
        f"--server.port={port}",
        "--server.enableCORS=false",
        "--server.enableXsrfProtection=false",
        "--server.enableWebsocketCompression=false",
        "--server.baseUrlPath=/"
    ]
    
    # Log the command we're about to run
    logger.info(f"Running Streamlit with command: {' '.join(command)}")
    
    # Start Streamlit in a subprocess rather than replacing the current process
    streamlit_process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1  # Line buffered
    )
    
    # Monitor the output in a separate thread
    def monitor_output():
        for line in iter(streamlit_process.stdout.readline, ''):
            logger.info(f"STREAMLIT: {line.rstrip()}")
    
    output_thread = threading.Thread(target=monitor_output, daemon=True)
    output_thread.start()
    
    # Verify that Streamlit is running
    if verify_streamlit_running(port):
        APP_HEALTHY = True
        logger.info("Application marked as HEALTHY")
    else:
        logger.error("Application could not be verified as running")
    
    # Wait for the Streamlit process to complete
    streamlit_process.wait()
    logger.warning(f"Streamlit process exited with code {streamlit_process.returncode}")
    APP_HEALTHY = False
    
    return streamlit_process.returncode

def setup_google_credentials():
    """Set up Google credentials from environment variable"""
    creds_json = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS_JSON')
    if not creds_json:
        logger.warning("GOOGLE_APPLICATION_CREDENTIALS_JSON environment variable not found")
        return False
    
    try:
        import json
        creds_dict = json.loads(creds_json)
        
        # Write credentials to a file
        creds_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'google_credentials.json')
        with open(creds_file, 'w') as f:
            json.dump(creds_dict, f, indent=2)
        
        # Set environment variable to point to the file
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = creds_file
        logger.info(f"Google credentials written to {creds_file}")
        return True
    except Exception as e:
        logger.error(f"Error setting up Google credentials: {str(e)}")
        return False

def main():
    """Main entry point"""
    logger.info("Starting Railway deployment script")
    
    # Wait a moment for system stabilization
    time.sleep(1)
    
    # Print the environment for debugging
    logger.info("Environment variables:")
    for key, value in os.environ.items():
        # Skip printing sensitive values
        if any(secret in key.lower() for secret in ['key', 'token', 'secret', 'password', 'credential']):
            logger.info(f"  {key}=********")
        else:
            logger.info(f"  {key}={value}")
    
    # Setup steps
    steps = [
        ("Setting up environment", setup_environment),
        ("Checking dependencies", check_dependencies),
        ("Setting up OpenCV libraries", setup_opencvlib),
        ("Setting up Google credentials", setup_google_credentials),
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
        exit_code = start_streamlit()
        logger.info(f"Streamlit exited with code {exit_code}")
        sys.exit(exit_code)
    except Exception as e:
        logger.error(f"Failed to start Streamlit: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 