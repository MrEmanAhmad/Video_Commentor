"""
Configuration file for Video Commentary Bot.
Store API keys and other settings here.
"""

import os
import json
from pathlib import Path

# Default API keys (override with environment variables or railway.json)
DEFAULT_CONFIG = {
    "DASHSCOPE_API_KEY": "",
    "OPENAI_API_KEY": ""
}

def load_config():
    """
    Load configuration from railway.json or environment variables.
    
    Returns:
        dict: Configuration dictionary with API keys
    """
    config = DEFAULT_CONFIG.copy()
    
    # Try to load from railway.json
    railway_config_path = Path("railway.json")
    if railway_config_path.exists():
        try:
            with open(railway_config_path, 'r') as f:
                railway_config = json.load(f)
            
            # Update config with values from railway.json
            for key in config.keys():
                if key in railway_config:
                    config[key] = railway_config[key]
        except Exception as e:
            print(f"Error loading railway.json: {e}")
    
    # Override with environment variables if available
    for key in config.keys():
        if key in os.environ:
            config[key] = os.environ[key]
    
    # Set environment variables for libraries that need them
    for key, value in config.items():
        if value:
            os.environ[key] = value
    
    return config

# Load configuration when module is imported
CONFIG = load_config()

# Provide access to individual settings
DASHSCOPE_API_KEY = CONFIG.get("DASHSCOPE_API_KEY", "")
OPENAI_API_KEY = CONFIG.get("OPENAI_API_KEY", "")

# Verify required API keys
if not DASHSCOPE_API_KEY:
    print("Warning: DASHSCOPE_API_KEY is not set")

if not OPENAI_API_KEY:
    print("Warning: OPENAI_API_KEY is not set") 