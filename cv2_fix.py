#!/usr/bin/env python
"""
Script to fix OpenCV library issues by copying the necessary libraries
from Nix store to a location where Python can find them.
"""

import os
import sys
import subprocess
import glob
import shutil
import platform

def main():
    """
    Find and copy OpenCV libraries to make them accessible to the Python environment.
    """
    print("Running OpenCV library fix script...")
    
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
    
    # Libraries to find and copy
    libraries = [
        'libGL.so*',
        'libopencv_*.so*',
        'libcuda*.so*',
        'libnvidia*.so*',
        'libavcodec.so*',
        'libavformat.so*',
        'libavutil.so*',
        'libswscale.so*'
    ]
    
    # Find and copy each library
    for lib_pattern in libraries:
        try:
            # Find all instances of the library in the Nix store
            find_cmd = ["find", "/nix/store", "-name", lib_pattern]
            print(f"Running: {' '.join(find_cmd)}")
            result = subprocess.run(find_cmd, capture_output=True, text=True, check=False)
            
            if result.stdout:
                lib_paths = result.stdout.strip().split('\n')
                print(f"Found {len(lib_paths)} matching libraries for pattern '{lib_pattern}'")
                
                # Copy each library to our lib directory
                for src_path in lib_paths:
                    if os.path.isfile(src_path):
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
            else:
                print(f"No libraries found matching pattern '{lib_pattern}'")
        
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
    
    # Print system information
    print("\nSystem information:")
    print(f"  Platform: {platform.platform()}")
    print(f"  Python: {sys.version}")
    print(f"  LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', 'Not set')}")
    
    return lib_dir

if __name__ == "__main__":
    main() 