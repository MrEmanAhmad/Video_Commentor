"""
Streamlit frontend for Video Commentary Bot
"""

import os
import sys
import subprocess
import glob
import json
import logging

# === FIX FOR OPENCV LIBRARY PATHS ===
# Set LD_LIBRARY_PATH for OpenCV before any imports
def setup_opencv_libraries():
    # Create a lib directory if it doesn't exist
    lib_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lib')
    os.makedirs(lib_dir, exist_ok=True)
    
    # Add this directory to LD_LIBRARY_PATH
    current_path = os.environ.get('LD_LIBRARY_PATH', '')
    if lib_dir not in current_path:
        if current_path:
            os.environ['LD_LIBRARY_PATH'] = f"{current_path}:{lib_dir}"
        else:
            os.environ['LD_LIBRARY_PATH'] = lib_dir
            
    # On Railway/Linux, make sure Nix libraries are in LD_LIBRARY_PATH
    if os.name != 'nt':  # Not Windows
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
        if expanded_paths:
            new_path = ':'.join(expanded_paths)
            if current_path:
                full_path = f"{current_path}:{new_path}"
            else:
                full_path = new_path
            
            os.environ['LD_LIBRARY_PATH'] = full_path
            print(f"LD_LIBRARY_PATH set to: {full_path}")

# Run the library path setup before any imports
setup_opencv_libraries()

# Now import time and other modules
import time
import asyncio
import streamlit as st
from pathlib import Path
import hashlib
import psutil
import tracemalloc

# Add the current directory to path to ensure imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import pipeline steps
from pipeline.Step_1_download_video import execute_step as download_video, download_from_url
from pipeline.Step_2_extract_frames import execute_step as extract_frames
from pipeline.Step_3_analyze_frames import execute_step as analyze_frames
from pipeline.Step_4_generate_commentary import execute_step as generate_commentary
from pipeline.Step_5_generate_audio import execute_step as generate_audio
from pipeline.Step_6_video_generation import execute_step as generate_video
from pipeline.Step_7_cleanup import execute_step as cleanup

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class ResourceMonitor:
    """Monitors resource usage during video processing."""
    
    def __init__(self, enabled=True):
        """Initialize the resource monitor.
        
        Args:
            enabled: Whether to enable monitoring
        """
        self.enabled = enabled
        self.steps = {}
        self.current_step = None
        self.start_time = None
        self.process = psutil.Process()
        
        if enabled:
            # Start memory tracking
            tracemalloc.start()
    
    def start_step(self, step_name):
        """Start monitoring a step."""
        if not self.enabled:
            return
            
        self.current_step = step_name
        self.start_time = time.time()
        _, self.start_memory = tracemalloc.get_traced_memory()
    
    def end_step(self):
        """End monitoring the current step."""
        if not self.enabled or not self.current_step:
            return
            
        elapsed = time.time() - self.start_time
        _, current_memory = tracemalloc.get_traced_memory()
        memory_used = current_memory - self.start_memory
        
        self.steps[self.current_step] = {
            'time': elapsed,
            'memory': memory_used,
            'cpu': self.process.cpu_percent()
        }
        
        logger.info(f"Step {self.current_step} completed in {elapsed:.2f}s, memory: {memory_used / 1024 / 1024:.2f}MB")
        self.current_step = None
    
    def get_summary(self):
        """Get a summary of resource usage."""
        if not self.enabled:
            return {}
            
        total_time = sum(step['time'] for step in self.steps.values())
        total_memory = sum(step['memory'] for step in self.steps.values())
        
        return {
            'steps': self.steps,
            'total_time': total_time,
            'total_memory': total_memory,
            'summary': {
                'time_seconds': f"{total_time:.2f}",
                'memory_mb': f"{total_memory / 1024 / 1024:.2f}"
            }
        }
    
    def stop(self):
        """Stop the resource monitor."""
        if self.enabled:
            tracemalloc.stop()
            
    def save_report(self, output_path):
        """Save a resource usage report to a file."""
        if not self.enabled:
            return
            
        summary = self.get_summary()
        
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Resource report saved to {output_path}")

async def process_video(
    video_url: str, 
    output_dir: str = "./output",
    language: str = "en",
    commentary_style: str = "descriptive",
    preserve_temp: bool = False,
    cleanup_temp: bool = True,
    watermark_text: str = None,
    watermark_size: int = 36,
    watermark_color: str = "white",
    watermark_font: str = "Arial",
    skip_analysis: bool = False,
    monitor_resources: bool = False
):
    """
    Process a video through the entire pipeline.
    
    Args:
        video_url: URL to the video to process
        output_dir: Directory to save output files
        language: Language for commentary (ISO code)
        commentary_style: Style for commentary generation
        preserve_temp: Whether to preserve temporary files
        cleanup_temp: Whether to run cleanup step
        watermark_text: Optional text to display as watermark
        watermark_size: Font size for watermark text
        watermark_color: Color of the watermark text
        watermark_font: Font to use for watermark
        skip_analysis: Whether to skip the frame analysis step
        monitor_resources: Whether to monitor resource usage
    
    Returns:
        Path to the final video file
    """
    # Initialize resource monitor if requested
    resource_monitor = ResourceMonitor(enabled=monitor_resources)
    
    try:
        # Prepare output directories
        base_output_dir = Path(output_dir)
        base_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create unique job ID based on input video
        job_id = hashlib.md5(video_url.encode()).hexdigest()[:8]
        job_dir = base_output_dir / job_id
        
        # Create step-specific directories
        steps_dirs = {
            "download": job_dir / "01_download",
            "frames": job_dir / "02_frames",
            "analysis": job_dir / "03_analysis",
            "commentary": job_dir / "04_commentary",
            "audio": job_dir / "05_audio",
            "final": job_dir / "06_final"
        }
        
        # Create all directories
        for dir_path in steps_dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Processing video: {video_url}")
        logger.info(f"Job ID: {job_id}")
        logger.info(f"Skip frame analysis: {skip_analysis}")
        
        # Step 1: Download Video
        resource_monitor.start_step("download_video")
        logger.info("Step 1: Downloading video...")
        # Use the async download function which returns the video path directly
        video_path = await download_from_url(video_url, steps_dirs["download"])
        resource_monitor.end_step()
        
        # Load the metadata from the JSON file
        metadata_file = steps_dirs["download"] / "video_metadata.json"
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # Add language to metadata
        metadata['language'] = language
        metadata['style'] = commentary_style
            
        # Step 2: Extract Frames
        resource_monitor.start_step("extract_frames")
        logger.info("Step 2: Extracting frames...")
        frames_result = extract_frames(video_path, steps_dirs["frames"])
        if "error" in frames_result:
            raise Exception(f"Frame extraction failed: {frames_result['error']}")
        resource_monitor.end_step()
        
        # Initialize frames_info with basic data
        frames_info = {
            "frames_dir": str(steps_dirs["frames"]),
            "metadata": metadata,
            "frames": frames_result["frames"] if "frames" in frames_result else []
        }
        
        # Step 3: Analyze Frames (optional)
        if not skip_analysis:
            resource_monitor.start_step("analyze_frames")
            logger.info("Step 3: Analyzing frames...")
            analysis_result = await analyze_frames(frames_info, steps_dirs["analysis"])
            frames_info = analysis_result
            resource_monitor.end_step()
        else:
            logger.info("Step 3: Skipping frame analysis as requested by skip_analysis flag...")
            # Create a basic analysis result file so downstream steps don't fail
            basic_analysis = {
                "frames_dir": str(steps_dirs["frames"]),
                "metadata": metadata,
                "frames": frames_info["frames"],
                "analysis_complete": False,
                "is_basic_analysis": True,
                "skipped_by_user": True
            }
            analysis_file = steps_dirs["analysis"] / "frame_analysis.json"
            logger.info(f"Creating basic analysis file at {analysis_file}")
            with open(analysis_file, 'w', encoding='utf-8') as f:
                json.dump(basic_analysis, f, indent=2)
            frames_info = basic_analysis
            logger.info("Basic analysis complete - no API calls were made")
        
        # Step 4: Generate Commentary
        resource_monitor.start_step("generate_commentary")
        logger.info("Step 4: Generating commentary...")
        commentary_result = await generate_commentary(
            frames_info=frames_info,
            output_dir=steps_dirs["commentary"],
            content_type=commentary_style
        )
        resource_monitor.end_step()
        
        # Step 5: Generate Audio
        resource_monitor.start_step("generate_audio")
        logger.info("Step 5: Generating audio...")
        audio_file = await generate_audio(
            frames_info=frames_info, 
            output_dir=steps_dirs["audio"],
            style=commentary_style
        )
        resource_monitor.end_step()
        
        # Step 6: Generate Final Video
        resource_monitor.start_step("generate_video")
        logger.info("Step 6: Generating final video...")
        # Log watermark info if provided
        if watermark_text:
            logger.info(f"Adding watermark: '{watermark_text}', size: {watermark_size}, color: {watermark_color}, font: {watermark_font}")
            
        video_result = generate_video(
            video_path=video_path,
            audio_path=audio_file,
            output_dir=steps_dirs["final"],
            watermark_text=watermark_text,
            watermark_size=watermark_size,
            watermark_color=watermark_color,
            watermark_font=watermark_font
        )
        resource_monitor.end_step()
        
        if "error" in video_result:
            raise Exception(f"Video generation failed: {video_result['error']}")
        
        # Step 7: Cleanup (optional)
        if cleanup_temp and not preserve_temp:
            resource_monitor.start_step("cleanup")
            logger.info("Step 7: Cleaning up temporary files...")
            cleanup_result = cleanup([job_dir], preserve=["final"])
            logger.info(f"Cleanup completed: {cleanup_result}")
            resource_monitor.end_step()
        
        # Save resource report if monitoring was enabled
        if monitor_resources:
            report_path = steps_dirs["final"] / "resource_report.json"
            try:
                # Ensure output directory exists
                report_path.parent.mkdir(parents=True, exist_ok=True)
                resource_monitor.save_report(str(report_path))
                
                # Add summary to log
                summary = resource_monitor.get_summary()['summary']
                logger.info(f"Process completed in {summary['time_seconds']}s using {summary['memory_mb']}MB of memory")
                
                # Log difference between skip_analysis and regular mode
                skip_status = "skipped" if skip_analysis else "performed"
                logger.info(f"Frame analysis was {skip_status}")
            except Exception as e:
                logger.warning(f"Could not save resource report: {str(e)}")
        
        # Stop resource monitoring
        resource_monitor.stop()
        
        # Return final video path
        final_video_path = video_result["output_path"]
        logger.info(f"Processing complete! Final video available at: {final_video_path}")
        
        return final_video_path
        
    except Exception as e:
        # Stop resource monitoring on error
        resource_monitor.stop()
        logger.error(f"Error processing video: {str(e)}")
        return {"error": str(e)}

# Set page configuration
st.set_page_config(
    page_title="Video Commentary Bot",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stButton button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        padding: 10px;
        font-size: 16px;
        border-radius: 5px;
    }
    .stProgress > div > div {
        background-color: #4CAF50;
    }
    .output-video {
        width: 100%;
        border-radius: 10px;
    }
    .watermark-options {
        background-color: #f8f9fa;
        padding: 10px;
        border-radius: 5px;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Define language options
LANGUAGE_OPTIONS = {
    "English (US)": "en",
    "English (UK)": "en-GB",
    "English (Australia)": "en-AU", 
    "English (India)": "en-IN",
    "Urdu": "ur",
    "Bengali": "bn",
    "Hindi": "hi",
    "Turkish": "tr",
}

# Define commentary style options based on supported backend styles
STYLE_OPTIONS = [
    "news", "funny", "nature", "documentary", "informative"
]

# Define font options
FONT_OPTIONS = [
    "Arial", "Times New Roman", "Verdana", "Comic Sans MS", "Impact", 
    "Courier New", "Tahoma", "Trebuchet MS", "Georgia", "Garamond"
]

# Define color options with nice predefined colors
COLOR_OPTIONS = {
    "White": "white",
    "Black": "black", 
    "Red": "red",
    "Blue": "blue",
    "Green": "green",
    "Yellow": "yellow",
    "Cyan": "cyan",
    "Magenta": "magenta",
    "Orange": "orange",
    "Purple": "purple",
    "Pink": "pink",
    "Gold": "gold",
    "Silver": "silver",
    "Lime": "lime",
    "Teal": "teal"
}

# Title and description
st.title("🎬 Video Commentary Bot")
st.markdown("""
Generate AI-powered commentary for any video. Just enter a URL and customize your settings!
""")

# Create two columns for inputs
col1, col2 = st.columns(2)

with col1:
    # Video URL input
    video_url = st.text_input(
        "Video URL",
        placeholder="Enter YouTube, Twitter, or direct video URL",
        help="Supports YouTube, Twitter/X, and direct video links"
    )
    
    # Commentary language
    lang_name = st.selectbox(
        "Commentary Language",
        options=list(LANGUAGE_OPTIONS.keys()),
        index=0,
        help="Select the language for your commentary. Full support for English, Urdu, Bengali, Hindi, and Turkish."
    )
    language = LANGUAGE_OPTIONS[lang_name]
    
    # Commentary style
    style = st.selectbox(
        "Commentary Style",
        options=STYLE_OPTIONS,
        index=0,
        help="Select the style of commentary to generate. Options include news reporting, funny commentary, nature documentary, or informative explanation."
    )

with col2:
    # Output directory
    output_dir = st.text_input(
        "Output Directory",
        value="./output",
        help="Directory where processed videos will be saved"
    )
    
    # Advanced options
    with st.expander("Advanced Options"):
        col1, col2 = st.columns(2)
        
        with col1:
            preserve_temp = st.checkbox(
                "Preserve Temporary Files",
                value=True,  # Changed to true by default to avoid cleanup issues
                help="Keep intermediate files generated during processing"
            )
            
            skip_analysis = st.checkbox(
                "Skip Frame Analysis",
                value=False,
                help="Skip the AI scene analysis step (faster but less detailed commentary)"
            )
        
        with col2:
            no_cleanup = st.checkbox(
                "Skip Cleanup",
                value=True,  # Changed to true by default to avoid cleanup issues
                help="Skip the cleanup step after processing"
            )
            
            monitor_resources = st.checkbox(
                "Monitor Resource Usage",
                value=False,
                help="Track processing time and memory usage for each step"
            )
            
            if monitor_resources:
                st.info("Resource usage report will be saved in the output directory")

    # Watermark options
    with st.expander("Watermark Options"):
        enable_watermark = st.checkbox(
            "Add Watermark Text",
            value=False,
            help="Add a text watermark to the center of the video"
        )
        
        if enable_watermark:
            # Container with background color for watermark options
            with st.container():
                st.markdown('<div class="watermark-options">', unsafe_allow_html=True)
                
                # Watermark text
                watermark_text = st.text_input(
                    "Watermark Text",
                    value="© Video Commentary Bot",
                    help="Text to display as watermark in the center of the video"
                )
                
                # Create two columns for size and color
                wm_col1, wm_col2 = st.columns(2)
                
                with wm_col1:
                    # Watermark size
                    watermark_size = st.slider(
                        "Font Size",
                        min_value=12,
                        max_value=72,
                        value=36,
                        step=2,
                        help="Size of the watermark text"
                    )
                
                with wm_col2:
                    # Watermark color
                    color_name = st.selectbox(
                        "Text Color",
                        options=list(COLOR_OPTIONS.keys()),
                        index=0,
                        help="Color of the watermark text"
                    )
                    watermark_color = COLOR_OPTIONS[color_name]
                
                # Watermark font
                watermark_font = st.selectbox(
                    "Font",
                    options=FONT_OPTIONS,
                    index=0,
                    help="Font to use for the watermark text"
                )
                
                # Preview of watermark
                st.markdown(
                    f"<div style='text-align:center; padding:10px; margin:10px 0; "
                    f"background-color:rgba(0,0,0,0.5); color:{watermark_color}; "
                    f"font-family:{watermark_font}; font-size:{watermark_size}px;'>"
                    f"{watermark_text}</div>", 
                    unsafe_allow_html=True
                )
                
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            watermark_text = None
            watermark_size = 36
            watermark_color = "white"
            watermark_font = "Arial"

# Submit button
if st.button("Generate Commentary"):
    if not video_url:
        st.error("Please enter a video URL")
    else:
        # Create a progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Create a placeholder for the final video
        video_placeholder = st.empty()
        
        try:
            # Define async function to update progress while processing
            async def process_with_progress():
                # Initialize progress stages
                stages = [
                    "Downloading video",
                    "Extracting frames"
                ]
                
                # Only include analysis stage if not skipped
                if not skip_analysis:
                    stages.append("Analyzing frames")
                
                # Add remaining stages
                stages.extend([
                    "Generating commentary",
                    "Creating audio",
                    "Rendering final video",
                    "Finalizing"
                ])
                
                # Show initial stages information
                if skip_analysis:
                    st.info("Skipping frame analysis for faster processing")
                
                # Log critical parameters
                st.session_state["params"] = {
                    "video_url": video_url,
                    "language": language,
                    "style": style,
                    "skip_analysis": skip_analysis,
                    "monitor_resources": monitor_resources
                }
                
                # Create task for video processing
                task = asyncio.create_task(
                    process_video(
                        video_url=video_url,
                        output_dir=output_dir,
                        language=language,
                        commentary_style=style,
                        preserve_temp=True,  # Always preserve temp files for Streamlit
                        cleanup_temp=False,   # Disable cleanup to ensure files persist
                        watermark_text=watermark_text if enable_watermark else None,
                        watermark_size=watermark_size,
                        watermark_color=watermark_color,
                        watermark_font=watermark_font,
                        skip_analysis=skip_analysis,
                        monitor_resources=monitor_resources
                    )
                )
                
                # Update progress while task is running
                for i, stage in enumerate(stages):
                    progress = (i / len(stages))
                    progress_bar.progress(progress)
                    status_text.text(f"Status: {stage}...")
                    
                    # Wait a bit to simulate progress
                    await asyncio.sleep(1)
                    
                    # Check if task is done, and if so, break
                    if task.done():
                        break
                
                # Wait for task to complete if it hasn't already
                result = await task
                
                # Final progress update
                progress_bar.progress(1.0)
                status_text.text("Status: Completed!")
                
                return result
            
            # Run the async function
            result = asyncio.run(process_with_progress())
            
            # Check if result is a dictionary with an error
            if isinstance(result, dict) and "error" in result:
                st.error(f"Error processing video: {result['error']}")
            else:
                # Display success message
                st.success(f"Processing completed successfully!")
                
                # Get the final video path
                final_video_path = str(result)
                
                if os.path.exists(final_video_path):
                    # Create a download button for the video
                    with open(final_video_path, "rb") as file:
                        file_name = os.path.basename(final_video_path)
                        st.download_button(
                            label="Download Video",
                            data=file,
                            file_name=file_name,
                            mime="video/mp4"
                        )
                    
                    # Display the video in the app
                    video_placeholder.video(final_video_path)
                else:
                    # If the specific path doesn't exist, try to find the video in output directories
                    st.warning(f"Video file not found at path: {final_video_path}")
                    
                    # Search for final video files
                    output_path = Path(output_dir)
                    final_videos = []
                    for job_dir in output_path.glob("*"):
                        final_dir = job_dir / "06_final"
                        if final_dir.exists():
                            videos = list(final_dir.glob("*.mp4"))
                            final_videos.extend(videos)
                    
                    if final_videos:
                        # Get most recent video
                        most_recent = max(final_videos, key=os.path.getmtime)
                        st.info(f"Found alternative video file: {most_recent}")
                        
                        # Create download button for the found video
                        with open(most_recent, "rb") as file:
                            file_name = os.path.basename(most_recent)
                            st.download_button(
                                label="Download Video",
                                data=file,
                                file_name=file_name,
                                mime="video/mp4"
                            )
                        
                        # Display the video in the app
                        video_placeholder.video(str(most_recent))
                    else:
                        st.error("No video files found in output directory")
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# Add information about the pipeline
with st.expander("About the Pipeline"):
    st.markdown("""
    ### Video Commentary Bot Pipeline
    
    This tool processes videos through a 6-step pipeline:
    
    1. **Download Video**: Handles YouTube, Twitter, and direct video URLs
    2. **Extract Frames**: Uses OpenCV to extract key frames from the video
    3. **Analyze Frames**: Uses Qwen Vision API for detailed scene understanding (optional)
    4. **Generate Commentary**: Uses Qwen LLMs for English commentary and translates to other languages if needed
    5. **Generate Audio**: Uses Google TTS for speech synthesis with extensive voice options
    6. **Generate Video**: Combines the original video with the generated audio
    
    ### Features
    
    - Multi-language support with high-quality TTS voices
    - Different commentary styles to match your content
    - Watermark option for adding custom text to videos
    - Optional AI scene detection for context-aware commentary
    
    ### Performance Options
    
    - **Skip Frame Analysis**: Bypass the AI scene analysis for faster processing but less detailed commentary
    
    ### Supported Languages
    
    - English (US, UK, Australia, India variants)
    - Urdu
    - Bengali
    - Hindi
    - Turkish
    
    ### Supported Commentary Styles
    
    - **News**: Formal journalistic style with objective reporting
    - **Funny**: Humorous commentary with witty observations
    - **Nature**: Nature documentary style with descriptive language
    - **Documentary**: Balanced educational and entertaining narration
    - **Informative**: Clear explanations focused on educational content
    """)

# Instructions for first-time setup
with st.sidebar:
    st.header("First-time Setup")
    st.markdown("""
    Before using this application, make sure you have:
    
    1. Installed all required dependencies:
    ```
    pip install streamlit google-cloud-texttospeech opencv-python yt-dlp openai dashscope tqdm
    ```
    
    2. Set up the following API keys as environment variables:
    ```
    OPENAI_API_KEY - For translations
    DASHSCOPE_API_KEY - For Qwen models
    ```
    
    3. Added your Google Cloud credentials JSON file to the root directory
    """)
    
    # Add a button to check environment setup
    if st.button("Check Environment Setup"):
        # Check installed packages
        packages = ["streamlit", "google-cloud-texttospeech", "opencv-python", "yt-dlp", "openai", "dashscope"]
        missing_packages = []
        
        for package in packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            st.error(f"Missing packages: {', '.join(missing_packages)}")
        else:
            st.success("All required packages are installed")
        
        # Check API keys
        if not os.getenv("OPENAI_API_KEY"):
            st.warning("OPENAI_API_KEY environment variable not set")
        else:
            st.success("OPENAI_API_KEY environment variable is set")
            
        if not os.getenv("DASHSCOPE_API_KEY"):
            st.warning("DASHSCOPE_API_KEY environment variable not set")
        else:
            st.success("DASHSCOPE_API_KEY environment variable is set")
        
        # Check Google credentials file
        if not os.path.exists("google_credentials.json"):
            st.warning("google_credentials.json file not found in the root directory")
        else:
            st.success("google_credentials.json file found")
            
    st.markdown("---")
    st.markdown("Made with ❤️ using Streamlit and AI") 