"""
Streamlit frontend for Video Commentary Bot
"""

import os
import sys
import subprocess
import glob
import json
import logging

# Create necessary directories at startup
os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output'), exist_ok=True)
os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tmp'), exist_ok=True)
os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lib'), exist_ok=True)

# Setup environment for Railway
if os.environ.get('RAILWAY_PROJECT_ID'):
    print("Running on Railway, setting up environment...")
    
    # Handle Google credentials from environment variable
    google_creds_json = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS_JSON')
    if google_creds_json:
        try:
            creds_dict = json.loads(google_creds_json)
            creds_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'google_credentials.json')
            with open(creds_file, 'w') as f:
                json.dump(creds_dict, f, indent=2)
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = creds_file
            print(f"Google credentials written to {creds_file}")
        except Exception as e:
            print(f"Error setting up Google credentials: {str(e)}")
            
    # Add current directory to path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.append(current_dir)

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
        print("Running on Linux, setting up library paths for OpenCV...")
        
        # Check for existing libGL.so.1
        try:
            result = subprocess.run(['find', '/', '-name', 'libGL.so.1', '-type', 'f'], 
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=10)
            found_paths = result.stdout.strip().split('\n')
            if found_paths and found_paths[0]:
                print(f"Found libGL.so.1 at: {found_paths}")
                # Add these directories to library path
                lib_dirs = [os.path.dirname(path) for path in found_paths if path]
                for lib_dir in lib_dirs:
                    if lib_dir not in current_path:
                        current_path = f"{current_path}:{lib_dir}" if current_path else lib_dir
            else:
                print("Warning: libGL.so.1 not found on system")
        except Exception as e:
            print(f"Error finding libGL.so.1: {str(e)}")
        
        # Common library paths in Nix
        lib_paths = [
            "/nix/store/*/lib",
            "/nix/store/*/mesa*/lib",
            "/nix/store/*/libglvnd*/lib",
            "/nix/store/*/libGL*/lib",
            "/nix/store/*/mesa*/*-drivers*/lib",
            "/nix/var/nix/profiles/default/lib",
            "/usr/lib",
            "/usr/lib/x86_64-linux-gnu",
            "/nix/store/v82vp13jf8lxaljmlar0qpl2z9pfsypi-mesa-24.2.5/lib",
            "/nix/store/lfxm7l9sgp6cxnl16pqglgl9mwhmb8xk-mesa-24.2.5-drivers/lib",
            "/nix/store/hinb0m8if8ic0dgm4h6dr2x3yk1f0qcr-libglvnd-1.7.0/lib"
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

        # Attempt to set LD_PRELOAD for libGL if needed
        try:
            # Look for libGL in the expanded paths
            for path in expanded_paths:
                libgl_path = os.path.join(path, 'libGL.so.1')
                if os.path.exists(libgl_path):
                    os.environ['LD_PRELOAD'] = libgl_path
                    print(f"LD_PRELOAD set to: {libgl_path}")
                    break
        except Exception as e:
            print(f"Error setting LD_PRELOAD: {str(e)}")
    else:
        # Windows-specific OpenCV library path handling
        print("Running on Windows, setting up library paths for OpenCV...")
        try:
            # Check if OpenCV is installed via pip or conda
            import importlib.util
            cv2_spec = importlib.util.find_spec("cv2")
            if cv2_spec:
                cv2_path = os.path.dirname(cv2_spec.origin)
                print(f"Found OpenCV at: {cv2_path}")
                
                # Find potential directories containing DLL files
                opencv_root = os.path.dirname(cv2_path)
                potential_lib_dirs = [
                    os.path.join(opencv_root, 'lib'),
                    os.path.join(opencv_root, 'Library', 'bin'),  # Conda path
                    os.path.join(opencv_root, 'Library', 'lib'),  # Conda path
                    cv2_path,  # Direct module directory
                ]
                
                # Add these directories to PATH environment variable
                for lib_dir in potential_lib_dirs:
                    if os.path.exists(lib_dir) and lib_dir not in os.environ.get('PATH', ''):
                        os.environ['PATH'] = f"{lib_dir}{os.pathsep}{os.environ.get('PATH', '')}"
                        print(f"Added to PATH: {lib_dir}")
                
                # Check for common OpenCV DLLs to verify successful setup
                opencv_dlls = ['opencv_world*.dll', 'libopencv_world*.dll']
                dll_found = False
                for lib_dir in potential_lib_dirs:
                    if os.path.exists(lib_dir):
                        for dll_pattern in opencv_dlls:
                            dll_matches = glob.glob(os.path.join(lib_dir, dll_pattern))
                            if dll_matches:
                                dll_found = True
                                print(f"Found OpenCV DLLs: {dll_matches}")
                                break
                
                if not dll_found:
                    print("Warning: No OpenCV DLLs found in the expected locations")
            else:
                print("Warning: OpenCV module not found")
                
        except Exception as e:
            print(f"Error setting up Windows OpenCV paths: {str(e)}")

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
# Import caption generator
from pipeline.generate_captions import generate_caption

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
    monitor_resources: bool = False,
    generate_captions: bool = False,
    caption_platform: str = "general",
    use_gpt4o_for_captions: bool = False
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
        generate_captions: Whether to generate social media captions
        caption_platform: Platform for captions
        use_gpt4o_for_captions: Whether to use GPT-4o for captions
    
    Returns:
        Path to the final video file and captions data
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
            "final": job_dir / "06_final",
            "captions": job_dir / "captions"  # Add directory for captions
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
        analysis_file = None
        if not skip_analysis:
            resource_monitor.start_step("analyze_frames")
            logger.info("Step 3: Analyzing frames...")
            analysis_result = await analyze_frames(frames_info, steps_dirs["analysis"])
            frames_info = analysis_result
            analysis_file = steps_dirs["analysis"] / "frame_analysis.json"
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
        
        # Optional step: Generate captions
        captions_data = None
        if generate_captions:
            resource_monitor.start_step("generate_captions")
            logger.info("Generating social media captions...")
            
            # Generate captions
            caption_result = await generate_caption(
                analysis_file=analysis_file,
                platform=caption_platform,
                use_gpt4o=use_gpt4o_for_captions
            )
            
            # Save caption result to file
            caption_file = steps_dirs["captions"] / "generated_caption.json"
            with open(caption_file, 'w', encoding='utf-8') as f:
                json.dump(caption_result, f, indent=2)
            
            captions_data = caption_result
            resource_monitor.end_step()
            logger.info("Caption generation complete!")
        
        # Step 7: Cleanup (optional)
        if cleanup_temp and not preserve_temp:
            resource_monitor.start_step("cleanup")
            logger.info("Step 7: Cleaning up temporary files...")
            cleanup_result = cleanup([job_dir], preserve=["final", "captions"])
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
        
        # Return final video path and captions data
        final_video_path = video_result["output_path"]
        logger.info(f"Processing complete! Final video available at: {final_video_path}")
        
        return {
            "video_path": final_video_path,
            "captions": captions_data
        }
        
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
    .copy-btn {
        background-color: #4CAF50;
        color: white;
        padding: 10px 15px;
        border: none;
        border-radius: 5px;
        cursor: pointer;
        font-size: 16px;
        margin: 10px 0;
    }
    .copy-btn:hover {
        background-color: #45a049;
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

# Define social media platform options
PLATFORM_OPTIONS = {
    "General": "general",
    "Instagram": "instagram",
    "TikTok": "tiktok",
    "Twitter/X": "twitter"
}

# Title and description
st.title("🎬 Video Commentary Bot")
st.markdown("""
Generate AI-powered commentary for any video. Just enter a URL and customize your settings!
""")

# Create a single tab instead of two tabs
# Video URL input
video_url = st.text_input(
    "Video URL",
    placeholder="Enter YouTube, Twitter, or direct video URL",
    help="Supports YouTube, Twitter/X, and direct video links"
)

# Create two columns for the main inputs
col1, col2 = st.columns(2)

with col1:
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

# Advanced options in expandable section
with st.expander("Advanced Options"):
    adv_col1, adv_col2 = st.columns(2)
    
    with adv_col1:
        preserve_temp = st.checkbox(
            "Preserve Temporary Files",
            value=False,
            help="Keep intermediate files generated during processing"
        )
        
        skip_analysis = st.checkbox(
            "Skip Frame Analysis",
            value=True,
            help="Skip the AI scene analysis step (faster but less detailed commentary)"
        )
    
    with adv_col2:
        no_cleanup = st.checkbox(
            "Skip Cleanup",
            value=True,
            help="Skip the cleanup step after processing"
        )
        
        monitor_resources = st.checkbox(
            "Monitor Resource Usage",
            value=False,
            help="Track processing time and memory usage for each step"
        )
        
        if monitor_resources:
            st.info("Resource usage report will be saved in the output directory")

    # Social Media Captions section within Advanced Options
    st.markdown("### Social Media Captions")
    st.markdown("Generate engaging captions with hashtags and emojis along with your video")
    
    # Generate captions checkbox
    generate_captions = st.checkbox(
        "Generate Social Media Captions",
        value=False,
        help="Also generate social media captions along with the video"
    )
    
    # Only show caption options if the checkbox is enabled
    if generate_captions:
        caption_col1, caption_col2 = st.columns(2)
        
        with caption_col1:
            # Social media platform selection
            platform_name = st.selectbox(
                "Social Media Platform",
                options=list(PLATFORM_OPTIONS.keys()),
                index=0,
                help="Select the target social media platform for your caption"
            )
            platform = PLATFORM_OPTIONS[platform_name]
        
        with caption_col2:
            # GPT-4o option
            use_gpt4o_for_captions = st.checkbox(
                "Use GPT-4o",
                value=False,
                help="Use OpenAI's GPT-4o model for better quality captions, especially in non-English languages"
            )
            
        st.info("Captions will be generated in the same language as the commentary")

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
                value="InterestingUrduVideos(Youtube logo)",
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
if st.button("Generate Video & Captions"):
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
                base_stages = [
                    "Generating commentary",
                    "Creating audio",
                    "Rendering final video"
                ]
                stages.extend(base_stages)
                
                # Add caption stage if needed
                if generate_captions:
                    stages.append("Generating captions")
                
                # Add finalizing stage
                stages.append("Finalizing")
                
                # Show initial stages information
                if skip_analysis:
                    st.info("Skipping frame analysis for faster processing")
                
                # Log critical parameters
                st.session_state["params"] = {
                    "video_url": video_url,
                    "language": language,
                    "style": style,
                    "skip_analysis": skip_analysis,
                    "monitor_resources": monitor_resources,
                    "generate_captions": generate_captions,
                    "caption_platform": platform if generate_captions else None,
                    "use_gpt4o_for_captions": use_gpt4o_for_captions if generate_captions else False
                }
                
                # Create task for video processing
                task = asyncio.create_task(
                    process_video(
                        video_url=video_url,
                        output_dir=output_dir,
                        language=language,
                        commentary_style=style,
                        preserve_temp=True,  # Keep temp files until user confirms
                        cleanup_temp=False,  # Don't clean up automatically
                        watermark_text=watermark_text if enable_watermark else None,
                        watermark_size=watermark_size,
                        watermark_color=watermark_color,
                        watermark_font=watermark_font,
                        skip_analysis=skip_analysis,
                        monitor_resources=monitor_resources,
                        generate_captions=generate_captions,
                        caption_platform=platform if generate_captions else "general",
                        use_gpt4o_for_captions=use_gpt4o_for_captions if generate_captions else False
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
                
                # Get the results
                if isinstance(result, dict):
                    final_video_path = result["video_path"]
                    captions_data = result.get("captions")
                else:
                    # Backward compatibility with older return format
                    final_video_path = str(result)
                    captions_data = None
                
                # Store the results in session state for persistent access
                st.session_state["final_video_path"] = final_video_path
                st.session_state["captions_data"] = captions_data
                st.session_state["download_ready"] = True
                st.session_state["cleanup_pending"] = True
                
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
                        
                        # Store the alternative video path in session state
                        st.session_state["final_video_path"] = str(most_recent)
                    else:
                        st.error("No video files found in output directory")
                
                # Display captions if they were generated
                if captions_data and generate_captions:
                    st.markdown("---")
                    st.subheader(f"Generated Caption for {platform_name}")
                    
                    # Create a container for the caption with styling
                    st.markdown("<div style='background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin: 10px 0;'>", unsafe_allow_html=True)
                    
                    # Display the caption
                    st.text_area("Caption", captions_data["caption"], height=150, key="video_caption_text")
                    
                    # Display hashtags if available
                    if "hashtags" in captions_data and captions_data["hashtags"]:
                        st.markdown("**Hashtags:**")
                        hashtags_text = " ".join(captions_data["hashtags"])
                        st.code(hashtags_text)
                    
                    # Display emojis if available
                    if "emojis" in captions_data and captions_data["emojis"]:
                        st.markdown("**Emojis used:**")
                        emojis_text = " ".join(captions_data["emojis"])
                        st.code(emojis_text)
                    
                    # JavaScript for copying to clipboard
                    copy_button_html = f"""
                    <button class="copy-btn" onclick="navigator.clipboard.writeText(`{captions_data['caption']}`)">
                        Copy Caption to Clipboard
                    </button>
                    """
                    st.markdown(copy_button_html, unsafe_allow_html=True)
                    
                    # Add info about the model used
                    model_used = captions_data.get("model_used", "Unknown")
                    st.markdown(f"<small>Generated using {model_used}</small>", unsafe_allow_html=True)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                
                # Add confirmation dialog with multiple options
                st.markdown("---")
                st.markdown("### What would you like to do next?")
                
                # Create a styled container for the confirmation dialog
                with st.container():
                    st.markdown("""
                    <div style='background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin: 10px 0;'>
                        <h4>Please choose an option:</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Create two columns for the buttons
                    conf_col1, conf_col2 = st.columns(2)
                    
                    with conf_col1:
                        if st.button("🗑️ Clean Up & Start New", help="Download your files first! This will clean up temporary files and reset the app."):
                            # If user confirms cleanup, clean up temporary files
                            if "final_video_path" in st.session_state:
                                try:
                                    # Extract the job directory from the video path
                                    video_path = st.session_state["final_video_path"]
                                    job_dir = Path(video_path).parents[1]  # Go up two levels to get to the job directory
                                    
                                    # Run cleanup to remove temporary files but preserve the final video
                                    cleanup_result = cleanup([job_dir], preserve=["final", "captions"])
                                    logger.info(f"Cleanup completed after user confirmation: {cleanup_result}")
                                    
                                    # Mark cleanup as completed
                                    st.session_state["cleanup_pending"] = False
                                    st.success("Temporary files cleaned up successfully!")
                                except Exception as e:
                                    logger.error(f"Error during post-confirmation cleanup: {str(e)}")
                                    st.error(f"Error cleaning up: {str(e)}")
                            
                            # Rerun the app to reset the UI
                            time.sleep(1)  # Short delay so the user can see the success message
                            st.experimental_rerun()
                    
                    with conf_col2:
                        if st.button("💾 Keep Files & Start New", help="Keep all temporary files for later use and reset the app."):
                            # Just rerun without cleanup
                            st.session_state["cleanup_pending"] = False
                            st.info("Keeping all files. Starting new session...")
                            time.sleep(1)  # Short delay so the user can see the info message
                            st.experimental_rerun()
                
                # Add explanation about temp files
                st.markdown("""
                <div style='font-size: 0.8em; color: #666; margin-top: 10px;'>
                    <p><strong>What are temporary files?</strong> These include downloaded video, extracted frames, 
                    analysis data, and other intermediate files. The final video and captions will always be preserved.</p>
                </div>
                """, unsafe_allow_html=True)
        
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.exception(e)

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
    
    ### Caption Generation
    
    The caption generation feature uses AI to create engaging social media captions with:
    
    - Relevant hashtags based on video content
    - Appropriate emojis that enhance the message
    - Platform-specific formatting (Instagram, TikTok, Twitter)
    - One-click copy functionality for easy sharing
    - Support for multiple languages
    
    You can choose between two AI models for caption generation:
    - **Qwen**: Faster processing, good for English captions
    - **GPT-4o**: Higher quality, better for non-English languages and more creative captions
    
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