"""
Step 6: Video generation module
Combines video and audio using FFmpeg for local video processing
"""

import os
import logging
import re
from pathlib import Path
from typing import Optional, Dict, List, Union, Tuple
import subprocess
import json
import shutil
import sys
import math

logger = logging.getLogger(__name__)

class VideoGenerator:
    """Handles video generation and audio overlay using FFmpeg."""
    
    def __init__(self, context):
        """Initialize the VideoGenerator.
        
        Args:
            context: The pipeline context
        """
        self._verify_ffmpeg()
        self.context = context
        self.video_input_dir = context.get_step_output_dir(1) / "video"
        self.audio_input_dir = context.get_step_output_dir(5)
        self.output_dir = context.create_step_dir("06_final")
        
    def _verify_ffmpeg(self):
        """Verify FFmpeg is installed and accessible."""
        try:
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
            logger.info("FFmpeg verified successfully")
        except subprocess.CalledProcessError:
            logger.error("FFmpeg not found or not working properly")
            raise RuntimeError("FFmpeg is required but not found. Please install FFmpeg.")
        except FileNotFoundError:
            logger.error("FFmpeg not found in system PATH")
            raise RuntimeError("FFmpeg is not installed or not in PATH. Please install FFmpeg.")
            
    def _ensure_directories(self, video_path: Path, output_path: Path):
        """Ensure all necessary directories exist."""
        video_dir = video_path.parent
        video_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured video directory exists: {video_dir}")

        output_dir = output_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured output directory exists: {output_dir}")

    def _verify_files_exist(self, video_path: str, audio_path: str) -> bool:
        """Verify that input files exist and are valid."""
        video_exists = os.path.exists(video_path)
        audio_exists = os.path.exists(audio_path)
        
        logger.info(f"Video file exists: {video_exists} ({video_path})")
        logger.info(f"Audio file exists: {audio_exists} ({audio_path})")
        
        if video_exists:
            video_size = os.path.getsize(video_path)
            logger.info(f"Video file size: {video_size} bytes")
            if video_size == 0:
                logger.error("Video file is empty")
                return False
        
        if audio_exists:
            audio_size = os.path.getsize(audio_path)
            logger.info(f"Audio file size: {audio_size} bytes")
            if audio_size == 0:
                logger.error("Audio file is empty")
                return False
            
        logger.info(f"Video absolute path: {os.path.abspath(video_path)}")
        logger.info(f"Audio absolute path: {os.path.abspath(audio_path)}")
        
        return video_exists and audio_exists
            
    def _get_audio_duration(self, audio_path: str) -> float:
        """Get duration of audio file in seconds."""
        try:
            cmd = [
                'ffprobe',
                '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                audio_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            duration = float(result.stdout.strip())
            logger.info(f"Audio duration: {duration} seconds")
            return duration
        except Exception as e:
            logger.warning(f"Could not get audio duration: {e}")
            return 0.0

    def generate_video(
        self,
        style: str = "default", 
        watermark_text: str = None,
        watermark_size: int = 36,
        watermark_color: str = "white",
        watermark_font: str = "Arial"
    ) -> Optional[Path]:
        """Generate final video with commentary audio.
        
        Args:
            style: The commentary style
            watermark_text: Optional text to display as watermark
            watermark_size: Font size for watermark text
            watermark_color: Color of the watermark text
            watermark_font: Font to use for watermark
            
        Returns:
            Path to the generated video file
        """
        try:
            # Get video file from context - first video in the video directory
            video_files = list(self.video_input_dir.glob("*.mp4"))
            if not video_files:
                raise FileNotFoundError("No video files found in input directory")
            video_path = str(video_files[0].absolute())
            
            # Get audio file
            audio_path = str((self.audio_input_dir / f"commentary_{style}.wav").absolute())
            
            # Generate output path
            output_path = self.output_dir / f"final_video_{style}.mp4"
            
            # Check if watermark parameters are in context state
            if 'watermark' in self.context.state:
                watermark_text = self.context.state.get('watermark_text', watermark_text)
                watermark_size = self.context.state.get('watermark_size', watermark_size)
                watermark_color = self.context.state.get('watermark_color', watermark_color)
                watermark_font = self.context.state.get('watermark_font', watermark_font)
            
            # Call the generate_video method with all required parameters
            return self._generate_video(
                video_path=video_path,
                audio_path=audio_path,
                output_path=output_path,
                style_name=style,
                watermark_text=watermark_text,
                watermark_size=watermark_size,
                watermark_color=watermark_color,
                watermark_font=watermark_font
            )
            
        except Exception as e:
            logger.error(f"Error generating video: {str(e)}")
            return None

    def _generate_video(
        self,
        video_path: str,
        audio_path: str,
        output_path: Path,
        style_name: str = None,
        watermark_text: str = None,
        watermark_size: int = 36,
        watermark_color: str = "white",
        watermark_font: str = "Arial"
    ) -> Optional[Path]:
        """Generate final video with optimized processing."""
        try:
            video_path = str(Path(video_path).absolute())
            audio_path = str(Path(audio_path).absolute())
            output_path = Path(output_path).absolute()
            
            self._ensure_directories(Path(video_path), output_path)
            
            if not self._verify_files_exist(video_path, audio_path):
                raise FileNotFoundError("Input video or audio file not found or empty")
            
            commentary_duration = self._get_audio_duration(audio_path)
            logger.info(f"Commentary duration: {commentary_duration} seconds")
            
            # Get video dimensions
            probe_cmd = [
                'ffprobe',
                '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=width,height',
                '-of', 'json',
                video_path
            ]
            probe_result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
            video_info = json.loads(probe_result.stdout)
            original_width = int(video_info['streams'][0]['width'])
            original_height = int(video_info['streams'][0]['height'])
            
            # Calculate padding while maintaining aspect ratio
            target_width = 720
            target_height = 1280
            scale_width = target_width
            scale_height = int(original_height * (target_width / original_width))
            
            if scale_height > target_height:
                scale_height = target_height
                scale_width = int(original_width * (target_height / original_height))
            
            # Calculate padding
            pad_x = (target_width - scale_width) // 2
            pad_y = (target_height - scale_height) // 2
            
            # Build filter complex with optional watermark
            if watermark_text:
                logger.info(f"Adding watermark text: '{watermark_text}', size: {watermark_size}, color: {watermark_color}, font: {watermark_font}")
                
                # Escape special characters in watermark text for drawtext filter
                escaped_text = watermark_text.replace("'", "\\'").replace(":", "\\:")
                
                # Add drawtext filter for watermark
                filter_complex = (
                    f'[0:v]scale={scale_width}:{scale_height}:flags=lanczos,'
                    f'pad={target_width}:{target_height}:{pad_x}:{pad_y}:color=black,'
                    f'drawtext=text=\'{escaped_text}\':'
                    f'fontfile={watermark_font}:'
                    f'fontsize={watermark_size}:'
                    f'fontcolor={watermark_color}:'
                    f'x=(w-text_w)/2:y=(h-text_h)/2:'
                    f'shadowcolor=black:shadowx=2:shadowy=2[v];'
                    f'[0:a]volume=1[original];[1:a]volume=0.7[commentary];'
                    f'[original]volume=0.3:enable=\'between(t,0,{commentary_duration})\':eval=frame[ducked];'
                    f'[ducked][commentary]amix=inputs=2:duration=longest[audio]'
                )
            else:
                # Standard filter without watermark
                filter_complex = (
                    f'[0:v]scale={scale_width}:{scale_height}:flags=lanczos,'
                    f'pad={target_width}:{target_height}:{pad_x}:{pad_y}:color=black[v];'
                    f'[0:a]volume=1[original];[1:a]volume=0.7[commentary];'
                    f'[original]volume=0.3:enable=\'between(t,0,{commentary_duration})\':eval=frame[ducked];'
                    f'[ducked][commentary]amix=inputs=2:duration=longest[audio]'
                )
            
            # Build FFmpeg command
            cmd = [
                'ffmpeg',
                '-y',  # Overwrite output
                '-i', video_path,
                '-i', audio_path,
                '-filter_complex', filter_complex,
                '-map', '[v]',  # Always map to [v] output from filter complex
                '-map', '[audio]',
                '-c:v', 'libx264',
                '-preset', 'medium',
                '-crf', '23',
                '-maxrate', '4M',
                '-bufsize', '8M',
                '-c:a', 'aac',
                '-b:a', '192k',
                '-ac', '2',
                '-ar', '44100',
                '-max_muxing_queue_size', '2048',
                '-movflags', '+faststart',
                str(output_path)
            ]
            
            logger.info(f"Current working directory: {os.getcwd()}")
            logger.info(f"FFmpeg command: {' '.join(cmd)}")
            
            # Run FFmpeg with improved error handling
            try:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True,
                    bufsize=1
                )
                
                try:
                    stdout, stderr = process.communicate(timeout=600)  # Increased timeout to 10 minutes
                    if process.returncode == 0:
                        if os.path.exists(str(output_path)) and os.path.getsize(str(output_path)) > 0:
                            logger.info(f"Video generated successfully: {output_path}")
                            return output_path
                        else:
                            logger.error("Output file is empty or does not exist")
                            if stderr:
                                logger.error(f"FFmpeg stderr: {stderr}")
                            return None
                    else:
                        logger.error(f"FFmpeg error: {stderr}")
                        return None
                except subprocess.TimeoutExpired:
                    process.kill()
                    logger.error("FFmpeg process timed out after 10 minutes")
                    return None
                    
            except subprocess.CalledProcessError as e:
                logger.error(f"FFmpeg command failed with error: {e.stderr}")
                return None
            except Exception as e:
                logger.error(f"Error generating video: {str(e)}")
                return None
                
        except Exception as e:
            logger.error(f"Error generating video: {str(e)}")
            return None

    @classmethod
    def execute_step(cls, context, preserve=None):
        """
        Execute the video generation step.
        
        Args:
            context: The pipeline context
            preserve: Optional list of temporary directories to preserve
            
        Returns:
            Updated context object
        """
        generator = cls(context)
        logger.info(f"Generating final video with audio...")
        
        # Set default styles if not specified
        if 'output_styles' not in context.state:
            context.state['output_styles'] = ['default']
        
        # Generate videos for each style
        final_video_paths = {}
        for style in context.state['output_styles']:
            logger.info(f"Generating video for style: {style}")
            # Get watermark parameters from context state
            watermark_text = context.state.get('watermark_text', None)
            watermark_size = context.state.get('watermark_size', 36)
            watermark_color = context.state.get('watermark_color', "white")
            watermark_font = context.state.get('watermark_font', "Arial")
            
            final_video_path = generator.generate_video(
                style, 
                watermark_text, 
                watermark_size, 
                watermark_color, 
                watermark_font
            )
            if final_video_path:
                final_video_paths[style] = final_video_path
            
        # Update context
        context.state['final_video_paths'] = final_video_paths
        return context

# Direct function implementation for process_video.py
def execute_step(video_path: str, audio_path: str, output_dir: Path, 
                 watermark_text: str = None, watermark_size: int = 36, 
                 watermark_color: str = "white", watermark_font: str = "Arial") -> dict:
    """Generate final video with audio.
    
    Args:
        video_path: Path to the input video file
        audio_path: Path to the audio file to add
        output_dir: Output directory for final video
        watermark_text: Optional text to display as watermark
        watermark_size: Font size for watermark text
        watermark_color: Color of the watermark text
        watermark_font: Font to use for watermark
        
    Returns:
        Dictionary with output path
    """
    try:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create simplified context for VideoGenerator
        class SimpleContext:
            def __init__(self, language, watermark_settings=None):
                self.state = {
                    "language": language
                }
                # Add watermark settings to state if provided
                if watermark_settings:
                    self.state.update(watermark_settings)
                
            def get_step_output_dir(self, step_num):
                return Path("./")
                
            def create_step_dir(self, name):
                return output_dir
        
        # Determine language from audio file name or use English by default
        language = "en"
        if "ur" in str(audio_path):
            language = "ur"
        elif "bn" in str(audio_path):
            language = "bn"
        elif "hi" in str(audio_path):
            language = "hi"
        elif "tr" in str(audio_path):
            language = "tr"
            
        # Create watermark settings if text is provided
        watermark_settings = {}
        if watermark_text:
            watermark_settings = {
                "watermark_text": watermark_text,
                "watermark_size": watermark_size,
                "watermark_color": watermark_color,
                "watermark_font": watermark_font
            }
            
        # Create a simplified context with watermark settings
        context = SimpleContext(language, watermark_settings)
        
        # Check if output path exists
        output_file = output_dir / "final_video.mp4"
        
        # Call the implementation directly
        generator = VideoGenerator(context)
        
        # Explicitly call with required parameters
        result = generator._generate_video(
            video_path=video_path,
            audio_path=audio_path,
            output_path=output_file,
            watermark_text=watermark_text,
            watermark_size=watermark_size,
            watermark_color=watermark_color,
            watermark_font=watermark_font
        )
        
        if result:
            logger.info(f"Successfully generated video: {result}")
            return {"output_path": str(result)}
        else:
            logger.error("Failed to generate video")
            return {"error": "Failed to generate video"}
            
    except Exception as e:
        logger.error(f"Error generating video: {str(e)}")
        return {"error": str(e)} 