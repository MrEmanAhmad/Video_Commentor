"""
Step 6: Video generation module
Combines video and audio using FFmpeg for local video processing
"""

import os
import logging
import re
from pathlib import Path
from typing import Optional, Dict
import subprocess
import json
import shutil

logger = logging.getLogger(__name__)

class VideoGenerator:
    """Handles video generation and audio overlay using FFmpeg."""
    
    def __init__(self):
        """Initialize the VideoGenerator."""
        self._verify_ffmpeg()
        
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
        # Create temp video directory
        video_dir = video_path.parent
        video_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured video directory exists: {video_dir}")

        # Create output directory
        output_dir = output_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured output directory exists: {output_dir}")

    def _verify_files_exist(self, video_path: str, audio_path: str) -> bool:
        """Verify that input files exist."""
        video_exists = os.path.exists(video_path)
        audio_exists = os.path.exists(audio_path)
        
        logger.info(f"Video file exists: {video_exists} ({video_path})")
        logger.info(f"Audio file exists: {audio_exists} ({audio_path})")
        
        # Add detailed file information
        if video_exists:
            video_size = os.path.getsize(video_path)
            logger.info(f"Video file size: {video_size} bytes")
        if audio_exists:
            audio_size = os.path.getsize(audio_path)
            logger.info(f"Audio file size: {audio_size} bytes")
            
        # Log absolute paths
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
            return float(result.stdout.strip())
        except Exception as e:
            logger.warning(f"Could not get audio duration: {e}")
            return 0.0

    async def generate_video(
        self,
        video_path: str,
        audio_path: str,
        output_path: Path,
        style_name: str = None
    ) -> Optional[Path]:
        """
        Generate final video with optimized processing.
        
        Args:
            video_path: Path to the input video
            audio_path: Path to the audio file
            output_path: Path to save the final video
            style_name: Name of the commentary style used
            
        Returns:
            Path to generated video if successful, None otherwise
        """
        try:
            # Convert paths to absolute paths
            video_path = str(Path(video_path).absolute())
            audio_path = str(Path(audio_path).absolute())
            output_path = Path(output_path).absolute()
            
            # Ensure directories exist
            self._ensure_directories(Path(video_path), output_path)
            
            # Verify input files exist
            if not self._verify_files_exist(video_path, audio_path):
                raise FileNotFoundError("Input video or audio file not found")
            
            # Get commentary duration
            commentary_duration = self._get_audio_duration(audio_path)
            logger.info(f"Commentary duration: {commentary_duration} seconds")
            
            # Build FFmpeg command with minimal complexity
            cmd = [
                'ffmpeg',
                '-y',
                '-i', video_path,
                '-i', audio_path,
                '-filter_complex',
                '[0:a]volume=1[original];'  # Original audio at full volume
                '[1:a]volume=0.7[commentary];'  # Commentary slightly lower
                f'[original]volume=0.3:enable=\'between(t,0,{commentary_duration})\':eval=frame[ducked];'  # Duck during commentary
                '[ducked][commentary]amix=inputs=2:duration=longest[audio]',  # Mix both audio streams
                '-vf', 'pad=720:1280:0:0:color=white',
                '-c:v', 'libx264',
                '-preset', 'ultrafast',
                '-crf', '28',
                '-c:a', 'aac',
                '-b:a', '192k',
                '-map', '0:v',
                '-map', '[audio]',
                '-max_muxing_queue_size', '1024',
                str(output_path)
            ]
            
            # Log command and current directory
            logger.info(f"Current working directory: {os.getcwd()}")
            logger.info(f"FFmpeg command: {' '.join(cmd)}")
            
            # Run FFmpeg with stderr capture
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=True
                )
                
                if result.returncode == 0:
                    logger.info(f"Video generated successfully: {output_path}")
                    return output_path
                else:
                    logger.error(f"FFmpeg error: {result.stderr}")
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

async def execute_step(
    video_file: Path,
    audio_file: Path,
    output_dir: Path,
    style_name: str
) -> Optional[Path]:
    """
    Execute video generation step.
    
    Args:
        video_file: Path to the input video file
        audio_file: Path to the generated audio file
        output_dir: Directory to save generated video
        style_name: Name of the commentary style used
        
    Returns:
        Path to the generated video if successful, None otherwise
    """
    logger.debug("Step 6: Generating final video...")
    
    try:
        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured output directory exists: {output_dir}")
        
        # Log input file paths
        logger.info(f"Input video file: {video_file} (exists: {video_file.exists()})")
        logger.info(f"Input audio file: {audio_file} (exists: {audio_file.exists()})")
        
        # Initialize video generator
        generator = VideoGenerator()
        
        # Generate final video
        output_file = output_dir / f"final_video_{style_name}.mp4"
        result = await generator.generate_video(
            str(video_file),
            str(audio_file),
            output_file,
            style_name
        )
        
        return result
    except Exception as e:
        logger.error(f"Error executing step: {str(e)}")
        return None 