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

    async def generate_video(
        self,
        video_path: str,
        audio_path: str,
        output_path: Path,
        style_name: str = None
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
            
            # Build FFmpeg command with improved stability
            cmd = [
                'ffmpeg',
                '-y',  # Overwrite output
                '-i', video_path,
                '-i', audio_path,
                '-filter_complex',
                f'[0:v]scale={scale_width}:{scale_height}:flags=lanczos,pad={target_width}:{target_height}:{pad_x}:{pad_y}:color=white[v];'
                f'[0:a]volume=1[original];'
                f'[1:a]volume=0.7[commentary];'
                f'[original]volume=0.3:enable=\'between(t,0,{commentary_duration})\':eval=frame[ducked];'
                '[ducked][commentary]amix=inputs=2:duration=longest[audio]',
                '-map', '[v]',
                '-map', '[audio]',
                '-c:v', 'libx264',
                '-preset', 'medium',  # Balance between speed and quality
                '-crf', '23',  # Maintain high quality
                '-maxrate', '4M',  # Allow higher bitrate for quality
                '-bufsize', '8M',
                '-c:a', 'aac',
                '-b:a', '192k',  # Good audio quality
                '-ac', '2',
                '-ar', '44100',
                '-max_muxing_queue_size', '2048',  # Increased queue size
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

async def execute_step(
    video_file: Path,
    audio_file: Path,
    output_dir: Path,
    style_name: str
) -> Optional[Path]:
    """Execute video generation step."""
    logger.debug("Step 6: Generating final video...")
    
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured output directory exists: {output_dir}")
        
        logger.info(f"Input video file: {video_file} (exists: {video_file.exists()})")
        logger.info(f"Input audio file: {audio_file} (exists: {audio_file.exists()})")
        
        generator = VideoGenerator()
        
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