"""
Step 3: Analyze extracted frames
Uses Qwen Vision API to analyze frames and generate insights
"""

import os
import base64
import json
import logging
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from openai import OpenAI
from tqdm import tqdm

# Configure logging
logger = logging.getLogger(__name__)

class VisionAnalyzer:
    """Class for analyzing frames using Qwen Vision API."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the vision analyzer.
        
        Args:
            api_key: API key for Qwen (optional, can use env variable)
        """
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            logger.warning("No DASHSCOPE_API_KEY found. Vision analysis will fail.")
    
    def _build_qwen_prompt(self) -> str:
        """Build the prompt for Qwen Vision API."""
        return (
            "Analyze this image in detail. Provide a comprehensive description including:\n"
            "1. Main subjects and objects visible\n"
            "2. Activities or actions taking place\n"
            "3. The setting or environment\n"
            "4. Noteworthy visual elements (colors, lighting, composition)\n"
            "5. Any text visible in the image\n"
            "6. Overall mood or atmosphere\n\n"
            "Format your analysis as a detailed paragraph focusing on what's most significant."
        )
    
    async def analyze_frame(self, frame_path: Path) -> Tuple[Optional[dict], bool]:
        """
        Analyze a frame using Qwen Vision API.
        Provides detailed scene understanding.
        
        Args:
            frame_path: Path to the frame image
            
        Returns:
            Tuple containing the analysis results dictionary and success flag
        """
        try:
            with open(frame_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            
            # Using OpenAI-compatible interface
            client = OpenAI(
                api_key=self.api_key,
                base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
            )
            
            response = client.chat.completions.create(
                model="qwen-vl-plus",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self._build_qwen_prompt()},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                },
                            },
                        ],
                    }
                ],
                max_tokens=500,
            )
            
            analysis_result = {
                "detailed_description": response.choices[0].message.content,
            }
            
            return analysis_result, True
        except Exception as e:
            logger.error(f"Qwen Vision API error: {str(e)}")
            return None, False
    
    async def analyze_frames(self, frames_dir: Path, max_frames: int = 10) -> Dict:
        """
        Analyze multiple frames using Qwen Vision API.
        
        Args:
            frames_dir: Directory containing frame images
            max_frames: Maximum number of frames to analyze
            
        Returns:
            Dictionary containing analysis results for each frame
        """
        results = {}
        frames = list(frames_dir.glob("*.jpg"))
        frames.sort(key=lambda x: int(x.stem.split('_')[1]))
        
        # Limit the number of frames to analyze
        if max_frames and len(frames) > max_frames:
            # Select frames evenly distributed across the video
            step = len(frames) // max_frames
            indices = [i * step for i in range(max_frames)]
            # Ensure we include the first and last frame
            if indices and indices[-1] != len(frames) - 1:
                indices[-1] = len(frames) - 1
            frames = [frames[i] for i in indices]
        
        logger.info(f"Analyzing {len(frames)} frames with Qwen Vision API")
        
        for frame in tqdm(frames, desc="Analyzing frames"):
            frame_number = int(frame.stem.split('_')[1])
            analysis, success = await self.analyze_frame(frame)
            
            if success:
                results[frame_number] = analysis
            else:
                logger.warning(f"Failed to analyze frame {frame_number}")
        
        return results

async def execute_step(frames_info: dict, output_dir: Path) -> dict:
    """
    Execute the frame analysis step.
    
    Args:
        frames_info: Dictionary containing frame extraction results
        output_dir: Directory to save output files
        
    Returns:
        Dictionary containing frame analysis results
    """
    try:
        analyzer = VisionAnalyzer()
        frames_dir = Path(frames_info["frames_dir"])
        
        # Get metadata
        metadata = frames_info.get("metadata", {})
        
        # Determine max frames to analyze based on video length
        duration = metadata.get("duration", 0)
        max_frames = min(10, max(5, int(duration / 10)))
        
        # Analyze frames
        logger.info(f"Analyzing up to {max_frames} frames")
        analysis_results = await analyzer.analyze_frames(frames_dir, max_frames)
        
        # Save analysis results
        output_file = output_dir / "frame_analysis.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump({
                "metadata": metadata,
                "analysis": analysis_results
            }, f, indent=2)
        
        logger.info(f"Frame analysis complete. Results saved to {output_file}")
        
        return {
            "metadata": metadata,
            "analysis": analysis_results,
            "analysis_file": str(output_file)
        }
        
    except Exception as e:
        logger.error(f"Error in frame analysis: {str(e)}")
        raise