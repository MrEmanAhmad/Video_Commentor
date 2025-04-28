"""
Video processing pipeline package
"""

from .Step_1_download_video import execute_step as download_from_url
from .Step_2_extract_frames import execute_step as extract_frames
from .Step_3_analyze_frames import execute_step as analyze_frames
from .Step_4_generate_commentary import execute_step as generate_commentary
from .Step_5_generate_audio import execute_step as generate_audio
from .Step_6_video_generation import execute_step as generate_video
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging
import os
from openai import OpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('video_commentary_bot.log', 'a')
    ]
)

logger = logging.getLogger(__name__)

async def translate_text(text: str, source_language: str = 'en', target_language: str = 'ur') -> str:
    """
    Translate text from source language to target language.
    Uses OpenAI exclusively for translation purposes.
    
    Args:
        text: Text to translate
        source_language: Source language code (e.g., 'en')
        target_language: Target language code (e.g., 'ur')
        
    Returns:
        Translated text
    """
    if source_language == target_language:
        return text
        
    # Use OpenAI for translation only
    try:
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": f"You are a professional translator from {source_language} to {target_language}."},
                {"role": "user", "content": f"Translate the following text to {target_language}, preserving the style and meaning:\n\n{text}"}
            ],
            temperature=0.3
        )
        
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Translation error: {str(e)}")
        return text  # Return original text if translation fails

__all__ = [
    "download_from_url",
    "extract_frames",
    "analyze_frames",
    "generate_commentary",
    "generate_audio",
    "generate_video",
    "translate_text"
] 