"""
Generate social media captions with hashtags and emojis.
Uses Qwen or GPT-4o based on user preference.
"""

import os
import json
import logging
import re
from pathlib import Path
from typing import Dict, Optional, List

# Import the prompt manager from the existing codebase
from .prompts import PromptManager, LLMProvider

logger = logging.getLogger(__name__)

class CaptionGenerator:
    """Generates social media captions with hashtags and emojis."""
    
    def __init__(self, use_gpt4o: bool = False):
        """
        Initialize caption generator.
        
        Args:
            use_gpt4o: Whether to use GPT-4o instead of Qwen
        """
        self.use_gpt4o = use_gpt4o
        if use_gpt4o:
            self.prompt_manager = PromptManager(provider=LLMProvider.OPENAI)
        else:
            self.prompt_manager = PromptManager(provider=LLMProvider.QWEN)
    
    def _build_system_prompt(self) -> str:
        """Build system prompt for caption generation."""
        return """You are a social media caption expert specialized in creating engaging, 
trending captions with relevant hashtags and emojis. Your captions are:
- Attention-grabbing and modern
- Optimized for social media engagement
- Includes relevant trending hashtags
- Uses appropriate emojis that enhance the message
- Concise and impactful"""
    
    def _build_user_prompt(self, analysis: Dict, platform: str) -> str:
        """Build the user prompt for caption generation."""
        # Get video metadata
        video_text = analysis['metadata'].get('text', '')
        video_title = analysis['metadata'].get('title', '')
        video_description = analysis['metadata'].get('description', '')
        
        # Get vision analysis
        vision_insights = {
            'objects': [],
            'descriptions': []
        }
        
        # Check if analysis was skipped
        analysis_skipped = analysis.get('is_basic_analysis', False)
        
        if not analysis_skipped and 'frames' in analysis:
            for frame in analysis['frames']:
                # Check for Qwen vision analysis
                if 'qwen_vision' in frame:
                    desc = frame['qwen_vision'].get('detailed_description', '')
                    if desc:
                        vision_insights['descriptions'].append(desc)
                # Backwards compatibility for OpenAI vision
                elif 'openai_vision' in frame:
                    desc = frame['openai_vision'].get('detailed_description', '')
                    if desc:
                        vision_insights['descriptions'].append(desc)
        
        # Extract objects from descriptions using NLP parsing
        objects_mentioned = set()
        for desc in vision_insights['descriptions']:
            # Use regex to find potential objects (nouns) in the description
            potential_objects = re.findall(r'\b([A-Z][a-z]+|[a-z]+)\b', desc)
            for obj in potential_objects:
                if len(obj) > 3 and obj.lower() not in ['with', 'this', 'that', 'there', 'here', 'from', 'what', 'when', 'where', 'which', 'while']:
                    objects_mentioned.add(obj)
        
        # Build content text
        content_text = "VIDEO CONTENT:\n"
        if video_title:
            content_text += f"Title: {video_title}\n"
        if video_description:
            content_text += f"Description: {video_description}\n"
        if video_text:
            content_text += f"Text in video: {video_text}\n"
            
        # Add visual insights
        if not analysis_skipped:
            content_text += "\nVISUAL ELEMENTS:\n"
            
            if objects_mentioned:
                content_text += "Objects/subjects detected: " + ", ".join(list(objects_mentioned)[:10]) + "\n"
            
            if vision_insights['descriptions']:
                content_text += "\nScene descriptions:\n"
                for i, desc in enumerate(vision_insights['descriptions'][:2]):
                    content_text += f"{i+1}. {desc}\n"
        
        # Get target language
        language = analysis['metadata'].get('language', 'en')
        
        # Build platform-specific instructions
        platform_instructions = ""
        if platform == "instagram":
            platform_instructions = """
- Create for Instagram
- Use 3-5 relevant hashtags
- Include 2-4 emojis
- Keep under 125 words
- Use line breaks for readability"""
        elif platform == "tiktok":
            platform_instructions = """
- Create for TikTok
- Use 4-6 trending hashtags
- Include 3-5 emojis
- Keep under 100 characters
- Include 2-3 call-to-action phrases"""
        elif platform == "twitter":
            platform_instructions = """
- Create for Twitter/X
- Use 2-3 hashtags
- Include 1-3 emojis
- Keep under 280 characters
- Make it shareable and engaging"""
        else:  # General social media
            platform_instructions = """
- Create for general social media use
- Use 3-4 relevant hashtags
- Include 2-3 appropriate emojis
- Keep concise and engaging
- Make it shareable across platforms"""
        
        # Final prompt
        prompt = f"""Generate an engaging social media caption in {language.upper()} for this video:

{content_text}

CAPTION REQUIREMENTS:{platform_instructions}
- Must include relevant hashtags
- Must include appropriate emojis
- Make it attention-grabbing
- Focus on the video's most interesting aspects
- Make it relevant to current social media trends
- Create in {language} language

Generate ONLY the caption and nothing else."""

        return prompt
    
    async def generate_caption(self, analysis_file: Path, platform: str = "general") -> Dict:
        """
        Generate a social media caption for the video.
        
        Args:
            analysis_file: Path to the analysis JSON file
            platform: Target social media platform (instagram, tiktok, twitter, or general)
            
        Returns:
            Dictionary with the generated caption
        """
        try:
            # Load analysis
            with open(analysis_file, encoding='utf-8') as f:
                analysis = json.load(f)
            
            # Get the target language from metadata
            language = analysis['metadata'].get('language', 'en')
            
            # Build prompts
            system_prompt = self._build_system_prompt()
            user_prompt = self._build_user_prompt(analysis, platform)
            
            # Log prompts
            logger.info("\n=== CAPTION GENERATION SYSTEM PROMPT ===")
            logger.info(system_prompt)
            logger.info("\n=== CAPTION GENERATION USER PROMPT ===")
            logger.info(user_prompt)
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # Select the appropriate model
            if self.use_gpt4o:
                model_name = "gpt-4o"
            else:
                model_name = "qwen-plus"
            
            # Generate the caption
            logger.info(f"\n=== GENERATING CAPTION USING {model_name} ===")
            
            caption_text = self.prompt_manager.generate_response(
                messages=messages,
                model=model_name,
                temperature=0.7,
                max_tokens=500
            )
            
            logger.info("\n=== RAW GENERATED CAPTION ===")
            logger.info(caption_text)
            
            # If language is not English and gpt4o is not being used, translate with OpenAI
            if language != 'en' and not self.use_gpt4o:
                logger.info("\n=== TRANSLATING CAPTION ===")
                caption_text = await self.prompt_manager.translate_text(
                    text=caption_text,
                    source_language='en',
                    target_language=language
                )
                logger.info("\n=== TRANSLATED CAPTION ===")
                logger.info(caption_text)
            
            # Extract hashtags from the caption
            hashtags = re.findall(r'#\w+', caption_text)
            
            # Extract emojis using a broad emoji detection pattern
            emoji_pattern = re.compile(
                "["
                "\U0001F600-\U0001F64F"  # emoticons
                "\U0001F300-\U0001F5FF"  # symbols & pictographs
                "\U0001F680-\U0001F6FF"  # transport & map symbols
                "\U0001F700-\U0001F77F"  # alchemical symbols
                "\U0001F780-\U0001F7FF"  # Geometric Shapes
                "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
                "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
                "\U0001FA00-\U0001FA6F"  # Chess Symbols
                "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
                "\U00002702-\U000027B0"  # Dingbats
                "\U000024C2-\U0001F251" 
                "]+"
            )
            emojis = emoji_pattern.findall(caption_text)
            
            # Create caption result
            caption_result = {
                "caption": caption_text,
                "language": language,
                "platform": platform,
                "hashtags": hashtags,
                "emojis": emojis,
                "model_used": model_name
            }
            
            return caption_result
            
        except Exception as e:
            logger.error(f"Error generating caption: {str(e)}")
            return {
                "error": str(e),
                "caption": "Error generating caption. Please try again."
            }

async def generate_caption(
    analysis_file: Path, 
    platform: str = "general",
    use_gpt4o: bool = False
) -> Dict:
    """
    Generate a social media caption for a video.
    
    Args:
        analysis_file: Path to the analysis JSON file
        platform: Target social media platform
        use_gpt4o: Whether to use GPT-4o instead of Qwen
        
    Returns:
        Dictionary with the generated caption
    """
    generator = CaptionGenerator(use_gpt4o=use_gpt4o)
    return await generator.generate_caption(analysis_file, platform) 