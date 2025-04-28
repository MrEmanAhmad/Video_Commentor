"""
Step 5: Audio generation module
Generates audio using Google Cloud Text-to-Speech API
"""

import os
import logging
import io
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Dict, List
from google.cloud import texttospeech

logger = logging.getLogger(__name__)

def select_tts_voice(language: str, gender: str = 'FEMALE', voice_type: str = 'Neural2') -> str:
    """
    Select appropriate TTS voice based on language, gender, and quality preference.
    
    Args:
        language: Language code (e.g., 'en', 'ur', 'hi', 'bn', 'tr')
        gender: 'FEMALE' or 'MALE'
        voice_type: 'Standard', 'Wavenet', 'Neural2', 'HD', 'Journey', 'News', 'Studio', 'Polyglot', or 'Casual'
    
    Returns:
        Voice name for Google TTS
    """
    voices = {
        'en': {
            'FEMALE': {
                'Standard': 'en-US-Standard-E',
                'Wavenet': 'en-US-Wavenet-F', 
                'Neural2': 'en-US-Neural2-F',
                'HD': 'en-US-Journey-F',
                'Journey': 'en-US-Journey-F',
                'News': 'en-US-News-K'
            },
            'MALE': {
                'Standard': 'en-US-Standard-D',
                'Wavenet': 'en-US-Wavenet-D',
                'Neural2': 'en-US-Neural2-D',
                'HD': 'en-US-Journey-D',
                'Journey': 'en-US-Journey-D',
                'News': 'en-US-News-N',
                'Casual': 'en-US-Casual-K',
                'Polyglot': 'en-US-Polyglot-1'
            }
        },
        'en-GB': {
            'FEMALE': {
                'Standard': 'en-GB-Standard-A',
                'Wavenet': 'en-GB-Wavenet-A',
                'Neural2': 'en-GB-Neural2-N',
                'HD': 'en-GB-Journey-F',
                'Journey': 'en-GB-Journey-F',
                'News': 'en-GB-News-G',
                'Studio': 'en-GB-Studio-C'
            },
            'MALE': {
                'Standard': 'en-GB-Standard-B',
                'Wavenet': 'en-GB-Wavenet-B',
                'Neural2': 'en-GB-Neural2-O',
                'HD': 'en-GB-Journey-D',
                'Journey': 'en-GB-Journey-D',
                'News': 'en-GB-News-J',
                'Studio': 'en-GB-Studio-B'
            }
        },
        'en-AU': {
            'FEMALE': {
                'Standard': 'en-AU-Standard-A',
                'Wavenet': 'en-AU-Wavenet-A',
                'Neural2': 'en-AU-Neural2-A',
                'HD': 'en-AU-Journey-F',
                'Journey': 'en-AU-Journey-F',
                'News': 'en-AU-News-E'
            },
            'MALE': {
                'Standard': 'en-AU-Standard-B',
                'Wavenet': 'en-AU-Wavenet-B',
                'Neural2': 'en-AU-Neural2-B',
                'HD': 'en-AU-Journey-D',
                'Journey': 'en-AU-Journey-D',
                'News': 'en-AU-News-G',
                'Polyglot': 'en-AU-Polyglot-1'
            }
        },
        'en-IN': {
            'FEMALE': {
                'Standard': 'en-IN-Standard-A',
                'Wavenet': 'en-IN-Wavenet-A',
                'Neural2': 'en-IN-Neural2-A',
                'HD': 'en-IN-Journey-F',
                'Journey': 'en-IN-Journey-F'
            },
            'MALE': {
                'Standard': 'en-IN-Standard-B',
                'Wavenet': 'en-IN-Wavenet-B',
                'Neural2': 'en-IN-Neural2-B',
                'HD': 'en-IN-Journey-D',
                'Journey': 'en-IN-Journey-D'
            }
        },
        'ur': {
            'FEMALE': {
                'Standard': 'ur-IN-Standard-A',
                'Wavenet': 'ur-IN-Wavenet-A',
                'HD': 'ur-IN-Standard-A'  # Fallback to Standard as HD isn't available yet
            },
            'MALE': {
                'Standard': 'ur-IN-Standard-B',
                'Wavenet': 'ur-IN-Wavenet-B',
                'HD': 'ur-IN-Standard-B'  # Fallback to Standard as HD isn't available yet
            }
        },
        'bn': {
            'FEMALE': {
                'Standard': 'bn-IN-Standard-A',
                'HD': 'bn-IN-Chirp3-HD-Aoede'
            },
            'MALE': {
                'Standard': 'bn-IN-Standard-B'
            }
        },
        'hi': {
            'FEMALE': {
                'Standard': 'hi-IN-Standard-A',
                'Wavenet': 'hi-IN-Wavenet-A',
                'HD': 'hi-IN-Chirp3-HD-Aoede'
            },
            'MALE': {
                'Standard': 'hi-IN-Standard-B',
                'Wavenet': 'hi-IN-Wavenet-B'
            }
        },
        'tr': {
            'FEMALE': {
                'Standard': 'tr-TR-Standard-A',
                'Wavenet': 'tr-TR-Wavenet-A',
                'HD': 'tr-TR-Chirp3-HD-Aoede'
            },
            'MALE': {
                'Standard': 'tr-TR-Standard-B',
                'Wavenet': 'tr-TR-Wavenet-B'
            }
        }
    }
    
    # Default fallback voice
    default_voice = 'en-US-Neural2-F'
    
    try:
        # Get language-specific voices
        language_voices = voices.get(language, voices['en'])
        
        # Get gender-specific voices
        gender_voices = language_voices.get(gender, language_voices['FEMALE'])
        
        # Get specific voice type or fall back to available type
        voice = gender_voices.get(voice_type, list(gender_voices.values())[0])
        
        return voice
    except (KeyError, IndexError):
        return default_voice

def get_google_voice_params(voice_name: str):
    """
    Get Google TTS voice parameters from voice name.
        
    Args:
        voice_name: Voice name (e.g., 'en-US-Neural2-F', 'en-GB-Journey-D')
            
    Returns:
        Tuple of (language_code, voice_name, ssml_gender)
    """
    # Parse voice name
    parts = voice_name.split('-')
            
    # Extract language code
    language_code = f"{parts[0]}-{parts[1]}"
    
    # Determine gender based on last character or naming convention
    gender = texttospeech.SsmlVoiceGender.FEMALE
    
    # Check for female indicators in voice name
    if any(part in ['A', 'C', 'E', 'F', 'G', 'K', 'N'] for part in [parts[-1]]):
        gender = texttospeech.SsmlVoiceGender.FEMALE
    # Check for male indicators in voice name
    elif any(part in ['B', 'D', 'I', 'J', 'M', 'O'] for part in [parts[-1]]) or "Polyglot" in voice_name or "Casual" in voice_name:
        gender = texttospeech.SsmlVoiceGender.MALE
    
    # Return parameters
    return language_code, voice_name, gender

def create_silent_audio(output_path: str, duration: float = 10.0) -> bool:
    """
    Create a silent audio file using ffmpeg.
    Used as a fallback if TTS fails.
    
    Args:
        output_path: Path to save the audio file
        duration: Duration of the silent audio in seconds
        
    Returns:
        True if successful, False otherwise
    """
    try:
        cmd = [
            'ffmpeg', '-f', 'lavfi', '-i', f'anullsrc=r=44100:cl=stereo', 
            '-t', str(duration), '-q:a', '0', '-ac', '2', '-ar', '44100', 
            '-y', output_path
        ]
        
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logger.info(f"Created silent audio file: {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error creating silent audio: {str(e)}")
        return False

def generate_tts_audio(text: str, output_path: str, language: str = 'en', gender: str = 'FEMALE', voice_type: str = 'Neural2') -> bool:
    """
    Generate audio using Google Cloud Text-to-Speech API.
    
    Args:
        text: Text to convert to speech
        output_path: Path to save the audio file
        language: Language code (e.g., 'en', 'ur', 'hi', 'bn', 'tr')
        gender: 'FEMALE' or 'MALE'
        voice_type: 'Standard', 'Wavenet', 'Neural2', 'HD', 'Journey', 'News', 'Studio', etc.
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Set environment variable for Google credentials
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "google_credentials.json"
        
        # Select voice
        voice_name = select_tts_voice(language, gender, voice_type)
        language_code, voice_name, ssml_gender = get_google_voice_params(voice_name)
        
        logger.info(f"Using Google TTS with voice: {voice_name}, language: {language_code}")
        
        # Initialize Text-to-Speech client
        client = texttospeech.TextToSpeechClient()
        
        # Format text as SSML for better control
        if language == 'ur':
            # Special handling for Urdu with proper SSML - improved prosody settings
            text = f"""<speak>
                <prosody rate="1.2" pitch="+2nd">
                {text}
                </prosody>
            </speak>"""
            input_text = texttospeech.SynthesisInput(ssml=text)
            
            # For Urdu, override the language code to ur-PK (which works according to user testing)
            language_code = "ur-PK"
            
            # Configure voice for Urdu
            voice = texttospeech.VoiceSelectionParams(
                language_code=language_code,
                ssml_gender=ssml_gender
            )
            
            # Urdu-specific audio config
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3,
                speaking_rate=1.0,  # Base rate (SSML will modify this)
                pitch=0.0,          # Base pitch (SSML will modify this)
                effects_profile_id=["headphone-class-device"]
            )
        else:
            # Standard text input for other languages
            input_text = texttospeech.SynthesisInput(text=text)
        
            # Configure voice for other languages
            voice = texttospeech.VoiceSelectionParams(
                language_code=language_code,
                name=voice_name,
                ssml_gender=ssml_gender
            )
        
            # Standard audio config for other languages
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.LINEAR16,
                sample_rate_hertz=24000,
                effects_profile_id=["medium-bluetooth-speaker-class-device"]
            )
        
        # Generate speech
        logger.info("Requesting speech synthesis from Google TTS")
        response = client.synthesize_speech(
            input=input_text,
            voice=voice,
            audio_config=audio_config
        )
        
        # Save audio to file
        with open(output_path, "wb") as out:
            out.write(response.audio_content)
            
        logger.info(f"Generated TTS audio file: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error generating TTS audio: {str(e)}")
        
        # Create silent audio as fallback
        logger.warning("Falling back to silent audio")
        return create_silent_audio(output_path, 10.0)

def detect_language(text: str) -> str:
    """
    Detect the language of the provided text.
    Returns ISO language code (e.g., 'en', 'ur', 'hi', 'bn', 'tr').
    """
    # Simple heuristics based on script detection
    if any('\u0600' <= c <= '\u06FF' for c in text):
        return 'ur'  # Urdu script detected
    elif any('\u0980' <= c <= '\u09FF' for c in text):
        return 'bn'  # Bengali script detected
    elif any('\u0900' <= c <= '\u097F' for c in text):
        return 'hi'  # Hindi script detected
    elif any('\u0C80' <= c <= '\u0CFF' for c in text):
        return 'tr'  # Turkish script detected (simplified, may need refinement)
    return 'en'  # Default to English

async def execute_step(frames_info: dict, output_dir: Path, style: str = None) -> str:
    """
    Generate audio from commentary text.
    
    Args:
        frames_info: Dictionary containing frame analysis and commentary
        output_dir: Directory to save output files
        style: Commentary style (optional)
        
    Returns:
        Path to generated audio file
    """
    try:
        # Get style from parameters or metadata
        style = style or frames_info['metadata'].get('style', 'documentary')
        logger.info(f"Using style: {style}")
        
        # Check if there's a commentary file in the commentary directory
        commentary_dir = Path(output_dir).parent / "04_commentary"
        commentary_file = commentary_dir / f"commentary_{style}.json"
        
        # If commentary file exists, load it
        if commentary_file.exists():
            logger.info(f"Loading commentary from {commentary_file}")
            with open(commentary_file, encoding='utf-8') as f:
                commentary = json.load(f)
        else:
            # If not, look for commentary in frames_info
            logger.info("No commentary file found, using commentary from frames_info")
            commentary = {
                'commentary': frames_info.get('commentary', ''),
                'language': frames_info['metadata'].get('language', 'en'),
                'style': style
            }
            
            # If still no commentary, create a default one
            if not commentary['commentary']:
                commentary['commentary'] = "This is a placeholder commentary."
                logger.warning("No commentary found, using placeholder")
        
        # Save commentary to output directory for audio generation
        output_commentary_file = output_dir / f"commentary_{style}.json"
        with open(output_commentary_file, 'w', encoding='utf-8') as f:
            json.dump(commentary, f, indent=2, ensure_ascii=False)
        
        # Get text and language
        text = commentary['commentary']
        language = commentary.get('language', 'en')
        
        logger.info(f"Generating audio for text: {text[:100]}...")
        
        # Generate audio file path
        audio_file = output_dir / f"commentary_{style}.wav"
        
        # Determine best voice type based on language
        voice_type = 'Neural2'  # Default for English
        
        # Language-specific voice types
        if language.startswith('en'):
            voice_type = 'Neural2'  # Best quality for English
        elif language == 'ur':
            voice_type = 'Wavenet'  # Best available for Urdu
        elif language in ['bn', 'hi', 'tr']:
            voice_type = 'HD'  # HD voices for Bengali, Hindi, Turkish
        
        # Generate TTS audio
        success = generate_tts_audio(
            text=text, 
            output_path=str(audio_file),
            language=language,
            gender='FEMALE',  # Could be configurable
            voice_type=voice_type
        )
        
        if success:
            logger.info(f"Successfully generated audio file: {audio_file}")
            return str(audio_file)
        else:
            logger.warning("TTS generation failed, using silent audio as fallback")
            success = create_silent_audio(str(audio_file), frames_info['metadata'].get('duration', 10.0))
            if success:
                return str(audio_file)
            else:
                raise Exception("Failed to generate audio")
            
    except Exception as e:
        logger.error(f"Error in audio generation: {str(e)}")
        raise 