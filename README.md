# Video Commentary Bot

An AI-powered tool that automatically generates engaging commentary for videos in multiple languages and styles.

![Video Commentary Bot](https://img.shields.io/badge/Video%20Commentary-Bot-brightgreen)

## Features

- **Multi-Platform Video Support**: Process videos from YouTube, Twitter/X, or direct video URLs
- **Multi-Language Commentary**: Generate commentary in English, Urdu, Bengali, Hindi, Turkish, and more
- **Style Options**: Choose from various commentary styles (news, funny, nature documentary, informative)
- **High-Quality TTS Voices**: Uses Google's WaveNet and Neural2 voices for natural-sounding narration
- **Intelligent Scene Analysis**: Uses Qwen Vision API to understand video content
- **Performance Options**: Skip costly AI analysis for faster processing
- **Resource Monitoring**: Track time and memory usage during processing
- **Custom Watermarks**: Add personalized text watermarks to your videos
- **Streamlit Interface**: User-friendly web interface for easy interaction

## Installation

### Prerequisites

- Python 3.8+
- FFmpeg installed on your system ([FFmpeg Installation Guide](https://ffmpeg.org/download.html))
- API keys for:
  - OpenAI (for translations)
  - Qwen/DashScope (for image analysis and commentary generation)
  - Google Cloud (for text-to-speech)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/video-commentary-bot.git
   cd video-commentary-bot
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   ```bash
   # Linux/Mac
   export OPENAI_API_KEY="your_openai_api_key"
   export DASHSCOPE_API_KEY="your_dashscope_api_key"
   
   # Windows (PowerShell)
   $env:OPENAI_API_KEY="your_openai_api_key"
   $env:DASHSCOPE_API_KEY="your_dashscope_api_key"
   ```

4. Place your Google Cloud credentials JSON file in the project root directory as `google_credentials.json`

## Usage

### Command Line Interface

```bash
python -m process_video --url VIDEO_URL --output OUTPUT_DIR --language LANGUAGE --style STYLE
```

Example:
```bash
python -m process_video --url https://www.youtube.com/watch?v=dQw4w9WgXcQ --output ./output --language en --style funny
```

#### Performance Options:

```bash
# Skip AI frame analysis for faster processing (less detailed commentary)
python -m process_video --url VIDEO_URL --skip-analysis

# Monitor resource usage during processing
python -m process_video --url VIDEO_URL --monitor
```

#### Advanced Options:

```bash
python -m process_video --url VIDEO_URL --output OUTPUT_DIR --language LANGUAGE --style STYLE --preserve-temp --no-cleanup --watermark-text "My Channel" --watermark-size 36 --watermark-color white --watermark-font "Arial" --skip-analysis --monitor
```

### Streamlit Web Interface

For a more user-friendly experience, run the Streamlit app:

```bash
streamlit run streamlit_app.py
```

This will open a web interface in your browser where you can:
- Enter video URLs
- Select language and style options
- Configure watermark settings with a live preview
- Enable performance options (skip analysis, monitor resources)
- Process videos and download the results

## Pipeline Overview

The video commentary bot processes videos through a 6-step pipeline:

1. **Download Video**: Handles YouTube, Twitter, and direct video URLs
2. **Extract Frames**: Uses OpenCV to extract key frames from the video
3. **Analyze Frames** (optional): Uses Qwen Vision API for detailed scene understanding
4. **Generate Commentary**: Uses Qwen LLMs for English commentary and translates to other languages
5. **Generate Audio**: Uses Google TTS for speech synthesis with extensive voice options
6. **Generate Video**: Combines the original video with the generated audio and applies watermarks

## Performance Options

- **Skip Frame Analysis**: Bypass the costly AI scene analysis for faster processing. The commentary will be based solely on video metadata instead of visual content.
- **Resource Monitoring**: Track processing time and memory usage for each step, with a full report saved to the output directory.

## Supported Languages

The bot currently supports the following languages:
- English (US, UK, Australia, India variants)
- Urdu
- Bengali
- Hindi
- Turkish

## Commentary Styles

- **News**: Formal journalistic style with objective reporting
- **Funny**: Humorous commentary with witty observations
- **Nature**: Nature documentary style with descriptive language
- **Documentary**: Balanced educational and entertaining narration
- **Informative**: Clear explanations focused on educational content

## Watermark Options

You can add custom watermarks to your videos with these options:
- **Watermark Text**: The text to display as a watermark
- **Font Size**: Control the size of the watermark text (12-72pt)
- **Text Color**: Choose from 15 predefined colors
- **Font**: Select from 10 common fonts for your watermark

## Example Output

The tool generates a final video file combining:
- The original video
- AI-generated commentary
- High-quality synthesized speech
- Optional custom watermark

## License

[MIT License](LICENSE)

## Acknowledgments

- [OpenAI](https://openai.com/) for translation capabilities
- [Qwen/DashScope](https://qianwen.aliyun.com/) for LLMs and image analysis
- [Google Cloud Text-to-Speech](https://cloud.google.com/text-to-speech) for voice synthesis
- [FFmpeg](https://ffmpeg.org/) for video processing
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) for video downloading 