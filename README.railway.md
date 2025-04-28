# Deploying Video Commentary Bot to Railway

This guide will help you deploy the Video Commentary Bot to Railway.com.

## Prerequisites

1. A Railway.com account
2. API keys for:
   - OpenAI
   - Dashscope (Qwen)
   - Google Cloud (for Text-to-Speech)

## Deployment Steps

### 1. Fork or Clone the Repository

Make sure you have your own copy of the repository.

### 2. Create a New Project on Railway

- Go to [Railway.com](https://railway.app/)
- Create a new project
- Choose "Deploy from GitHub repo"
- Connect your GitHub account and select your repository

### 3. Configure Environment Variables

In your Railway project, go to the "Variables" tab and add the following environment variables:

```
OPENAI_API_KEY=your-openai-api-key
DASHSCOPE_API_KEY=your-dashscope-api-key
```

### 4. Add Google Cloud Credentials

For Google Cloud Text-to-Speech, you need to add your service account credentials as an environment variable:

1. Convert your Google credentials JSON file to a single line (no line breaks)
2. In Railway, add a variable named `GOOGLE_APPLICATION_CREDENTIALS_JSON` with the JSON string as the value

### 5. Deploy

- Railway will automatically detect the Procfile and deploy your application
- The application will be accessible at the URL provided by Railway

## Configuration Options

You can customize your deployment with these additional environment variables:

- `PORT`: The port for the Streamlit app to listen on (default: 8501)

## Troubleshooting

If you encounter any issues:

1. Check the logs in the Railway dashboard
2. Ensure all required environment variables are set
3. Verify that your Google Cloud credentials JSON is correctly formatted

## Notes

- The application uses FFmpeg for video processing, which is automatically installed by Railway
- Temporary files are stored in the container's file system, which is ephemeral
- For production use, consider setting up cloud storage for video outputs 