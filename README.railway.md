# Railway Deployment Guide for Video Commentary Bot

This guide explains how to deploy Video Commentary Bot on Railway.app platform.

## Prerequisites

Before deploying, make sure you have:

1. A Railway.app account
2. Required API keys:
   - OpenAI API key
   - DashScope API key
   - Google Cloud API credentials (for text-to-speech)

## Setup Steps

### 1. Create a New Project on Railway

- Go to [Railway.app](https://railway.app/) and log in
- Click "New Project" and select "Deploy from GitHub repo"
- Connect your GitHub account and select the Video Commentary Bot repository

### 2. Configure Environment Variables

Add the following environment variables in your Railway project's settings:

```
OPENAI_API_KEY=your_openai_api_key
DASHSCOPE_API_KEY=your_dashscope_api_key
GOOGLE_APPLICATION_CREDENTIALS_JSON={"your":"google_credentials_json_content"}
```

For Google credentials, you need to convert your JSON file to a single line and paste it as the value.

### 3. Configure Service Settings

Railway will automatically detect the deployment configuration from:
- railway.json
- railway.toml
- Procfile

No additional configuration is needed.

## Troubleshooting

If you encounter OpenCV errors in logs, check that all system dependencies are properly installed:

1. Make sure the Railway build is using the Nixpacks build pack
2. Verify that `apt.txt` contains all necessary system libraries

## Resource Configuration

Recommended minimum resources:
- Memory: 1GB
- CPU: 1 vCPU

For faster video processing:
- Memory: 2GB
- CPU: 2 vCPU

## Monitoring

Check the Railway logs to troubleshoot any issues during deployment or runtime.

## Updating the Deployment

Railway will automatically redeploy when you push changes to the connected GitHub repository.

## Note about Disk Space

Railway has ephemeral storage, so any files created during the execution will be deleted when the service restarts. The app is designed to handle this by storing only the necessary files during processing. 