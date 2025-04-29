#!/bin/bash

# Set default port if PORT is not set
PORT=${PORT:-8080}
echo "Starting Streamlit on port $PORT"

# Run streamlit
streamlit run streamlit_app.py --server.address=0.0.0.0 --server.port=$PORT --server.enableCORS=false 