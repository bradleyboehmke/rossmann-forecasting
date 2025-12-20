#!/bin/bash
# Script to run the Streamlit application

# Change to streamlit directory
cd "$(dirname "$0")"

echo "================================================"
echo "Starting Rossmann Sales Forecasting Dashboard"
echo "================================================"
echo ""
echo "Make sure the FastAPI server is running at http://localhost:8000"
echo "If not, run: cd ../api && python main.py"
echo ""
echo "Dashboard will open at: http://localhost:8501"
echo ""

# Run streamlit
streamlit run Home.py
