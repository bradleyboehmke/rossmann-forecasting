#!/bin/bash
# Launch script for Streamlit dashboard

set -e  # Exit on error

# Colors for output
GREEN='\033[0.32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}  Rossmann Forecasting Dashboard${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo -e "${YELLOW}⚠️  Virtual environment not found. Please run 'uv venv' first.${NC}"
    exit 1
fi

# Activate virtual environment
echo -e "${GREEN}✓ Activating virtual environment...${NC}"
source .venv/bin/activate

# Check if required packages are installed
echo -e "${GREEN}✓ Checking dependencies...${NC}"
if ! python -c "import streamlit" 2>/dev/null; then
    echo -e "${YELLOW}⚠️  Streamlit not installed. Installing dependencies...${NC}"
    uv pip install streamlit plotly
fi

# Navigate to Streamlit directory
cd deployment/streamlit

# Check if MLflow is accessible
echo -e "${GREEN}✓ Checking MLflow connection...${NC}"
if ! python -c "from src.models.model_registry import list_registered_models; list_registered_models()" 2>/dev/null; then
    echo -e "${YELLOW}⚠️  Warning: Could not connect to MLflow or no models registered.${NC}"
    echo -e "${YELLOW}   Make sure MLflow tracking server is running and models are registered.${NC}"
fi

# Launch Streamlit dashboard
echo ""
echo -e "${GREEN}🎨 Starting Streamlit dashboard...${NC}"
echo ""
echo -e "${BLUE}Dashboard will be available at: ${GREEN}http://localhost:8501${NC}"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop the dashboard${NC}"
echo ""

# Run Streamlit with custom config
streamlit run Home.py \
    --server.port 8501 \
    --server.address localhost \
    --browser.gatherUsageStats false \
    --server.headless false
