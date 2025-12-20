#!/bin/bash
# Unified launcher for Rossmann Sales Forecasting Dashboard
# Starts FastAPI backend and Streamlit frontend in sequence

set -e  # Exit on any error

# Color codes for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Get the project root directory (parent of scripts/)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEPLOYMENT_DIR="$PROJECT_ROOT/deployment"

echo ""
echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}  Rossmann Sales Forecasting Dashboard Launcher${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""

# Check if FastAPI is already running
if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo -e "${YELLOW}⚠️  FastAPI server is already running on port 8000${NC}"
    echo -e "${YELLOW}   Skipping FastAPI launch...${NC}"
    echo ""
else
    echo -e "${GREEN}🚀 Starting FastAPI backend server...${NC}"
    echo -e "${BLUE}   URL: http://localhost:8000${NC}"
    echo ""

    # Start FastAPI in background
    cd "$DEPLOYMENT_DIR/api"
    nohup python main.py > /tmp/rossmann_api.log 2>&1 &
    API_PID=$!

    echo -e "${GREEN}   ✓ FastAPI started (PID: $API_PID)${NC}"
    echo -e "${BLUE}   Logs: /tmp/rossmann_api.log${NC}"
    echo ""

    # Wait for API to be ready (max 30 seconds)
    echo -e "${YELLOW}⏳ Waiting for API to be ready...${NC}"
    COUNTER=0
    until curl -s http://localhost:8000/health > /dev/null 2>&1 || [ $COUNTER -eq 30 ]; do
        sleep 1
        COUNTER=$((COUNTER + 1))
        echo -ne "${YELLOW}   Waiting... ${COUNTER}s\r${NC}"
    done
    echo ""

    if [ $COUNTER -eq 30 ]; then
        echo -e "${RED}❌ FastAPI failed to start within 30 seconds${NC}"
        echo -e "${RED}   Check logs at: /tmp/rossmann_api.log${NC}"
        exit 1
    fi

    echo -e "${GREEN}   ✓ FastAPI is ready!${NC}"
    echo ""
fi

# Launch Streamlit
echo -e "${GREEN}🚀 Starting Streamlit dashboard...${NC}"
echo -e "${BLUE}   URL: http://localhost:8501${NC}"
echo ""

cd "$DEPLOYMENT_DIR/streamlit"
streamlit run Home.py

# Note: When Streamlit exits (Ctrl+C), the script continues here
echo ""
echo -e "${YELLOW}================================================${NC}"
echo -e "${YELLOW}  Dashboard Shutdown${NC}"
echo -e "${YELLOW}================================================${NC}"
echo ""
echo -e "${BLUE}Streamlit has been stopped.${NC}"
echo ""
echo -e "${YELLOW}FastAPI server is still running in the background.${NC}"
echo -e "${YELLOW}To stop it, run: lsof -ti:8000 | xargs kill -9${NC}"
echo ""
