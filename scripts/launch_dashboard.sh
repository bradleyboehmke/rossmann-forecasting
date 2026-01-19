#!/bin/bash
# Unified launcher for Rossmann Sales Forecasting Dashboard
# Starts MLflow, FastAPI backend, and Streamlit frontend in sequence

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
VENV_DIR="$PROJECT_ROOT/.venv"

# Activate virtual environment
if [ -d "$VENV_DIR" ]; then
    source "$VENV_DIR/bin/activate"
    echo -e "${GREEN}✓ Virtual environment activated${NC}"
else
    echo -e "${RED}❌ Virtual environment not found at $VENV_DIR${NC}"
    echo -e "${RED}   Please create it with: uv venv${NC}"
    exit 1
fi

echo ""
echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}  Rossmann Sales Forecasting Dashboard Launcher${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""

# ============================================================================
# STEP 1: Start MLflow Tracking Server
# ============================================================================
if lsof -Pi :5000 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo -e "${YELLOW}⚠️  MLflow server is already running on port 5000${NC}"
    echo -e "${YELLOW}   Skipping MLflow launch...${NC}"
    echo ""
else
    echo -e "${GREEN}🚀 Starting MLflow tracking server...${NC}"
    echo -e "${BLUE}   URL: http://localhost:5000${NC}"
    echo ""

    # Parse MLflow configuration from params.yaml
    cd "$PROJECT_ROOT"
    TRACKING_URI=$(grep -A 10 "^mlflow:" config/params.yaml | grep "tracking_uri:" | awk '{print $2}')
    ARTIFACT_LOCATION=$(grep -A 10 "^mlflow:" config/params.yaml | grep "artifact_location:" | awk '{print $2}')
    HOST=$(grep -A 10 "^mlflow:" config/params.yaml | grep "host:" | awk '{print $2}')
    PORT=$(grep -A 10 "^mlflow:" config/params.yaml | grep "port:" | awk '{print $2}')

    # Use defaults if not found
    TRACKING_URI=${TRACKING_URI:-./mlruns}
    ARTIFACT_LOCATION=${ARTIFACT_LOCATION:-./mlartifacts}
    HOST=${HOST:-127.0.0.1}
    PORT=${PORT:-5000}

    # Create directories if they don't exist
    mkdir -p "$TRACKING_URI"
    mkdir -p "$ARTIFACT_LOCATION"

    # Start MLflow in background
    nohup mlflow server \
        --backend-store-uri "$TRACKING_URI" \
        --default-artifact-root "$ARTIFACT_LOCATION" \
        --host "$HOST" \
        --port "$PORT" \
        > /tmp/rossmann_mlflow.log 2>&1 &
    MLFLOW_PID=$!

    echo -e "${GREEN}   ✓ MLflow started (PID: $MLFLOW_PID)${NC}"
    echo -e "${BLUE}   Logs: /tmp/rossmann_mlflow.log${NC}"
    echo ""

    # Wait for MLflow to be ready (max 15 seconds)
    echo -e "${YELLOW}⏳ Waiting for MLflow to be ready...${NC}"
    COUNTER=0
    until curl -s http://localhost:5000 > /dev/null 2>&1 || [ $COUNTER -eq 15 ]; do
        sleep 1
        COUNTER=$((COUNTER + 1))
        echo -ne "${YELLOW}   Waiting... ${COUNTER}s\r${NC}"
    done
    echo ""

    if [ $COUNTER -eq 15 ]; then
        echo -e "${RED}❌ MLflow failed to start within 15 seconds${NC}"
        echo -e "${RED}   Check logs at: /tmp/rossmann_mlflow.log${NC}"
        exit 1
    fi

    echo -e "${GREEN}   ✓ MLflow is ready!${NC}"
    echo ""
fi

# ============================================================================
# STEP 2: Start FastAPI Backend
# ============================================================================
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

# ============================================================================
# STEP 3: Launch Streamlit Dashboard (Foreground)
# ============================================================================
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
echo -e "${YELLOW}MLflow and FastAPI servers are still running in the background.${NC}"
echo -e "${YELLOW}To stop them, run:${NC}"
echo -e "${YELLOW}  MLflow:  lsof -ti:5000 | xargs kill -9${NC}"
echo -e "${YELLOW}  FastAPI: lsof -ti:8000 | xargs kill -9${NC}"
echo ""
