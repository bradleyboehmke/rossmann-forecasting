#!/bin/bash
# Launch script for FastAPI server

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}  Rossmann Forecasting API Server${NC}"
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
if ! python -c "import fastapi" 2>/dev/null; then
    echo -e "${YELLOW}⚠️  FastAPI not installed. Installing dependencies...${NC}"
    uv pip install fastapi uvicorn pydantic
fi

# Navigate to API directory
cd deployment/api

# Launch FastAPI server
echo ""
echo -e "${GREEN}🚀 Starting FastAPI server...${NC}"
echo ""
echo -e "${BLUE}Server will be available at:${NC}"
echo -e "  - API: ${GREEN}http://localhost:8000${NC}"
echo -e "  - Docs: ${GREEN}http://localhost:8000/docs${NC}"
echo -e "  - ReDoc: ${GREEN}http://localhost:8000/redoc${NC}"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop the server${NC}"
echo ""

# Run with auto-reload for development
uvicorn main:app --host 0.0.0.0 --port 8000 --reload --log-level info
