#!/bin/bash
# Test script for Rossmann Forecasting API

set -e

echo "========================================"
echo "Testing Rossmann Forecasting API"
echo "========================================"
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 1. Health check
echo -e "${BLUE}1. Testing health endpoint...${NC}"
curl -s http://localhost:8000/health | jq .
echo ""

# 2. Model info
echo -e "${BLUE}2. Testing model info endpoint...${NC}"
curl -s http://localhost:8000/model/info | jq .
echo ""

# 3. Load Production model
echo -e "${BLUE}3. Loading Production model...${NC}"
curl -s http://localhost:8000/model/load?stage=Production -X POST | jq .
echo ""

# 4. Single prediction
echo -e "${BLUE}4. Testing single prediction...${NC}"
curl -s http://localhost:8000/predict -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [
      {
        "Store": 1,
        "DayOfWeek": 5,
        "Date": "2015-08-01",
        "Open": 1,
        "Promo": 1,
        "StateHoliday": "0",
        "SchoolHoliday": 0
      }
    ],
    "model_stage": "Production"
  }' | jq .
echo ""

# 5. Batch prediction (3 inputs)
echo -e "${BLUE}5. Testing batch prediction (3 inputs)...${NC}"
curl -s http://localhost:8000/predict -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [
      {
        "Store": 1,
        "DayOfWeek": 5,
        "Date": "2015-08-01",
        "Open": 1,
        "Promo": 1,
        "StateHoliday": "0",
        "SchoolHoliday": 0
      },
      {
        "Store": 1,
        "DayOfWeek": 6,
        "Date": "2015-08-02",
        "Open": 1,
        "Promo": 0,
        "StateHoliday": "0",
        "SchoolHoliday": 0
      },
      {
        "Store": 2,
        "DayOfWeek": 5,
        "Date": "2015-08-01",
        "Open": 1,
        "Promo": 1,
        "StateHoliday": "0",
        "SchoolHoliday": 0
      }
    ],
    "model_stage": "Production"
  }' | jq .
echo ""

echo -e "${GREEN}✓ All tests completed successfully!${NC}"
echo ""
echo "API Documentation:"
echo "  - Swagger UI: http://localhost:8000/docs"
echo "  - ReDoc: http://localhost:8000/redoc"
