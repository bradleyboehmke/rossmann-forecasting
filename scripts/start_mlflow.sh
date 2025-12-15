#!/bin/bash
# Start MLflow tracking server
#
# This script starts the MLflow tracking server with the configuration
# specified in config/params.yaml. The server provides a web UI for
# visualizing experiments, comparing runs, and managing models.
#
# Usage:
#   bash scripts/start_mlflow.sh
#
# The UI will be available at http://127.0.0.1:5000

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

cd "$PROJECT_ROOT"

# Check if config file exists
if [ ! -f "config/params.yaml" ]; then
    echo "Error: config/params.yaml not found"
    exit 1
fi

# Parse configuration from params.yaml
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

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Starting MLflow Tracking Server${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${GREEN}Configuration:${NC}"
echo "  Backend Store URI: $TRACKING_URI"
echo "  Artifact Location: $ARTIFACT_LOCATION"
echo "  Host: $HOST"
echo "  Port: $PORT"
echo ""
echo -e "${GREEN}Web UI will be available at:${NC}"
echo "  http://$HOST:$PORT"
echo ""
echo -e "${BLUE}========================================${NC}"
echo ""

# Start MLflow server
mlflow server \
    --backend-store-uri "$TRACKING_URI" \
    --default-artifact-root "$ARTIFACT_LOCATION" \
    --host "$HOST" \
    --port "$PORT"
