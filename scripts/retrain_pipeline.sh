#!/bin/bash

################################################################################
# Rossmann Model Retraining Pipeline
#
# This script orchestrates the end-to-end model retraining workflow:
# 1. Data validation
# 2. Model training
# 3. Model validation & promotion to Staging
# 4. (Optional) Inference testing
#
# Usage:
#   bash scripts/retrain_pipeline.sh [--strict] [--no-validate] [--test-inference]
#
# Options:
#   --strict          Use strict validation threshold (RMSPE < 0.09856)
#   --no-validate     Skip model validation step
#   --test-inference  Run inference test after validation
################################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse arguments
STRICT_MODE=false
SKIP_VALIDATION=false
TEST_INFERENCE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --strict)
            STRICT_MODE=true
            shift
            ;;
        --no-validate)
            SKIP_VALIDATION=true
            shift
            ;;
        --test-inference)
            TEST_INFERENCE=true
            shift
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo -e "${YELLOW}Warning: Virtual environment not activated${NC}"
    echo -e "${YELLOW}Attempting to activate .venv...${NC}"
    if [ -f ".venv/bin/activate" ]; then
        source .venv/bin/activate
    else
        echo -e "${RED}Error: Virtual environment not found${NC}"
        echo -e "${YELLOW}Please activate your virtual environment and try again${NC}"
        exit 1
    fi
fi

echo -e "${BLUE}======================================================================${NC}"
echo -e "${BLUE}ROSSMANN MODEL RETRAINING PIPELINE${NC}"
echo -e "${BLUE}======================================================================${NC}"
echo -e "Started at: $(date)"
echo -e "Project root: $PROJECT_ROOT"
echo -e "Strict mode: $STRICT_MODE"
echo -e "Skip validation: $SKIP_VALIDATION"
echo -e "Test inference: $TEST_INFERENCE"
echo -e "${BLUE}======================================================================${NC}"

################################################################################
# Step 1: Data Validation
################################################################################

echo -e "\n${BLUE}[Step 1/4] Data Validation${NC}"
echo -e "${BLUE}----------------------------------------------------------------------${NC}"

# Check if processed data exists
if [ ! -f "data/processed/train_features.parquet" ]; then
    echo -e "${RED}Error: Processed data not found${NC}"
    echo -e "${YELLOW}Please run feature engineering first:${NC}"
    echo -e "  python src/features/build_features.py"
    exit 1
fi

# Validate data (basic checks)
echo -e "${GREEN}✓ Found processed data: data/processed/train_features.parquet${NC}"

# Optional: Run data validation script if available
if [ -f "src/data/validate_data.py" ]; then
    echo -e "Running data quality checks..."
    python src/data/validate_data.py --stage processed || {
        echo -e "${YELLOW}Warning: Data validation checks failed${NC}"
        echo -e "${YELLOW}Continuing anyway, but review data quality${NC}"
    }
fi

################################################################################
# Step 2: Model Training
################################################################################

echo -e "\n${BLUE}[Step 2/4] Model Training${NC}"
echo -e "${BLUE}----------------------------------------------------------------------${NC}"

# Check if best hyperparameters exist
if [ ! -f "config/best_hyperparameters.json" ]; then
    echo -e "${RED}Error: Best hyperparameters not found${NC}"
    echo -e "${YELLOW}Please run hyperparameter tuning first:${NC}"
    echo -e "  python notebooks/04-advanced-models-and-ensembles.ipynb"
    exit 1
fi

echo -e "Training ensemble model..."
python src/models/train_ensemble.py || {
    echo -e "${RED}Error: Model training failed${NC}"
    exit 1
}

echo -e "${GREEN}✓ Model training complete${NC}"

################################################################################
# Step 3: Model Validation & Promotion
################################################################################

if [ "$SKIP_VALIDATION" = false ]; then
    echo -e "\n${BLUE}[Step 3/4] Model Validation & Promotion${NC}"
    echo -e "${BLUE}----------------------------------------------------------------------${NC}"

    # Build validation command
    VALIDATE_CMD="python src/models/validate_model.py"

    if [ "$STRICT_MODE" = true ]; then
        VALIDATE_CMD="$VALIDATE_CMD --strict"
        echo -e "Using strict validation (RMSPE < 0.09856)"
    else
        echo -e "Using standard validation (RMSPE < 0.10)"
    fi

    # Run validation
    $VALIDATE_CMD || {
        echo -e "${RED}Error: Model validation failed${NC}"
        echo -e "${YELLOW}Model was trained but did not pass validation${NC}"
        echo -e "${YELLOW}Review metrics and consider:${NC}"
        echo -e "  - Retraining with adjusted hyperparameters"
        echo -e "  - Collecting more training data"
        echo -e "  - Adjusting ensemble weights"
        exit 1
    }

    echo -e "${GREEN}✓ Model validation complete and promoted to Staging${NC}"
else
    echo -e "\n${YELLOW}[Step 3/4] Model Validation - SKIPPED${NC}"
fi

################################################################################
# Step 4: Inference Testing (Optional)
################################################################################

if [ "$TEST_INFERENCE" = true ]; then
    echo -e "\n${BLUE}[Step 4/4] Inference Testing${NC}"
    echo -e "${BLUE}----------------------------------------------------------------------${NC}"

    # Test inference on Staging model
    echo -e "Running inference test on Staging model..."
    python src/models/predict.py \
        --stage Staging \
        --output-path outputs/predictions/staging_test_predictions.csv \
        || {
        echo -e "${YELLOW}Warning: Inference test failed${NC}"
        echo -e "${YELLOW}Model is in Staging but inference test had issues${NC}"
    }

    echo -e "${GREEN}✓ Inference test complete${NC}"
    echo -e "  Predictions saved to: outputs/predictions/staging_test_predictions.csv"
else
    echo -e "\n${YELLOW}[Step 4/4] Inference Testing - SKIPPED${NC}"
fi

################################################################################
# Summary
################################################################################

echo -e "\n${BLUE}======================================================================${NC}"
echo -e "${GREEN}✓ RETRAINING PIPELINE COMPLETE${NC}"
echo -e "${BLUE}======================================================================${NC}"
echo -e "Completed at: $(date)"
echo -e "\n${GREEN}Next Steps:${NC}"
echo -e "  1. Review model in MLflow UI:"
echo -e "     ${BLUE}bash scripts/start_mlflow.sh${NC}"
echo -e "\n  2. Test Staging model in production-like environment"
echo -e "\n  3. If satisfied, promote to Production:"
echo -e "     ${BLUE}python src/models/validate_model.py --promote-to-production${NC}"
echo -e "\n  4. Run production inference:"
echo -e "     ${BLUE}python src/models/predict.py --stage Production${NC}"
echo -e "${BLUE}======================================================================${NC}"

exit 0
