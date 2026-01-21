#!/bin/bash
# Complete DataOps workflow: Process, Validate, and Version Control Data
#
# Usage: bash scripts/dataops_workflow.sh

set -e  # Exit on error

echo "=================================================="
echo "   DATAOPS WORKFLOW"
echo "   Process → Validate → Version Control"
echo "=================================================="
echo ""

# Activate virtual environment
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
else
    echo "Warning: No virtual environment found. Assuming Python is in PATH."
fi

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if in project root
if [ ! -f "pyproject.toml" ]; then
    echo -e "${RED}Error: Must run from project root${NC}"
    exit 1
fi

# Check if raw data exists
echo -e "${BLUE}Checking for raw data...${NC}"
if [ ! -f "data/raw/train.csv" ] || [ ! -f "data/raw/store.csv" ]; then
    echo -e "${RED}Error: Raw data not found!${NC}"
    echo "Please ensure these files exist:"
    echo "  - data/raw/train.csv"
    echo "  - data/raw/store.csv"
    exit 1
fi
echo -e "${GREEN}✓ Raw data found${NC}"
echo ""

# Step 1: Validate Raw Data
echo -e "${BLUE}Step 1/6: Validating raw data...${NC}"
python src/data/validate_data.py --stage raw --fail-on-error
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Raw data validation passed${NC}"
else
    echo -e "${RED}✗ Raw data validation failed${NC}"
    exit 1
fi
echo ""

# Step 2: Process Data
echo -e "${BLUE}Step 2/6: Processing data...${NC}"
python -m src.data.make_dataset
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Data processing complete${NC}"
else
    echo -e "${RED}✗ Data processing failed${NC}"
    exit 1
fi
echo ""

# Step 3: Validate Processed Data
echo -e "${BLUE}Step 3/8: Validating processed data...${NC}"
python src/data/validate_data.py --stage processed --fail-on-error
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Processed data validation passed${NC}"
else
    echo -e "${RED}✗ Processed data validation failed${NC}"
    exit 1
fi
echo ""

# Step 4: Build Standard Features
echo -e "${BLUE}Step 4/8: Building standard features...${NC}"
python -m src.features.build_features
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Feature engineering complete${NC}"
else
    echo -e "${RED}✗ Feature engineering failed${NC}"
    exit 1
fi
echo ""

# Step 5: Validate Features
echo -e "${BLUE}Step 5/8: Validating features...${NC}"
python src/data/validate_data.py --stage features --fail-on-error
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Feature validation passed${NC}"
else
    echo -e "${RED}✗ Feature validation failed${NC}"
    exit 1
fi
echo ""

# Step 6: Track with DVC
echo -e "${BLUE}Step 6/8: Version controlling with DVC...${NC}"

# Strategy: Track processed/features data with DVC
# - Raw data: stays in Git (for accessibility and simple versioning)
# - Processed data: excluded from Git, tracked ONLY by DVC

# Track processed data (excluded from Git via .gitignore)
if [ -f "data/processed/train_clean.parquet" ]; then
    if [ ! -f "data/processed/train_clean.parquet.dvc" ]; then
        echo "  Tracking data/processed/train_clean.parquet with DVC..."
        dvc add data/processed/train_clean.parquet
        echo -e "${GREEN}  ✓ Added train_clean.parquet to DVC${NC}"
    else
        echo "  Updating data/processed/train_clean.parquet tracking..."
        dvc add data/processed/train_clean.parquet
        echo -e "${GREEN}  ✓ Updated train_clean.parquet in DVC${NC}"
    fi
else
    echo -e "${YELLOW}  ⊙ No processed data to track yet${NC}"
fi

# Track features if they exist
if [ -f "data/processed/train_features.parquet" ]; then
    if [ ! -f "data/processed/train_features.parquet.dvc" ]; then
        echo "  Tracking data/processed/train_features.parquet with DVC..."
        dvc add data/processed/train_features.parquet
        echo -e "${GREEN}  ✓ Added train_features.parquet to DVC${NC}"
    else
        echo "  Updating data/processed/train_features.parquet tracking..."
        dvc add data/processed/train_features.parquet
        echo -e "${GREEN}  ✓ Updated train_features.parquet in DVC${NC}"
    fi
fi

echo -e "${GREEN}✓ DVC tracking complete${NC}"
echo -e "${YELLOW}Note: Raw data versioned in Git, processed data in DVC${NC}"
echo ""

# Step 7: Generate Data Profile
echo -e "${BLUE}Step 7/8: Generating data profile...${NC}"
python -c "
import pandas as pd
import json

df = pd.read_parquet('data/processed/train_clean.parquet')

profile = {
    'rows': int(len(df)),
    'columns': int(len(df.columns)),
    'stores': int(df['Store'].nunique()),
    'date_range': {
        'min': str(df['Date'].min()),
        'max': str(df['Date'].max())
    },
    'missing_pct': round(float((df.isna().sum().sum() / df.size) * 100), 2),
}

print(json.dumps(profile, indent=2))
" > /tmp/data_profile.json

cat /tmp/data_profile.json
echo -e "${GREEN}✓ Data profile generated${NC}"
echo ""

# Step 8: Commit to Git
echo -e "${BLUE}Step 8/8: Committing to git...${NC}"

# Check if there are changes
if git diff --quiet data/ && git diff --quiet .gitignore; then
    echo -e "${YELLOW}⊙ No changes to commit${NC}"
else
    # Get data stats for commit message
    DATA_STATS=$(cat /tmp/data_profile.json | python -c "
import sys, json
data = json.load(sys.stdin)
print(f\"Rows: {data['rows']:,}, Stores: {data['stores']}, Date: {data['date_range']['min']} to {data['date_range']['max']}\")
")

    # Stage files
    git add data/processed/*.dvc .gitignore 2>/dev/null || true

    # Create commit message
    COMMIT_MSG="chore: complete DataOps workflow with standard features

Processed and validated data:
- ${DATA_STATS}
- Validation: PASSED (raw + processed + features)
- Missing data: $(cat /tmp/data_profile.json | python -c "import sys,json; print(json.load(sys.stdin)['missing_pct'])")%

Data versioning strategy:
- Raw data: versioned in Git (accessible to all users)
- Processed/features data: excluded from Git, tracked with DVC

DVC tracking:
- data/processed/train_clean.parquet
- data/processed/train_features.parquet (standard features)"

    git commit -m "$COMMIT_MSG"
    echo -e "${GREEN}✓ Changes committed to git${NC}"
fi
echo ""

# Optional: Push to DVC remote
echo -e "${BLUE}DVC Remote:${NC}"
if dvc remote list | grep -q "myremote"; then
    read -p "Push data to DVC remote? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        dvc push
        echo -e "${GREEN}✓ Data pushed to DVC remote${NC}"
    fi
else
    echo -e "${YELLOW}⊙ No DVC remote configured${NC}"
    echo "  To add a remote:"
    echo "    dvc remote add -d myremote s3://bucket/path"
fi
echo ""

# Summary
echo "=================================================="
echo -e "${GREEN}   ✓ DATAOPS WORKFLOW COMPLETE!${NC}"
echo "=================================================="
echo ""
echo "Summary:"
echo "  ✓ Raw data validated"
echo "  ✓ Data processed successfully"
echo "  ✓ Processed data validated"
echo "  ✓ Standard features built"
echo "  ✓ Features validated"
echo "  ✓ Data tracked with DVC"
echo "  ✓ Changes committed to git"
echo ""
echo "Next steps:"
echo "  - Train models: python -m src.models.train_baselines"
echo "  - Experiment with new features (ModelOps)"
echo "  - Push to GitHub: git push"
echo ""
