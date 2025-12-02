"""
Simple script to run Phase 5 final model training and evaluation.
Run from project root: python run_phase5.py
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Now run the main function
from models.train_final import main

if __name__ == "__main__":
    main()
