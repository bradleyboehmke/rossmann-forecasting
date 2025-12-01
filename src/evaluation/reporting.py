"""
Reporting utilities for model evaluation and results presentation.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Any
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.log import get_logger
from utils.io import ensure_dir

logger = get_logger(__name__)


def save_cv_results(
    results: Dict[str, Any],
    model_name: str,
    output_dir: str = "outputs/metrics/baseline"
) -> None:
    """
    Save cross-validation results to JSON file.

    Parameters
    ----------
    results : dict
        Dictionary with CV results including:
        - fold_scores: list of per-fold scores
        - mean_score: average score across folds
        - std_score: standard deviation across folds
        - model_name: name of the model
        - metric: name of the metric
    model_name : str
        Name of the model (used for filename)
    output_dir : str, default="outputs/metrics/baseline"
        Directory to save results
    """
    ensure_dir(output_dir)

    output_path = Path(output_dir) / f"{model_name}_cv_results.json"

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Saved CV results to {output_path}")


def print_cv_summary(
    results: Dict[str, Any],
    model_name: str = None
) -> None:
    """
    Print a formatted summary of cross-validation results.

    Parameters
    ----------
    results : dict
        Dictionary with CV results
    model_name : str, optional
        Model name to display (overrides results['model_name'])
    """
    name = model_name or results.get('model_name', 'Model')
    metric = results.get('metric', 'Score')
    fold_scores = results.get('fold_scores', [])
    mean_score = results.get('mean_score', np.nan)
    std_score = results.get('std_score', np.nan)

    print("=" * 60)
    print(f"{name} - Cross-Validation Results")
    print("=" * 60)
    print(f"Metric: {metric}")
    print(f"\nPer-fold scores:")
    for i, score in enumerate(fold_scores, 1):
        print(f"  Fold {i}: {score:.6f}")
    print(f"\nMean {metric}: {mean_score:.6f}")
    print(f"Std  {metric}: {std_score:.6f}")
    print(f"CV Range: [{mean_score - std_score:.6f}, {mean_score + std_score:.6f}]")
    print("=" * 60)


def compare_models(
    results_dict: Dict[str, Dict[str, Any]],
    metric: str = 'RMSPE'
) -> pd.DataFrame:
    """
    Compare multiple models and create a summary dataframe.

    Parameters
    ----------
    results_dict : dict
        Dictionary mapping model names to their CV results
    metric : str, default='RMSPE'
        Metric name for display

    Returns
    -------
    pd.DataFrame
        Comparison dataframe sorted by mean score (ascending for RMSPE)
    """
    comparison = []

    for model_name, results in results_dict.items():
        # Ensure fold_scores are floats (in case loaded from JSON as strings)
        fold_scores = [float(s) for s in results.get('fold_scores', [])]

        comparison.append({
            'Model': model_name,
            f'Mean {metric}': float(results.get('mean_score', np.nan)),
            f'Std {metric}': float(results.get('std_score', np.nan)),
            f'Min {metric}': min(fold_scores) if fold_scores else np.nan,
            f'Max {metric}': max(fold_scores) if fold_scores else np.nan,
            'Num Folds': len(fold_scores)
        })

    df = pd.DataFrame(comparison)

    # Sort by mean score (ascending for error metrics like RMSPE)
    df = df.sort_values(f'Mean {metric}', ascending=True).reset_index(drop=True)

    return df


def create_cv_summary_table(
    fold_results: List[Dict[str, Any]]
) -> pd.DataFrame:
    """
    Create a summary table from detailed fold results.

    Parameters
    ----------
    fold_results : list of dict
        List of dictionaries, one per fold, with keys:
        - fold: fold number
        - train_size: number of training samples
        - val_size: number of validation samples
        - score: fold score
        - train_time: training time (optional)

    Returns
    -------
    pd.DataFrame
        Summary table
    """
    return pd.DataFrame(fold_results)


def save_predictions(
    predictions: pd.DataFrame,
    model_name: str,
    output_dir: str = "outputs/predictions"
) -> None:
    """
    Save model predictions to CSV file.

    Parameters
    ----------
    predictions : pd.DataFrame
        Predictions dataframe with columns like:
        - Store, Date, Sales (actual), Predictions
    model_name : str
        Model name (used for filename)
    output_dir : str, default="outputs/predictions"
        Directory to save predictions
    """
    ensure_dir(output_dir)

    output_path = Path(output_dir) / f"{model_name}_predictions.csv"
    predictions.to_csv(output_path, index=False)

    logger.info(f"Saved predictions to {output_path}")


def calculate_summary_stats(
    scores: List[float]
) -> Dict[str, float]:
    """
    Calculate summary statistics for a list of scores.

    Parameters
    ----------
    scores : list of float
        List of scores

    Returns
    -------
    dict
        Dictionary with mean, std, min, max, median
    """
    return {
        'mean': np.mean(scores),
        'std': np.std(scores),
        'min': np.min(scores),
        'max': np.max(scores),
        'median': np.median(scores)
    }
