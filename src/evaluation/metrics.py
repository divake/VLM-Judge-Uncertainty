"""Evaluation metrics wrapper.

Wraps existing conformal_predictors/evaluation_metrics.py and
adds VLM-specific metrics (SSC, midpoint correlation).
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Import existing utilities
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from conformal_predictors.evaluation_metrics import (
    calculate_correlations,
    calculate_errors,
)
from conformal_predictors.interval_processing import (
    calculate_coverage_and_width,
)


def compute_all_metrics(intervals_df: pd.DataFrame, gt_col: str = "y_test",
                        low_col: str = "low", up_col: str = "up") -> dict:
    """Compute all metrics for a set of prediction intervals.

    Args:
        intervals_df: DataFrame with columns for low, up, y_test
        gt_col: Column name for ground truth
        low_col: Column name for lower bound
        up_col: Column name for upper bound

    Returns:
        Dict with all metrics.
    """
    y_test = intervals_df[gt_col].values
    low = intervals_df[low_col].values
    up = intervals_df[up_col].values
    midpoints = (low + up) / 2

    # Coverage and width
    width_mean, coverage = calculate_coverage_and_width(low, up, y_test)

    # Correlations of midpoint with GT
    pearson_r, pearson_p = pearsonr(y_test, midpoints)
    spearman_r, spearman_p = spearmanr(y_test, midpoints)
    kendall_r, kendall_p = kendalltau(y_test, midpoints)

    # Errors
    mse = mean_squared_error(y_test, midpoints)
    mae = mean_absolute_error(y_test, midpoints)
    rmse = mse ** 0.5

    # Size-stratified coverage (SSC)
    ssc = compute_ssc(low, up, y_test, n_strata=5)

    return {
        "coverage": float(coverage),
        "width_mean": float(width_mean),
        "pearson_r": float(pearson_r),
        "spearman_r": float(spearman_r),
        "kendall_tau": float(kendall_r),
        "mse": float(mse),
        "mae": float(mae),
        "rmse": float(rmse),
        "ssc_violation": float(ssc["max_violation"]),
        "ssc_per_stratum": ssc["per_stratum"],
    }


def compute_point_prediction_metrics(judge_scores: np.ndarray,
                                     gt_scores: np.ndarray) -> dict:
    """Compute correlation and error metrics for point predictions.

    Compares judge VLM predicted scores against human GT scores.

    Args:
        judge_scores: Judge-assigned scores (extracted from generated text)
        gt_scores: Human ground truth scores

    Returns:
        Dict with Pearson, Spearman, Kendall, MSE, MAE, RMSE, accuracy.
    """
    # Filter out any NaN
    mask = ~(np.isnan(judge_scores) | np.isnan(gt_scores))
    js = judge_scores[mask]
    gs = gt_scores[mask]

    if len(js) < 3:
        return {"error": "Too few valid samples"}

    pearson_r, pearson_p = pearsonr(gs, js)
    spearman_r, spearman_p = spearmanr(gs, js)
    kendall_r, kendall_p = kendalltau(gs, js)

    mse = mean_squared_error(gs, js)
    mae = mean_absolute_error(gs, js)
    rmse = mse ** 0.5

    # Exact match accuracy
    accuracy = float(np.mean(js.round() == gs.round()))

    return {
        "n_samples": int(len(js)),
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
        "kendall_tau": float(kendall_r),
        "kendall_p": float(kendall_p),
        "mse": float(mse),
        "mae": float(mae),
        "rmse": float(rmse),
        "accuracy": accuracy,
    }


def compute_ssc(low: np.ndarray, up: np.ndarray, y_test: np.ndarray,
                n_strata: int = 5, target_coverage: float = 0.90) -> dict:
    """Compute Size-Stratified Coverage (SSC).

    Groups samples by interval width and checks coverage within each stratum.
    Reports max violation (worst stratum deviation from target).

    Args:
        low, up: Interval bounds
        y_test: Ground truth scores
        n_strata: Number of width strata
        target_coverage: Target coverage level

    Returns:
        Dict with per-stratum coverage and max violation.
    """
    widths = up - low
    in_interval = (low <= y_test) & (y_test <= up)

    # Sort by width and split into strata
    sort_idx = np.argsort(widths)
    stratum_size = len(sort_idx) // n_strata
    per_stratum = []

    for i in range(n_strata):
        start = i * stratum_size
        end = start + stratum_size if i < n_strata - 1 else len(sort_idx)
        stratum_idx = sort_idx[start:end]
        stratum_coverage = float(np.mean(in_interval[stratum_idx]))
        stratum_width = float(np.mean(widths[stratum_idx]))
        per_stratum.append({
            "stratum": i,
            "coverage": stratum_coverage,
            "avg_width": stratum_width,
            "n_samples": len(stratum_idx),
        })

    violations = [target_coverage - s["coverage"] for s in per_stratum]
    max_violation = max(0, max(violations))

    return {
        "per_stratum": per_stratum,
        "max_violation": max_violation,
    }
