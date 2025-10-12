"""Evaluation metrics for multi-task regression"""

import numpy as np
from typing import Dict, List, Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def masked_mse(y_true: np.ndarray, y_pred: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """
    Calculate MSE ignoring missing values

    Args:
        y_true: True values (n_samples, n_tasks)
        y_pred: Predicted values (n_samples, n_tasks)
        mask: Boolean mask for valid values (n_samples, n_tasks)

    Returns:
        MSE score
    """
    if mask is None:
        mask = ~np.isnan(y_true)

    valid_true = y_true[mask]
    valid_pred = y_pred[mask]

    if len(valid_true) == 0:
        return np.nan

    return mean_squared_error(valid_true, valid_pred)


def masked_mae(y_true: np.ndarray, y_pred: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """
    Calculate MAE ignoring missing values

    Args:
        y_true: True values (n_samples, n_tasks)
        y_pred: Predicted values (n_samples, n_tasks)
        mask: Boolean mask for valid values (n_samples, n_tasks)

    Returns:
        MAE score
    """
    if mask is None:
        mask = ~np.isnan(y_true)

    valid_true = y_true[mask]
    valid_pred = y_pred[mask]

    if len(valid_true) == 0:
        return np.nan

    return mean_absolute_error(valid_true, valid_pred)


def masked_r2(y_true: np.ndarray, y_pred: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """
    Calculate R2 score ignoring missing values

    Args:
        y_true: True values (n_samples, n_tasks)
        y_pred: Predicted values (n_samples, n_tasks)
        mask: Boolean mask for valid values (n_samples, n_tasks)

    Returns:
        R2 score
    """
    if mask is None:
        mask = ~np.isnan(y_true)

    valid_true = y_true[mask]
    valid_pred = y_pred[mask]

    if len(valid_true) == 0:
        return np.nan

    return r2_score(valid_true, valid_pred)


def compute_multitask_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task_names: List[str],
    mask: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute metrics for multi-task regression

    Args:
        y_true: True values (n_samples, n_tasks)
        y_pred: Predicted values (n_samples, n_tasks)
        task_names: List of task names
        mask: Boolean mask for valid values (n_samples, n_tasks)

    Returns:
        Dictionary of metrics
    """
    if mask is None:
        mask = ~np.isnan(y_true)

    metrics = {}

    # Overall metrics
    metrics['overall_mse'] = masked_mse(y_true, y_pred, mask)
    metrics['overall_mae'] = masked_mae(y_true, y_pred, mask)
    metrics['overall_r2'] = masked_r2(y_true, y_pred, mask)
    metrics['overall_rmse'] = np.sqrt(metrics['overall_mse'])

    # Per-task metrics
    for i, task_name in enumerate(task_names):
        task_mask = mask[:, i] if mask.ndim > 1 else mask
        task_true = y_true[:, i]
        task_pred = y_pred[:, i]

        metrics[f'{task_name}_mse'] = masked_mse(task_true, task_pred, task_mask)
        metrics[f'{task_name}_mae'] = masked_mae(task_true, task_pred, task_mask)
        metrics[f'{task_name}_r2'] = masked_r2(task_true, task_pred, task_mask)
        metrics[f'{task_name}_rmse'] = np.sqrt(metrics[f'{task_name}_mse'])

        # Count of valid samples
        metrics[f'{task_name}_n_samples'] = np.sum(task_mask)

    return metrics


def print_metrics(metrics: Dict[str, float], prefix: str = ""):
    """Pretty print metrics"""
    if prefix:
        print(f"\n{prefix}")
        print("=" * 60)

    # Print overall metrics first
    overall_keys = ['overall_rmse', 'overall_mae', 'overall_r2']
    print("\nOverall Metrics:")
    print("-" * 60)
    for key in overall_keys:
        if key in metrics:
            print(f"  {key:20s}: {metrics[key]:.6f}")

    # Print per-task metrics
    print("\nPer-Task Metrics:")
    print("-" * 60)
    task_names = set()
    for key in metrics:
        if '_' in key and key.split('_')[0] not in ['overall']:
            task_name = key.rsplit('_', 1)[0]
            task_names.add(task_name)

    for task_name in sorted(task_names):
        print(f"\n  {task_name}:")
        for metric_type in ['rmse', 'mae', 'r2', 'n_samples']:
            key = f'{task_name}_{metric_type}'
            if key in metrics:
                value = metrics[key]
                if metric_type == 'n_samples':
                    print(f"    {metric_type:15s}: {int(value)}")
                else:
                    print(f"    {metric_type:15s}: {value:.6f}")
