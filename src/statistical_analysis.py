"""
Statistical analysis utilities for rigorous benchmarking.
Provides bootstrap confidence intervals and significance tests.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats
import logging

logger = logging.getLogger(__name__)


def bootstrap_confidence_interval(
    data: List[float], 
    n_bootstrap: int = 1000, 
    confidence: float = 0.95,
    seed: int = 42
) -> Tuple[float, float, float]:
    """
    Calculate bootstrap confidence interval for a metric.
    
    Returns:
        (mean, lower_ci, upper_ci)
    """
    if len(data) == 0:
        return (0.0, 0.0, 0.0)
    
    np.random.seed(seed)
    data_array = np.array(data)
    
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data_array, size=len(data_array), replace=True)
        bootstrap_means.append(np.mean(sample))
    
    mean = np.mean(data_array)
    lower = np.percentile(bootstrap_means, (1 - confidence) / 2 * 100)
    upper = np.percentile(bootstrap_means, (1 + confidence) / 2 * 100)
    
    return (mean, lower, upper)


def bootstrap_f1_confidence_interval(
    predictions: List[Dict],
    ground_truth: List[Dict],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42
) -> Tuple[float, float, float]:
    """
    Calculate bootstrap CI for F1 score by resampling contracts.
    
    Returns:
        (f1_mean, f1_lower, f1_upper)
    """
    np.random.seed(seed)
    n_contracts = len(predictions)
    
    # Calculate per-contract TP/FP/FN
    per_contract_metrics = []
    for pred, gt in zip(predictions, ground_truth):
        pred_vulns = set(v.get('vulnerability_type', '') for v in pred.get('vulnerabilities', []))
        gt_vulns = set(gt.get('vulnerability_types', []))
        
        tp = len(pred_vulns & gt_vulns)
        fp = len(pred_vulns - gt_vulns)
        fn = len(gt_vulns - pred_vulns)
        
        per_contract_metrics.append({'tp': tp, 'fp': fp, 'fn': fn})
    
    # Bootstrap F1
    bootstrap_f1s = []
    for _ in range(n_bootstrap):
        indices = np.random.choice(n_contracts, size=n_contracts, replace=True)
        
        total_tp = sum(per_contract_metrics[i]['tp'] for i in indices)
        total_fp = sum(per_contract_metrics[i]['fp'] for i in indices)
        total_fn = sum(per_contract_metrics[i]['fn'] for i in indices)
        
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        bootstrap_f1s.append(f1)
    
    f1_mean = np.mean(bootstrap_f1s)
    f1_lower = np.percentile(bootstrap_f1s, (1 - confidence) / 2 * 100)
    f1_upper = np.percentile(bootstrap_f1s, (1 + confidence) / 2 * 100)
    
    return (f1_mean, f1_lower, f1_upper)


def mcnemar_test(model1_correct: List[bool], model2_correct: List[bool]) -> Tuple[float, float]:
    """
    Perform McNemar's test to compare two models.
    
    Returns:
        (chi2_statistic, p_value)
    """
    # Build contingency table
    b = sum(1 for c1, c2 in zip(model1_correct, model2_correct) if c1 and not c2)
    c = sum(1 for c1, c2 in zip(model1_correct, model2_correct) if not c1 and c2)
    
    if b + c == 0:
        return (0.0, 1.0)  # No difference
    
    # McNemar's test statistic (with continuity correction)
    chi2 = (abs(b - c) - 1) ** 2 / (b + c) if (b + c) > 0 else 0
    p_value = 1 - stats.chi2.cdf(chi2, df=1)
    
    return (chi2, p_value)


def cohens_kappa(y_true: List[int], y_pred: List[int]) -> float:
    """Calculate Cohen's Kappa for inter-rater agreement."""
    if len(y_true) != len(y_pred) or len(y_true) == 0:
        return 0.0
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Observed agreement
    po = np.mean(y_true == y_pred)
    
    # Expected agreement
    p_true_1 = np.mean(y_true == 1)
    p_pred_1 = np.mean(y_pred == 1)
    pe = p_true_1 * p_pred_1 + (1 - p_true_1) * (1 - p_pred_1)
    
    if pe == 1.0:
        return 1.0
    
    kappa = (po - pe) / (1 - pe)
    return kappa


def calculate_all_statistics(
    results: Dict[str, Dict],
    predictions_by_model: Dict[str, List[Dict]],
    ground_truth: List[Dict],
    n_bootstrap: int = 1000,
    seed: int = 42
) -> Dict[str, Dict]:
    """
    Calculate comprehensive statistics for all models.
    
    Returns:
        Dictionary with CIs and significance tests for each model.
    """
    stats_results = {}
    
    model_names = list(results.keys())
    
    for model_name in model_names:
        preds = predictions_by_model.get(model_name, [])
        
        # Bootstrap CI for F1
        f1_mean, f1_lower, f1_upper = bootstrap_f1_confidence_interval(
            preds, ground_truth, n_bootstrap=n_bootstrap, seed=seed
        )
        
        stats_results[model_name] = {
            'f1_bootstrap': {
                'mean': f1_mean,
                'ci_lower': f1_lower,
                'ci_upper': f1_upper,
                'ci_width': f1_upper - f1_lower
            }
        }
    
    # Pairwise McNemar's tests
    for i, model1 in enumerate(model_names):
        for model2 in model_names[i+1:]:
            preds1 = predictions_by_model.get(model1, [])
            preds2 = predictions_by_model.get(model2, [])
            
            if len(preds1) != len(preds2) or len(preds1) == 0:
                continue
            
            # Build correctness vectors
            correct1 = []
            correct2 = []
            
            for p1, p2, gt in zip(preds1, preds2, ground_truth):
                gt_vulns = set(gt.get('vulnerability_types', []))
                
                pred1_vulns = set(v.get('vulnerability_type', '') for v in p1.get('vulnerabilities', []))
                pred2_vulns = set(v.get('vulnerability_type', '') for v in p2.get('vulnerabilities', []))
                
                # Correct if any ground truth vulnerability detected
                correct1.append(len(pred1_vulns & gt_vulns) > 0)
                correct2.append(len(pred2_vulns & gt_vulns) > 0)
            
            chi2, p_value = mcnemar_test(correct1, correct2)
            
            comparison_key = f"{model1}_vs_{model2}"
            if comparison_key not in stats_results:
                stats_results[comparison_key] = {}
            
            stats_results[comparison_key]['mcnemar'] = {
                'chi2': chi2,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
    
    return stats_results


def format_ci_for_latex(mean: float, lower: float, upper: float) -> str:
    """Format confidence interval for LaTeX table."""
    return f"{mean:.3f} [{lower:.3f}, {upper:.3f}]"
