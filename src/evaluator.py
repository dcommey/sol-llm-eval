"""
Evaluator for vulnerability detection performance metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import logging

logger = logging.getLogger(__name__)


class EvaluationMetrics:
    """Container for evaluation metrics."""
    
    def __init__(self, precision: float, recall: float, f1: float,
                 false_positive_rate: float, false_negative_rate: float,
                 true_positives: int, false_positives: int,
                 true_negatives: int, false_negatives: int):
        self.precision = precision
        self.recall = recall
        self.f1 = f1
        self.false_positive_rate = false_positive_rate
        self.false_negative_rate = false_negative_rate
        self.tp = true_positives
        self.fp = false_positives
        self.tn = true_negatives
        self.fn = false_negatives
    
    def to_dict(self) -> Dict:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1,
            "false_positive_rate": self.false_positive_rate,
            "false_negative_rate": self.false_negative_rate,
            "true_positives": self.tp,
            "false_positives": self.fp,
            "true_negatives": self.tn,
            "false_negatives": self.fn
        }


class VulnerabilityEvaluator:
    """Evaluates LLM performance on vulnerability detection."""
    
    def __init__(self, config: Dict):
        """
        Initialize evaluator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config['evaluation']
        self.line_tolerance = self.config['matching']['line_number_tolerance']
        self.type_matching = self.config['matching']['type_matching']
    
    def _normalize_vulnerability_type(self, vuln_type: str) -> str:
        """Normalize vulnerability type names for matching."""
        vuln_type = vuln_type.lower().strip()
        
        # Remove common prefixes/suffixes
        vuln_type = vuln_type.replace('vulnerability', '').strip()
        vuln_type = vuln_type.replace('attack', '').strip()
        
        # Map common variations to canonical names
        # CRITICAL: Must include all variations LLMs commonly return
        mappings = {
            'reentrancy': [
                'reentrancy', 'reentrant', 're-entrancy', 're entrancy',
                'reentrancy attack', 'recursive call', 'callback'
            ],
            'integer_overflow': [
                'integer_overflow', 'integer overflow', 'arithmetic', 
                'overflow', 'underflow', 'integer underflow',
                'arithmetic overflow', 'integer wraparound', 'wraparound'
            ],
            'unchecked_low_level_calls': [
                'unchecked_low_level_calls', 'unchecked low level calls',
                'unchecked-send', 'unchecked send', 'unchecked_call',
                'unchecked external call', 'unchecked return value',
                'unchecked call return', 'low level call',
                'unchecked low-level call', 'unchecked transfer',
                'unsafe external call', 'unhandled exception',
                'missing return check', 'call return not checked',
                'unchecked call', 'external call'
            ]
        }
        
        for canonical, variants in mappings.items():
            for variant in variants:
                if variant in vuln_type or vuln_type in variant:
                    return canonical
        
        return vuln_type
    
    def _match_vulnerability(self, predicted: Dict, ground_truth: List[str],
                            contract_lines: int) -> bool:
        """
        Check if a predicted vulnerability matches ground truth.
        
        Args:
            predicted: Predicted vulnerability dictionary
            ground_truth: List of ground truth vulnerability types
            contract_lines: Total lines in contract
            
        Returns:
            True if match found
        """
        pred_type = self._normalize_vulnerability_type(predicted['vulnerability_type'])
        
        # Check if type matches
        for gt_type in ground_truth:
            gt_type_norm = self._normalize_vulnerability_type(gt_type)
            if pred_type == gt_type_norm:
                return True
        
        return False
    
    def evaluate_predictions(
        self, 
        predictions: pd.DataFrame,
        ground_truth: pd.DataFrame
    ) -> EvaluationMetrics:
        """
        Evaluate predictions against ground truth.
        
        Args:
            predictions: DataFrame with model predictions
            ground_truth: DataFrame with ground truth labels
            
        Returns:
            EvaluationMetrics object
        """
        tp = 0  # True positives
        fp = 0  # False positives
        tn = 0  # True negatives
        fn = 0  # False negatives
        
        for idx, row in ground_truth.iterrows():
            contract_name = row['contract_name']
            gt_vulns = row['ground_truth_vulnerabilities']
            num_lines = row.get('num_lines', 1000)
            
            # Find predictions for this contract
            pred_row = predictions[predictions['contract_name'] == contract_name]
            
            if len(pred_row) == 0:
                # No predictions for this contract
                fn += len(gt_vulns)
                continue
            
            pred_vulns = pred_row.iloc[0]['vulnerabilities']
            
            if not pred_vulns or len(pred_vulns) == 0:
                # Model found no vulnerabilities, but there are some
                fn += len(gt_vulns)
            else:
                # Track which ground truth vulns have been matched
                matched_gt_indices = set()
                
                for pred_vuln in pred_vulns:
                    pred_type = self._normalize_vulnerability_type(pred_vuln['vulnerability_type'])
                    matched = False
                    
                    # Try to match with an unmatched ground truth vulnerability
                    for gt_idx, gt_vuln in enumerate(gt_vulns):
                        if gt_idx in matched_gt_indices:
                            continue  # Already matched
                        
                        gt_type = self._normalize_vulnerability_type(gt_vuln)
                        if pred_type == gt_type:
                            tp += 1
                            matched_gt_indices.add(gt_idx)
                            matched = True
                            break
                    
                    if not matched:
                        # Prediction doesn't match any unmatched ground truth
                        fp += 1
                
                # Unmatched ground truth = false negatives
                fn += len(gt_vulns) - len(matched_gt_indices)
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        
        logger.info(f"Evaluation: TP={tp}, FP={fp}, TN={tn}, FN={fn}")
        logger.info(f"Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")
        
        return EvaluationMetrics(
            precision=precision,
            recall=recall,
            f1=f1,
            false_positive_rate=fpr,
            false_negative_rate=fnr,
            true_positives=tp,
            false_positives=fp,
            true_negatives=tn,
            false_negatives=fn
        )
    
    def per_vulnerability_analysis(
        self,
        predictions: pd.DataFrame,
        ground_truth: pd.DataFrame,
        vulnerability_types: List[str]
    ) -> Dict[str, EvaluationMetrics]:
        """
        Compute metrics broken down by vulnerability type.
        
        Args:
            predictions: DataFrame with model predictions
            ground_truth: DataFrame with ground truth
            vulnerability_types: List of vulnerability types to analyze
            
        Returns:
            Dictionary mapping vulnerability type to metrics
        """
        results = {}
        
        for vuln_type in vulnerability_types:
            # Filter to only this vulnerability type
            gt_filtered = ground_truth[
                ground_truth['vulnerability_type'] == vuln_type
            ].copy()
            
            # Filter predictions to contracts in this subset
            contract_names = set(gt_filtered['contract_name'])
            pred_filtered = predictions[
                predictions['contract_name'].isin(contract_names)
            ].copy()
            
            # Evaluate
            metrics = self.evaluate_predictions(pred_filtered, gt_filtered)
            results[vuln_type] = metrics
            
            logger.info(f"Metrics for {vuln_type}: F1={metrics.f1:.3f}")
        
        return results
    
    def compute_confusion_matrix(
        self,
        predictions: pd.DataFrame,
        ground_truth: pd.DataFrame
    ) -> np.ndarray:
        """
        Compute confusion matrix for binary classification.
        
        Args:
            predictions: DataFrame with predictions
            ground_truth: DataFrame with ground truth
            
        Returns:
            2x2 confusion matrix
        """
        y_true = []
        y_pred = []
        
        for idx, row in ground_truth.iterrows():
            contract_name = row['contract_name']
            has_vuln = len(row['ground_truth_vulnerabilities']) > 0
            
            pred_row = predictions[predictions['contract_name'] == contract_name]
            if len(pred_row) > 0:
                pred_has_vuln = len(pred_row.iloc[0]['vulnerabilities']) > 0
            else:
                pred_has_vuln = False
            
            y_true.append(1 if has_vuln else 0)
            y_pred.append(1 if pred_has_vuln else 0)
        
        return confusion_matrix(y_true, y_pred)
