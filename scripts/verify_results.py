
import json
import pandas as pd
import numpy as np
from pathlib import Path

# --- independent re-implementation of metric logic ---

def normalize_type(vuln_type):
    v = vuln_type.lower().strip()
    v = v.replace('vulnerability', '').replace('attack', '').strip()
    
    mapping = {
        'reentrancy': ['reentrancy', 'reentrant', 'recursive call'],
        'integer_overflow': ['integer_overflow', 'integer overflow', 'overflow', 'underflow', 'arithmetic'],
        'unchecked_low_level_calls': ['unchecked', 'low level call', 'external call', 'return value', 'unhandled exception']
    }
    
    for key, variants in mapping.items():
        if key in v: return key
        for var in variants:
            if var in v: return key
    return v

def calculate_metrics(predictions, ground_truth):
    # predictions: dict {contract_name: [vuln1, vuln2]}
    # ground_truth: dict {contract_name: [vuln1, vuln2]}
    
    tp = 0; fp = 0; fn = 0
    
    # For FP rate on clean contracts
    clean_contracts = [c for c, v in ground_truth.items() if len(v) == 0]
    
    fp_clean = 0
    tn_clean = 0
    
    for contract, gt_vulns in ground_truth.items():
        pred_vulns = predictions.get(contract, [])
        
        # Normalize
        gt_norm = [normalize_type(v) for v in gt_vulns]
        pred_norm = [normalize_type(v['vulnerability_type']) for v in pred_vulns if 'vulnerability_type' in v]
        
        # Matching
        matched_gt = set()
        contract_tp = 0
        contract_fp = 0
        
        for p in pred_norm:
            match_found = False
            for i, g in enumerate(gt_norm):
                if i not in matched_gt and p == g:
                    matched_gt.add(i)
                    match_found = True
                    break
            if match_found:
                contract_tp += 1
            else:
                contract_fp += 1
        
        contract_fn = len(gt_norm) - len(matched_gt)
        
        tp += contract_tp
        fp += contract_fp
        fn += contract_fn
        
        # Contract-level stats for Clean contracts
        if len(gt_vulns) == 0:
            if len(pred_vulns) > 0:
                fp_clean += 1
            else:
                tn_clean += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    fpr_clean = fp_clean / (fp_clean + tn_clean) if (fp_clean + tn_clean) > 0 else 0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr_clean": fpr_clean,
        "tp": tp, "fp": fp, "fn": fn
    }

def main():
    base_dir = Path("/Volumes/ExtSystem/Users/seraphic/Documents/dev/sol-llm-eval")
    raw_data_path = base_dir / "data/raw/combined_dataset.json"
    pred_dir = base_dir / "results/predictions"
    
    # Load Ground Truth
    with open(raw_data_path, 'r') as f:
        dataset = json.load(f)
    
    # Build GT dict
    gt_map = {}
    for entry in dataset:
        name = entry.get('contract_name')
        if not name:
             # Try to get from path
             name = Path(entry['contract_path']).name
             
        vulns = entry.get('ground_truth_vulnerabilities', [])
        # Extract types
        v_types = []
        for v in vulns:
            if isinstance(v, dict):
                v_types.append(v.get('vulnerability_type', ''))
            else:
                v_types.append(v)
        gt_map[name] = v_types


    # DEBUG: Print first contract details
    print(f"Loaded {len(gt_map)} contracts from ground truth.")
    
    models = ['qwen', 'mistral', 'deepseek', 'codellama']

    # DEBUG: Print first contract details
    first_contract = list(gt_map.keys())[0]
    print(f"\nDEBUG: Contract '{first_contract}'")
    print(f"  GT: {gt_map[first_contract]}")
    
    for m in models:
        pred_path = pred_dir / f"{m}_predictions.json"
        if not pred_path.exists():
            continue
        with open(pred_path, 'r') as f:
            preds_raw = json.load(f)
        
        # Find prediction for this contract
        for p in preds_raw:
            if p.get('contract_name') == first_contract:
                print(f"  {m} Raw Pred: {p.get('vulnerabilities')}")
                break

    print("-" * 60)
    
    for m in models:

        pred_path = pred_dir / f"{m}_predictions.json"
        if not pred_path.exists():
            print(f"Skipping {m}: not found")
            continue
            
        with open(pred_path, 'r') as f:
            preds_raw = json.load(f)
            
        # Build Pred map
        pred_map = {}
        for p in preds_raw:
            c_name = p.get('contract_name')
            vulns = p.get('vulnerabilities', [])
            pred_map[c_name] = vulns
            
        metrics = calculate_metrics(pred_map, gt_map)
        
        print(f"{m:<15} {metrics['precision']:.3f}      {metrics['recall']:.3f}      {metrics['f1']:.3f}      {metrics['fpr_clean']:.3f}")

if __name__ == "__main__":
    main()
