#!/usr/bin/env python3
"""Recompute all manuscript measurements; fail on partial or mismatched runs."""
import itertools
import copy
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.run_ccnc_experiment import ROOT, MODELS, CLASSES, save, digest, parse
from src.evaluator import VulnerabilityEvaluator

def align(rows,data):
    by_name={r['contract_name']:r for r in rows}
    if len(by_name)!=len(rows) or set(by_name)!={r['contract_name'] for r in data}:
        raise ValueError('Incomplete, duplicate, or foreign predictions')
    ordered=[by_name[r['contract_name']] for r in data]
    if any(p['source_sha256']!=r['source_sha256'] for p,r in zip(ordered,data)):
        raise ValueError('Prediction/source hash mismatch')
    return ordered

def counts(rows,data,classes=CLASSES):
    permitted=set(classes)
    values=[]
    for p,g in zip(rows,data):
        pred={v['vulnerability_type'] for v in p['vulnerabilities']} & permitted
        truth=set(g['ground_truth_vulnerabilities']) & permitted
        values.append([len(pred&truth),len(pred-truth),len(truth-pred)])
    return np.array(values,dtype=int)

def f1(c):
    return np.divide(2*c[...,0],2*c[...,0]+c[...,1]+c[...,2],
                     out=np.zeros(c.shape[:-1],dtype=float),where=(2*c[...,0]+c[...,1]+c[...,2])!=0)

def summarize(rows,data,classes=CLASSES,cluster_pairs=False):
    c=counts(rows,data,classes);tp,fp,fn=c.sum(axis=0)
    precision=tp/(tp+fp) if tp+fp else 0
    recall=tp/(tp+fn) if tp+fn else 0
    rng=np.random.default_rng(42)
    if cluster_pairs:
        groups={}
        for index,r in enumerate(data):groups.setdefault(r['pair_id'],[]).append(index)
        clusters=np.array([c[indices].sum(axis=0) for indices in groups.values()])
    else:clusters=c
    draws=rng.integers(0,len(clusters),size=(10000,len(clusters)))
    bootstrap=f1(clusters[draws].sum(axis=1))
    clean=[i for i,g in enumerate(data) if not g['ground_truth_vulnerabilities']]
    clean_fp=sum(bool({v['vulnerability_type'] for v in rows[i]['vulnerabilities']} & set(classes)) for i in clean)
    failed=lambda row:bool(row.get('parse_error')) or not row.get('success',True)
    clean_failed=sum(failed(rows[i]) for i in clean)
    clean_attention=sum(failed(rows[i]) or bool({v['vulnerability_type'] for v in rows[i]['vulnerabilities']} & set(classes)) for i in clean)
    implement=[i for i in clean if re.search(r'\b(?:contract|library)\s+\w+',data[i]['contract_code'])]
    implement_fp=sum(bool({v['vulnerability_type'] for v in rows[i]['vulnerabilities']} & set(classes)) for i in implement)
    implement_failed=sum(failed(rows[i]) for i in implement)
    n=len(clean);p=clean_fp/n if n else 0;z=1.959963984540054
    denom=1+z*z/n if n else 1
    center=(p+z*z/(2*n))/denom if n else 0
    half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/denom if n else 0
    latency=np.array([r['inference_time'] for r in rows])
    return dict(tp=int(tp),fp=int(fp),fn=int(fn),precision=float(precision),recall=float(recall),f1=float(f1(c.sum(axis=0))),
                f1_ci=[float(np.quantile(bootstrap,.025)),float(np.quantile(bootstrap,.975))],
                bootstrap_iterations=10000,bootstrap_unit='pair' if cluster_pairs else 'contract',
                clean_n=n,clean_fp=clean_fp,clean_failed=clean_failed,clean_attention=clean_attention,clean_valid_coverage=(n-clean_failed)/n if n else None,clean_fpr=p,clean_fpr_wilson95=[center-half,center+half],
                implementation_negative_n=len(implement),implementation_negative_fp=implement_fp,implementation_negative_failed=implement_failed,
                implementation_negative_fpr=implement_fp/len(implement) if implement else None,
                parse_failures=sum(bool(r.get('parse_error')) for r in rows),
                truncated_outputs=sum(r.get('done_reason')=='length' for r in rows),
                tool_failures=sum(not r.get('success',True) for r in rows),
                mean_latency=float(latency.mean()),median_latency=float(np.median(latency)),
                p95_latency=float(np.quantile(latency,.95)),std_latency=float(latency.std(ddof=1)))

def paired(a,b,data,classes=CLASSES):
    ca=counts(a,data,classes);cb=counts(b,data,classes)
    rng=np.random.default_rng(42);draws=rng.integers(0,len(data),size=(10000,len(data)))
    deltas=f1(ca[draws].sum(axis=1))-f1(cb[draws].sum(axis=1))
    return dict(delta=float(f1(ca.sum(axis=0))-f1(cb.sum(axis=0))),ci=[float(np.quantile(deltas,.025)),float(np.quantile(deltas,.975))],
                n=len(data),iterations=10000,unit='paired contract',confidence=.95,multiplicity='unadjusted descriptive intervals')

def pair_metrics(rows,data):
    groups={}
    for i,g in enumerate(data):groups.setdefault(g['pair_id'],[]).append(i)
    label_match=lambda i:{v['vulnerability_type'] for v in rows[i]['vulnerabilities']}==set(data[i]['ground_truth_vulnerabilities'])
    valid=lambda i:not rows[i].get('parse_error') and rows[i].get('success',True)
    return dict(pairs_label_match=sum(all(label_match(i) for i in indices) for indices in groups.values()),
                pairs_exact=sum(all(valid(i) and label_match(i) for i in indices) for indices in groups.values()),
                pairs_n=len(groups),pairs_definition='Both sources yield valid successful outputs with exact target-category matches; invalid repair output is unanswered, not correct.')

CLOUD='claude'
FENCE=re.compile(r'```(?:json)?\s*(\[[\s\S]*?\])\s*```')

def last_json_block(raw):
    """Sensitivity parser for the cloud reference: the final fenced JSON array, prose ignored."""
    blocks=FENCE.findall(raw)
    return parse(blocks[-1]) if blocks else parse(raw)

def main():
    root=ROOT/'results/ccnc2027';report={};validation=[];sensitivity={}
    provenance_checks=[]
    for model in MODELS:
        main_meta=json.loads((root/f'metadata/main_{model}.json').read_text())
        fresh_meta=json.loads((root/f'metadata/fresh_{model}.json').read_text())
        for field in ['tag','digest','template','parameters','ollama_version','hardware','memory_bytes','platform','protocol_sha256']:
            if main_meta[field]!=fresh_meta[field]:raise ValueError(f'Main/fresh provenance mismatch: {model}/{field}')
        provenance_checks.append(dict(model=model,main_fresh_artifact_and_runtime_match=True))
    allrows={};all_data={}
    for suite in ['main','fresh']:
        data=json.loads((root/f'datasets/{suite}.json').read_text());all_data[suite]=data
        summary={};rows_by_model={};sensitivity[suite]={}
        for model in list(MODELS)+[CLOUD,'slither']:
            rows=align(json.loads((root/f'predictions/{suite}_{model}.json').read_text()),data)
            rows_by_model[model]=rows
            metrics=summarize(rows,data,cluster_pairs=suite=='fresh')
            metrics['per_class']={c:summarize(rows,data,[c],cluster_pairs=suite=='fresh') for c in CLASSES}
            # Independent reconciliation against the repository evaluator.
            evaluator=VulnerabilityEvaluator({'evaluation':{'matching':{'line_number_tolerance':0,'type_matching':'exact'}}})
            original=evaluator.evaluate_predictions(pd.DataFrame(rows),pd.DataFrame(data))
            assert np.isclose(original.f1,metrics['f1'])
            assert original.tp==metrics['tp'] and original.fp==metrics['fp'] and original.fn==metrics['fn']
            assert np.isclose(original.false_positive_rate,metrics['clean_fpr'])
            if model=='slither':
                indices=[i for i,r in enumerate(rows) if r['success']]
                metrics['coverage']=len(indices)/len(data)
                metrics['conditional']=summarize([rows[i] for i in indices],[data[i] for i in indices],cluster_pairs=False)
            if model in MODELS:
                recovered=copy.deepcopy(rows);changed=[]
                for i,row in enumerate(recovered):
                    raw=row['raw_response'].strip()
                    if row['parse_error'] and raw.startswith('[') and raw.endswith('```'):
                        reports,error=parse(raw[:-3].strip())
                        if error is None:
                            row['vulnerabilities']=reports;row['parse_error']=None;changed.append(row['contract_name'])
                alternative=summarize(recovered,data,cluster_pairs=suite=='fresh')
                verification=evaluator.evaluate_predictions(pd.DataFrame(recovered),pd.DataFrame(data))
                assert verification.tp==alternative['tp'] and verification.fp==alternative['fp'] and verification.fn==alternative['fn']
                if suite=='fresh':
                    alternative.update(pair_metrics(recovered,data))
                sensitivity[suite][model]=dict(recovered_count=len(changed),recovered_sources=changed,primary_f1=metrics['f1'],alternative=alternative)
            if model==CLOUD:
                recovered=copy.deepcopy(rows);changed=[]
                for row in recovered:
                    if row['parse_error'] and row['parse_error']!='refusal':
                        reports,error=last_json_block(row['raw_response'])
                        if error is None:
                            row['vulnerabilities']=reports;row['parse_error']=None;changed.append(row['contract_name'])
                alternative=summarize(recovered,data,cluster_pairs=suite=='fresh')
                alternative['per_class']={c:summarize(recovered,data,[c],cluster_pairs=suite=='fresh') for c in CLASSES}
                verification=evaluator.evaluate_predictions(pd.DataFrame(recovered),pd.DataFrame(data))
                assert verification.tp==alternative['tp'] and verification.fp==alternative['fp'] and verification.fn==alternative['fn']
                if suite=='fresh':
                    alternative.update(pair_metrics(recovered,data))
                sensitivity[suite][model]=dict(rule='Final fenced JSON array, surrounding prose ignored; identical schema validation',recovered_count=len(changed),recovered_sources=changed,primary_f1=metrics['f1'],alternative=alternative)
                allrows.setdefault('cloud_lenient',{})[suite]=recovered
            if suite=='fresh':
                metrics.update(pair_metrics(rows,data))
            summary[model]=metrics
            validation.append(dict(suite=suite,model=model,contracts=len(rows),hash_alignment=True,independent_count_reconciliation=True))
        report[suite]=dict(contracts=len(data),positive_labels=sum(len(r['ground_truth_vulnerabilities']) for r in data),models=summary,dataset_sha256=digest(data))
        allrows[suite]=rows_by_model
    data=all_data['main'];rows=allrows['main']
    supported=[i for i,r in enumerate(data) if r['ground_truth_vulnerabilities']!=['integer_overflow']]
    restricted=[data[i] for i in supported]
    categories=['reentrancy','unchecked_low_level_calls']
    report['supported_categories']=dict(contracts=len(restricted),categories=categories,models={},comparisons={})
    for model,rr in rows.items():
        subset=[rr[i] for i in supported]
        report['supported_categories']['models'][model]=summarize(subset,restricted,categories)
        if model!='slither':report['supported_categories']['comparisons'][model+'_minus_slither']=paired(subset,[rows['slither'][i] for i in supported],restricted,categories)
    lenient=allrows['cloud_lenient']['main'];subset=[lenient[i] for i in supported]
    report['supported_categories']['cloud_lenient']=dict(model=CLOUD,metrics=summarize(subset,restricted,categories),
        minus_slither=paired(subset,[rows['slither'][i] for i in supported],restricted,categories))
    repeat=ROOT/'results/ccnc2027/cloud_repeat'
    agreement={}
    for suite in ['main','fresh']:
        path=repeat/f'{suite}_{CLOUD}_run2.json'
        if path.exists() and len(json.loads(path.read_text()))==len(all_data[suite]):
            second=align(json.loads(path.read_text()),all_data[suite]);first=allrows[suite][CLOUD]
            same_validity=sum(bool(a['parse_error'])==bool(b['parse_error']) for a,b in zip(first,second))
            fa=[last_json_block(r['raw_response'])[0] for r in first];fb=[last_json_block(r['raw_response'])[0] for r in second]
            same_sets=sum({v['vulnerability_type'] for v in a}=={v['vulnerability_type'] for v in b} for a,b in zip(fa,fb))
            agreement[suite]=dict(n=len(first),same_validity=same_validity,same_lenient_category_sets=same_sets,
                run2_primary=summarize(second,all_data[suite],cluster_pairs=suite=='fresh'))
    report['cloud_repeat_agreement']=agreement
    report['three_class_comparisons']={a+'_minus_'+b:paired(rows[a],rows[b],data) for a,b in itertools.combinations(rows,2)}
    report['schema_version']=2
    save(root/'evaluations/summary.json',report)
    save(root/'evaluations/parsing_sensitivity.json',dict(primary_unchanged=True,exploratory_posthoc=True,rule='For a failed response beginning with [ and ending with a lone closing Markdown fence, remove only that fence and apply the identical full schema parser. No missing delimiters or prose repaired.',results=sensitivity,independent_count_reconciliation=True))
    save(root/'validation/metrics_reconciliation.json',dict(status='passed',checks=validation,provenance_checks=provenance_checks,
        definitions='Micro TP/FP/FN at deduplicated target-category level; clean FPR at contract level; tool failures are abstentions/empty operational predictions.'))
    for suite in ['main','fresh']:
        print(suite)
        for model,m in report[suite]['models'].items():print(model,'P/R/F1',round(m['precision'],3),round(m['recall'],3),round(m['f1'],3),'FP',m['clean_fp'],'mean_s',round(m['mean_latency'],2))

if __name__=='__main__':main()
