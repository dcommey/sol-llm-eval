#!/usr/bin/env python3
"""Targeted Slither baseline; preserve failures, compiler versions, and raw output."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.run_ccnc_experiment import ROOT, save, digest
from src.slither_baseline import SlitherBaseline

DETECTORS = {'reentrancy-eth':'reentrancy','reentrancy-no-eth':'reentrancy',
             'unchecked-lowlevel':'unchecked_low_level_calls','unchecked-send':'unchecked_low_level_calls'}

def run(root,suite):
    data=json.loads((root/f'datasets/{suite}.json').read_text())
    destination=root/f'predictions/{suite}_slither.json'
    rows=json.loads(destination.read_text()) if destination.exists() else []
    done={r['contract_name'] for r in rows}
    env={**os.environ,'PATH':str(ROOT/'.venv/bin')+os.pathsep+os.environ['PATH']}
    version=subprocess.check_output(['slither','--version'],env=env,text=True).strip()
    manifest=dict(slither_version=version,detectors=DETECTORS,compiler_bridge_sha256=hashlib.sha256((ROOT/'tools/solc-js/solc-bridge.cjs').read_bytes()).hexdigest(),
                  dataset_sha256=digest(data),failure_policy='operational: empty prediction; conditional: exclude failed analysis',
                  compiler_versions=['0.4.25','0.5.17','0.8.27'],compiler_distribution='official pinned solc-js npm packages',optimizer=dict(enabled=True,runs=200))
    manifest_path=root/f'metadata/{suite}_slither.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text())!=manifest:
        raise RuntimeError('Baseline provenance changed; cannot resume')
    save(manifest_path,manifest)
    for row in data:
        if row['contract_name'] in done:continue
        # Main input is semantically identical apart from comment removal.
        source=ROOT/row['contract_path']
        compiler=SlitherBaseline._compiler_version_for(str(source))
        start=time.perf_counter()
        cmd=['slither',str(source),'--compile-force-framework','solc','--solc',str(ROOT/'tools/solc-js/solc-bridge.cjs'),
             '--detect',','.join(DETECTORS),'--json','-','--fail-none']
        proc=subprocess.run(cmd,capture_output=True,text=True,timeout=120,env={**env,'SOLC_VERSION':compiler})
        raw=None
        try:raw=json.loads(proc.stdout)
        except ValueError:pass
        success=bool(raw and raw.get('success') and proc.returncode==0)
        reports=[]
        if success:
            for d in raw.get('results',{}).get('detectors',[]):
                if d['check'] in DETECTORS:
                    reports.append(dict(vulnerability_type=DETECTORS[d['check']],detector=d['check'],severity=d.get('impact'),explanation=d['description']))
        rows.append(dict(contract_name=row['contract_name'],vulnerabilities=reports,success=success,
                         compiler_version=compiler,inference_time=time.perf_counter()-start,
                         source_sha256=row['source_sha256'],error=None if success else ((raw.get('error') if raw else None) or proc.stderr or f'Exit {proc.returncode} without diagnostic')[-3000:],
                         command=cmd))
        save(root/f'raw/slither/{suite}_{row["contract_name"]}.json',dict(returncode=proc.returncode,stdout=raw or proc.stdout,stderr=proc.stderr))
        save(destination,rows)
        if len(rows)%10==0 or not success: print(f'{suite}/slither {len(rows)}/{len(data)}; failures {sum(not r["success"] for r in rows)}',flush=True)
    print(f'COMPLETE {suite}/slither',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='results/ccnc2027');p.add_argument('--suite',default='main');a=p.parse_args()
    os.chdir(ROOT);run((ROOT/a.root).resolve(),a.suite)
