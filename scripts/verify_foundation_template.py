#!/usr/bin/env python3
"""Compare publisher Jinja and installed Ollama Go prompt strings, without inference."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from jinja2 import Template
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.run_ccnc_experiment import ROOT,save

def main():
    root=ROOT/'results/ccnc2027';protocol=json.loads((root/'protocol.json').read_text());data=[]
    for suite in ['main','fresh']:data.extend(json.loads((root/f'datasets/{suite}.json').read_text()))
    cases=[[{'role':'system','content':protocol['system']},
            {'role':'user','content':protocol['instruction'].format(code=x['contract_code'])}] for x in data]
    upstream=Template((ROOT/'tools/foundationsec-chat_template.jinja').read_text())
    expected=[upstream.render(messages=x,eos_token='<|end_of_text|>',add_generation_prompt=True) for x in cases]
    proc=subprocess.run(['go','run',str(ROOT/'tools/verify-foundation-template.go'),str(ROOT/'tools/foundationsec-native.Modelfile')],
                        input=json.dumps(cases),capture_output=True,text=True,check=True,cwd=ROOT)
    actual=json.loads(proc.stdout)
    if actual!=expected:raise AssertionError('Native template differs from publisher template')
    checks=[dict(source_sha256=x['source_sha256'],rendered_prompt_sha256=hashlib.sha256(y.encode()).hexdigest()) for x,y in zip(data,actual)]
    save(root/'validation/foundationsec_template_equivalence.json',dict(status='passed',cases=len(checks),
         comparison='Publisher Jinja rendered by Jinja2 versus actual local Go template rendered by Go text/template; all study single-turn system/user messages; exact string equality',checks=checks))
    print(f'{len(checks)}/{len(checks)} native-template renders exactly match the pinned publisher template.')

if __name__=='__main__':main()
