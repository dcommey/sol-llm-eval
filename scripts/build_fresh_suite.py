#!/usr/bin/env python3
"""Create and execute nine vulnerable/fixed pairs before evaluating those fixtures.

These are controlled synthetic witnesses, not a production benchmark or a claim
that the underlying vulnerability patterns are absent from pretraining.
"""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.run_ccnc_experiment import ROOT, save, digest
from web3 import Web3, EthereumTesterProvider

HEADER='pragma solidity ^0.8.27;\n'

def fixtures():
    cases=[]
    for variant in range(3):
        for fixed in [False,True]:
            reset='credit[msg.sender] = 0;'
            transfer='(bool ok,) = msg.sender.call{value: amount}(""); require(ok);'
            if variant==1: transfer='_pay(amount);'
            if variant==2: transfer='address payable recipient = payable(msg.sender); (bool ok,) = recipient.call{value: amount}(""); require(ok);'
            body=(reset+transfer) if fixed else (transfer+reset)
            helper='function _pay(uint256 amount) internal { (bool ok,) = msg.sender.call{value: amount}(""); require(ok); }' if variant==1 else ''
            source=HEADER+f'''contract Unit {{
    mapping(address => uint256) public credit;
    function deposit() external payable {{ credit[msg.sender] += msg.value; }}
    function withdraw() external {{
        uint256 amount = credit[msg.sender]; require(amount > 0);
        {body}
    }}
    {helper}
}}
'''
            cases.append(dict(kind='reentrancy',variant=variant,fixed=fixed,source=source))
    arithmetic=[
        ('uint8 public used;','function add(uint8 amount) external { BODY }','used += amount;','add',[250,10]),
        ('uint256 public remaining = 20;','function reserve(uint256 amount) external { BODY }','remaining -= amount;','reserve',[21]),
        ('uint256 public total;','function price(uint256 quantity, uint256 unitPrice) external { BODY }','total = quantity * unitPrice;','price',[(2**256)//2,2]),
    ]
    for variant,(state,fn,operation,method,arguments) in enumerate(arithmetic):
        for fixed in [False,True]:
            body=operation if fixed else 'unchecked { '+operation+' }'
            source=HEADER+'contract Unit {\n    '+state+'\n    '+fn.replace('BODY',body)+'\n}\n'
            cases.append(dict(kind='integer_overflow',variant=variant,fixed=fixed,source=source,method=method,arguments=arguments))
    for variant in range(3):
        for fixed in [False,True]:
            check='require(ok);' if fixed else ''
            if variant==0:
                call='(bool ok,) = receiver.call{value: msg.value}("");'
                helper=''
            elif variant==1:
                call='bool ok = receiver.send(msg.value);';helper=''
            else:
                call='bool ok = _deliver(receiver, msg.value);'
                helper='function _deliver(address payable receiver, uint256 value) internal returns (bool) { (bool ok,) = receiver.call{value: value}(""); return ok; }'
            source=HEADER+f'''contract Unit {{
    bool public completed;
    function execute(address payable receiver) external payable {{
        completed = true; {call} {check}
    }}
    {helper}
}}
'''
            cases.append(dict(kind='unchecked_low_level_calls',variant=variant,fixed=fixed,source=source))
    return cases

WITNESS=HEADER+'''interface IUnit { function deposit() external payable; function withdraw() external; }
contract Probe {
    IUnit public unit; uint256 public rounds;
    constructor(address target) { unit = IUnit(target); }
    function begin() external payable { unit.deposit{value:msg.value}(); unit.withdraw(); }
    receive() external payable {
        if (rounds < 1) { rounds++; (bool ok,) = address(unit).call(abi.encodeWithSignature("withdraw()")); ok; }
    }
}
contract Rejector { receive() external payable { revert(); } }
'''

def compile_source(source):
    inp=dict(language='Solidity',sources={'Unit.sol':dict(content=source)},
             settings=dict(outputSelection={'*':{'*':['abi','evm.bytecode.object']}}))
    proc=subprocess.run([str(ROOT/'tools/solc-js/solc-bridge.cjs'),'--standard-json'],
        input=json.dumps(inp),text=True,capture_output=True,check=True,env={**os.environ,'SOLC_VERSION':'0.8.27'})
    result=json.loads(proc.stdout)
    errors=[r for r in result.get('errors',[]) if r['severity']=='error']
    if errors:raise RuntimeError(errors)
    return result['contracts']['Unit.sol']

def deploy(w3,compiled,name,*args):
    unit=compiled[name]
    factory=w3.eth.contract(abi=unit['abi'],bytecode=unit['evm']['bytecode']['object'])
    receipt=w3.eth.wait_for_transaction_receipt(factory.constructor(*args).transact({'from':w3.eth.accounts[0],'gas':5000000}))
    assert receipt.status==1
    return w3.eth.contract(address=receipt.contractAddress,abi=unit['abi'])

def validate(case):
    w3=Web3(EthereumTesterProvider())
    account=w3.eth.accounts[0]
    unit=deploy(w3,compile_source(case['source']),'Unit')
    witness=compile_source(WITNESS)
    if case['kind']=='reentrancy':
        unit.functions.deposit().transact({'from':account,'value':5*10**18})
        probe=deploy(w3,witness,'Probe',unit.address)
        receipt=w3.eth.wait_for_transaction_receipt(probe.functions.begin().transact({'from':account,'value':10**18,'gas':2000000}))
        assert receipt.status==1
        observed=w3.eth.get_balance(probe.address)
        expected=(1 if case['fixed'] else 2)*10**18
        assert observed==expected,(case,observed,expected)
        return dict(test='one recursive withdrawal',observed_attacker_payout_wei=observed,expected_payout_wei=expected,passed=True)
    if case['kind']=='integer_overflow':
        fn=getattr(unit.functions,case['method'])
        if case['variant']==0:
            fn(250).transact({'from':account})
            args=[10];view=unit.functions.used
            expected=250 if case['fixed'] else 4
        elif case['variant']==1:
            args=[21];view=unit.functions.remaining
            expected=20 if case['fixed'] else 2**256-1
        else:
            args=case['arguments'];view=unit.functions.total;expected=0
        receipt=w3.eth.wait_for_transaction_receipt(fn(*args).transact({'from':account,'gas':1000000}))
        assert receipt.status==(0 if case['fixed'] else 1)
        observed=view().call();assert observed==expected
        return dict(test='overflow or underflow boundary',transaction_status=receipt.status,observed_value=observed,expected_value=expected,passed=True)
    recipient=deploy(w3,witness,'Rejector')
    receipt=w3.eth.wait_for_transaction_receipt(unit.functions.execute(recipient.address).transact({'from':account,'value':10**18,'gas':1000000}))
    assert receipt.status==(0 if case['fixed'] else 1)
    observed=unit.functions.completed().call();assert observed==(not case['fixed'])
    assert w3.eth.get_balance(recipient.address)==0
    return dict(test='recipient rejects payment',transaction_status=receipt.status,observed_completion=observed,expected_completion=not case['fixed'],passed=True)

def main():
    root=ROOT/'results/ccnc2027';data=[];checks=[]
    for i,case in enumerate(fixtures(),1):
        name=f'unit_{i:02d}'
        result=validate(case)
        source=case['source'];path=ROOT/f'data/fresh-ccnc2027/{name}.sol'
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(source)
        data.append(dict(contract_name=name,contract_path=str(path.relative_to(ROOT)),contract_code=source,
            source_sha256=hashlib.sha256(source.encode()).hexdigest(),source='fresh_synthetic',
            ground_truth_vulnerabilities=[] if case['fixed'] else [case['kind']],
            pair_id=f'{case["kind"]}_{case["variant"]}',is_vulnerable=not case['fixed']))
        checks.append(dict(contract_name=name,pair_id=data[-1]['pair_id'],target=case['kind'],fixed=case['fixed'],**result))
        print(name,case['kind'],'fixed' if case['fixed'] else 'vulnerable','PASS',flush=True)
    save(root/'datasets/fresh.json',data)
    save(root/'validation/fresh_witnesses.json',dict(dataset_sha256=digest(data),contracts=len(data),pairs=9,
        source_origin='Generated in this revision on 2026-10-01; not externally sourced',
        limitation='Small designed contrasts; wrapping arithmetic labels assume overflow/underflow is unintended. Not evidence of production prevalence or absence of pattern exposure.',
        compiler='official solc-js 0.8.27',checks=checks))
    (ROOT/'data/fresh-ccnc2027/Probe.sol').write_text(WITNESS)

if __name__=='__main__':main()
