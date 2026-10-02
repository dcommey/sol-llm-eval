#!/usr/bin/env node
// Reproduce solc combined-json from pinned official solc-js standard-json.
// Needed because this ARM Mac has no Rosetta for Solidity 0.4/0.5 binaries.
const fs = require('fs');
const path = require('path');
const versions = {'0.4.25':'solc-0425','0.5.17':'solc-0517','0.8.27':'solc-0827'};
const compiler = require(versions[process.env.SOLC_VERSION || '0.8.27']);
const args = process.argv.slice(2);
if (args.includes('--version')) {
  process.stdout.write('solc, the solidity compiler commandline interface\nVersion: '+compiler.version()+'\n');
  process.exit(0);
}
const imports = filename => {
  try { return {contents:fs.readFileSync(filename,'utf8')}; }
  catch(e) { return {error:e.message}; }
};
function compile(input) {
  if (compiler.compileStandardWrapper) return JSON.parse(compiler.compileStandardWrapper(JSON.stringify(input),imports));
  return JSON.parse(compiler.compile(JSON.stringify(input),{import:imports}));
}
if (args.includes('--standard-json')) {
  process.stdout.write(JSON.stringify(compile(JSON.parse(fs.readFileSync(0,'utf8')))));
  process.exit(0);
}
const filename = args.find(a=>a.endsWith('.sol'));
if (!filename || !args.includes('--combined-json')) throw new Error('Only version, standard-json, and combined-json are supported');
const input = {language:'Solidity',sources:{[filename]:{content:fs.readFileSync(filename,'utf8')}},
  settings:{optimizer:{enabled:true,runs:200},outputSelection:{'*':{'*':['abi','evm.bytecode','evm.deployedBytecode','userdoc','devdoc'],'':['ast']}}}};
const result=compile(input);
for(const e of result.errors||[]) process.stderr.write(e.formattedMessage+'\n');
if ((result.errors||[]).some(e=>e.severity==='error')) process.exit(1);
const combined={contracts:{},sources:{},sourceList:[],version:compiler.version()};
for(const [name,source] of Object.entries(result.sources||{})) {
  combined.sources[name]={AST:source.ast}; combined.sourceList[source.id]=name;
}
for(const [file,contracts] of Object.entries(result.contracts||{})) {
  for(const [name,contract] of Object.entries(contracts)) {
    combined.contracts[file+':'+name]={abi:process.env.SOLC_VERSION==='0.8.27'?contract.abi:JSON.stringify(contract.abi),
      bin:contract.evm.bytecode.object,'bin-runtime':contract.evm.deployedBytecode.object,
      srcmap:contract.evm.bytecode.sourceMap,'srcmap-runtime':contract.evm.deployedBytecode.sourceMap,
      userdoc:process.env.SOLC_VERSION==='0.8.27'?contract.userdoc:JSON.stringify(contract.userdoc),
      devdoc:process.env.SOLC_VERSION==='0.8.27'?contract.devdoc:JSON.stringify(contract.devdoc)};
  }
}
process.stdout.write(JSON.stringify(combined));
