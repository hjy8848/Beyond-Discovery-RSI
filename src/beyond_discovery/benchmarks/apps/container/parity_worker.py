"""Trusted golden programs ONLY, in OCI. Executes pinned official run_test.

The tiny RuntimeModule compatibility shim supplies the original evaluator's sole
pyext dependency. It does not change run_test, wrapping or comparison code.
"""
import sys,types,io,contextlib,json
sys.path.insert(0,'/opt/apps')
class RuntimeModule:
    @staticmethod
    def from_string(name,doc,source):
        module=types.ModuleType(name,doc);exec(source,module.__dict__);return module
shim=types.ModuleType('pyext');shim.RuntimeModule=RuntimeModule;sys.modules['pyext']=shim
from testing_util import run_test
CASES=[
    ('stdin-ac',None,['2 3','4 7'],['5','11'],'a,b=map(int,input().split())\nprint(a+b)'),
    ('stdin-wa',None,['2 3','4 7'],['5','11'],'print(0)'),
    ('call-ac','add',[[2,3],[4,7]],[5,11],'def add(a,b): return a+b'),
    ('call-wa','add',[[2,3],[4,7]],[5,11],'def add(a,b): return a-b'),
    ('class-ac','add',[[2,3],[4,7]],[5,11],'class Solution:\n def add(self,a,b): return a+b'),
    ('call-re','add',[[2,3]],[5],'def add(a,b): raise ValueError()'),
]
if __name__=='__main__':
    results={}
    for name,fn,inputs,outputs,code in CASES:
        with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
            verdicts=run_test(problem={'input_output':{'fn_name':fn,'inputs':inputs,'outputs':outputs}},test=code)
        results[name]=[bool(x) if type(x).__name__ in ('bool','bool_') else int(x) for x in verdicts]
    print(json.dumps(results))
