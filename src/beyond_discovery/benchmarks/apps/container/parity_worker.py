"""Trusted golden programs ONLY, in OCI. Executes pinned official run_test.

The tiny RuntimeModule compatibility shim supplies the original evaluator's sole
pyext dependency. It does not change run_test, wrapping or comparison code.
"""
import sys,types,io,contextlib,json,os,subprocess,builtins
sys.path.insert(0,'/opt/apps')
class RuntimeModule:
    @staticmethod
    def from_string(name,doc,source):
        module=types.ModuleType(name,doc);exec(source,module.__dict__);return module
shim=types.ModuleType('pyext');shim.RuntimeModule=RuntimeModule;sys.modules['pyext']=shim
from testing_util import run_test
# Preserve the exact APPS guard while supplying the dict form of builtins its
# source indexes directly under Python 3.11.
run_test.__globals__['__builtins__']=vars(builtins)
CASES=[
    ('stdin-ac',None,['2 3','4 7'],['5','11'],'a,b=map(int,input().split())\nprint(a+b)'),
    ('stdin-wa',None,['2 3','4 7'],['5','11'],'print(0)'),
    ('call-ac','add',[[2,3],[4,7]],[5,11],'def add(a,b): return a+b'),
    ('call-wa','add',[[2,3],[4,7]],[5,11],'def add(a,b): return a-b'),
    ('class-ac','add',[[2,3],[4,7]],[5,11],'class Solution:\n def add(self,a,b): return a+b'),
    ('call-re','add',[[2,3]],[5],'def add(a,b): raise ValueError()'),
]
def run_one(case):
    name,fn,inputs,outputs,code=case
    # The pinned evaluator disables os.putenv() in its reliability guard. A
    # plain copy preserves the inherited container environment while allowing
    # its OMP_NUM_THREADS assignment to complete.
    os.environ=dict(os.environ)
    with contextlib.redirect_stdout(io.StringIO()):
        verdicts=run_test(problem={'input_output':{'fn_name':fn,'inputs':inputs,'outputs':outputs}},test=code)
    return [bool(x) if type(x).__name__ in ('bool','bool_') else int(x) for x in verdicts]

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--one':
        print(json.dumps(run_one(json.loads(sys.stdin.readline()))))
    else:
        results={}
        for case in CASES:
            # One process per program matches the isolation used by APPS and
            # prevents the evaluator's guard from leaking state between cases.
            completed=subprocess.run([sys.executable,'-I',__file__,'--one'],input=json.dumps(case),
                text=True,capture_output=True,timeout=60,check=True)
            results[case[0]]=json.loads(completed.stdout)
        print(json.dumps(results))
