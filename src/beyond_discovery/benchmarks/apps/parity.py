"""Live OCI semantic parity; no fallback simulation if the engine is unavailable."""
import json,subprocess,ast
from pathlib import Path
from ...core.protocol import ProtocolError,CapabilityUnsupported


def check_official_parity(judge):
    # Constants are trusted golden fixtures, not Agent candidates. AST literal
    # extraction avoids importing/executing the container-only official runner.
    path=Path(__file__).parent/'container'/'parity_worker.py'
    tree=ast.parse(path.read_text())
    cases=ast.literal_eval(next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CASES' for t in n.targets)))
    name='apps-official-parity'
    command=judge.command(name,sum(len(c[2]) for c in cases))
    command=command[:-1]+['--entrypoint','python',judge.image,'-I','/opt/apps/parity_worker.py']
    try:
        completed=subprocess.run(command,capture_output=True,timeout=60,check=True)
        official=json.loads(completed.stdout)
    except (subprocess.SubprocessError,OSError,ValueError):raise CapabilityUnsupported('official parity image/runner failed; no simulated parity result')
    finally:
        try:subprocess.run([judge.engine,'rm','-f',name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=10)
        except (subprocess.SubprocessError,OSError):pass
    rows=[]
    for name,fn,inputs,outputs,code in cases:
        result=judge.grade(code,[{'input':x,'output':y} for x,y in zip(inputs,outputs)],fn)
        if list(result.verdicts)!=official[name]:raise ProtocolError('official parity mismatch: '+name)
        rows.append({'case':name,'verdicts':result.verdicts,'judge_launches':result.launches,'judge_cases':result.cases})
    return {'status':'VERIFIED','official_pyext_runtime_module':'documented compatibility shim','image':judge.image,'rows':rows}
