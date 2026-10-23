"""Explicit protocol fixtures. Never run candidate code on the host."""
from pathlib import Path
import hashlib,io,json,tarfile
from beyond_discovery.benchmarks.apps.data import import_archive,APPSStore
from beyond_discovery.benchmarks.apps.judge import JudgeResult


def fixture_store(root):
    root=Path(root);archive=root/'fixture.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for split,n in (('train',20),('test',3)):
            for i in range(n):
                files={'question.txt':f'PROTOCOL FIXTURE {split} task {i}: add two numbers',
                       'input_output.json':json.dumps({'inputs':[[2,3],[4,7]],'outputs':[5,11],'fn_name':'add'}),
                       'solutions.json':json.dumps(['PRIVATE_REFERENCE_SENTINEL']),
                       'public_tests.json':json.dumps([{'input':[1,2],'output':3}])}
                for name,data in files.items():
                    raw=data.encode();info=tarfile.TarInfo(f'APPS/{split}/{i:04d}/{name}');info.size=len(raw);tar.addfile(info,io.BytesIO(raw))
    digest=hashlib.sha256(archive.read_bytes()).hexdigest()
    import_archive(archive,root/'store',digest)
    return APPSStore(root/'store')


class ProtocolJudge:
    identity='EXPLICIT_TEST_PROTOCOL_FIXTURE_NOT_PYTHON_EXECUTION'
    def __init__(self):self.records=[]
    def grade(self,code,cases,fn_name=None):
        # This mapping is only a transport/judging protocol fixture. No eval/exec.
        correct='return a+b' in code or 'return sum' in code
        outputs=tuple(sum(c['input']) if correct else -999 for c in cases)
        result=JudgeResult(tuple(x==c['output'] for x,c in zip(outputs,cases)),'COMPLETE',len(cases),1,0.,outputs)
        self.records.append({'code':code,'result':result});return result
