"""Host-side trustworthy comparison; arbitrary Python only enters OCI containers."""
from dataclasses import dataclass
import json,os,re,selectors,shutil,subprocess,time,uuid
from ...core.protocol import CapabilityUnsupported,ProtocolError
from ...core.contracts import digest
from . import semantics
from .wire import encode,decode


@dataclass(frozen=True)
class JudgeResult:
    verdicts: tuple
    status: str
    cases: int
    launches: int
    elapsed_seconds: float
    outputs: tuple=()
    @property
    def accepted(self): return bool(self.verdicts) and all(x is True for x in self.verdicts)


class ContainerFailure(RuntimeError): pass


class DockerSession:
    def __init__(self,judge,source,fn_name,total_cases):
        self.judge=judge; self.name='rsi-apps-'+uuid.uuid4().hex;self.process=None
        self.started=time.monotonic(); self.total_cases=total_cases
        self.source=source; self.fn_name=fn_name
    def __enter__(self):
        command=self.judge.command(self.name,self.total_cases)
        self.process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
        for stream in (self.process.stdin,self.process.stdout,self.process.stderr): os.set_blocking(stream.fileno(),False)
        self.pending=b''
        try:
            response=self.exchange({'prepared_source':semantics.prepare(self.source,self.fn_name),'fn_name':self.fn_name,
                                    'solution_class':bool(self.fn_name and 'class Solution' in self.source)},self.judge.startup_timeout)
            self.ready=response['status'];return self
        except Exception: self.close();raise
    def exchange(self,message,timeout):
        payload=json.dumps(message,allow_nan=False).encode()+b'\n'
        if len(payload)>1048576: raise ContainerFailure('INPUT_LIMIT')
        deadline=min(time.monotonic()+timeout,self.started+self.judge.problem_timeout)
        selection=selectors.DefaultSelector(); offset=0; data=self.pending; error=b''
        selection.register(self.process.stdin,selectors.EVENT_WRITE,'in')
        selection.register(self.process.stdout,selectors.EVENT_READ,'out')
        selection.register(self.process.stderr,selectors.EVENT_READ,'err')
        try:
            while True:
                if b'\n' in data and offset==len(payload):
                    line,self.pending=data.split(b'\n',1)
                    try: response=json.loads(line)
                    except (ValueError,RecursionError): raise ContainerFailure('PROTOCOL_ERROR')
                    if not isinstance(response,dict) or response.get('status') not in ('READY','OK','CE','RE','OUTPUT_LIMIT','RETURN_TYPE_UNSUPPORTED'):
                        raise ContainerFailure('PROTOCOL_ERROR')
                    return response
                remaining=deadline-time.monotonic()
                if remaining<=0: raise ContainerFailure('TIMEOUT')
                for key,_ in selection.select(min(remaining,.1)):
                    if key.data=='in':
                        try: count=os.write(key.fd,payload[offset:offset+65536])
                        except (BrokenPipeError,OSError): raise ContainerFailure('WORKER_EXIT')
                        offset+=count
                        if offset==len(payload): selection.unregister(key.fileobj)
                    else:
                        chunk=os.read(key.fd,65536)
                        if not chunk:
                            selection.unregister(key.fileobj)
                            if key.data=='out': raise ContainerFailure('WORKER_EXIT')
                        elif key.data=='out': data+=chunk
                        else: error+=chunk
                        if len(data)+len(error)>131072: raise ContainerFailure('OUTPUT_LIMIT')
        finally: selection.close()
    def close(self):
        # Killing just docker's client does NOT kill the untrusted container.
        try: subprocess.run([self.judge.engine,'rm','-f',self.name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=10)
        except (OSError,subprocess.TimeoutExpired): pass
        if self.process:
            if self.process.poll() is None: self.process.kill()
            self.process.wait(timeout=10)
            for f in (self.process.stdin,self.process.stdout,self.process.stderr): f.close()
    def __exit__(self,*args): self.close()


class DockerJudge:
    def __init__(self,image,engine='docker',case_timeout=4,memory_mb=512,problem_timeout=300,startup_timeout=20):
        if not re.fullmatch(r'[a-zA-Z0-9_./:-]+@sha256:[0-9a-f]{64}',image or ''): raise ValueError('Judge requires an immutable image digest, never a mutable tag')
        if not 1<=case_timeout<=30 or not 128<=memory_mb<=4096 or not 1<=problem_timeout<=3600: raise ValueError('Judge resource limits')
        self.image=image;self.engine=shutil.which(engine)
        if not self.engine: raise CapabilityUnsupported('OCI container engine is not installed')
        self.case_timeout=case_timeout;self.memory_mb=memory_mb;self.problem_timeout=problem_timeout;self.startup_timeout=startup_timeout
        self.identity=digest({'image':image,'official':semantics.VENDOR_SHA256,'case_timeout':case_timeout,'memory_mb':memory_mb,'problem_timeout':problem_timeout})
        self.records=[]
        self.preflight()
    def preflight(self):
        try:
            info=subprocess.run([self.engine,'info','--format','{{json .SecurityOptions}}'],capture_output=True,timeout=15,check=True)
            if 'seccomp' not in info.stdout.decode().lower(): raise CapabilityUnsupported('container daemon has no seccomp support')
            subprocess.run([self.engine,'image','inspect',self.image],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=15,check=True)
        except (OSError,subprocess.SubprocessError): raise CapabilityUnsupported('container daemon/image unavailable; no host execution fallback')
    def command(self,name,cases):
        return [self.engine,'run','--rm','--pull=never','--name',name,'--network=none','--read-only','--cap-drop=ALL',
                '--security-opt=no-new-privileges','--pids-limit=32','--memory='+str(self.memory_mb)+'m',
                '--memory-swap='+str(self.memory_mb)+'m','--cpus=1','--ulimit','nofile=64:64',
                '--ulimit','fsize=1048576:1048576','--ulimit','cpu='+str(int(min(self.problem_timeout,self.case_timeout*(cases+1))+1)),
                '--user=65534:65534','--tmpfs','/tmp:rw,noexec,nosuid,nodev,size=16777216','--workdir=/tmp',
                '--log-driver=none','--env','PYTHONHASHSEED=0','--env','OMP_NUM_THREADS=1','-i',self.image]
    def grade(self,code,cases,fn_name=None):
        if type(code) is not str or len(code.encode())>65536 or not cases or len(cases)>10000: raise ProtocolError('APPS code/case size bound')
        start=time.monotonic(); verdicts=[];outputs=[];executed=0;status='COMPLETE'
        try:
            with DockerSession(self,code,fn_name,len(cases)) as session:
                if session.ready!='READY': verdicts=[-2];status='CE'
                else:
                    for case in cases:
                        inputs,expected=semantics.normalize_call(case['input'],case['output']) if fn_name else (case['input'],case['output'])
                        executed+=1
                        response=session.exchange({'input':encode(inputs)},self.case_timeout)
                        if response['status']=='RETURN_TYPE_UNSUPPORTED':raise CapabilityUnsupported('APPS call return is not representable as safe JSON/tuple data')
                        if response['status']!='OK': verdicts.append(-1);outputs.append(None);status=response['status'];continue
                        try: output=decode(response['output'])
                        except (ValueError,TypeError,RecursionError): raise ContainerFailure('RETURN_PROTOCOL_ERROR')
                        outputs.append(output);verdicts.append(semantics.compare(output,expected,fn_name))
        except ContainerFailure as exc:
            status=str(exc);verdicts.append(-1)
        result=JudgeResult(tuple(verdicts),status,executed,1,time.monotonic()-start,tuple(outputs))
        self.records.append({'code_hash':digest(code),'case_hash':digest(cases),'fn_name':fn_name,'result':result,'judge':self.identity})
        return result
