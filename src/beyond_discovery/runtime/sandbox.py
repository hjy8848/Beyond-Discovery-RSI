"""Restricted worker for pure research ASTs; no arbitrary Python is accepted.

Spawn (not fork) excludes evaluator-private memory. Only bounded JSON AST/context
cross the pipe. Worker denies file/network/process/ctypes audit operations after
startup and applies OS CPU/address-space/file-size limits plus parent timeouts.
"""
from contextvars import ContextVar
import json
import multiprocessing
import threading
import sys
import os
import tempfile
import time
from ..core.protocol import ProtocolError, CapabilityUnsupported

executor = ContextVar('research_expression_executor',default=None)


def _worker(connection,directory,memory,cpu):
    import os
    import sys
    try:
        import resource
        from .expression import evaluate,freeze
        from .control import validate as validate_control, execute as execute_control
        resource.setrlimit(resource.RLIMIT_CPU,(cpu,cpu))
        try:
            resource.setrlimit(resource.RLIMIT_AS,(memory,memory))
            memory_mode='address-space-limit'
        except ValueError:
            # macOS rejects RLIMIT_AS. Parent enforces resident memory instead.
            memory_mode='parent-rss-monitor'
        resource.setrlimit(resource.RLIMIT_FSIZE,(0,0))
        os.chdir(directory)
        def deny(event,args):
            if event=='open' or event.startswith(('socket.','subprocess.','ctypes.')) or event in ('os.system','os.fork','os.posix_spawn','os.exec','os.remove','os.rename','os.mkdir','os.rmdir','os.chdir'):
                raise PermissionError('sandbox host capability denied')
        sys.addaudithook(deny)
        connection.send_bytes(json.dumps({'ready':True,'memory_mode':memory_mode}).encode())
        while True:
            payload=connection.recv_bytes(262144)
            if payload==b'close': break
            try:
                data=json.loads(payload)
                if 'control' in data:
                    program=validate_control(data['control'])
                    def effect(name,args):
                        packet=json.dumps({'effect':{'name':name,'arguments':args}},allow_nan=False).encode()
                        if len(packet)>262144: raise ProtocolError('control effect request size')
                        connection.send_bytes(packet)
                        reply=json.loads(connection.recv_bytes(262144))
                        if 'effect_error' in reply: raise ProtocolError('control capability rejected')
                        return reply['value']
                    result=execute_control(program,data['context'],effect,data['fuel'])
                else:
                    result=evaluate(freeze(data['node']),data['context'],data['fuel'])
                response={'value':result}
            except (ValueError,TypeError,KeyError,IndexError,MemoryError,PermissionError,ProtocolError) as exc:
                response={'error':type(exc).__name__}
            connection.send_bytes(json.dumps(response,allow_nan=False).encode())
    except Exception:
        try: connection.send_bytes(b'{"unsupported":true}')
        except Exception: pass
    finally: connection.close()


class ResearchSandbox:
    def __init__(self,memory=512*1024*1024,cpu=60,wall=300,timeout=2):
        self.memory,self.cpu,self.wall,self.timeout=memory,cpu,wall,timeout
        self.process=None

    def __enter__(self):
        self.directory=tempfile.TemporaryDirectory(prefix='rsi-worker-')
        ctx=multiprocessing.get_context('spawn')
        self.connection,child=ctx.Pipe(duplex=True)
        self.process=ctx.Process(target=_worker,args=(child,self.directory.name,self.memory,self.cpu),daemon=True)
        self.process.start(); child.close(); self.started=time.monotonic()
        try:
            if not self.connection.poll(10): raise CapabilityUnsupported('restricted worker startup timeout')
            ready=json.loads(self.connection.recv_bytes())
            if ready.get('ready') is not True:
                raise CapabilityUnsupported('CAPABILITY_UNSUPPORTED: restricted worker/resource limits unavailable')
        except Exception:
            self.close(); raise
        self.memory_mode=ready['memory_mode']
        self.monitor_stop=threading.Event()
        if self.memory_mode=='parent-rss-monitor':
            if self.resident_bytes()<=0:
                self.close(); raise CapabilityUnsupported('CAPABILITY_UNSUPPORTED: resident-memory monitoring unavailable')
            def monitor():
                while not self.monitor_stop.wait(.02):
                    try:
                        if self.resident_bytes()>self.memory:
                            self.process.terminate(); return
                    except Exception:
                        self.process.terminate(); return
            self.monitor=threading.Thread(target=monitor,daemon=True); self.monitor.start()
        self.token=executor.set(self)
        return self

    def resident_bytes(self):
        if sys.platform=='darwin':
            import ctypes
            library=ctypes.CDLL('/usr/lib/libproc.dylib')
            buffer=ctypes.create_string_buffer(96)
            size=library.proc_pidinfo(self.process.pid,4,0,buffer,96)
            if size!=96: raise RuntimeError('cannot read worker resident memory')
            return int.from_bytes(buffer.raw[8:16],sys.byteorder)
        if sys.platform.startswith('linux'):
            from pathlib import Path
            return int((Path('/proc')/str(self.process.pid)/'statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE')
        raise CapabilityUnsupported('resident memory monitor unsupported')

    def evaluate(self,node,context,fuel):
        return self.request({'node':node,'context':context,'fuel':fuel})

    def control(self,program,context,effect,fuel=4096):
        return self.request({'control':program,'context':context,'fuel':fuel},effect)

    def request(self,data,effect=None):
        if not self.process.is_alive() or time.monotonic()-self.started>self.wall:
            raise ProtocolError('SANDBOX_RESOURCE_LIMIT')
        payload=json.dumps(data,allow_nan=False).encode()
        if len(payload)>262144: raise ProtocolError('SANDBOX_INPUT_SIZE')
        capability_error=None
        try:
            self.connection.send_bytes(payload)
            effects=0
            while True:
                if not self.connection.poll(self.timeout):
                    self.process.terminate(); raise ProtocolError('SANDBOX_TIMEOUT')
                response=json.loads(self.connection.recv_bytes(262144))
                if 'effect' not in response: break
                effects+=1
                if effect is None or effects>64:
                    self.process.terminate(); raise ProtocolError('SANDBOX_CAPABILITY_BUDGET')
                try:
                    request=response['effect']
                    answer=effect(request['name'],request['arguments'])
                    encoded=json.dumps({'value':answer},allow_nan=False).encode()
                    if len(encoded)>262144: raise ProtocolError('SANDBOX_EFFECT_RESULT_SIZE')
                except Exception as exc:
                    capability_error=exc
                    self.connection.send_bytes(b'{"effect_error":true}')
                    if self.connection.poll(self.timeout): self.connection.recv_bytes(262144)
                    else: self.process.terminate()
                    raise
                self.connection.send_bytes(encoded)
        except (EOFError,OSError):
            if capability_error is not None: raise capability_error
            raise ProtocolError('SANDBOX_WORKER_FAILED')
        if 'error' in response: raise ProtocolError('SANDBOX_CODE_INVALID:'+response['error'])
        if 'value' not in response: raise ProtocolError('SANDBOX_PROTOCOL_INVALID')
        return response['value']

    def close(self):
        if hasattr(self,'monitor_stop'): self.monitor_stop.set()
        if hasattr(self,'monitor'): self.monitor.join(1)
        if self.process is not None:
            try:
                if self.process.is_alive(): self.connection.send_bytes(b'close')
            except (OSError,EOFError): pass
            self.process.join(1)
            if self.process.is_alive(): self.process.terminate(); self.process.join(1)
            if self.process.is_alive(): self.process.kill(); self.process.join(1)
            self.connection.close()
        if hasattr(self,'directory'): self.directory.cleanup()

    def __exit__(self,*args):
        executor.reset(self.token); self.close()
