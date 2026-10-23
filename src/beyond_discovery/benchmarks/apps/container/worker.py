"""Runs ONLY inside an isolated OCI container. No labels/references are supplied."""
import types, builtins
import ast, contextlib, io, json, sys, os, platform, faulthandler, signal
from unittest.mock import patch,mock_open
import numpy as np
sys.path.insert(0,'/opt/apps')
from wire import encode,decode

protocol_in=sys.stdin; protocol_out=sys.stdout
# Trusted helper source is in the immutable image, never in a candidate mount.
source=ast.parse(open('/opt/apps/testing_util.py').read())
helpers=[n for n in source.body if isinstance(n,ast.FunctionDef) and n.name in ('call_method','reliability_guard')]
exec(compile(ast.Module(body=helpers,type_ignores=[]),'<official-helpers>','exec'),globals())
# ``exec`` receives the module's builtins module on some Python builds, while
# the pinned APPS guard indexes ``__builtins__`` as a dict. Supply the equivalent
# builtin namespace mapping expected by that original implementation.
reliability_guard.__globals__['__builtins__']=vars(builtins)
StringIO=io.StringIO

def reply(value):
    message=json.dumps(value,allow_nan=False)
    if len(message.encode())>65536: message='{"status":"OUTPUT_LIMIT"}'
    protocol_out.write(message+'\n'); protocol_out.flush()


def main():
    init=json.loads(protocol_in.readline(1048577))
    # APPS' reliability_guard disables os.putenv(), which backs mutations on
    # os.environ. Keep the already-inherited, container-fixed environment as a
    # plain mapping so its subsequent OMP_NUM_THREADS assignment cannot abort
    # every Judge session. The Docker launch has already set this value to 1.
    os.environ=dict(os.environ)
    reliability_guard()
    namespace=types.ModuleType('tmp_sol','').__dict__
    def timeout_handler(*args):raise TimeoutError('APPS time limit')
    signal.signal(signal.SIGALRM,timeout_handler)
    try:
        signal.alarm(init['timeout'])
        with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
            exec(init['prepared_source'],namespace)
            obj=namespace['Solution']() if init['solution_class'] else None
            method=getattr(obj,init['fn_name']) if obj is not None else namespace[init['fn_name'] or 'code']
        signal.alarm(0)
        reply({'status':'READY'})
    except BaseException:
        signal.alarm(0);reply({'status':'CE'}); return
    for line in protocol_in:
        try:
            inputs=decode(json.loads(line)['input'])
            capture=io.StringIO();capture.close=lambda *args:None
            signal.alarm(init['timeout'])
            with contextlib.redirect_stdout(capture),contextlib.redirect_stderr(io.StringIO()):
                value=method(*inputs) if init['fn_name'] else call_method(method,inputs)
            signal.alarm(0)
            output=value if init['fn_name'] else capture.getvalue().splitlines()
            try: encoded=encode(output)
            except (TypeError,ValueError) as exc:
                status='EVALUATOR_TYPE_UNSUPPORTED' if 'ndarray' in str(exc) else 'RETURN_TYPE_UNSUPPORTED'
                reply({'status':status});continue
            reply({'status':'OK','output':encoded})
        except (TypeError,ValueError): signal.alarm(0);reply({'status':'RE'})
        except BaseException: signal.alarm(0);reply({'status':'RE'})

if __name__=='__main__': main()
