"""Runs ONLY inside an isolated OCI container. No labels/references are supplied."""
import ast, contextlib, io, json, sys, os, platform, faulthandler, signal
from unittest.mock import patch,mock_open
sys.path.insert(0,'/opt/apps')
from wire import encode,decode

protocol_in=sys.stdin; protocol_out=sys.stdout
# Trusted helper source is in the immutable image, never in a candidate mount.
source=ast.parse(open('/opt/apps/testing_util.py').read())
helpers=[n for n in source.body if isinstance(n,ast.FunctionDef) and n.name in ('call_method','reliability_guard')]
exec(compile(ast.Module(body=helpers,type_ignores=[]),'<official-helpers>','exec'),globals())
StringIO=io.StringIO

def reply(value):
    message=json.dumps(value,allow_nan=False)
    if len(message.encode())>65536: message='{"status":"OUTPUT_LIMIT"}'
    protocol_out.write(message+'\n'); protocol_out.flush()


def main():
    init=json.loads(protocol_in.readline(1048577))
    reliability_guard()
    namespace={}
    try:
        with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
            exec(init['prepared_source'],namespace)
            obj=namespace['Solution']() if init['solution_class'] else None
            method=getattr(obj,init['fn_name']) if obj is not None else namespace[init['fn_name'] or 'code']
        reply({'status':'READY'})
    except BaseException:
        reply({'status':'CE'}); return
    for line in protocol_in:
        try:
            inputs=decode(json.loads(line)['input'])
            capture=io.StringIO()
            with contextlib.redirect_stdout(capture),contextlib.redirect_stderr(io.StringIO()):
                value=method(*inputs) if init['fn_name'] else call_method(method,inputs)
            output=value if init['fn_name'] else capture.getvalue().splitlines()
            try: encoded=encode(output)
            except (TypeError,ValueError): reply({'status':'RETURN_TYPE_UNSUPPORTED'});continue
            reply({'status':'OK','output':encoded})
        except (TypeError,ValueError): reply({'status':'RE'})
        except BaseException: reply({'status':'RE'})

if __name__=='__main__': main()
