"""Trusted projections of pinned official APPS code, not candidate execution.

Comparison and source preparation are extracted from the original AST, avoiding
an independently invented 'close enough' judging rule. Only SHA-verified vendor
code is compiled here. Candidate source remains a STRING on the host.
"""
import ast
from copy import deepcopy
import contextlib
import hashlib
import io
import math
from pathlib import Path
from ...core.protocol import ProtocolError

REVISION='362aedc3c71cd7d9bd2fc96a6c80e11dbc38c7a5'
VENDOR=Path(__file__).parent/'vendor'/'testing_util.py'
VENDOR_SHA256='9a4e58ff2634ef606c42597457c0733910862e0588ea750d901265bbfe65d36f'


class _NumpyComparison:
    @staticmethod
    def allclose(a,b):
        # Official code converts to one-dimensional float lists before this call.
        # NumPy defaults: rtol=1e-5, atol=1e-8, equal_nan=False.
        return len(a)==len(b) and all(x==y or (math.isfinite(x) and math.isfinite(y) and abs(x-y)<=1e-8+1e-5*abs(y)) for x,y in zip(a,b))


def _build():
    raw=VENDOR.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=VENDOR_SHA256: raise ProtocolError('official APPS source checksum mismatch')
    tree=ast.parse(raw)
    run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run_test')
    namespace={'np':_NumpyComparison,'deepcopy':deepcopy}
    helpers=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('custom_compare_','stripped_string_compare')]
    exec(compile(ast.Module(body=helpers,type_ignores=[]),str(VENDOR),'exec'),namespace)
    # Extract the two type branches containing the ORIGINAL comparison statements.
    blocks=[n for n in ast.walk(run) if isinstance(n,ast.If) and isinstance(n.test,ast.Compare) and ast.unparse(n.test.left)=='which_type']
    standard=next(n for n in blocks if 'CODE_TYPE.standard_input' in ast.unparse(n.test) and any(isinstance(x,ast.With) for x in n.body))
    first=next(i for i,n in enumerate(standard.body) if isinstance(n,ast.If) and 'custom_compare_' in ast.unparse(n.test))
    prefix=ast.parse('def compare_stdout(output, expected):\n in_outs={"outputs":[deepcopy(expected)]}\n debug=False\n results=[]\n if isinstance(in_outs["outputs"][0],list): in_outs["outputs"][0]="\\n".join(in_outs["outputs"][0])\n for index in [0]:\n  pass\n return bool(results[0])').body[0]
    loop=next(x for x in prefix.body if isinstance(x,ast.For));loop.body=deepcopy(standard.body[first:])
    exec(compile(ast.fix_missing_locations(ast.Module(body=[prefix],type_ignores=[])),str(VENDOR),'exec'),namespace)
    # Call-return comparisons, including the official tuple/nested-list behavior.
    call=next(n for n in blocks if 'CODE_TYPE.call_based' in ast.unparse(n.test) and any(isinstance(x,ast.Try) for x in n.body) and any('output = method' in ast.unparse(x) for x in n.body))
    trial=next(n for n in call.body if isinstance(n,ast.Try))
    statements=trial.body
    begin=next(i for i,n in enumerate(statements) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='output' for t in n.targets))+1
    end=next(i for i,n in enumerate(statements) if isinstance(n,ast.Expr) and 'results.append' in ast.unparse(n))
    fn=ast.parse('def compare_call(output,expected):\n in_outs={"outputs":[deepcopy(expected)]}\n index=0\n pass').body[0]
    fn.body=fn.body[:2]+deepcopy(statements[begin:end])+[ast.Return(value=ast.Call(func=ast.Name(id='bool',ctx=ast.Load()),args=[ast.Name(id='tmp_result',ctx=ast.Load())],keywords=[]))]
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(VENDOR),'exec'),namespace)
    # Original preamble and stdin wrapping are string transformations, not eval.
    sol=next(n for n in ast.walk(run) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='sol' for t in n.targets) and isinstance(n.value,ast.Constant))
    namespace['PREAMBLE']=sol.value.value
    prep=next(n for n in blocks if 'CODE_TYPE.standard_input' in ast.unparse(n.test) and any(isinstance(x,ast.Assign) and 'tmp_test' in ast.unparse(x.targets) for x in n.body))
    stop=next(i for i,n in enumerate(prep.body) if isinstance(n,ast.AugAssign) and isinstance(n.target,ast.Name) and n.target.id=='sol')
    fn=ast.parse('def prepare_stdin(test):\n pass').body[0]
    fn.body=deepcopy(prep.body[:stop])+[ast.Return(value=ast.Name(id='tmp_test',ctx=ast.Load()))]
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(VENDOR),'exec'),namespace)
    return namespace

_CACHE=None

def official():
    global _CACHE
    if _CACHE is None: _CACHE=_build()
    return _CACHE


def prepare(source,fn_name):
    ns=official()
    return ns['PREAMBLE']+(source if fn_name else ns['prepare_stdin'](source))


def compare(output,expected,fn_name):
    ns=official()
    with contextlib.redirect_stdout(io.StringIO()):
        try: return ns['compare_call' if fn_name else 'compare_stdout'](deepcopy(output),deepcopy(expected))
        except (ValueError,TypeError,IndexError,KeyError): return False


def normalize_call(input_value,expected):
    # These are the exact input/output dictionary conversions before method(*inputs).
    inputs=deepcopy(input_value); expected=deepcopy(expected)
    try:
        if isinstance(inputs[0],dict): inputs=[{int(k):v for k,v in inputs[0].items()}]
    except Exception: pass
    try:
        if isinstance(expected,dict): expected=[{int(k):v for k,v in expected.items()}]
    except Exception: pass
    try:
        if isinstance(expected[0],dict): expected=[{int(k):v for k,v in expected[0].items()}]
    except Exception: pass
    return inputs,expected
