"""Fuel-bounded research control language: functions, branches, folds and effects.

This permits new diagnostic/data-processing algorithms and feedback-dependent
research workflows, not just selection of a fixed ranking formula. It remains a
restricted language, never arbitrary Python or a claim of unrestricted invention.
"""
import json
from ..core.contracts import canonical
from ..core.protocol import ProtocolError, JSONValue

LANGUAGE='research-control-v1'
OPERATIONS={'get','list','object','append','concat','length','add','sub','eq','lt','and','or','not'}
EFFECTS={'invent','experiment','evaluate','predict','model','emit',
         'code_generate','code_test','code_submit','failure_analysis','code_experiment'}


def load(artifact):
    if artifact.kind!='meta' or artifact.language!=LANGUAGE: raise ProtocolError('control artifact language/kind')
    def pairs(items):
        result={}
        for key,value in items:
            if key in result: raise ProtocolError('duplicate control key')
            result[key]=value
        return result
    data=json.loads(artifact.source,object_pairs_hook=pairs)
    return validate(data)


def validate(data):
    if not isinstance(data,dict) or set(data)!={'score','functions','entry'}: raise ProtocolError('control document fields')
    from .expression import freeze,typecheck
    facts={k:'integer' for k in ('errors','complexity','cost','novelty','coverage')}
    if typecheck(freeze(data['score']),facts)!='integer': raise ProtocolError('control score type')
    functions=data['functions']
    if not isinstance(functions,dict) or len(functions)>16: raise ProtocolError('control function bound')
    nodes=[0]
    def walk(node,scope,depth=0):
        nodes[0]+=1
        if nodes[0]>512 or depth>24 or not isinstance(node,list) or not node or type(node[0]) is not str:
            raise ProtocolError('control AST bound/type')
        tag=node[0]
        if tag=='const':
            if len(node)!=2: raise ProtocolError('constant arity')
            JSONValue.of(node[1]); return
        if tag=='var':
            if len(node)!=2 or type(node[1]) is not str or node[1] not in scope: raise ProtocolError('unbound control variable')
            return
        if tag=='let':
            if len(node)!=4 or type(node[1]) is not str: raise ProtocolError('let arity')
            walk(node[2],scope,depth+1); walk(node[3],scope|{node[1]},depth+1); return
        if tag=='fold':
            if len(node)!=6 or type(node[2]) is not str or type(node[3]) is not str or node[2]==node[3]: raise ProtocolError('fold bindings')
            walk(node[1],scope,depth+1); walk(node[4],scope,depth+1); walk(node[5],scope|{node[2],node[3]},depth+1); return
        if tag=='if':
            if len(node)!=4: raise ProtocolError('branch arity')
            for n in node[1:]: walk(n,scope,depth+1)
            return
        allowed=functions if tag=='call' else OPERATIONS if tag=='op' else EFFECTS if tag=='effect' else ()
        if len(node)<2 or type(node[1]) is not str or node[1] not in allowed: raise ProtocolError('unknown control instruction/capability')
        if tag=='call' and len(node)-2!=len(functions[node[1]]['parameters']): raise ProtocolError('control call arity')
        for n in node[2:]: walk(n,scope,depth+1)
    for name,fn in functions.items():
        if type(name) is not str or not isinstance(fn,dict) or set(fn)!={'parameters','body'}: raise ProtocolError('control function schema')
        params=fn['parameters']
        if not isinstance(params,list) or len(params)>8 or any(type(p) is not str for p in params) or len(set(params))!=len(params): raise ProtocolError('function parameters')
    for fn in functions.values(): walk(fn['body'],set(fn['parameters']))
    walk(data['entry'],{'input'})
    return data


def execute(data,context,effect,fuel=4096):
    """Only effect is a capability; in production it is a sandbox RPC, not a host object."""
    if type(fuel) is not int or not 1<=fuel<=10000: raise ProtocolError('control fuel bound')
    remaining=[fuel]
    def bounded(value):
        if isinstance(value,(list,dict)) and len(value)>256: raise ProtocolError('control collection bound')
        if type(value) is int and abs(value)>1000000: raise ProtocolError('control numeric bound')
        if len(canonical(value).encode())>262144: raise ProtocolError('control wire bound')
        return value
    def run(node,env,depth=0):
        remaining[0]-=1
        if remaining[0]<0 or depth>32: raise ProtocolError('control fuel/depth exhausted')
        tag=node[0]
        if tag=='const': value=node[1]
        elif tag=='var': value=env[node[1]]
        elif tag=='let': return run(node[3],{**env,node[1]:run(node[2],env,depth+1)},depth+1)
        elif tag=='if':
            cond=run(node[1],env,depth+1)
            if type(cond) is not bool: raise ProtocolError('control condition must be boolean')
            return run(node[2] if cond else node[3],env,depth+1)
        elif tag=='fold':
            seq=run(node[1],env,depth+1)
            if not isinstance(seq,list) or len(seq)>256: raise ProtocolError('fold requires bounded sequence')
            value=run(node[4],env,depth+1)
            for item in seq: value=run(node[5],{**env,node[2]:value,node[3]:item},depth+1)
        elif tag=='call':
            fn=data['functions'][node[1]]
            args=[run(n,env,depth+1) for n in node[2:]]
            value=run(fn['body'],dict(zip(fn['parameters'],args)),depth+1)
        else:
            args=[run(n,env,depth+1) for n in node[2:]]
            if tag=='effect': value=effect(node[1],args)
            else:
                op=node[1]
                if op=='list': value=args
                elif op=='object':
                    if len(args)%2 or any(type(x) is not str for x in args[::2]) or len(set(args[::2]))!=len(args)//2: raise ProtocolError('object pairs')
                    value=dict(zip(args[::2],args[1::2]))
                elif op=='get':
                    if len(args)!=2 or not isinstance(args[0],(list,dict)): raise ProtocolError('get arity/type')
                    if isinstance(args[0],list) and (type(args[1]) is not int or not 0<=args[1]<len(args[0])): raise ProtocolError('get sequence index')
                    value=args[0][args[1]]
                elif op=='length':
                    if len(args)!=1 or not isinstance(args[0],(list,dict,str)): raise ProtocolError('length type')
                    value=len(args[0])
                elif op=='not':
                    if len(args)!=1 or type(args[0]) is not bool: raise ProtocolError('boolean type')
                    value=not args[0]
                else:
                    if len(args)!=2: raise ProtocolError('binary operation arity')
                    a,b=args
                    if op in ('add','sub','lt'):
                        if type(a) is not int or type(b) is not int: raise ProtocolError('integer operation type')
                        value=a+b if op=='add' else a-b if op=='sub' else a<b
                    elif op=='eq': value=type(a) is type(b) and a==b
                    elif op in ('and','or'):
                        if type(a) is not bool or type(b) is not bool: raise ProtocolError('boolean operation type')
                        value=(a and b) if op=='and' else (a or b)
                    elif op=='append':
                        if not isinstance(a,list): raise ProtocolError('append type')
                        value=a+[b]
                    elif op=='concat':
                        if type(a) is not type(b) or not isinstance(a,(list,str)): raise ProtocolError('concat type')
                        value=a+b
                    else: raise ProtocolError('unknown operation')
        return bounded(value)
    try: return run(data['entry'],{'input':context})
    except (KeyError,IndexError,TypeError,RecursionError) as exc: raise ProtocolError('control runtime data/type error') from exc
