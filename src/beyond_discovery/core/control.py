"""Train-only research control artifacts and structural program mutation."""
from copy import deepcopy
import json
from .contracts import canonical,digest
from .protocol import ResearchArtifact,Schema,ProtocolError
from ..runtime.control import LANGUAGE,load,validate


def is_control(artifact): return artifact is not None and artifact.language==LANGUAGE


def control_artifact(data,provenance,version='1'):
    if provenance.split!='train': raise PermissionError('research control generation is train-only')
    validate(data)
    return ResearchArtifact('meta',LANGUAGE,version,canonical(data),Schema.of({'type':'object'}),Schema.of({'oneOf':[{'type':t} for t in ('object','array','string','number','boolean','null')]}),provenance)


def seed_control(parent,provenance):
    """Compatibility starting program; search/model proposals may replace any body.

    This is an initial policy, not a fabricated discovery or a menu of workflows.
    """
    score=json.loads(parent.source)['score' if is_control(parent) else 'expression']
    V=lambda n:['var',n]
    C=lambda v:['const',v]
    get=lambda node,key:['op','get',node,C(key)]
    diagnose=get(V('record'),'refuted')
    research=['if',['call','diagnose',V('record')],
              ['effect','invent',C('REFRAME'),get(V('record'),'index')],C({'emitted':0})]
    loop=['fold',get(V('input'),'evidence'),'count','record',C(0),
          ['op','add',V('count'),get(['call','research',V('record')],'emitted')]]
    body=['let','discovered',loop,['if',['op','eq',V('discovered'),C(0)],['effect','experiment',C(3)],V('discovered')]]
    entry=['if',['op','eq',get(V('input'),'role'),C('challenger')],['effect','experiment',C(3)],body]
    return control_artifact({'score':score,'functions':{
        'diagnose':{'parameters':['record'],'body':diagnose},
        'research':{'parameters':['record'],'body':research}},'entry':entry},provenance)


def _nodes(node,path=()):
    yield path,node
    if node[0]=='const': return
    for i,child in enumerate(node[1:],1):
        if isinstance(child,list) and child and type(child[0]) is str:
            yield from _nodes(child,path+(i,))


def mutate_control(parent,provenance,budget=16,record_fields=None):
    """Generic AST edits/crossover, not a preauthored catalog of research methods.

    Mutations are hypotheses only. The caller must run them on fixed train
    opportunities and require actual trace/product differences before proposing.
    """
    data=load(parent); seen={parent.source}; results=[]
    roots=[('entry',None)]+[('functions',n) for n in data['functions']]
    def root(d,kind,name): return d['entry'] if kind=='entry' else d['functions'][name]['body']
    def changed(kind,name,path,replacement):
        new=deepcopy(data)
        if not path:
            if kind=='entry': new['entry']=replacement
            else: new['functions'][name]['body']=replacement
        else:
            n=root(new,kind,name)
            for i in path[:-1]: n=n[i]
            n[path[-1]]=replacement
        return new
    edits=[]
    for kind,name in roots:
        nodes=list(_nodes(root(data,kind,name)))
        if kind=='functions' and record_fields:
            # Data-schema projections and typed predicate composition generate
            # new subprogram bodies; no correct diagnostic rule is supplied.
            integers=[['const',0],['const',1]]
            booleans=[['const',True],['const',False]]
            for param in data['functions'][name]['parameters']:
                for field,field_type in record_fields.items():
                    projection=['op','get',['var',param],['const',field]]
                    if field_type=='integer': integers.append(projection)
                    if field_type=='boolean': booleans.append(projection)
                    if field_type in ('array','string'): integers.append(['op','length',projection])
            for left in integers:
                for right in integers:
                    for op in ('lt','eq'): booleans.append(['op',op,left,right])
            for body in booleans:
                edits.append((kind,name,(),body))
        for path,node in nodes:
            if node[0]=='if': edits.append((kind,name,path,[node[0],node[1],node[3],node[2]]))
            if node[0]=='op' and node[1] in ('and','or','lt','eq','add','sub'):
                for op in ('and','or','lt','eq','add','sub'):
                    if op!=node[1]: edits.append((kind,name,path,['op',op]+node[2:]))
            if node[0]=='const' and node[1]=='REFRAME':
                for op in ('RECOMBINE','RELAX_ASSUMPTION','ANALOGIZE'): edits.append((kind,name,path,['const',op]))
            # Lexically valid subtree transplantation can replace diagnostics,
            # remove/reorder research stages or change data-dependent branches.
            for _,donor in nodes:
                if donor!=node: edits.append((kind,name,path,donor))
    import random
    random.Random(int(digest((parent.content_hash,provenance.agent_version)),16)).shuffle(edits)
    for kind,name,path,node in edits:
        candidate=changed(kind,name,path,deepcopy(node))
        try: artifact=control_artifact(candidate,provenance,str(int(parent.version)+1))
        except (ProtocolError,TypeError,ValueError): continue
        if artifact.source in seen: continue
        seen.add(artifact.source); results.append(artifact)
        if len(results)>=budget: break
    return tuple(results)
