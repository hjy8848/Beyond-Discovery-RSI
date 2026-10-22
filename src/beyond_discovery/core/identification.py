"""Independent bounded identifiability proof for executable worlds.

Scope is the complete registered finite comparison grammar plus proposed world.
This proves an executable distinguishing experiment within the action budget,
not identifiability among every possible program or outside declared probes.
"""
from dataclasses import dataclass, replace
from itertools import product
from .contracts import Status, digest
from .protocol import ProtocolError
from .synthesis import enumerate_programs
from ..runtime.expression import typecheck


@dataclass(frozen=True)
class IdentificationCertificate:
    status: Status
    family_hash: str
    experiment: tuple
    remaining_classes: int
    evaluations: int
    scope: str = 'complete registered grammar(max_nodes<=3) + proposal; behavior on probes of length<=2'


def identify_world(program,grammar,actions,initial,action_budget,evaluation_budget,semantics):
    enumeration=enumerate_programs(replace(grammar,max_nodes=min(3,grammar.max_nodes),features=()),2048)
    if not enumeration.complete: return IdentificationCertificate(Status.PENDING,'',(),0,0)
    pool=tuple(p for p in enumeration.programs if typecheck(p,dict(grammar.variables))==grammar.output_type)
    pool=tuple(dict.fromkeys(pool+(program,)))
    count=0
    def trace(node,sequence,state,offset=0):
        nonlocal count
        result=[]
        for t,action in enumerate(sequence):
            if count>=evaluation_budget: raise ProtocolError('identification budget')
            count+=1
            state=semantics(node,{'input':action.arguments.value,'previous':state,'time':offset+t})
            result.append(state)
        return tuple(result)
    experiment=[]
    family_hash=digest(pool)
    try:
        probes=tuple((a,) for a in actions)+tuple(product(actions,repeat=2))
        classes={p:digest(tuple(trace(p,s,initial) for s in probes)) for p in pool}
        target=classes[program]
        remaining=pool; states={p:initial for p in pool}; current=initial
        while len({classes[p] for p in remaining})>1 and len(experiment)<action_budget:
            options=[]
            for seq in probes:
                if len(seq)>action_budget-len(experiment): continue
                observed=trace(program,seq,current,len(experiment))
                survivors=tuple(p for p in remaining if trace(p,seq,states[p],len(experiment))==observed)
                options.append(((len(remaining)-len(survivors))/len(seq),-len(seq),digest(seq),seq))
            _,_,_,sequence=max(options)
            # Execute the entire independently checked protocol fragment.
            for action in sequence:
                observed=trace(program,(action,),current,len(experiment))[0]
                updated={p:trace(p,(action,),states[p],len(experiment))[0] for p in remaining}
                remaining=tuple(p for p in remaining if updated[p]==observed)
                states={p:updated[p] for p in remaining}; current=observed; experiment.append(action)
        n=len({classes[p] for p in remaining})
        return IdentificationCertificate(Status.VERIFIED if n==1 else Status.PENDING,family_hash,tuple(experiment),n,count)
    except ProtocolError:
        return IdentificationCertificate(Status.PENDING,family_hash,tuple(experiment),0,count)
