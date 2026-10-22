"""Narrow, independently checked train capabilities for research workflow code."""
from dataclasses import replace,asdict
from ..core.contracts import digest,plain
from ..core.protocol import Action,ProtocolError
from ..core.control import is_control
from ..core.synthesis import ArtifactInventor,enumerate_programs,errors
from ..core.experiments import design_experiment,protocol_actions,rollout
from ..core.worlds import expression_from
from .control import load
from .expression import freeze,typecheck
from .sandbox import executor,ResearchSandbox


def run_control(runtime,parent,training,context,meta,limit,mode):
    if any(e.task.split!='train' or not e.score.valid or e.failure for e in training):
        raise PermissionError('control capabilities require independently verified train evidence')
    program=load(meta.program)
    grammar=runtime.grammar(parent)
    public_actions=tuple(Action.create(a['operation'],a['arguments']) for a in context['actions'])
    emitted=[]; trace=[]; provenance=runtime.provenance(parent)
    records=[{'index':i,'refuted':any(getattr(c.kind,'value',c.kind)=='SPACE_REFUTED' for c in e.boundary_certificates),
              'errors':len(e.boundaries),'cluster':e.task.source_cluster,'observations':plain(e.observations)} for i,e in enumerate(training)]
    visible_records=[{k:v for k,v in r.items() if k!='refuted'} for r in records] if mode in ('program_evolution','no_diagnosis','reflection') else records
    input_data={'role':parent.role,'evidence':visible_records,'actions':plain(public_actions),
                'models':plain(runtime.models(training)),'grammar':asdict(grammar)}
    def accept(node):
        if len(emitted)>=limit: return {'emitted':0}
        if typecheck(node,dict(grammar.variables))!=grammar.output_type: raise ProtocolError('control emitted wrong program type')
        examples=tuple(x for e in training for x in runtime.examples(e))
        runtime.charge_search(runtime.search_budget)
        existing=enumerate_programs(grammar,runtime.search_budget).programs
        before=min((errors(p,examples) for p in existing),default=len(examples))
        after=errors(node,examples)
        if after>=before: return {'emitted':0,'errors':after}
        artifact=runtime.hypothesis(node,provenance)
        updated=replace(grammar,features=grammar.features+(node,))
        representation=runtime.artifact('representation',{'grammar':asdict(updated)},provenance)
        child=replace(parent,artifacts=(representation,artifact)+tuple(a for a in parent.artifacts[1:] if a.kind=='experiment'),
                      parent_hash=parent.content_hash,generation=parent.generation+1,
                      knowledge=parent.knowledge+tuple(digest(e) for e in training))
        if any(c.artifacts==child.artifacts for c,_ in emitted): return {'emitted':0}
        emitted.append((child,{'operator':'CONTROL_PROGRAM','workflow':meta.program.content_hash,'train_errors':after}))
        return {'emitted':1,'errors':after}
    def experiment(artifact):
        if mode=='no_falsification': raise ProtocolError('experiment capability disabled by preregistered ablation')
        actions=protocol_actions(artifact)
        if not actions or len(actions)>parent.experiment.action_budget or any(a not in public_actions for a in actions):
            raise ProtocolError('control experiment actions/budget')
        if len(emitted)>=limit or any(a.kind=='experiment' and a.source==artifact.source for a in parent.artifacts): return {'emitted':0}
        child=replace(parent,artifacts=tuple(a for a in parent.artifacts if a.kind!='experiment')+(artifact,),
                      parent_hash=parent.content_hash,generation=parent.generation+1,
                      knowledge=parent.knowledge+tuple(digest(e) for e in training))
        emitted.append((child,{'operator':'CONTROL_EXPERIMENT','workflow':meta.program.content_hash}))
        return {'emitted':1}
    def effect(name,args):
        if len(trace)>=64: raise ProtocolError('control effect budget')
        row={'name':name,'arguments_hash':digest(args)}; trace.append(row)
        if name=='invent':
            if mode=='no_restructuring': raise ProtocolError('representation capability disabled')
            if len(args)!=2 or args[0] not in meta.operators or args[0]=='INVENT_EXPERIMENT' or type(args[1]) is not int or not 0<=args[1]<len(training): raise ProtocolError('invent capability arguments')
            evidence=training[args[1]]
            # Independent certificates retain their meaning; workflow heuristics
            # cannot relabel an uncertified space as refuted.
            if mode not in ('program_evolution','no_diagnosis','reflection') and not records[args[1]]['refuted']:
                result={'emitted':0,'reason':'NO_SPACE_REFUTATION'}
            else:
                runtime.charge_search(runtime.search_budget)
                pool=enumerate_programs(grammar,runtime.search_budget).programs
                inventions=ArtifactInventor(grammar,runtime.search_budget).invent(args[0],pool,runtime.examples(evidence),provenance)
                result={'emitted':0}
                for invention in inventions:
                    result=accept(expression_from(invention.artifact))
                    if result['emitted']: break
        elif name=='evaluate':
            if len(args)!=2 or type(args[1]) is not int or not 0<=args[1]<len(training): raise ProtocolError('evaluate train scope')
            node=freeze(args[0]); typecheck(node,dict(grammar.variables))
            examples=runtime.examples(training[args[1]]); runtime.charge_search(max(1,len(examples)))
            result={'errors':errors(node,examples),'examples':len(examples)}
        elif name=='predict':
            if len(args)!=2 or not isinstance(args[1],list): raise ProtocolError('predict schema')
            node=freeze(args[0]); typecheck(node,dict(grammar.variables))
            actions=tuple(Action.create(a['operation'],a['arguments']) for a in args[1])
            if len(actions)>parent.experiment.action_budget or any(a not in public_actions for a in actions): raise ProtocolError('predict public actions/budget')
            runtime.charge_search(max(1,len(actions))); result=plain(rollout(node,actions,context['initial']))
        elif name=='experiment':
            if len(args)!=1 or type(args[0]) is not int or not 1<=args[0]<=parent.experiment.action_budget: raise ProtocolError('experiment horizon')
            runtime.charge_search(runtime.search_budget)
            models=runtime.models(training)
            result={'emitted':0} if len(models)<2 else experiment(design_experiment(models,public_actions,context['initial'],args[0],runtime.search_budget,provenance))
        elif name=='model':
            if not hasattr(runtime,'clients') or not hasattr(runtime,'_call'): raise ProtocolError('CAPABILITY_UNSUPPORTED: no real model client')
            if len(args)!=1 or type(args[0]) is not int or not 1<=args[0]<=getattr(runtime,'adaptation_candidates',5): raise ProtocolError('model capability candidate bound')
            if getattr(runtime,'adaptation_limit',None) is not None:
                if runtime.adaptation_model_calls>=1: raise ProtocolError('matched meta model-call budget exhausted')
                runtime.adaptation_model_calls+=1
            result=runtime._call(parent,'candidate',training,context,args[0])
            if not isinstance(result,list) or len(result)>args[0]: raise ProtocolError('model capability response schema')
        elif name=='emit':
            if len(args)!=1 or not isinstance(args[0],dict): raise ProtocolError('emit schema')
            data=args[0]
            if set(data)=={'expression'}:
                if mode=='no_restructuring': raise ProtocolError('representation capability disabled')
                result=accept(freeze(data['expression']))
            elif set(data)=={'grammar'}:
                if mode=='no_restructuring': raise ProtocolError('representation capability disabled')
                from .synthesis import grammar_from
                candidate=grammar_from(data['grammar']); allowed=grammar_from(context['grammar'])
                if candidate.variables!=allowed.variables or candidate.output_type!=allowed.output_type or not set(candidate.operators)<=set(allowed.operators): raise ProtocolError('control grammar domain mismatch')
                runtime.charge_search(runtime.search_budget)
                programs=enumerate_programs(candidate,runtime.search_budget).programs
                examples=tuple(x for e in training for x in runtime.examples(e))
                best=min(programs,key=lambda p:(errors(p,examples),digest(p)))
                result=accept(best)
            elif set(data)=={'actions'}:
                artifact=runtime.artifact('experiment',{'actions':data['actions'],'hypotheses':[],'search_cost':0},provenance)
                result=experiment(replace(artifact,language='research-protocol-v1'))
            else: raise ProtocolError('emit cannot change trusted state')
        else: raise ProtocolError('unknown research capability')
        row['result']=result; return result
    outer=executor.get()
    try:
        # A separate tool worker avoids reentrant RPC into the suspended control
        # worker. Neither worker receives private evaluation data or host objects.
        if outer is None:
            with ResearchSandbox() as outer:
                with ResearchSandbox(): result=outer.control(program,input_data,effect)
        else:
            with ResearchSandbox(): result=outer.control(program,input_data,effect)
    finally:
        runtime.events.append({'type':'research-control-trace','workflow':meta.program.content_hash,
                               'role':parent.role,'train_evidence':tuple(digest(e) for e in training),'trace':trace})
    return tuple(emitted),{'result':result,'trace':trace}
