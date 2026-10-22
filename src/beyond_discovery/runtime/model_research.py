"""Model-generated executable research programs under the same independent gates."""
from dataclasses import replace
import json
from .synthesis import SynthesisRuntime, grammar_from
from .expression import typecheck, freeze
from .trainer import LoRABackend, TrainingRecord
from ..core.contracts import ChallengeProposal, Usage, canonical, digest, plain
from ..core.experiments import protocol_actions
from ..core.meta import meta_artifact, META_VARIABLES
from ..core.protocol import ProtocolError, CapabilityUnsupported
from ..core.training import train_role
from ..core.worlds import expression_from


def strict_json(text):
    def pairs(items):
        result={}
        for k,v in items:
            if k in result: raise ProtocolError('duplicate model JSON key')
            result[k]=v
        return result
    if len(text)>100000: raise ProtocolError('model source size exceeded')
    return json.loads(text,object_pairs_hook=pairs,parse_constant=lambda _: (_ for _ in ()).throw(ProtocolError('nonfinite JSON')))


class ModelResearchRuntime(SynthesisRuntime):
    # Artifact execution is identical to synthesis; replay never repeats API calls.
    runtime_id='evidence-synthesis-v1'
    def __init__(self,context,clients,training=None):
        super().__init__(context)
        if set(clients)!={'scientist','challenger'}: raise ValueError('two role clients required')
        self.clients=clients; self.training_config=training
        self.model_events=[]; self.sampled={}; self.total_training_steps=0

    @property
    def total_model_tokens(self):
        ledgers={id(c.ledger):c.ledger for c in self.clients.values()}
        return sum(x.used for x in ledgers.values())

    def _call(self,parent,kind,training,context,limit,meta=None):
        if any(e.task.split!='train' for e in training): raise PermissionError('model generation is train-only')
        client=self.clients[parent.role]
        if client.model!=parent.model.base_model: raise ProtocolError('shared model identity mismatch')
        if parent.model.delta_checkpoint and not hasattr(client,'prepare'):
            raise CapabilityUnsupported('CAPABILITY_UNSUPPORTED: remote client cannot load local adapter')
        if hasattr(client,'prepare'): client.prepare(parent.model)
        prompt=canonical({'task':kind,'role':parent.role,'public_context':context,'parent':parent,
                          'meta_policy':meta or parent.meta,'train_evidence':[e.agent_view() for e in training],
                          'limit':limit,'contract':{'world':'JSON object: expression (typed AST), track (A/B)',
                          'candidate':'JSON list of objects: grammar (optional complete typed grammar), experiment (optional action list), meta (optional scoring AST). Pure code only; no answers or verifier changes.'}})
        start=len(client.calls)
        try:
            result=client(prompt,2048)
            if result.truncated or result.model!=parent.model.base_model: raise ProtocolError('truncated or mismatched model result')
            data=strict_json(result.text)
            self.sampled[(parent.content_hash,kind)]=(prompt,result,tuple(digest(e) for e in training))
            return data
        finally:
            self.model_events.extend({'role':parent.role,'call':call} for call in client.calls[start:])

    def propose(self,public_context,challenger,scientist,training,generation):
        self.public=public_context; self.training=training
        data=self._call(challenger,'world',training,public_context,1)
        if not isinstance(data,dict) or set(data)!={'expression','track'} or data['track'] not in ('A','B'):
            raise ProtocolError('model world schema')
        node=freeze(data['expression']); grammar=grammar_from(public_context['grammar'])
        if typecheck(node,dict(grammar.variables))!=grammar.output_type: raise ProtocolError('world type mismatch')
        code=self.artifact('world',{'expression':node,'models':self.models(training),'scientist_hash':scientist.content_hash,
                                  'evidence':[digest(e) for e in training]},self.provenance(challenger))
        return ChallengeProposal(public_context['domain_id'],code,data['track'])

    def candidates(self,parent,diagnosis,training,public_context,limit,mode):
        self.training=training
        if mode=='fixed': return ()
        data=self._call(parent,'candidate',training,public_context,limit)
        if not isinstance(data,list) or len(data)>limit: raise ProtocolError('model candidate count')
        results=[]
        for item in data:
            if not isinstance(item,dict) or not item or not set(item)<={'grammar','experiment','meta'}:
                raise ProtocolError('model candidate schema')
            artifacts=list(parent.artifacts); meta=parent.meta; provenance=self.provenance(parent)
            if 'grammar' in item and mode!='no_restructuring':
                grammar=grammar_from(item['grammar']); allowed=grammar_from(public_context['grammar'])
                if grammar.variables!=allowed.variables or grammar.output_type!=allowed.output_type or not set(grammar.operators)<=set(allowed.operators):
                    raise ProtocolError('model changed domain semantics')
                artifacts[0]=self.artifact('representation',{'grammar':plain(grammar)},provenance)
            if 'experiment' in item and mode!='no_falsification':
                experiment=self.artifact('experiment',{'actions':item['experiment'],'hypotheses':[],'search_cost':0},provenance)
                # Protocol language differs from expression grammar, executable VM validates.
                experiment=replace(experiment,language='research-protocol-v1')
                actions=protocol_actions(experiment)
                if len(actions)>parent.experiment.action_budget: raise ProtocolError('model experiment budget')
                for a in actions:
                    if plain(a) not in public_context['actions']: raise ProtocolError('model action outside public domain')
                artifacts=[a for a in artifacts if a.kind!='experiment']+[experiment]
            if 'meta' in item and mode!='no_meta':
                program=meta_artifact(freeze(item['meta']),provenance)
                if typecheck(freeze(item["meta"]),dict(META_VARIABLES))!="integer": raise ProtocolError("meta type mismatch")
                meta=replace(meta,program=program,revision=meta.revision+1)
            child=replace(parent,artifacts=tuple(artifacts),meta=meta,parent_hash=parent.content_hash,generation=parent.generation+1)
            results.append((child,{'operator':'MODEL_CODE','request_hash':digest(self.sampled[(parent.content_hash,'candidate')][0])}))
        return tuple(results)

    def trained_candidates(self,parent,attempts,training,evaluate,root):
        if self.training_config is None or not attempts: return attempts
        if not hasattr(self.clients[parent.role],'prepare'): raise CapabilityUnsupported('CAPABILITY_UNSUPPORTED: training requires a checkpoint-capable local client')
        prompt,result,hashes=self.sampled[(parent.content_hash,'candidate')]
        # Rewards come exclusively from independent TRAIN execution, not agent answers.
        evaluated=[evaluate(child) for child,_ in attempts]
        reward=sum(e.score.value if e.score.valid else 0 for batch in evaluated for e in batch)/sum(len(b) for b in evaluated)
        if reward<=0:
            self.events.append({'type':'training-skipped','role':parent.role,'reason':'no useful verified train rollout'})
            return attempts
        record=TrainingRecord(prompt,result.text,hashes,reward,behavior_logprob=result.behavior_logprob,policy_id=result.policy_id,prompt_ids=result.prompt_ids,completion_ids=result.completion_ids)
        steps=self.training_config.get('steps',1)
        if self.total_training_steps+steps>self.training_config['max_steps']: raise ProtocolError('TRAINING_STEP_BUDGET_EXHAUSTED')
        # Charge attempted steps conservatively even if a backend fails mid-update.
        self.total_training_steps+=steps
        backend=LoRABackend((record,),**{k:v for k,v in self.training_config.items() if k!='max_steps'})
        trained=train_role(parent,training,backend,root)
        self.events.extend({'type':'training-receipt',**r} for r in backend.receipts)
        generated=self.candidates(trained,None,training,self.public,len(attempts),'full')
        return tuple((child,{'operator':'TRAINED_MODEL_CODE','checkpoint':trained.model.delta_checkpoint,'patch':patch}) for child,patch in generated)

    def adapt(self,parent,diagnosis,public_context,meta):
        # Fixed parent weights/program, fixed train opportunity and output cap.
        attempts=self.candidates(replace(parent,meta=meta),diagnosis,self.training,public_context,1,'full')
        return attempts[0] if attempts else (parent,None)
