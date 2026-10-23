"""APPS DomainAdapter. Private cases/solutions never cross its public interface."""
from dataclasses import dataclass
import json
from ...core.contracts import Usage,Observation,Score,Status,Certification,digest,plain
from ...core.protocol import DomainCapabilities,Schema,ProtocolError,JSONValue
from .data import APPSStore
from .judge import DockerJudge

LANGUAGE='python-apps-v1'

@dataclass
class Session:
    task: object
    budget: int
    sequence: int=0


class APPSAdapter:
    domain_id='apps';verifier_version='apps-independent-judge-v1'
    def __init__(self,store,judge,episode_model_tokens=69632,task_limit=None,testing=False):
        if not isinstance(judge,DockerJudge) and not testing: raise TypeError('production APPS requires DockerJudge, not ResearchSandbox or a fixture')
        self.store=store if isinstance(store,APPSStore) else APPSStore(store)
        self.judge=judge;self.episode_model_tokens=episode_model_tokens;self.task_limit=task_limit
        self._usage=Usage();self._sessions={};self.judge_events=[];self._generated={}
        self._submissions={};self.frozen_scientist=None
    def describe(self):
        action=Schema.of({'type':'object','properties':{'operation':{'type':'string','enum':['public_test']},'arguments':{'type':'object','properties':{'code':{'type':'string','maxLength':65536},'indexes':{'type':'array','items':{'type':'integer','minimum':0}}},'required':['code'],'additionalProperties':False}},'required':['operation','arguments'],'additionalProperties':False})
        return DomainCapabilities('apps',action,Schema.of({'type':'object'}),Schema.of({'type':'object'}),(LANGUAGE,),('A',),8,
            features=('audit','sealed-test','stateful','archived-inference-replay','frozen-challenge-binding'),deterministic=False,
            score_definition='official APPS strict problem accuracy: every official case must pass',
            limitations=('APPS is public and may occur in model pretraining','Generated tasks require an independent legality validator and verified Oracle','No host execution fallback'))
    def public_context(self):
        return {'domain_id':'apps','manifest_hash':self.store.manifest['manifest_hash'],'judge':self.judge.identity,
                'episode_model_tokens':self.episode_model_tokens,'language':LANGUAGE,'verifier_version':self.verifier_version}
    def sample(self,split,seed):
        refs=self.store.sample(split,seed)
        return refs[:self.task_limit] if self.task_limit else refs
    def _data(self,task): return self._generated[task.task_id]['data'] if task.task_id in self._generated else self.store.load(task)
    def public_task(self,task):
        data=self._data(task)
        return {k:data[k] for k in ('question','starter','fn_name','public_cases')}
    def estimate_episode(self,action_budget,search_budget):
        maximum=max(r['case_count']+r['public_case_count'] for r in self.store.manifest['tasks'].values())
        return Usage(experiments=2*action_budget+4,model_tokens=self.episode_model_tokens,verification_outputs=(2*action_budget+4)*max(1,maximum))
    def estimate_certification(self):return Usage(experiments=32,verification_outputs=320000)
    def meter(self):return self._usage
    def begin(self,task,budget):
        self._data(task)
        if type(budget) is not int or not 1<=budget<=8:raise ProtocolError('APPS action budget')
        session=Session(task,budget);self._sessions[id(session)]=session;return session
    def _session(self,session):
        if self._sessions.get(id(session)) is not session:raise PermissionError('closed/foreign APPS session')
    def close(self,session):self._sessions.pop(id(session),None)
    def initial_observation(self,session):self._session(session);return JSONValue.of(self.public_task(session.task))
    def _judge(self,code,cases,fn_name,label):
        result=self.judge.grade(code,cases,fn_name)
        self._usage=self._usage+Usage(experiments=result.launches,verification_outputs=result.cases)
        self.judge_events.append({'label':label,'code_hash':digest(code),'cases_hash':digest(cases),'judge':self.judge.identity,'result':result})
        return result
    def step(self,session,action):
        self._session(session);self.describe().action_schema.validate(action)
        if session.sequence>=session.budget:raise ProtocolError('APPS action budget exhausted')
        args=action.arguments.value;data=self._data(session.task);cases=data['public_cases']
        indexes=args.get('indexes',list(range(len(cases))))
        if len(set(indexes))!=len(indexes) or any(i>=len(cases) for i in indexes):raise ProtocolError('not a public test index')
        chosen=[cases[i] for i in indexes]
        if chosen:
            result=self._judge(args['code'],chosen,data['fn_name'],'public-test')
            feedback={'status':result.status,'verdicts':plain(result.verdicts),'outputs':plain(result.outputs),'test_scope':'declared-public-tests'}
        else:feedback={'status':'NO_PUBLIC_TESTS','verdicts':[],'outputs':[],'test_scope':'declared-public-tests'}
        session.sequence+=1
        return Observation(action,JSONValue.of(feedback),session.sequence)
    def evaluate(self,claim,task,split):
        if split!=task.split:raise PermissionError('APPS evaluation split mismatch')
        data=self._data(task);program=claim.program
        if program.language!=LANGUAGE or program.provenance.domain_id!='apps':raise ProtocolError('APPS expects a Python ResearchArtifact')
        result=self._judge(program.source,data['private_cases'],data['fn_name'],'private-official')
        if task.split=='train':self._submissions[(task.task_id,program.content_hash)]=program
        return Score(float(result.accepted),True,result.cases,result.status if result.status!='COMPLETE' else ('AC' if result.accepted else 'WA'))
    def audit(self,evidence):
        if evidence.provenance.domain_id!='apps' or evidence.provenance.split!=evidence.task.split:return False
        if evidence.failure:return not evidence.score.valid
        if self.evaluate(evidence.new_claim,evidence.task,evidence.task.split)!=evidence.score:return False
        for i,observation in enumerate(evidence.observations):
            if observation.sequence!=i+1:return False
            session=self.begin(evidence.task,8)
            try:
                regenerated=self.step(session,observation.action)
                if regenerated.outcome!=observation.outcome:return False
            finally:self.close(session)
        return True
    def compile_challenge(self,proposal):
        return Certification(Status.PENDING,None,'',digest(proposal),self.verifier_version,'CAPABILITY_UNSUPPORTED: independent APPS Oracle/legality validator not configured')
    def conformance_actions(self):return ()
    def unseal_for_final(self):self.store.sealed=False
    def bind_frozen_scientist(self,content_hash):self.frozen_scientist=content_hash
