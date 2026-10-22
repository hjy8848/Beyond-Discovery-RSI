from dataclasses import replace
import unittest
from beyond_discovery.core.contracts import Provenance,digest
from beyond_discovery.core.control import control_artifact,seed_control,mutate_control
from beyond_discovery.core.protocol import ProtocolError
from beyond_discovery.core.meta import meta_artifact,score
from beyond_discovery.runtime.control import load,execute
from beyond_discovery.runtime.sandbox import ResearchSandbox

C=lambda x:['const',x]
V=lambda x:['var',x]
G=lambda n,k:['op','get',n,C(k)]


class ResearchControlTests(unittest.TestCase):
    def setUp(self): self.p=Provenance('control','train','test',0,'s',0,'v')
    def artifact(self,entry,functions=None):
        return control_artifact({'score':['sub',['lit',0],['ref','errors']],'functions':functions or {},'entry':entry},self.p)

    def test_invented_diagnostic_function_changes_actual_control_flow(self):
        # Same evidence, same scoring expression; a different executable
        # diagnostic subprogram changes WHICH tool runs, not candidate ordering.
        fn={'diagnose':{'parameters':['record'],'body':['op','lt',C(1),G(V('record'),'errors')]}}
        body=['fold',G(V('input'),'evidence'),'result','record',C([]),
              ['if',['call','diagnose',V('record')],['op','append',V('result'),['effect','evaluate',C(['ref','input.x']),G(V('record'),'index')]],V('result')]]
        a=self.artifact(body,fn)
        fn2={'diagnose':{'parameters':['record'],'body':['op','lt',C(0),G(V('record'),'errors')]}}
        b=self.artifact(body,fn2)
        ctx={'evidence':[{'index':0,'errors':1},{'index':1,'errors':2}]}
        traces=[]
        for artifact in (a,b):
            trace=[]
            with ResearchSandbox() as sandbox:
                result=sandbox.control(load(artifact),ctx,lambda name,args:trace.append((name,args)) or {'errors':0})
            traces.append(trace)
            self.assertEqual(len(result),len(trace))
        self.assertEqual(len(traces[0]),1); self.assertEqual(len(traces[1]),2)
        self.assertEqual(score(a,dict(errors=2,complexity=1,cost=1,novelty=0,coverage=1)),score(b,dict(errors=2,complexity=1,cost=1,novelty=0,coverage=1)))

    def test_sequential_feedback_creates_a_new_experiment_program(self):
        body=['let','observation',['effect','predict',C(['ref','input.x']),G(V('input'),'actions')],
              ['if',['op','eq',['op','length',V('observation')],C(2)],
               ['effect','emit',['op','object',C('actions'),G(V('input'),'actions')]],C(None)]]
        a=self.artifact(body); trace=[]
        def effect(name,args):
            trace.append((name,args))
            return [0,1] if name=='predict' else {'emitted':1}
        with ResearchSandbox() as sandbox:
            result=sandbox.control(load(a),{'actions':[{'operation':'probe','arguments':{'x':0}},{'operation':'probe','arguments':{'x':1}}]},effect)
        self.assertEqual([name for name,_ in trace],['predict','emit'])
        self.assertEqual(result,{'emitted':1})

    def test_general_functions_can_construct_code_not_selected_from_operator_menu(self):
        body=['effect','emit',['op','object',C('expression'),['op','list',C('add'),['op','list',C('ref'),C('input.x')],['op','list',C('ref'),C('previous')]]]]
        a=self.artifact(body); values=[]
        with ResearchSandbox() as sandbox: sandbox.control(load(a),{},lambda name,args:values.append(args[0]) or {'emitted':1})
        self.assertEqual(values[0]['expression'],['add',['ref','input.x'],['ref','previous']])

    def test_capability_lexical_scope_fuel_and_private_provenance_fail_closed(self):
        for body in (['effect','open',C('/etc/passwd')],V('missing')):
            with self.assertRaises(ProtocolError): self.artifact(body)
        fn={'loop':{'parameters':[],'body':['call','loop']}}
        recursive=self.artifact(['call','loop'],fn)
        with ResearchSandbox() as sandbox:
            with self.assertRaises(ProtocolError): sandbox.control(load(recursive),{},lambda *_:None,fuel=10)
        with self.assertRaises(PermissionError): control_artifact(load(self.artifact(C(0))),replace(self.p,split='dev'))

    def test_effect_rejection_does_not_desynchronize_worker(self):
        artifact=self.artifact(['effect','emit',C({'expression':['lit',1]})])
        with ResearchSandbox() as sandbox:
            with self.assertRaises(PermissionError): sandbox.control(load(artifact),{},lambda *_:(_ for _ in ()).throw(PermissionError('denied')))
            self.assertEqual(sandbox.evaluate(('lit',7),{},256),7)

    def test_structural_mutations_change_workflow_source_not_only_score(self):
        seed=seed_control(meta_artifact(('ref','novelty'),self.p),self.p)
        mutations=mutate_control(seed,self.p,10)
        self.assertTrue(mutations)
        self.assertTrue(any(load(x)['entry']!=load(seed)['entry'] for x in mutations))
        self.assertTrue(all(load(x)['score']==load(seed)['score'] for x in mutations))

    def test_real_adapter_executes_workflow_derived_protocol_and_rejects_private_data(self):
        from beyond_discovery.adapters.transition import StringTransitionAdapter
        from beyond_discovery.runtime.research import resolve_runtime
        from beyond_discovery.core.controller import RunConfig
        from beyond_discovery.core.interaction import episode
        from beyond_discovery.core.experiments import protocol_actions
        adapter=StringTransitionAdapter(); runtime=resolve_runtime(adapter)
        c,s=runtime.initial_pair(RunConfig())
        train=tuple(episode(adapter,t,s,c,0,runtime=runtime) for t in adapter.sample('train',0))
        body=['effect','emit',['op','object',C('actions'),['op','list',['op','get',G(V('input'),'actions'),C(1)]]]]
        program=self.artifact(body)
        runtime.training=train
        child,patch=runtime.adapt(s,None,adapter.public_context(),replace(s.meta,program=program))
        self.assertNotEqual(child.artifacts,s.artifacts)
        artifact=next(a for a in child.artifacts if a.kind=='experiment')
        observations=episode(adapter,adapter.sample('train',0)[0],child,c,1,runtime=runtime).observations
        self.assertTrue(all(o.action==protocol_actions(artifact)[0] for o in observations))
        private=episode(adapter,adapter.sample('dev',0)[0],s,c,0,runtime=runtime)
        runtime.training=(private,)
        with self.assertRaises(PermissionError): runtime.adapt(s,None,adapter.public_context(),replace(s.meta,program=program))

    def test_invalid_later_effect_discards_partial_research_product(self):
        from beyond_discovery.adapters.transition import StringTransitionAdapter
        from beyond_discovery.runtime.research import resolve_runtime
        from beyond_discovery.core.controller import RunConfig
        from beyond_discovery.core.interaction import episode
        adapter=StringTransitionAdapter(); runtime=resolve_runtime(adapter)
        c,s=runtime.initial_pair(RunConfig())
        runtime.training=tuple(episode(adapter,t,s,c,0,runtime=runtime) for t in adapter.sample('train',0))
        emit=['effect','emit',['op','object',C('actions'),['op','list',['op','get',G(V('input'),'actions'),C(0)]]]]
        body=['let','first',emit,['effect','emit',C({'score':1})]]
        parent_hash=s.content_hash
        child,_=runtime.adapt(s,None,adapter.public_context(),replace(s.meta,program=self.artifact(body)))
        self.assertEqual(child,s); self.assertEqual(s.content_hash,parent_hash)
        self.assertTrue(any(e['type']=='control-execution-invalid' for e in runtime.events))

    def test_workflow_can_call_model_capability_and_emit_checked_code(self):
        from beyond_discovery.adapters.transition import StringTransitionAdapter
        from beyond_discovery.runtime.model_research import ModelResearchRuntime
        from beyond_discovery.runtime.models import TokenLedger
        from beyond_discovery.runtime.llm import Completion
        from beyond_discovery.core.controller import RunConfig
        from beyond_discovery.core.interaction import episode
        import json
        adapter=StringTransitionAdapter()
        class Client:
            model='protocol-fixture'
            def __init__(self): self.ledger=TokenLedger(10000); self.calls=[]
            def __call__(self,prompt,limit):
                request=json.loads(prompt)
                assert all(e['task']['split']=='train' for e in request['train_evidence'])
                result=Completion(json.dumps([{'experiment':[adapter.public_context()['actions'][1]]}]),self.model,1,1)
                self.calls.append({'request':prompt,'response':result.__dict__})
                return result
        runtime=ModelResearchRuntime(adapter.public_context(),{r:Client() for r in ('scientist','challenger')})
        c,s=runtime.initial_pair(RunConfig(base_model='protocol-fixture'))
        runtime.training=tuple(episode(adapter,t,s,c,0,runtime=runtime) for t in adapter.sample('train',0))
        body=['let','proposals',['effect','model',C(1)],['effect','emit',['op','object',C('actions'),G(['op','get',V('proposals'),C(0)],'experiment')]]]
        meta=replace(s.meta,program=self.artifact(body))
        child,_=runtime.adapt(s,None,adapter.public_context(),meta)
        self.assertTrue(any(a.kind=='experiment' for a in child.artifacts))
        self.assertEqual(len(runtime.clients['scientist'].calls),1)
        self.assertIn((s.content_hash,'candidate'),runtime.sampled)

    def test_oversized_effect_packet_rejects_without_poisoning_worker(self):
        artifact=self.artifact(['effect','emit',V('input'),V('input'),V('input')])
        with ResearchSandbox() as sandbox:
            with self.assertRaises(ProtocolError): sandbox.control(load(artifact),{'text':'x'*100000},lambda *_:None)
            self.assertEqual(sandbox.evaluate(('lit',3),{},256),3)

    def test_workflow_cannot_buy_extra_model_calls_in_fixed_adaptation(self):
        from beyond_discovery.adapters.transition import StringTransitionAdapter
        from beyond_discovery.runtime.model_research import ModelResearchRuntime
        from beyond_discovery.runtime.models import TokenLedger
        from beyond_discovery.runtime.llm import Completion
        from beyond_discovery.core.controller import RunConfig
        from beyond_discovery.core.interaction import episode
        adapter=StringTransitionAdapter()
        class Client:
            model='protocol-fixture'
            def __init__(self): self.ledger=TokenLedger(10000); self.calls=[]
            def __call__(self,prompt,limit):
                self.calls.append({'request':prompt})
                return Completion('[]',self.model,0,0)
        runtime=ModelResearchRuntime(adapter.public_context(),{r:Client() for r in ('scientist','challenger')})
        c,s=runtime.initial_pair(RunConfig(base_model='protocol-fixture'))
        runtime.training=tuple(episode(adapter,t,s,c,0,runtime=runtime) for t in adapter.sample('train',0))
        program=self.artifact(['let','first',['effect','model',C(1)],['effect','model',C(1)]])
        child,_=runtime.adapt(s,None,adapter.public_context(),replace(s.meta,program=program))
        self.assertEqual(child,s)
        self.assertEqual(len(runtime.clients['scientist'].calls),1)
        allocation=next(e for e in runtime.events if e['type']=='meta-adaptation-allocation')
        self.assertEqual(allocation['model_calls'],1)
        self.assertLessEqual(allocation['search_used'],allocation['search_limit'])
