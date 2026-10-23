import json,tempfile,unittest
from pathlib import Path
from apps_fixtures import fixture_store,ProtocolJudge,ScriptedModel
from beyond_discovery.benchmarks.apps.adapter import APPSAdapter
from beyond_discovery.benchmarks.apps.data import APPSStore
from beyond_discovery.runtime.coding import CodingResearchRuntime,CodingLimits
from beyond_discovery.core.controller import RunConfig,run_rsi
from beyond_discovery.core.replay import replay
from beyond_discovery.core.archive import read_verified
from beyond_discovery.core.conformance import check_conformance

class ControllerTests(unittest.TestCase):
    def test_no_aggregate_model_token_cap_preserves_actual_accounting(self):
        from beyond_discovery.runtime.models import TokenLedger
        with tempfile.TemporaryDirectory() as root:
            limits=CodingLimits(2,256,64);store=fixture_store(root);ledger=TokenLedger(None)
            a=APPSAdapter(store,ProtocolJudge(),limits.tokens,task_limit=1,testing=True)
            client=ScriptedModel(['def add(a,b): return a+b']*200,ledger=ledger)
            r=CodingResearchRuntime({'scientist':client,'challenger':ScriptedModel([],ledger=ledger)},limits,method='self-repair',testing=True)
            out=Path(root)/'unbounded-run'
            cfg=RunConfig(generations=1,base_model=client.model,required_world_tracks=('A',),
                final_evaluation=False,max_model_tokens=None)
            result=run_rsi(a,out,cfg,runtime=r)
            self.assertEqual(result['status'],'SMOKE_COMPLETE')
            self.assertIsNone(result['test_score'])
            self.assertIsNone(cfg.max_model_tokens)
            self.assertGreater(result['usage']['model_tokens'],0)
            self.assertEqual(result['usage']['model_tokens'],ledger.used)
            self.assertIsNone(ledger.limit)

    def test_direct_controller_cannot_open_official_test(self):
        from beyond_discovery.core.protocol import ProtocolError
        with tempfile.TemporaryDirectory() as root:
            limits=CodingLimits(2,256,64);a=APPSAdapter(fixture_store(root),ProtocolJudge(),limits.tokens,testing=True)
            r=CodingResearchRuntime({'scientist':ScriptedModel([]),'challenger':ScriptedModel([])},limits,'self-repair',testing=True)
            with self.assertRaises(ProtocolError):run_rsi(a,Path(root)/'run',RunConfig(base_model='TEST_MODEL_FIXTURE'),runtime=r)
            self.assertTrue(a.store.sealed)
    def test_smoke_uses_no_test_costs_match_actual_calls_and_replay_is_offline(self):
        with tempfile.TemporaryDirectory() as root:
            limits=CodingLimits(2,256,64);store=fixture_store(root)
            a=APPSAdapter(store,ProtocolJudge(),limits.tokens,task_limit=2,testing=True)
            client=ScriptedModel(['def add(a,b): return a+b']*200)
            r=CodingResearchRuntime({'scientist':client,'challenger':ScriptedModel([])},limits,method='self-repair',testing=True)
            out=Path(root)/'run';cfg=RunConfig(generations=1,base_model=client.model,required_world_tracks=('A',),final_evaluation=False,max_model_tokens=2000000)
            summary=run_rsi(a,out,cfg,runtime=r)
            self.assertEqual(summary['status'],'SMOKE_COMPLETE');self.assertIsNone(summary['test_score'])
            self.assertTrue(store.sealed);self.assertFalse((out/'final_started.json').exists())
            self.assertFalse(any(p.startswith('test/') for p in store.loaded))
            records=read_verified(out);episodes=[v['evidence'] for k,v in records if k=='episode']
            self.assertEqual(summary['usage']['model_tokens'],client.ledger.used)
            self.assertEqual(client.ledger.used,12*len(client.calls))
            self.assertEqual(summary['usage']['experiments'],a.meter().experiments)
            self.assertEqual(sum(e['usage']['model_tokens'] for e in episodes),summary['usage']['model_tokens'])
            self.assertTrue(any(k=='runtime_judge_events' for k,_ in records))
            fresh=APPSAdapter(APPSStore(store.root),ProtocolJudge(),limits.tokens,2,testing=True)
            result=replay(out,fresh)
            self.assertEqual(result['episodes_reexecuted'],len(episodes));self.assertTrue(fresh.store.sealed)
            self.assertEqual(len(client.calls),len(episodes)*2)
    def test_generic_conformance_exercises_apps_protocol(self):
        from beyond_discovery.core.protocol import Action
        with tempfile.TemporaryDirectory() as root:
            limits=CodingLimits(2,256,64);store=fixture_store(root)
            a=APPSAdapter(store,ProtocolJudge(),limits.tokens,testing=True)
            r=CodingResearchRuntime({'scientist':ScriptedModel(['def add(a,b): return a+b']*4),'challenger':ScriptedModel([])},limits,testing=True)
            report=check_conformance(a,actions=(Action.create('public_test',{'code':'def add(a,b): return a+b'}),),runtime=r,config=RunConfig(base_model='TEST_MODEL_FIXTURE'))
            self.assertTrue(report.passed);self.assertIn('evidence_tamper_rejection',report.checks)
    def test_failed_inference_retains_ambiguous_token_charge_and_prior_episodes(self):
        from beyond_discovery.runtime.models import TokenLedger
        class Failure(ScriptedModel):
            def __call__(self,prompt,max_tokens):
                self.ledger.reserve(self.input_bound+max_tokens)
                self.calls.append({'status':'FAILED_EXPLICIT_FIXTURE'})
                raise RuntimeError('explicit transport fixture failure')
        with tempfile.TemporaryDirectory() as root:
            limits=CodingLimits(2,256,64);store=fixture_store(root);a=APPSAdapter(store,ProtocolJudge(),limits.tokens,task_limit=2,testing=True)
            r=CodingResearchRuntime({'scientist':Failure([]),'challenger':ScriptedModel([])},limits,testing=True)
            out=Path(root)/'run'
            with self.assertRaises(RuntimeError):run_rsi(a,out,RunConfig(generations=1,base_model='TEST_MODEL_FIXTURE',required_world_tracks=('A',),final_evaluation=False,max_model_tokens=2000000),runtime=r)
            records=read_verified(out);costs=[v for k,v in records if k=='evaluation_cost']
            self.assertEqual(costs[-1]['usage']['model_tokens'],320)
            self.assertTrue(any(k=='runtime_model_events' for k,_ in records));self.assertTrue(store.sealed)
    def test_final_requires_completed_freeze_and_is_one_shot(self):
        from beyond_discovery.benchmarks.apps.final import validate_frozen,run_final
        from beyond_discovery.core.protocol import ProtocolError
        with tempfile.TemporaryDirectory() as root:
            limits=CodingLimits(2,256,64);store=fixture_store(root)
            def make_adapter():return APPSAdapter(APPSStore(store.root),ProtocolJudge(),limits.tokens,2,testing=True)
            def make_runtime():return CodingResearchRuntime({'scientist':ScriptedModel(['def add(a,b): return a+b']*100),'challenger':ScriptedModel([])},limits,'self-repair',testing=True)
            a=make_adapter();r=make_runtime();out=Path(root)/'smoke'
            cfg=RunConfig(generations=1,base_model='TEST_MODEL_FIXTURE',required_world_tracks=('A',),final_evaluation=False,max_model_tokens=2000000)
            run_rsi(a,out,cfg,runtime=r)
            source=validate_frozen(out,make_adapter(),limits)
            final_adapter=make_adapter()
            result=run_final(source,final_adapter,make_runtime(),Path(root)/'final',cfg.max_model_tokens,cfg.max_experiments,cfg.max_verification_outputs)
            self.assertEqual(result['status'],'COMPLETE');self.assertFalse(final_adapter.store.sealed)
            with self.assertRaises(ProtocolError):validate_frozen(out,make_adapter(),limits)
            replay_adapter=make_adapter();self.assertTrue(replay(Path(root)/'final',replay_adapter)['verified'])
    def test_evolution_gate_restores_meta_with_frozen_parent_model(self):
        from beyond_discovery.runtime.llm import Completion
        class AdaptiveFixture(ScriptedModel):
            def __call__(self,prompt,max_tokens):
                data=json.loads(prompt);contract=data['contract']
                direct={'score':['lit',0],'functions':{},'entry':['let','v',['effect','code_generate',['const',False]],['effect','code_submit',['var','v']]]}
                if contract.startswith('Generate a complete Python'):
                    text='def add(a,b): return a+b' if data['instructions']==['FIXTURE_GOOD'] else 'def add(a,b): return 0'
                elif contract.startswith('Return JSON object'):
                    e=data['train_evidence'][0];text=json.dumps({'parent_task_id':e['task_id'],'old_artifact_hash':e['old_artifact_hash'],'input':[8,9]})
                else:
                    item={'instructions':'FIXTURE_GOOD'}
                    if data['role']=='scientist':item['control']=direct
                    text=json.dumps([item])
                reserved=self.ledger.reserve(self.input_bound+max_tokens);self.ledger.settle(reserved,12)
                self.calls.append({'request':prompt,'status':'EXPLICIT_ADAPTIVE_FIXTURE'})
                return Completion(text,self.model,8,4)
        with tempfile.TemporaryDirectory() as root:
            limits=CodingLimits(2,256,1024);store=fixture_store(root)
            a=APPSAdapter(store,ProtocolJudge(),limits.tokens,4,testing=True)
            r=CodingResearchRuntime({'scientist':AdaptiveFixture([]),'challenger':AdaptiveFixture([])},limits,testing=True)
            out=Path(root)/'run';cfg=RunConfig(generations=1,base_model='TEST_MODEL_FIXTURE',required_world_tracks=('A',),final_evaluation=False,max_model_tokens=2000000)
            result=run_rsi(a,out,cfg,runtime=r);records=read_verified(out)
            gates=[v for k,v in records if k=='gate' and v['role']=='scientist']
            self.assertEqual(gates[0]['task_gate']['decision'],'ACCEPT')
            self.assertTrue(gates[0]['meta_rollback']);self.assertEqual(gates[0]['meta_gate']['decision'],'INCONCLUSIVE')
            self.assertEqual(result['scientist']['representation'],['FIXTURE_GOOD'])
            attempts=[v for k,v in records if k=='world_attempt'];self.assertEqual(attempts[0]['certification']['status'],'PENDING')
            self.assertEqual(result['usage']['model_tokens'],r.total_model_tokens)
            fresh=APPSAdapter(APPSStore(store.root),ProtocolJudge(),limits.tokens,4,testing=True)
            self.assertTrue(replay(out,fresh)['verified'])
