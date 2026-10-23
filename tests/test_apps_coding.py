import json,tempfile,unittest
from dataclasses import replace
from apps_fixtures import fixture_store,ProtocolJudge,ScriptedModel
from beyond_discovery.benchmarks.apps.adapter import APPSAdapter
from beyond_discovery.runtime.coding import CodingResearchRuntime,CodingLimits,python_source
from beyond_discovery.core.controller import RunConfig
from beyond_discovery.core.contracts import plain
from beyond_discovery.core.protocol import CapabilityUnsupported,ProtocolError

class CodingTests(unittest.TestCase):
    def test_model_context_bound_can_use_full_reported_gateway_window(self):
        limits=CodingLimits(2,32768,32768)
        self.assertEqual(limits.tokens,262144)
        with self.assertRaises(ValueError):CodingLimits(2,32768,32769)

    def setup_runtime(self,root,texts,method='self-repair'):
        limits=CodingLimits(2,256,64)
        adapter=APPSAdapter(fixture_store(root),ProtocolJudge(),limits.tokens,testing=True)
        client=ScriptedModel(texts)
        runtime=CodingResearchRuntime({'scientist':client,'challenger':ScriptedModel([])},limits,method,testing=True)
        c,s=runtime.initial_pair(RunConfig(base_model=client.model))
        return adapter,runtime,c,s,client
    def test_code_revisions_public_feedback_cost_and_offline_replay(self):
        with tempfile.TemporaryDirectory() as root:
            a,r,c,s,client=self.setup_runtime(root,['def add(a,b): return 0','def add(a,b): return a+b'])
            task=a.sample('train',0)[0];e=r.episode(a,task,s,c,0,256)
            self.assertEqual(e.score.value,1);self.assertNotEqual(e.old_claim,e.new_claim)
            self.assertEqual(e.usage.model_tokens,24);self.assertEqual(e.usage.experiments,8)
            self.assertEqual(a.public_context()["episode_model_tokens"],CodingLimits(2,256,64).tokens)
            self.assertEqual(e.boundary_certificates[0].kind,'SOLUTION_REFUTED')
            self.assertIn('no claim',e.boundary_certificates[0].scope)
            prompts=json.dumps(client.calls)
            for secret in ('PRIVATE_REFERENCE_SENTINEL','private_cases','"score"','[2, 3]'):
                self.assertNotIn(secret,prompts);self.assertNotIn(secret,json.dumps(e.agent_view()))
            replay=CodingResearchRuntime(limits=r.limits,replay_only=True)
            actual=replay.replay_episode(plain(e),a,task,s,c,0,256)
            self.assertEqual(plain(e),plain(actual));self.assertEqual(len(client.calls),2)
            broken=plain(e);broken['execution_records'][0]['records'][0]['request']['prompt']='tampered'
            with self.assertRaises(ProtocolError):replay.replay_episode(broken,a,task,s,c,0,256)
    def test_dev_cannot_generate_candidates_or_expose_agent_view(self):
        with tempfile.TemporaryDirectory() as root:
            a,r,c,s,_=self.setup_runtime(root,['def add(a,b): return a+b']*2)
            e=r.episode(a,a.sample('dev',0)[0],s,c,0,256)
            with self.assertRaises(PermissionError):e.agent_view()
            with self.assertRaises(PermissionError):r.candidates(s,None,(e,),{},1,'full')
    def test_invalid_code_and_unavailable_real_model_fail_explicitly(self):
        with self.assertRaises(CapabilityUnsupported):CodingResearchRuntime().validate_mode('full')
        with self.assertRaises(ProtocolError):python_source('this is invalid Python :')
        with tempfile.TemporaryDirectory() as root:
            a,r,c,s,_=self.setup_runtime(root,['invalid Python :'])
            with self.assertRaises(ProtocolError):r.episode(a,a.sample('train',0)[0],s,c,0,256)
            self.assertEqual(r.total_model_tokens,12)
            self.assertFalse(a._sessions)
    def test_metapolicy_cannot_exceed_model_call_cap(self):
        from beyond_discovery.core.control import control_artifact
        from beyond_discovery.runtime.coding import provenance
        with tempfile.TemporaryDirectory() as root:
            a,r,c,s,client=self.setup_runtime(root,['def add(a,b): return a+b']*3)
            program={'score':['lit',0],'functions':{},'entry':['fold',['const',[0,1,2]],'acc','item',['const',None],['effect','code_generate',['const',False]]]}
            s=replace(s,meta=replace(s.meta,program=control_artifact(program,provenance())))
            with self.assertRaises(ProtocolError):r.episode(a,a.sample('train',0)[0],s,c,0,256)
            self.assertEqual(len(client.calls),2);self.assertEqual(r.total_model_tokens,24)
