import json,tempfile,unittest
from dataclasses import replace
from apps_fixtures import fixture_store,ProtocolJudge,ScriptedModel
from beyond_discovery.benchmarks.apps.adapter import APPSAdapter
from beyond_discovery.benchmarks.apps.judge import JudgeResult
from beyond_discovery.benchmarks.apps.challenges import RestrictedOracle,solution_boundary
from beyond_discovery.runtime.coding import CodingResearchRuntime,CodingLimits,provenance
from beyond_discovery.core.contracts import ChallengeProposal,Status,plain
from beyond_discovery.core.controller import RunConfig
from beyond_discovery.core.protocol import ResearchArtifact,Schema
from beyond_discovery.core.control import control_artifact

class OracleProtocolJudge(ProtocolJudge):
    """No Python execution: explicit output-protocol fixture for certificate tests."""
    def grade(self,code,cases,fn_name=None):
        if code=='REVIEWED_VALIDATOR_FIXTURE':return JudgeResult((True,), 'COMPLETE',1,1,0,(True,))
        if code.startswith('REF_FIXTURE_'):
            outs=tuple(sum(c['input']) for c in cases)
            return JudgeResult(tuple(o==c['output'] for o,c in zip(outs,cases)),'COMPLETE',len(cases),1,0,outs)
        return super().grade(code,cases,fn_name)

class ChallengeTests(unittest.TestCase):
    def setup_case(self,root):
        store=fixture_store(root);task=store.sample('train',0)[0];limits=CodingLimits(2,256,64)
        oracle=RestrictedOracle({'format':'apps-reviewed-oracles-v1','tasks':{task.task_id:{'validator':'REVIEWED_VALIDATOR_FIXTURE','references':['REF_FIXTURE_1','REF_FIXTURE_2'],'review_id':'EXPLICIT_TEST_ONLY'}}})
        a=APPSAdapter(store,OracleProtocolJudge(),limits.tokens,testing=True,oracle=oracle)
        r=CodingResearchRuntime({'scientist':ScriptedModel(['def add(a,b): return 0']*2),'challenger':ScriptedModel([])},limits,testing=True)
        c,s=r.initial_pair(RunConfig(base_model='TEST_MODEL_FIXTURE'));a.bind_frozen_scientist(s.content_hash)
        e=r.episode(a,task,s,c,0,256)
        return a,r,c,s,e
    def proposal(self,e,data=None):
        data=data or {'parent_task_id':e.task.task_id,'old_artifact_hash':e.new_claim.program.content_hash,'input':[8,9]}
        return ChallengeProposal('apps',ResearchArtifact('world','apps-challenge-v1','1',json.dumps(data),Schema.of({'type':'object'}),Schema.of({'type':'object'}),provenance('challenger')),'A')
    def test_oracle_certifies_new_input_but_agent_answer_and_unbound_fail(self):
        with tempfile.TemporaryDirectory() as root:
            a,r,c,s,e=self.setup_case(root)
            cert=a.compile_challenge(self.proposal(e));self.assertEqual(cert.status,Status.VERIFIED)
            self.assertEqual(cert.task.source_cluster,e.task.source_cluster)
            self.assertEqual(a.public_task(cert.task)['public_cases'][0]['output'],17)
            bad=json.loads(self.proposal(e).program.source);bad['answer']=17
            self.assertEqual(a.compile_challenge(self.proposal(e,bad)).status,Status.INVALID)
            a.bind_frozen_scientist('another-parent')
            self.assertEqual(a.compile_challenge(self.proposal(e)).status,Status.INVALID)
            a.oracle=None
            self.assertEqual(a.compile_challenge(self.proposal(e)).status,Status.PENDING)
    def test_single_solution_certificate_is_verified_and_tampering_fails(self):
        with tempfile.TemporaryDirectory() as root:
            a,r,c,s,e=self.setup_case(root)
            self.assertEqual(e.boundary_certificates[0].kind,'SOLUTION_REFUTED');self.assertTrue(a.audit(e))
            cert=replace(e.boundary_certificates[0],program_hash='forged')
            self.assertFalse(a.audit(replace(e,boundary_certificates=(cert,))))
            from beyond_discovery.core.diagnosis import diagnose
            # It must not turn a program WA into a representation refutation.
            d=plain(diagnose(s.meta,(e,)))
            self.assertNotIn('SPACE_REFUTED',json.dumps(d))
    def test_model_generated_meta_has_behavior_witness_and_fixed_adaptation(self):
        with tempfile.TemporaryDirectory() as root:
            a,r,c,s,e=self.setup_case(root)
            direct={'score':['lit',0],'functions':{},'entry':['let','v',['effect','code_generate',['const',False]],['effect','code_submit',['var','v']]]}
            proposal=json.dumps([{'instructions':s.representation[0],'control':direct}])
            r.clients['scientist']=ScriptedModel([proposal]);r.method='evolved-meta-rsi'
            attempts=r.candidates(s,None,(e,),{},1,'full')
            self.assertEqual(len(attempts),1);child,patch=attempts[0]
            self.assertTrue(patch['meta_witnesses']);self.assertNotEqual(child.meta,s.meta)
            adapted,_=r.adapt(s,None,{},child.meta)
            self.assertEqual(adapted.model,s.model);self.assertEqual(adapted.representation,s.representation)
            self.assertEqual(adapted.experiment,s.experiment)
            r.clients['scientist']=ScriptedModel([json.dumps([{'instructions':'unchanged','control':json.loads(s.meta.program.source)}])])
            self.assertFalse(r.candidates(s,None,(e,),{},1,'full'))
