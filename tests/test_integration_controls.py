from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from beyond_discovery.adapters.transition import TransitionAdapter,StringTransitionAdapter
from beyond_discovery.adapters.registry import external_adapter
from beyond_discovery.core.conformance import check_conformance
from beyond_discovery.core.controller import RunConfig, run_rsi
from beyond_discovery.core.archive import read_verified
from beyond_discovery.core.protocol import CapabilityUnsupported
from beyond_discovery.core.contracts import ChallengeProposal, Status
from beyond_discovery.core.diagnosis import diagnose
from beyond_discovery.core.interaction import episode
from beyond_discovery.runtime.research import resolve_runtime
from beyond_discovery.experiments.suite import METHODS, comparison_suite


class IntegrationControlTests(unittest.TestCase):
    def test_stateful_conformance_both_mechanisms(self):
        for cls in (TransitionAdapter,StringTransitionAdapter):
            adapter=cls()
            result=check_conformance(adapter,actions=adapter.conformance_actions())
            self.assertTrue(result.passed); self.assertEqual(len(result.checks),13)

    def test_generation_budget_stops_before_overspend(self):
        with tempfile.TemporaryDirectory() as root:
            result=run_rsi(StringTransitionAdapter(),Path(root)/'run',RunConfig(generations=1,max_search=80))
            self.assertLessEqual(result['usage']['search'],80)
            self.assertIn('budget',result['stop_reason'])

    def test_baseline_selection_and_ablations_change_actual_behavior(self):
        adapter=StringTransitionAdapter(); runtime=resolve_runtime(adapter)
        c,s=runtime.initial_pair(RunConfig(search_budget=64))
        training=tuple(episode(adapter,t,s,c,0,64,runtime=runtime) for t in adapter.sample('train',0))
        d=diagnose(s.meta,training); context=runtime.context(adapter)
        for mode in METHODS:
            runtime.validate_mode(mode); RunConfig(mode=mode).validate()
        for mode in ('fixed','fixed_challenger','random_challenger','hardness_first'):
            self.assertEqual(runtime.candidates(c,d,training,context,1,mode),())
        attempts=runtime.candidates(s,d,training,context,2,'no_restructuring')
        self.assertTrue(attempts)
        self.assertTrue(all(child.artifacts[0]==s.artifacts[0] for child,_ in attempts))
        attempts=runtime.candidates(s,d,training,context,2,'no_meta')
        self.assertTrue(all(child.meta==s.meta for child,_ in attempts))
        attempts=runtime.candidates(s,d,training,context,2,'no_falsification')
        self.assertTrue(all(not any(a.kind=='experiment' for a in child.artifacts) for child,_ in attempts))

    def test_malformed_world_invalid_and_external_missing_explicit(self):
        adapter=TransitionAdapter()
        self.assertEqual(adapter.compile_challenge(ChallengeProposal(adapter.domain_id,{'answers':[1]},'B')).status,Status.INVALID)
        with self.assertRaises(CapabilityUnsupported): external_adapter('missing-real-benchmark')

    def test_comparison_uses_separate_sealed_runs(self):
        with tempfile.TemporaryDirectory() as root:
            result=comparison_suite(StringTransitionAdapter,Path(root)/'suite',RunConfig(generations=1),('fixed','no_meta'))
            self.assertEqual(set(result['results']),{'fixed','no_meta'})
            for mode in result['results']:
                records=read_verified(Path(root)/'suite'/mode)
                frozen=next(i for i,(k,_) in enumerate(records) if k=='freeze')
                self.assertTrue(all(i>frozen for i,(k,d) in enumerate(records) if k=='episode' and d['evidence']['task']['split']=='test'))

    def test_vm_rejects_malformed_nodes_and_has_no_host_execution(self):
        from beyond_discovery.runtime.expression import evaluate, freeze
        from beyond_discovery.core.protocol import ProtocolError
        bad=(('lit',1,'ignored'),('ref',1),('lit',{}),('__import__',('lit','os')),(['lit'],1))
        for node in bad:
            with self.subTest(node=node),self.assertRaises(ProtocolError): evaluate(node,{})
        with self.assertRaises(ProtocolError): evaluate(('mul',('lit',1000000),('lit',1000000)),{})

    def test_weight_controls_require_real_backend_and_fixed_prompt(self):
        from beyond_discovery.runtime.model_research import ModelResearchRuntime
        from beyond_discovery.runtime.models import TokenLedger
        from beyond_discovery.runtime.llm import Completion
        from beyond_discovery.core.contracts import ModelState
        adapter=StringTransitionAdapter()
        class Client:
            model='fixture'
            def __init__(self): self.ledger=TokenLedger(1000); self.calls=[]; self.prompts=[]
            def prepare(self,state): self.state=state
            def __call__(self,prompt,limit):
                self.prompts.append(prompt)
                return Completion('[{"grammar":'+json.dumps(adapter.public_context()['grammar'])+'}]','fixture',0,0)
        clients={r:Client() for r in ('scientist','challenger')}
        with self.assertRaises(ValueError): ModelResearchRuntime(adapter.public_context(),clients,None,'lora-only')
        runtime=ModelResearchRuntime(adapter.public_context(),clients,{'steps':1,'max_steps':2},'lora-only')
        c,s=runtime.initial_pair(RunConfig(base_model='fixture'))
        runtime.candidates(s,None,(),adapter.public_context(),1,'full')
        updated=replace(s,model=replace(s.model,delta_checkpoint='/test/adapter'))
        runtime.candidates(updated,None,(),adapter.public_context(),1,'full')
        self.assertEqual(clients['scientist'].prompts[0],clients['scientist'].prompts[1])

    def test_spawned_sandbox_enforces_language_resources_and_cleanup(self):
        from beyond_discovery.runtime.sandbox import ResearchSandbox
        from beyond_discovery.runtime.expression import evaluate
        from beyond_discovery.core.protocol import ProtocolError
        import os
        with ResearchSandbox() as sandbox:
            self.assertNotEqual(sandbox.process.pid,os.getpid())
            self.assertEqual(evaluate(('add',('lit',2),('lit',3)),{}),5)
            with self.assertRaises(ProtocolError): evaluate(('__import__',('lit','os')),{})
            with self.assertRaises(ProtocolError): evaluate(('lit','x'*65537),{})
            with self.assertRaises(ProtocolError): evaluate(('lit',1),{},fuel=0)
            sandbox.process.terminate(); sandbox.process.join(1)
            with self.assertRaises(ProtocolError): evaluate(('lit',1),{})
        self.assertFalse(sandbox.process.is_alive())
