import tempfile
from pathlib import Path
import unittest
from beyond_discovery.adapters.transition import TransitionAdapter,StringTransitionAdapter
from beyond_discovery.core.controller import RunConfig,run_rsi
from beyond_discovery.core.replay import replay
from beyond_discovery.core.archive import read_verified
from beyond_discovery.core.interaction import episode
from beyond_discovery.core.boundary import BoundaryKind
from beyond_discovery.runtime.research import resolve_runtime


class SynthesisRuntimeTests(unittest.TestCase):
    def test_complete_dual_role_invention_meta_gates_and_replay_in_both_mechanisms(self):
        for cls in (TransitionAdapter,StringTransitionAdapter):
            with self.subTest(domain=cls.domain_id),tempfile.TemporaryDirectory() as root:
                out=Path(root)/"run"
                result=run_rsi(cls(),out,RunConfig(generations=1,candidate_limit=2))
                self.assertEqual(result["status"],"COMPLETE")
                records=read_verified(out)
                barrier=next(d for k,d in records if k=="candidate_barrier")
                self.assertTrue(barrier["scientist"])
                self.assertTrue(barrier["challenger"])
                self.assertEqual({d["role"] for k,d in records if k=="gate"},{"scientist","challenger"})
                self.assertTrue(any(d["meta_gate"] is not None for k,d in records if k=="gate"))
                self.assertTrue(any(d["meta_source_diff"] for k,d in records if k=="gate"))
                self.assertTrue(replay(out,cls())["verified"])

    def test_local_search_produces_actual_source_from_audited_train_and_retains_proof_scope(self):
        adapter=TransitionAdapter(); runtime=resolve_runtime(adapter)
        c,s=runtime.initial_pair(RunConfig())
        es=tuple(episode(adapter,t,s,c,0,runtime=runtime) for t in adapter.sample("train",7))
        self.assertTrue(any(e.boundary_certificates[0].kind==BoundaryKind.SPACE_REFUTED for e in es))
        from beyond_discovery.core.diagnosis import diagnose
        candidates=runtime.candidates(s,diagnose(s.meta,es),es,runtime.context(adapter),2,"full")
        self.assertTrue(candidates)
        self.assertNotEqual(candidates[0][0].artifacts[0].source,s.artifacts[0].source)
        private=episode(adapter,adapter.sample("dev",0)[0],s,c,0,runtime=runtime)
        with self.assertRaises(PermissionError): runtime.candidates(s,None,(private,),runtime.context(adapter),1,"full")
