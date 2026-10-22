from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from beyond_discovery import RunConfig, run_rsi
from beyond_discovery.adapters.causal import CausalDiscoveryAdapter
from beyond_discovery.adapters.vector import VectorProgramAdapter
from beyond_discovery.adapters.kernel import CapabilityUnsupported, KernelBenchAdapter
from beyond_discovery.core.archive import Archive, read_verified
from beyond_discovery.core.contracts import (ChallengeProposal, Decision, MetaPolicy, ModelState,
                                           Status, digest, plain)
from beyond_discovery.core.controller import initial_pair
from beyond_discovery.core.diagnosis import diagnose
from beyond_discovery.core.evolution import candidates
from beyond_discovery.core.gate import GateConfig, paired_gate
from beyond_discovery.core.interaction import episode
from beyond_discovery.core.replay import replay, snapshot_from
from beyond_discovery.core.training import train_role, training_batch
from beyond_discovery.runtime.llm import Completion, JSONPolicyProposer
from beyond_discovery.runtime.policy_runtime import PolicyError, compile_program, predict, validate_meta


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.adapter = CausalDiscoveryAdapter()
        self.c, self.s = initial_pair(RunConfig())

    def test_private_test_and_forged_split_are_rejected(self):
        with self.assertRaises(PermissionError): self.adapter.sample("test", 1)
        task = self.adapter.sample("train", 1)[0]
        with self.assertRaises(ValueError): self.adapter.begin(replace(task, split="dev"), 1)
        with self.assertRaises(PermissionError): self.adapter.evaluate(episode(self.adapter, task, self.s, self.c, 0).new_claim, task, "dev")

    def test_world_tracks_and_independent_certification(self):
        for track in ("A", "B"):
            adapter = CausalDiscoveryAdapter()
            p = ChallengeProposal(adapter.domain_id, ("xor", ("x", 0), ("add", ("x", 1), ("x", 2))), track)
            cert = adapter.compile_challenge(p)
            self.assertEqual(cert.status, Status.VERIFIED)
            self.assertEqual(cert.task.split, "train")
            self.assertEqual(adapter.compile_challenge(p).status, Status.INVALID)
        bad = ChallengeProposal(self.adapter.domain_id, ("const", 1), "B")
        self.assertEqual(self.adapter.compile_challenge(bad).status, Status.INVALID)

    def test_certification_never_reads_dev_or_test_oracles(self):
        original = self.adapter._oracle
        private = {id(p) for key, p in self.adapter._worlds.items() if self.adapter._refs[key].split != "train"}
        def guarded(program, values):
            self.assertNotIn(id(program), private)
            return original(program, values)
        self.adapter._oracle = guarded
        p = ChallengeProposal(self.adapter.domain_id, ("add", ("x", 0), ("add", ("x", 1), ("x", 2))), "B")
        self.assertEqual(self.adapter.compile_challenge(p).status, Status.VERIFIED)

    def test_audit_rejects_fabricated_observation_score_and_behavior(self):
        e = episode(self.adapter, self.adapter.sample("train", 2)[0], self.s, self.c, 0)
        self.assertTrue(self.adapter.audit(e))
        self.assertFalse(self.adapter.audit(replace(e, score=replace(e.score, value=0.123))))
        self.assertFalse(self.adapter.audit(replace(e, behavior_hash="fake")))
        fake_obs = replace(e.observations[0], outcome=999)
        self.assertFalse(self.adapter.audit(replace(e, observations=(fake_obs,) + e.observations[1:])))

    def test_train_only_candidate_and_training_inputs(self):
        e = episode(self.adapter, self.adapter.sample("dev", 2)[0], self.s, self.c, 0)
        with self.assertRaises(PermissionError): e.agent_view()
        with self.assertRaises(PermissionError): diagnose(self.s.meta, (e,))
        with self.assertRaises(PermissionError): training_batch(self.s, (e,))
        with self.assertRaises(PermissionError): candidates(self.s, None, (e,), self.adapter.capabilities())

    def test_no_text_only_fixation_trigger(self):
        d = diagnose(self.s.meta, ())
        self.assertEqual(d.action, "insufficient_evidence")
        self.assertEqual(candidates(self.s, d, (), self.adapter.capabilities()), ())

    def test_session_budget_and_invalid_action(self):
        task = self.adapter.sample("train", 0)[0]
        session = self.adapter.begin(task, 1)
        self.adapter.step(session, (0,) * 6)
        with self.assertRaises(ValueError): self.adapter.step(session, (0,) * 6)
        with self.assertRaises(ValueError): self.adapter.step(self.adapter.begin(task, 1), (3,) * 6)

    def test_vm_has_no_host_language_escape(self):
        for program in (("__import__", "os"), ("x", -1), ("const", 999), ("open", "/tmp/private")):
            with self.assertRaises(PolicyError): compile_program(program, 6)
        program = ("x", 0)
        for _ in range(8): program = ("add", program, ("const", 1))
        with self.assertRaises(PolicyError): compile_program(program, 6)
        self.assertEqual(predict(("mul", ("x", 0), ("x", 1)), (2, 3)), 6)

    def test_meta_validation(self):
        with self.assertRaises(PolicyError): validate_meta(MetaPolicy(operators=("CHANGE_VERIFIER",)))

    def test_kernel_does_not_silently_fallback(self):
        with self.assertRaises(CapabilityUnsupported): KernelBenchAdapter()

    def test_role_isolated_training_produces_candidate_only(self):
        e = episode(self.adapter, self.adapter.sample("train", 0)[0], self.s, self.c, 0)
        class Backend:
            def fit(self, state, batch, directory):
                path = directory / "delta.bin"
                path.write_bytes(b"test-fixture-checkpoint")
                return replace(state, delta_checkpoint=str(path), training_data_hash=digest(batch))
        with tempfile.TemporaryDirectory() as root:
            child = train_role(self.s, (e,), Backend(), root)
            self.assertEqual(child.parent_hash, self.s.content_hash)
            self.assertIn("scientist", child.model.delta_checkpoint)
            self.assertIsNone(self.s.model.delta_checkpoint)


class GateTests(unittest.TestCase):
    def rows(self, value, clusters=5):
        return tuple((str(i), value, True, 100) for i in range(clusters))

    def test_three_states(self):
        old = self.rows(0.4)
        self.assertEqual(paired_gate(old, self.rows(0.6), GateConfig()).decision, Decision.ACCEPT)
        self.assertEqual(paired_gate(old, self.rows(0.4), GateConfig()).decision, Decision.INCONCLUSIVE)
        self.assertEqual(paired_gate(old, self.rows(0.1), GateConfig()).decision, Decision.REJECT)

    def test_invalid_nan_and_budget_do_not_become_weighted_success(self):
        for row in (("0", 1.0, False, 100), ("0", float("nan"), True, 100), ("0", 1.0, True, 101)):
            new = (row,) + self.rows(0.9)[1:]
            self.assertEqual(paired_gate(self.rows(0.4), new, GateConfig()).decision, Decision.REJECT)

    def test_cluster_unit_prevents_pseudoreplication(self):
        old = (("one-world", 0.1, True, 1),) * 2304
        new = (("one-world", 0.8, True, 1),) * 2304
        result = paired_gate(old, new, GateConfig())
        self.assertEqual(result.clusters, 1)
        self.assertEqual(result.decision, Decision.INCONCLUSIVE)

    def test_exact_task_pairing(self):
        old = (("cluster", 0.1, True, 1, "a"),)
        new = (("cluster", 0.9, True, 1, "b"),)
        self.assertEqual(paired_gate(old, new, GateConfig()).decision, Decision.REJECT)


class EndToEndTests(unittest.TestCase):
    def test_both_domains_evolve_replay_and_use_frozen_opponents(self):
        for cls in (CausalDiscoveryAdapter, VectorProgramAdapter):
            with self.subTest(domain=cls.domain_id), tempfile.TemporaryDirectory() as root:
                out = Path(root) / "run"
                result = run_rsi(cls(), out)
                self.assertEqual(result["status"], "COMPLETE")
                self.assertGreaterEqual(result["test_score"], 0)
                self.assertGreater(result["usage"]["verification_outputs"], 0)
                records = read_verified(out)
                barriers = {x["generation"]: x for k, x in records if k == "candidate_barrier"}
                for kind, data in records:
                    if kind != "episode" or not data["label"].startswith("dev-"): continue
                    generation = data["evidence"]["provenance"]["generation"]
                    c_parent, s_parent = barriers[generation]["parents"]
                    if data["label"].startswith("dev-challenger"):
                        self.assertEqual(data["scientist"], s_parent)
                    elif data["label"].startswith("dev-scientist"):
                        self.assertEqual(data["challenger"], c_parent)
                test_records = [x for k, x in records if k == "episode" and x["evidence"]["task"]["split"] == "test"]
                self.assertEqual(len(test_records), len({x["evidence"]["task"]["task_id"] for x in test_records}))
                self.assertTrue(replay(out, cls())["verified"])
                with self.assertRaises(FileExistsError): run_rsi(cls(), out)

    def test_inconclusive_retains_parent_and_unproven_meta_is_reverted(self):
        with tempfile.TemporaryDirectory() as root:
            c, s = initial_pair(RunConfig())
            result = run_rsi(CausalDiscoveryAdapter(), Path(root) / "run",
                             RunConfig(generations=1, gate=GateConfig(min_clusters=99)))
            self.assertEqual(result["challenger"], plain(c))
            self.assertEqual(result["scientist"], plain(s))
        with tempfile.TemporaryDirectory() as root:
            out = Path(root) / "run"
            run_rsi(CausalDiscoveryAdapter(), out)
            found = False
            for kind, data in read_verified(out):
                if kind == "gate" and data["meta_gate"] and data["meta_gate"]["decision"] != "ACCEPT":
                    if data["role"] == "scientist":
                        self.assertEqual(data["candidate"]["meta"], plain(MetaPolicy()))
                        found = True
            self.assertTrue(found)

    def test_budget_stop_freezes_without_exceeding_budget(self):
        with tempfile.TemporaryDirectory() as root:
            out = Path(root) / "run"
            result = run_rsi(CausalDiscoveryAdapter(), out, RunConfig(max_experiments=1))
            self.assertEqual(result["status"], "BUDGET_EXHAUSTED")
            self.assertEqual(result["usage"]["experiments"], 0)
            self.assertTrue((out / "frozen.json").exists())

    def test_archive_detects_tampering(self):
        with tempfile.TemporaryDirectory() as root:
            archive = Archive(Path(root) / "run")
            key = archive.append("evidence", {"truth": 1})
            (archive.objects / (key + ".json")).write_text('{"truth": 2}')
            with self.assertRaises(ValueError): read_verified(archive.path)

    def test_fixed_baseline(self):
        with tempfile.TemporaryDirectory() as root:
            result = run_rsi(CausalDiscoveryAdapter(), Path(root) / "run", RunConfig(generations=1, mode="fixed"))
            self.assertEqual(result["scientist"]["generation"], 0)


class ModelBoundaryTests(unittest.TestCase):
    def test_json_model_is_train_only_and_cannot_change_outer_rules(self):
        adapter = CausalDiscoveryAdapter()
        c, s = initial_pair(RunConfig(base_model="fixture"))
        e = episode(adapter, adapter.sample("train", 0)[0], s, c, 0)
        def complete(request, limit):
            data = json.loads(request)
            self.assertEqual(data["train_evidence"][0]["task"]["split"], "train")
            return Completion('[{"verifier": "mine"}]', "fixture", 100, 20)
        proposer = JSONPolicyProposer(complete, "fixture")
        with self.assertRaises(PolicyError): proposer.propose(s, diagnose(s.meta, (e,)), (e,), adapter.capabilities(), 1, "full")
        self.assertEqual(len(proposer.calls), 1)

    def test_truncation_records_actual_usage(self):
        proposer = JSONPolicyProposer(lambda *_: Completion("[", "fixture", 50, 10, True), "fixture")
        adapter = CausalDiscoveryAdapter()
        c, s = initial_pair(RunConfig(base_model="fixture"))
        with self.assertRaises(PolicyError): proposer.propose(s, None, (), adapter.capabilities(), 1, "full")
        self.assertEqual(proposer.total_tokens, 60)

    def test_injected_model_proposals_run_through_the_same_gates(self):
        response = json.dumps([{"representation": ["add", "xor"],
                               "experiment": {"strategy": "disagreement", "action_budget": 8},
                               "meta": {"operators": ["REFRAME", "RECOMBINE", "RELAX_ASSUMPTION", "ANALOGIZE", "INVENT_EXPERIMENT"],
                                        "required_conflicts": 2, "required_stagnation": 2, "search_order": "forward"}}])
        proposer = JSONPolicyProposer(lambda *_: Completion(response, "fixture", 100, 100), "fixture")
        with tempfile.TemporaryDirectory() as root:
            out = Path(root) / "run"
            result = run_rsi(CausalDiscoveryAdapter(), out, RunConfig(generations=1, base_model="fixture"), proposer)
            self.assertEqual(result["usage"]["model_tokens"], 400)
            self.assertEqual(len([x for k, x in read_verified(out) if k == "model_call"]), 2)
            self.assertTrue(replay(out, CausalDiscoveryAdapter())["verified"])


if __name__ == "__main__":
    unittest.main()
