"""Behavioral M1 acceptance tests, including a genuinely stateful wire protocol."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from beyond_discovery.core.contracts import Provenance, Usage, plain
from beyond_discovery.core.protocol import (
    Action, CapabilityUnsupported, JSONValue, ProtocolError, ResearchArtifact,
    Schema, negotiate,
)
from beyond_discovery.core.conformance import check_conformance
from beyond_discovery.core.interaction import episode
from beyond_discovery.core.controller import RunConfig, run_rsi
from beyond_discovery.core.replay import replay
from beyond_discovery.core.archive import read_verified
from beyond_discovery.adapters.document import DocumentAdapter
from beyond_discovery.adapters.causal import CausalDiscoveryAdapter
from beyond_discovery.runtime.research import ArtifactResearchRuntime


def artifact(domain="document-edit", split="train", instructions=None, kind="experiment"):
    return ResearchArtifact(kind=kind, language="research-workflow-v1", version="1",
                            source=json.dumps(instructions or [
                                {"call": {"operation": "append", "arguments": {"text": "ab"}}},
                                {"call": {"operation": "append", "arguments": {"text": {"$ref": "last.text"}}}},
                                {"call": {"operation": "finish", "arguments": {}}},
                                {"return": {"$ref": "last"}},
                            ]), input_schema=Schema.of({"type": "object"}),
                            output_schema=Schema.of({"type": "object"}),
                            provenance=Provenance(domain, split, "authored-protocol-test", 0, "test-author", 1, "document-reference-v1"))


class DomainProtocolTests(unittest.TestCase):
    def test_schema_checks_structure_types_unknown_fields_and_nonfinite_numbers(self):
        schema = Schema.of({"type": "object", "properties": {"x": {"type": "integer"}},
                            "required": ["x"], "additionalProperties": False})
        schema.validate({"x": 2})
        for value in ({"x": True}, {"x": "2"}, {}, {"x": 2, "secret": 1}, {"x": float("nan")}):
            with self.assertRaises(ProtocolError): schema.validate(value)
        with self.assertRaises(ProtocolError): Schema.of({"type": "object", "unimplementedConstraint": True})

    def test_payloads_are_immutable_canonical_values(self):
        data = {"nested": ["a", {"b": 2}]}
        value = JSONValue.of(data)
        data["nested"].append("mutated")
        copy = value.value
        copy["nested"].append("also mutated")
        self.assertEqual(value.value, {"nested": ["a", {"b": 2}]})
        self.assertEqual(value, JSONValue.of({"nested": ["a", {"b": 2}]}))

    def test_negotiation_fails_before_execution_for_missing_languages_features_or_budget(self):
        caps = DocumentAdapter().describe()
        self.assertEqual(negotiate(caps, ("research-workflow-v1",), 4).language, "research-workflow-v1")
        for languages, budget, worlds in ((("python-unrestricted",), 4, ()), (("research-workflow-v1",), 999, ()),
                                           (("research-workflow-v1",), 4, ("A", "B"))):
            with self.assertRaises(CapabilityUnsupported): negotiate(caps, languages, budget, worlds)
        with self.assertRaises(CapabilityUnsupported): negotiate(replace(caps, protocol_version="unknown"), ("research-workflow-v1",), 4)
        with self.assertRaises(CapabilityUnsupported): negotiate(replace(caps, features=("audit", "sealed-test")),
                                                                 ("research-workflow-v1",), 4, required_features=("stateful",))

    def test_stateful_actions_observations_and_executable_artifact(self):
        adapter = DocumentAdapter()
        runtime = ArtifactResearchRuntime(artifact())
        c, s = runtime.initial_pair(RunConfig(action_budget=4))
        evidence = episode(adapter, adapter.sample("train", 0)[0], s, c, 0, runtime=runtime)
        self.assertEqual([o.outcome.value["text"] for o in evidence.observations], ["ab", "abab", "abab"])
        self.assertEqual([o.sequence for o in evidence.observations], [1, 2, 3])
        self.assertTrue(evidence.observations[-1].terminal)
        self.assertEqual(evidence.new_claim.program.kind, "result")
        self.assertTrue(adapter.audit(evidence))
        self.assertEqual(evidence.score.value, 1.0)

    def test_invalid_action_is_rejected_before_state_mutation_and_terminal_is_enforced(self):
        adapter = DocumentAdapter()
        session = adapter.begin(adapter.sample("train", 0)[0], 4)
        with self.assertRaises(ProtocolError): adapter.step(session, Action.create("append", {"text": 5}))
        self.assertEqual(adapter.step(session, Action.create("read", {})).outcome.value["revision"], 0)
        adapter.step(session, Action.create("finish", {}))
        with self.assertRaises(ProtocolError): adapter.step(session, Action.create("read", {}))

    def test_session_ownership_split_and_budget_are_enforced(self):
        adapter = DocumentAdapter()
        task = adapter.sample("train", 0)[0]
        session = adapter.begin(task, 1)
        adapter.step(session, Action.create("read", {}))
        with self.assertRaises(ProtocolError): adapter.step(session, Action.create("read", {}))
        with self.assertRaises(ProtocolError): DocumentAdapter().step(session, Action.create("read", {}))
        with self.assertRaises(PermissionError): adapter.sample("test", 0)
        with self.assertRaises(ProtocolError): adapter.begin(replace(task, split="dev"), 1)
        finite = CausalDiscoveryAdapter()
        finite_session = finite.begin(finite.sample("train", 0)[0], 1)
        with self.assertRaises(ValueError): CausalDiscoveryAdapter().step(finite_session, (0,) * 6)
        finite.close(finite_session)
        with self.assertRaises(ValueError): finite.step(finite_session, (0,) * 6)

    def test_artifact_tampering_wrong_domain_and_wrong_schema_fail(self):
        adapter = DocumentAdapter()
        for art in (artifact(domain="other"), replace(artifact(), language="python"),
                    replace(artifact(), output_schema=Schema.of({"type": "integer"}))):
            runtime = ArtifactResearchRuntime(art)
            c, s = runtime.initial_pair(RunConfig(action_budget=4))
            with self.assertRaises((ProtocolError, CapabilityUnsupported)):
                episode(adapter, adapter.sample("train", 0)[0], s, c, 0, runtime=runtime)

    def test_no_scoring_authority_in_an_artifact(self):
        adapter = DocumentAdapter()
        bad = artifact(instructions=[{"return": {"text": "wrong", "revision": 0, "finished": True}}])
        runtime = ArtifactResearchRuntime(bad)
        c, s = runtime.initial_pair(RunConfig(action_budget=4))
        evidence = episode(adapter, adapter.sample("train", 0)[0], s, c, 0, runtime=runtime)
        self.assertEqual(evidence.score.value, 0.0)
        self.assertTrue(adapter.audit(evidence))

    def test_artifact_vm_rejects_host_code_references_and_overlong_execution(self):
        adapter = DocumentAdapter()
        for instructions in ([{"exec": "open('/etc/passwd')"}], [{"return": {"$ref": "oracle.secret"}}],
                             [{"call": {"operation": "read", "arguments": {}}}] * 5):
            runtime = ArtifactResearchRuntime(artifact(instructions=instructions))
            c, s = runtime.initial_pair(RunConfig(action_budget=4))
            with self.assertRaises(ProtocolError):
                episode(adapter, adapter.sample("train", 0)[0], s, c, 0, runtime=runtime)

    def test_conformance_really_runs_audit_and_budget_failures(self):
        adapter = DocumentAdapter()
        report = check_conformance(adapter, artifact(), (Action.create("append", {"text": "ab"}),
                                                         Action.create("finish", {})))
        self.assertTrue(report.passed)
        self.assertIn("budget_rejection", report.checks)
        self.assertIn("evidence_tamper_rejection", report.checks)
        class BadAudit(DocumentAdapter):
            def audit(self, evidence): return True
        with self.assertRaises(ProtocolError): check_conformance(BadAudit(), artifact(), (Action.create("read", {}),))
        class IgnoresBudget(DocumentAdapter):
            def step(self, session, action):
                session.remaining += 1
                return super().step(session, action)
        with self.assertRaises(ProtocolError): check_conformance(IgnoresBudget(), artifact(), (Action.create("read", {}),))

    def test_resource_reservations_and_failure_cleanup(self):
        class UnderstatesResources(DocumentAdapter):
            def estimate_episode(self, *args): return Usage()
        runtime = ArtifactResearchRuntime(artifact())
        c, s = runtime.initial_pair(RunConfig(action_budget=4))
        with self.assertRaises(ProtocolError): episode(UnderstatesResources(), DocumentAdapter().sample("train", 0)[0], s, c, 0, runtime=runtime)
        adapter = DocumentAdapter()
        bad = ArtifactResearchRuntime(artifact(instructions=[{"call": {"operation": "__invalid__", "arguments": {}}}, {"return": {}}]))
        c, s = bad.initial_pair(RunConfig(action_budget=4))
        with self.assertRaises(ProtocolError): episode(adapter, adapter.sample("train", 0)[0], s, c, 0, runtime=bad)
        self.assertFalse(adapter._sessions)
        self.assertEqual(adapter.meter().experiments, 0)

    def test_out_of_order_observations_are_rejected_and_sessions_are_closed(self):
        class BadSequence(DocumentAdapter):
            def step(self, session, action):
                return replace(super().step(session, action), sequence=100)
        adapter = BadSequence()
        runtime = ArtifactResearchRuntime(artifact())
        c, s = runtime.initial_pair(RunConfig(action_budget=4))
        with self.assertRaises(ProtocolError): episode(adapter, adapter.sample("train", 0)[0], s, c, 0, runtime=runtime)
        self.assertFalse(adapter._sessions)

    def test_full_controller_detects_capability_changes_during_execution(self):
        class ChangesCapabilities(DocumentAdapter):
            def describe(self):
                caps = super().describe()
                return replace(caps, limitations=("changed",)) if self.meter().experiments else caps
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(ProtocolError):
                run_rsi(ChangesCapabilities(), Path(root) / "run", RunConfig(generations=1, mode="fixed", required_world_tracks=()),
                        runtime=ArtifactResearchRuntime(artifact()))

    def test_unsupported_generic_invention_is_never_silently_a_fixed_agent(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(CapabilityUnsupported):
                run_rsi(DocumentAdapter(), Path(root) / "run", RunConfig(required_world_tracks=()),
                        runtime=ArtifactResearchRuntime(artifact()))
            self.assertFalse((Path(root) / "run").exists())

    def test_only_train_evidence_and_public_context_reach_generation(self):
        class CheckingRuntime(ArtifactResearchRuntime):
            def candidates(self, parent, diagnosis, training, public_context, limit, mode):
                if any(e.task.split != "train" for e in training): raise AssertionError("private evidence leaked")
                if not isinstance(public_context, dict) or public_context.get("domain_id") != "document-edit":
                    raise AssertionError("generation received evaluator instead of public context")
                return super().candidates(parent, diagnosis, training, public_context, limit, mode)
        with tempfile.TemporaryDirectory() as root:
            run_rsi(DocumentAdapter(), Path(root) / "run", RunConfig(generations=1, mode="fixed", required_world_tracks=()),
                    runtime=CheckingRuntime(artifact()))

    def test_decision_lock_matches_original_spec(self):
        import hashlib
        root = Path(__file__).parents[1]
        lock = json.loads((root / "design/Beyond-Discovery-RSI-Decision-Lock-v1.0.json").read_text())
        self.assertTrue(lock["locked"])
        self.assertEqual(len(lock["choices"]), 24)
        self.assertEqual(hashlib.sha256((root / "design" / lock["source_spec"]).read_bytes()).hexdigest(), lock["spec_sha256"])

    def test_generic_controller_archive_replay_without_finite_keys(self):
        with tempfile.TemporaryDirectory() as root:
            out = Path(root) / "run"
            runtime = ArtifactResearchRuntime(artifact())
            cfg = RunConfig(generations=1, action_budget=4, mode="fixed", required_world_tracks=())
            result = run_rsi(DocumentAdapter(), out, cfg, runtime=runtime)
            self.assertEqual(result["status"], "COMPLETE")
            self.assertTrue(replay(out, DocumentAdapter())["verified"])
            self.assertTrue(result["limitations"])
            episodes = [data for kind, data in read_verified(out) if kind == "episode"]
            self.assertEqual(result["usage"]["experiments"], sum(len(data["evidence"]["observations"]) for data in episodes))
            self.assertGreater(result["usage"]["experiments"], 0)
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(CapabilityUnsupported):
                run_rsi(DocumentAdapter(), Path(root) / "unsupported", RunConfig(action_budget=4), runtime=runtime)
            self.assertFalse((Path(root) / "unsupported").exists())

    def test_existing_finite_protocol_conforms_without_exposing_finite_semantics_in_core(self):
        adapter = CausalDiscoveryAdapter()
        caps = adapter.describe()
        self.assertNotIn("input_space", plain(caps))
        self.assertEqual(negotiate(caps, ("finite-expression-v1",), 8, ("A", "B")).language, "finite-expression-v1")
        core = Path(__file__).parents[1] / "src" / "beyond_discovery" / "core"
        for file in core.glob("*.py"):
            source = file.read_text()
            for token in ('["input_space"]', '["width"]', '["expression_ops"]'):
                self.assertNotIn(token, source, str(file))


if __name__ == "__main__": unittest.main()
