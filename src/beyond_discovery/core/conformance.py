"""Reusable behavioral checks. Success asserts exercised protocol paths only.

The caller supplies executable probes, not trusted answers. No test set is
unsealed; verification recomputes actual executions using the adapter's audit.
"""
from dataclasses import dataclass, replace
from .controller import RunConfig
from .interaction import episode
from .protocol import Action, ProtocolError, negotiate
from ..runtime.research import ArtifactResearchRuntime, resolve_runtime


@dataclass(frozen=True)
class ConformanceReport:
    passed: bool
    domain_id: str
    checks: tuple


def _must_reject(operation, name):
    try:
        operation()
    except (ValueError, PermissionError):
        return
    raise ProtocolError("conformance failed: " + name)


def check_conformance(adapter, experiment=None, actions=(),runtime=None,config=None):
    checks = []
    caps = adapter.describe()
    runtime = runtime or (ArtifactResearchRuntime(experiment) if experiment is not None else resolve_runtime(adapter))
    budget = min(8, caps.max_action_budget)
    negotiate(caps, runtime.languages, budget)
    train = adapter.sample("train", 17)
    dev = adapter.sample("dev", 18)
    if not train or not dev or train != adapter.sample("train", 17):
        raise ProtocolError("conformance failed: stable nonempty partitions required")
    if {t.task_id for t in train} & {t.task_id for t in dev}:
        raise ProtocolError("conformance failed: overlapping partitions")
    if any(t.split != split or t.domain_id != adapter.domain_id for split, group in (("train", train), ("dev", dev)) for t in group):
        raise ProtocolError("conformance failed: task provenance mismatch")
    checks.append("deterministic_partition_sampling")
    _must_reject(lambda: adapter.sample("test", 0), "sealed-test access")
    _must_reject(lambda: adapter.begin(replace(train[0], split="dev"), budget), "forged task")
    checks += ["sealed_test_rejection", "forged_task_rejection"]
    if actions:
        first = adapter.begin(train[0], budget)
        second = adapter.begin(train[0], budget)
        try:
            initial = adapter.initial_observation(second)
            for action in actions:
                caps.action_schema.validate(action)
                observation = adapter.step(first, action)
                caps.observation_schema.validate(observation.outcome)
            if adapter.initial_observation(second) != initial:
                raise ProtocolError("conformance failed: state crossed isolated sessions")
            _must_reject(lambda: adapter.step(second, Action.create("__invalid__", {})), "invalid action")
        finally:
            adapter.close(first)
            adapter.close(second)
        checks += ["typed_action_observation", "session_isolation", "invalid_action_rejection"]
        session = adapter.begin(train[0], 1)
        try:
            adapter.step(session, actions[0])
            _must_reject(lambda: adapter.step(session, actions[0]), "action budget")
        finally:
            adapter.close(session)
        _must_reject(lambda: adapter.step(session, actions[0]), "closed session")
        checks += ["budget_rejection", "closed_session_rejection"]
    c, s = runtime.initial_pair(config or RunConfig(action_budget=budget))
    evidence = episode(adapter, train[0], s, c, 0, runtime=runtime)
    if not adapter.audit(evidence): raise ProtocolError("conformance failed: valid execution audit")
    try:
        forged_accepted = adapter.audit(replace(evidence, behavior_hash="tampered"))
    except (ValueError, PermissionError):
        forged_accepted = False
    if forged_accepted: raise ProtocolError("conformance failed: evidence tampering accepted")
    if adapter.audit(replace(evidence, score=replace(evidence.score, value=-123.0))):
        raise ProtocolError("conformance failed: fabricated score accepted")
    _must_reject(lambda: adapter.evaluate(evidence.new_claim, train[0], "dev"), "evaluation split mismatch")
    checks += ["execution_and_independent_audit", "evidence_tamper_rejection", "score_tamper_rejection", "evaluation_split_rejection"]
    # This intentionally checks train-only admission using a real dev execution.
    private = episode(adapter, dev[0], s, c, 0, runtime=runtime)
    _must_reject(private.agent_view, "dev evidence generation access")
    checks.append("dev_visibility_rejection")
    return ConformanceReport(True, adapter.domain_id, tuple(checks))
