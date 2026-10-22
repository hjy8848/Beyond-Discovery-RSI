"""Stateful document-editing protocol reference environment.

This is an M1 executable conformance environment, not an invention agent or an
independent scientific benchmark. The fixed authored workflow is supplied by
the caller. No answer supplied by that workflow is trusted by the scorer.
"""
from __future__ import annotations

from dataclasses import dataclass
from ..core.contracts import Observation, Score, TaskRef, Usage, digest
from ..core.protocol import (Action, CapabilityUnsupported, DomainCapabilities,
                             JSONValue, ProtocolError, Schema)
from ..runtime.research import execute_result


def _object(properties, required=None):
    return {"type": "object", "properties": properties,
            "required": list(properties) if required is None else required, "additionalProperties": False}


@dataclass
class _Session:
    task: TaskRef
    remaining: int
    text: str = ""
    revision: int = 0
    sequence: int = 0
    terminal: bool = False


class DocumentAdapter:
    domain_id = "document-edit"
    verifier_version = "document-reference-v1"

    def __init__(self):
        self._sessions = {}
        self._sealed = True
        self._usage = Usage()
        self._tasks = {split: TaskRef(digest((self.domain_id, split)), self.domain_id, split,
                                      "document-protocol-fixture", 0) for split in ("train", "dev", "test")}
        # Evaluator-owned specification, never furnished by a candidate.
        self._reference_actions = (Action.create("append", {"text": "ab"}),
                                   Action.create("append", {"text": "ab"}), Action.create("finish", {}))

    def describe(self):
        cases = []
        for op, arguments in (("append", _object({"text": {"type": "string", "maxLength": 4096}})),
                              ("read", _object({})), ("finish", _object({}))):
            cases.append(_object({"operation": {"type": "string", "enum": [op]}, "arguments": arguments}))
        observation = Schema.of(_object({"text": {"type": "string", "maxLength": 65536},
                                        "revision": {"type": "integer", "minimum": 0},
                                        "finished": {"type": "boolean"}}))
        return DomainCapabilities(self.domain_id, Schema.of({"oneOf": cases}), observation, observation,
                                  ("research-workflow-v1",), (), 32,
                                  features=("audit", "sealed-test", "stateful", "structured-observations"),
                                  score_definition="exact agreement with evaluator-owned edit specification",
                                  limitations=("stateful protocol reference; no world creation or invention backend",))

    def sample(self, split, seed):
        if split not in self._tasks: raise ProtocolError("unknown split")
        if split == "test" and self._sealed: raise PermissionError("test is sealed")
        return (self._tasks[split],)

    def unseal_for_final(self):
        if not self._sealed: raise PermissionError("test already unsealed")
        self._sealed = False

    def _check_task(self, task):
        if task != self._tasks.get(task.split): raise ProtocolError("unknown/forged task")
        if task.split == "test" and self._sealed: raise PermissionError("test is sealed")

    def begin(self, task, budget):
        self._check_task(task)
        if type(budget) is not int or not 1 <= budget <= self.describe().max_action_budget:
            raise ProtocolError("action budget invalid")
        session = _Session(task, budget)
        self._sessions[id(session)] = session
        return session

    def _session(self, session):
        if self._sessions.get(id(session)) is not session: raise ProtocolError("foreign or closed session")
        self._check_task(session.task)

    def initial_observation(self, session):
        self._session(session)
        return JSONValue.of({"text": session.text, "revision": session.revision, "finished": session.terminal})

    def step(self, session, action):
        self._session(session)
        if session.terminal: raise ProtocolError("action after termination")
        if session.remaining <= 0: raise ProtocolError("action budget exhausted")
        if not isinstance(action, Action): raise ProtocolError("typed action required")
        self.describe().action_schema.validate(action)
        args = action.arguments.value
        if action.operation == "append" and len(session.text) + len(args["text"]) > 65536:
            raise ProtocolError("document size bound exceeded")
        if action.operation == "append":
            session.text += args["text"]
            session.revision += 1
        if action.operation == "finish": session.terminal = True
        session.remaining -= 1
        session.sequence += 1
        self._usage = self._usage + Usage(experiments=1)
        return Observation(action, self.initial_observation(session), session.sequence, session.terminal)

    def close(self, session):
        self._sessions.pop(id(session), None)

    def meter(self): return self._usage

    def estimate_episode(self, action_budget, search_budget):
        return Usage(experiments=action_budget, predictions=2, search=search_budget,
                     verification_outputs=action_budget + 2 * len(self._reference_actions))

    def estimate_certification(self): raise CapabilityUnsupported("document world certification unavailable")

    def conformance_actions(self):
        return self._reference_actions

    def compile_challenge(self, proposal): raise CapabilityUnsupported("document world creation unavailable")

    def _reference(self, actions):
        # Independent semantics: pure sequence transformation, no Session/step.
        chunks = []
        revision = 0
        terminal = False
        trace = []
        for action in actions:
            if terminal: raise ProtocolError("invalid audit trace after termination")
            self.describe().action_schema.validate(action)
            if action.operation == "append":
                chunks.append(action.arguments.value["text"])
                revision += 1
            elif action.operation == "finish": terminal = True
            value = JSONValue.of({"text": "".join(chunks), "revision": revision, "finished": terminal})
            trace.append(value)
            self._usage = self._usage + Usage(verification_outputs=1)
        return tuple(trace)

    def evaluate(self, submission, task, split):
        self._check_task(task)
        if split != task.split: raise PermissionError("split mismatch")
        self._usage = self._usage + Usage(predictions=1)
        try:
            if submission.program.provenance.domain_id != self.domain_id: raise ProtocolError("submission domain mismatch")
            actual = execute_result(submission.program)
            self.describe().result_schema.validate(actual)
        except (ProtocolError, AttributeError, TypeError) as exc:
            return Score(0.0, False, 1, str(exc))
        expected = self._reference(self._reference_actions)[-1]
        return Score(float(actual == expected), True, 1)

    def audit(self, evidence):
        self._check_task(evidence.task)
        if (evidence.provenance.domain_id != self.domain_id or evidence.provenance.split != evidence.task.split
                or evidence.provenance.verifier_version != self.verifier_version): return False
        try:
            reference_trace = self._reference(tuple(o.action for o in evidence.observations))
            for index, (observed, expected) in enumerate(zip(evidence.observations, reference_trace)):
                if (observed.outcome != expected or observed.sequence != index + 1
                        or observed.terminal != expected.value["finished"]): return False
            actual = execute_result(evidence.new_claim.program)
            if digest(actual) != evidence.behavior_hash: return False
            # M1 does not certify cognitive boundaries for this adapter.
            if evidence.boundaries: return False
            return evidence.score == self.evaluate(evidence.new_claim, evidence.task, evidence.task.split)
        except (ProtocolError, AttributeError, TypeError): return False
