"""Trusted finite semantics and session/score isolation, without a solver."""
from __future__ import annotations

from dataclasses import dataclass
import itertools
import random

from ..core.contracts import (Certification, ChallengeProposal, Claim, Observation,
                              Score, Status, TaskRef, Usage, digest)
from ..core.protocol import DomainCapabilities, JSONValue, Schema
from ..runtime.policy_runtime import PolicyError, compile_program, execute


def reference(program, values):
    """Oracle interpreter, separate from candidate bytecode execution."""
    tag, *args = program
    if tag == "x": return values[args[0]]
    if tag == "const": return args[0]
    a, b = (reference(x, values) for x in args)
    return {"add": lambda: a + b, "sub": lambda: a - b,
            "mul": lambda: a * b, "xor": lambda: a ^ b,
            "max": lambda: a if a > b else b,
            "min": lambda: a if a < b else b}[tag]()


@dataclass
class Session:
    task: TaskRef
    remaining: int


class FiniteAdapter:
    verifier_version = "finite-exhaustive-v1"
    width = 6
    values = (0, 1)
    operators = ("add", "xor")
    domain_id = "abstract"

    def __init__(self):
        self.verification_outputs = 0
        self._worlds = {}
        self._refs = {}
        self._sessions = {}
        self._sealed = True
        self.inputs = tuple(itertools.product(self.values, repeat=self.width))
        pairs = tuple(itertools.combinations(range(self.width), 2))
        for split, group in zip(("train", "dev", "test"), (pairs[:5], pairs[5:10], pairs[10:])):
            for i, j in group:
                cluster = "%s/%s/%s-%s" % (self.domain_id, split, i, j)
                for op in self.operators:
                    program = (op, ("x", i), ("x", j))
                    task_id = digest((self.domain_id, program))
                    task = TaskRef(task_id, self.domain_id, split, cluster, 0)
                    self._worlds[task_id] = program
                    self._refs[task_id] = task

    def _oracle(self, program, values):
        self.verification_outputs += 1
        return reference(program, values)

    def certification_budget(self):
        return (len(self._worlds) + 1) * len(self.inputs)

    def describe(self):
        return DomainCapabilities(self.domain_id,
            Schema.of({"type": "array", "items": {"type": "integer"}, "minItems": self.width, "maxItems": self.width}),
            Schema.of({"type": "integer"}), Schema.of({"type": "integer"}),
            ("finite-expression-v1",), ("A", "B"), 16,
            limitations=("retained finite baseline; track B is bounded expression DSL",))

    def estimate_episode(self, action_budget, search_budget):
        return Usage(experiments=2 * action_budget, predictions=4 * len(self.inputs) + search_budget * action_budget,
                     search=search_budget, verification_outputs=4 * len(self.inputs) + 6 * action_budget)

    def estimate_certification(self):
        return Usage(verification_outputs=self.certification_budget())

    def meter(self):
        return Usage(verification_outputs=self.verification_outputs)

    def close(self, session):
        self._sessions.pop(id(session), None)
        session.remaining = 0

    def initial_observation(self, session):
        if self._sessions.get(id(session)) is not session:
            raise ValueError("foreign or closed session")
        return JSONValue.of(None)

    def conformance_actions(self):
        return (self.inputs[0], self.inputs[-1])

    def capabilities(self):
        return {"domain_id": self.domain_id, "width": self.width,
                "input_space": self.inputs, "expression_ops": self.operators,
                "actions": ("probe",), "world_creation": ("A", "B-bounded-DSL"),
                "score_direction": "higher", "score_meaning": "exhaustive exact prediction fraction",
                "evaluation_randomness": "deterministic", "max_action_budget": 16,
                "limitations": "finite synthetic domain; no external benchmark validation"}

    def sample(self, split, seed):
        if split not in ("train", "dev", "test"):
            raise ValueError("unknown split")
        if split == "test" and self._sealed:
            raise PermissionError("test is sealed")
        tasks = [x for x in self._refs.values() if x.split == split and not x.source_cluster.startswith("generated/")]
        random.Random(seed).shuffle(tasks)
        return tuple(tasks)

    def unseal_for_final(self):
        if not self._sealed:
            raise PermissionError("test already unsealed")
        self._sealed = False

    def _check(self, task):
        if task.domain_id != self.domain_id or self._refs.get(task.task_id) != task:
            raise ValueError("unregistered/forged task ref")
        if task.split == "test" and self._sealed:
            raise PermissionError("test is sealed")

    def compile_challenge(self, proposal: ChallengeProposal):
        world_hash = digest(proposal.program)
        if proposal.domain_id != self.domain_id or proposal.track not in ("A", "B"):
            return Certification(Status.INVALID, None, world_hash, digest(proposal), self.verifier_version, "domain/track mismatch")
        try:
            code = compile_program(proposal.program, self.width)
            # Only adapter-sanctioned semantics, not arbitrary author Python.
            if any(x[0] not in {"x", "const", *self.operators} for x in code):
                raise PolicyError("unsupported semantics")
            behavior = tuple(self._oracle(proposal.program, x) for x in self.inputs)
            if behavior != tuple(execute(code, x) for x in self.inputs):
                raise PolicyError("independent interpreters disagree")
            if len(set(behavior)) < 2:
                raise PolicyError("trivial constant world")
            # Fixed worlds depend on exactly two variables. Generated worlds
            # must depend on >=3, proven from this world's own truth table.
            # Never compare with dev/test answers during train certification.
            table = dict(zip(self.inputs, behavior))
            influential = 0
            for index in range(self.width):
                if any(table[x] != table[x[:index] + (value,) + x[index + 1:]]
                       for x in self.inputs for value in self.values):
                    influential += 1
            if influential < 3:
                raise PolicyError("generated source namespace requires >=3 influential variables")
            for task_id, existing in self._worlds.items():
                if self._refs[task_id].split != "train":
                    continue
                if behavior == tuple(self._oracle(existing, x) for x in self.inputs):
                    raise PolicyError("duplicate training world")
            task_id = digest((self.domain_id, proposal.program))
            task = TaskRef(task_id, self.domain_id, "train", "generated/" + digest(behavior), 0)
            self._worlds[task_id] = proposal.program
            self._refs[task_id] = task
            return Certification(Status.VERIFIED, task, world_hash, digest(proposal), self.verifier_version,
                                 "exhaustive agreement of independent semantics", (digest(behavior), len(behavior)))
        except (PolicyError, ValueError, TypeError, IndexError, KeyError, OverflowError) as exc:
            return Certification(Status.INVALID, None, world_hash, digest(proposal), self.verifier_version, str(exc))

    def begin(self, task, budget):
        self._check(task)
        if type(budget) is not int or not 1 <= budget <= 16:
            raise ValueError("EXPERIMENT_BUDGET")
        session = Session(task, budget)
        self._sessions[id(session)] = session
        return session

    def step(self, session, action):
        if self._sessions.get(id(session)) is not session:
            raise ValueError("foreign or closed session")
        self._check(session.task)
        if session.remaining <= 0:
            raise ValueError("EXPERIMENT_BUDGET")
        if action not in self.inputs:
            raise ValueError("ACTION_INVALID")
        session.remaining -= 1
        return Observation(action, self._oracle(self._worlds[session.task.task_id], action))

    def evaluate(self, submission: Claim, task, split):
        self._check(task)
        if task.split != split:
            raise PermissionError("split mismatch")
        try:
            code = compile_program(submission.program, self.width)
            count = sum(execute(code, x) == self._oracle(self._worlds[task.task_id], x) for x in self.inputs)
            return Score(count / len(self.inputs), True, len(self.inputs))
        except PolicyError as exc:
            return Score(0.0, False, len(self.inputs), str(exc))

    def audit(self, evidence):
        self._check(evidence.task)
        if (evidence.provenance.domain_id != self.domain_id or evidence.provenance.split != evidence.task.split
                or evidence.provenance.verifier_version != self.verifier_version):
            return False
        program = self._worlds[evidence.task.task_id]
        if any(self._oracle(program, o.action) != o.outcome for o in evidence.observations):
            return False
        for boundary in evidence.boundaries:
            if not boundary.reproduced or boundary.witness not in evidence.observations:
                return False
            if self._oracle(program, boundary.witness.action) != boundary.witness.outcome:
                return False
            if execute(compile_program(evidence.old_claim.program, self.width), boundary.witness.action) != boundary.old_prediction:
                return False
            if boundary.old_prediction == boundary.witness.outcome:
                return False
        code = compile_program(evidence.new_claim.program, self.width)
        if digest(tuple(execute(code, x) for x in self.inputs)) != evidence.behavior_hash:
            return False
        return self.evaluate(evidence.new_claim, evidence.task, evidence.task.split) == evidence.score
