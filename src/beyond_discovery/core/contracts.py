"""Immutable public protocol. Oracle contents never enter an agent context."""
from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum
import hashlib
import json
from typing import Any, Optional, Protocol, Tuple


def plain(value: Any) -> Any:
    if hasattr(value, "to_json"):
        return plain(value.to_json())
    if hasattr(value, "__dataclass_fields__"):
        return {f.name: plain(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def canonical(value: Any) -> str:
    return json.dumps(plain(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


class Decision(str, Enum):
    ACCEPT = "ACCEPT"
    REJECT = "REJECT"
    INCONCLUSIVE = "INCONCLUSIVE"


class Status(str, Enum):
    VERIFIED = "VERIFIED"
    UNVERIFIED = "UNVERIFIED"
    INVALID = "INVALID"
    PENDING = "PENDING"


@dataclass(frozen=True)
class Usage:
    experiments: int = 0
    predictions: int = 0
    search: int = 0
    model_tokens: int = 0
    training_steps: int = 0
    verification_outputs: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(*(getattr(self, k) + getattr(other, k) for k in self.__dataclass_fields__))


@dataclass(frozen=True)
class Provenance:
    domain_id: str
    split: str
    origin: str
    generation: int
    agent_version: str
    seed: int
    verifier_version: str
    parent_hash: Optional[str] = None


@dataclass(frozen=True)
class TaskRef:
    task_id: str
    domain_id: str
    split: str
    source_cluster: str
    seed: int


@dataclass(frozen=True)
class Claim:
    program: Any
    assumption: str
    scope: str = "adapter-declared experimental scope"


@dataclass(frozen=True)
class ExperimentProgram:
    strategy: str = "sequential"
    action_budget: int = 6


@dataclass(frozen=True)
class ChallengeProposal:
    domain_id: str
    program: Any
    track: str = "A"
    # No answer field: authors have no authority to certify a world.


@dataclass(frozen=True)
class Certification:
    status: Status
    task: Optional[TaskRef]
    world_hash: str
    generator_hash: str
    semantics_version: str
    reason: str
    certificate: tuple = ()


@dataclass(frozen=True)
class Observation:
    action: Any
    outcome: Any
    sequence: int = 0
    terminal: bool = False


@dataclass(frozen=True)
class Score:
    value: float
    valid: bool
    cost: int
    reason: str = "OK"


@dataclass(frozen=True)
class Boundary:
    assumption: str
    witness: Observation
    old_prediction: Any
    alternative_prediction: Optional[Any]
    reproduced: bool
    source_cluster: str


@dataclass(frozen=True)
class EvidenceBundle:
    provenance: Provenance
    task: TaskRef
    observations: Tuple[Observation, ...]
    old_claim: Claim
    new_claim: Claim
    score: Score
    boundaries: Tuple[Boundary, ...]
    behavior_hash: str
    usage: Usage
    failure: Optional[str] = None
    allocation: Optional[Usage] = None

    def agent_view(self) -> dict:
        if self.task.split != "train":
            raise PermissionError("dev/test evidence cannot enter candidate generation")
        # Score is independent evaluation feedback; no private probe labels/programs.
        return plain(self)


@dataclass(frozen=True)
class ModelState:
    base_model: str = "bounded-enumerator-v1"
    role: str = "scientist"
    delta_checkpoint: Optional[str] = None
    training_data_hash: Optional[str] = None


@dataclass(frozen=True)
class MetaPolicy:
    # Executable bounded DSL: condition + ordered operator dispatch.
    operators: Tuple[str, ...] = ("RECOMBINE", "REFRAME", "RELAX_ASSUMPTION", "ANALOGIZE", "INVENT_EXPERIMENT")
    required_conflicts: int = 2
    required_stagnation: int = 2
    search_order: str = "forward"
    revision: int = 0


@dataclass(frozen=True)
class Snapshot:
    role: str
    model: ModelState
    meta: MetaPolicy
    representation: Tuple[str, ...]
    experiment: ExperimentProgram
    generation: int = 0
    parent_hash: Optional[str] = None
    knowledge: Tuple[str, ...] = ()
    artifacts: Tuple[Any, ...] = ()

    @property
    def content_hash(self) -> str:
        return digest(self)


@dataclass(frozen=True)
class RepresentationPatch:
    operator: str
    before: Tuple[str, ...]
    after: Tuple[str, ...]
    reason: str


@dataclass(frozen=True)
class GateResult:
    decision: Decision
    delta: float
    interval: Tuple[float, float]
    clusters: int
    reason: str


class DomainAdapter(Protocol):
    domain_id: str
    verifier_version: str
    def describe(self) -> Any: ...
    def estimate_episode(self, action_budget: int, search_budget: int) -> Usage: ...
    def estimate_certification(self) -> Usage: ...
    def meter(self) -> Usage: ...
    def close(self, session: Any) -> None: ...
    def sample(self, split: str, seed: int) -> Tuple[TaskRef, ...]: ...
    def compile_challenge(self, proposal: ChallengeProposal) -> Certification: ...
    def begin(self, task: TaskRef, budget: int) -> Any: ...
    def initial_observation(self, session: Any) -> Any: ...
    def step(self, session: Any, action: Any) -> Observation: ...
    def evaluate(self, submission: Claim, task: TaskRef, split: str) -> Score: ...
    def audit(self, evidence: EvidenceBundle) -> bool: ...
    def unseal_for_final(self) -> None: ...
