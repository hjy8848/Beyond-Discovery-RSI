"""Independent, resource-bounded refutation of an explicitly declared family.

An incorrect prediction does not prove a representation limit. Exhaustion is
relative to the registered finite family and audited observations only.
"""
from dataclasses import dataclass
from enum import Enum
from .contracts import digest, plain
from .protocol import ProtocolError


class BoundaryKind(str, Enum):
    SPACE_REFUTED = "SPACE_REFUTED"
    SEARCH_FAILURE = "SEARCH_FAILURE"
    CONSISTENT = "CONSISTENT"
    UNDETERMINED = "UNDETERMINED"


@dataclass(frozen=True)
class HypothesisFamily:
    name: str
    version: str
    members: tuple
    scope: str
    complete: bool = True

    def __post_init__(self):
        if not self.name or not self.version or not self.scope or not self.members:
            raise ProtocolError("finite family requires name, version, members and scope")
        if len({digest(x) for x in self.members}) != len(self.members):
            raise ProtocolError("duplicate hypotheses in declared family")

    @property
    def content_hash(self): return digest(self)


@dataclass(frozen=True)
class BoundaryCertificate:
    kind: BoundaryKind
    family_hash: str
    evidence_hash: str
    scope: str
    examined: int
    total: int
    evaluations: int
    consistent_member: str
    refutations: tuple
    verifier_version: str = "finite-family-proof-v1"


def certify_boundary(adapter, evidence, family, predictor, budget):
    if type(budget) is not int or budget < 0: raise ProtocolError("invalid proof budget")
    if not adapter.audit(evidence): raise ProtocolError("unaudited observations cannot certify a boundary")
    # Bind to immutable observations, not self-referential certificate fields.
    evidence_hash = digest((evidence.provenance, evidence.task, evidence.observations, evidence.old_claim))
    refutations = []
    examined = evaluations = 0
    consistent = ""
    if evidence.observations:
        selected_hash = digest(evidence.old_claim.program)
        members = sorted(family.members, key=lambda member: digest(member) != selected_hash)
        for member in members:
            if examined >= budget: break
            mismatch = None
            for index, observation in enumerate(evidence.observations):
                value = predictor(member, observation.action, evidence.observations[:index])
                evaluations += 1
                if plain(value) != plain(observation.outcome):
                    mismatch = index
                    break
            examined += 1
            if mismatch is None:
                consistent = digest(member)
                break
            refutations.append((digest(member), mismatch))
    if consistent:
        selected = digest(evidence.old_claim.program)
        kind = BoundaryKind.CONSISTENT if consistent == selected else BoundaryKind.SEARCH_FAILURE
    elif examined == len(family.members) and evidence.observations and family.complete:
        kind = BoundaryKind.SPACE_REFUTED
    else:
        kind = BoundaryKind.UNDETERMINED
    return BoundaryCertificate(kind, family.content_hash, evidence_hash, family.scope,
                               examined, len(family.members), evaluations, consistent, tuple(refutations))


def verify_certificate(certificate, adapter, evidence, family, predictor):
    # Recompute all proofs independently; do not trust supplied witnesses/counts.
    expected = certify_boundary(adapter, evidence, family, predictor, certificate.examined)
    return expected == certificate
