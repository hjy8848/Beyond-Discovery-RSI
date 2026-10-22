from dataclasses import dataclass


@dataclass(frozen=True)
class Diagnosis:
    action: str
    assumptions: tuple
    evidence_hashes: tuple
    conflicting_clusters: int
    stagnant_clusters: int
    confidence: float


def diagnose(meta, evidence):
    """Require independent clusters, recurring failures AND executable conflict."""
    from .contracts import digest
    if any(e.task.split != "train" for e in evidence):
        raise PermissionError("diagnosis accepts train only")
    valid = [e for e in evidence if e.failure is None and e.score.valid]
    failures = {e.task.source_cluster for e in valid if e.score.value < 1.0}
    conflicts = {e.task.source_cluster for e in valid if any(b.reproduced for b in e.boundaries)}
    shared = {}
    for e in valid:
        if e.boundaries:
            shared.setdefault(e.old_claim.assumption, set()).add(e.task.source_cluster)
    assumptions = tuple(k for k, clusters in shared.items() if len(clusters) >= meta.required_conflicts)
    enough = len(conflicts) >= meta.required_conflicts and len(failures) >= meta.required_stagnation and bool(assumptions)
    action = "propose_restructure" if enough else ("continue_local_search" if valid else "insufficient_evidence")
    return Diagnosis(action, assumptions, tuple(digest(e) for e in valid if e.boundaries),
                     len(conflicts), len(failures), min(1.0, len(conflicts) / max(1, meta.required_conflicts)))
