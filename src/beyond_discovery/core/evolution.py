from dataclasses import replace
from .contracts import digest
from .invention import invent
from ..runtime.policy_runtime import validate_meta


def candidates(parent, diagnosis, evidence, capabilities, limit=3, mode="full"):
    """Generate ALL attempts before any dev feedback, from the same parent."""
    if any(e.task.split != "train" for e in evidence):
        raise PermissionError("candidate generation accepts train only")
    validate_meta(parent.meta)
    if mode == "fixed":
        return ()
    if mode in ("full", "no_restructuring", "no_meta") and diagnosis.action != "propose_restructure" and parent.role == "scientist":
        return ()
    results = []
    operators = parent.meta.operators
    for operator in operators[:limit]:
        child, patch = invent(parent, diagnosis, capabilities, operator)
        if mode == "no_restructuring" and child.representation != parent.representation:
            continue
        if child.representation == parent.representation and child.experiment == parent.experiment:
            continue
        # MetaPolicy actually changes future dispatch ordering. Its causal effect
        # is evaluated separately using a fixed parent task representation.
        updated_order = (operator,) + tuple(x for x in parent.meta.operators if x != operator)
        meta = parent.meta if mode == "no_meta" else replace(parent.meta, operators=updated_order, revision=parent.meta.revision + 1)
        child = replace(child, meta=meta, parent_hash=parent.content_hash,
                        generation=parent.generation + 1,
                        knowledge=parent.knowledge + tuple(digest(e) for e in evidence if e.boundaries))
        results.append((child, patch))
    return tuple(results)
