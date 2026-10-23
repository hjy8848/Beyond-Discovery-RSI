"""Domain-neutral episode dispatch and fixed selection metrics."""
from dataclasses import replace
from .protocol import negotiate, usage_within, ProtocolError


def episode(adapter, task, scientist_snapshot, challenger_snapshot, generation, search_budget=256, runtime=None):
    from ..runtime.research import resolve_runtime
    runtime = runtime or resolve_runtime(adapter)
    budget = min(scientist_snapshot.experiment.action_budget, challenger_snapshot.experiment.action_budget)
    negotiate(adapter.describe(), runtime.languages, budget)
    if task.domain_id != adapter.domain_id:
        raise ProtocolError("task domain mismatch")
    reserved = adapter.estimate_episode(budget, search_budget)
    if not usage_within(reserved, reserved):
        raise ProtocolError("adapter declared an invalid resource reservation")
    evidence = runtime.episode(adapter, task, scientist_snapshot, challenger_snapshot, generation, search_budget)
    if not usage_within(evidence.usage, reserved):
        raise ProtocolError("runtime exceeded its predeclared resource reservation")
    return replace(evidence, allocation=reserved)


def evaluate_pair(adapter, tasks, scientist, challenger, generation, search_budget, archive, label, runtime=None,on_evidence=None):
    evidence=[]
    for task in tasks:
        item=episode(adapter,task,scientist,challenger,generation,search_budget,runtime=runtime)
        archive.append("episode",{"label":label,"scientist":scientist,"challenger":challenger,"evidence":item})
        if on_evidence:on_evidence(item)
        evidence.append(item)
    return tuple(evidence)


def rows(evidence, role):
    result = []
    # Cluster-level de-duplication: repeated counterexamples in one mechanism do
    # not multiply Challenger credit. Same budget ALLOCATION for both arms;
    # actual costs (including independent reproduction) stay in Usage.
    seen = set()
    for e in evidence:
        if role == "scientist":
            value = e.score.value
        else:
            key = (e.task.source_cluster, e.behavior_hash)
            novel = key not in seen
            seen.add(key)
            certified = any(getattr(c.kind,"value",c.kind)in ("SPACE_REFUTED","SOLUTION_REFUTED") for c in e.boundary_certificates)
            value = float(novel and certified and any(b.reproduced for b in e.boundaries)) / max(1, e.usage.experiments)
        allocated_cost = allocation_cost(e)
        result.append((e.task.source_cluster, value, e.score.valid and e.failure is None, allocated_cost, e.task.task_id))
    return tuple(result)


def invention_rows(frozen, adapted, role):
    """Success rate of one-shot inventions, separate from task score changes.

    Each opportunity has the same adaptation allocation. The baseline is the
    exact same frozen task policy, not the other arm's learned task program.
    """
    if len(frozen) != len(adapted):
        raise ValueError("unpaired invention opportunities")
    result = []
    for old, new in zip(frozen, adapted):
        if old.task != new.task:
            raise ValueError("invention task mismatch")
        if role == "scientist":
            success = old.score.value < 1.0 and new.score.value == 1.0 and old.behavior_hash != new.behavior_hash
        else:
            success = (old.observations != new.observations and bool(new.boundaries)
                       and any(getattr(c.kind,"value",c.kind)in ("SPACE_REFUTED","SOLUTION_REFUTED") for c in new.boundary_certificates)
                       and all(b.reproduced for b in new.boundaries)
                       and not old.boundaries)
        result.append((new.task.source_cluster, float(success), new.score.valid,
                       allocation_cost(new), new.task.task_id))
    return tuple(result)


def allocation_cost(evidence):
    # Gate consumes a fixed budget-allocation scalar; preserve its comparison
    # rule while sourcing allocation from domain resource contracts.
    allocation = evidence.allocation or evidence.usage
    return sum(getattr(allocation, name) for name in allocation.__dataclass_fields__)
