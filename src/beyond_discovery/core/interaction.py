from __future__ import annotations

from dataclasses import replace

from ..agents.scientist import Scientist
from ..agents.challenger import Challenger
from ..runtime.policy_runtime import predict
from .contracts import Boundary, EvidenceBundle, Provenance, Usage, digest


def episode(adapter, task, scientist_snapshot, challenger_snapshot, generation, search_budget=256):
    caps = adapter.capabilities()
    scientist = Scientist(scientist_snapshot, caps, search_budget)
    experiment_driver = challenger_snapshot
    if scientist_snapshot.experiment.strategy == "disagreement":
        experiment_driver = replace(challenger_snapshot, experiment=replace(challenger_snapshot.experiment, strategy="disagreement"))
    challenger = Challenger(experiment_driver)
    budget = min(scientist_snapshot.experiment.action_budget, challenger_snapshot.experiment.action_budget)
    session = adapter.begin(task, budget)
    observations = []
    for _ in range(budget):
        action = challenger.discriminate(caps, scientist, observations)
        observations.append(adapter.step(session, action))
    claim = scientist.hypothesize(observations)
    # A contradiction to the fitted restricted family, not an LLM introspection.
    boundaries = []
    for observation in observations:
        predicted = predict(claim.program, observation.action)
        if predicted != observation.outcome:
            reproduced = adapter.step(adapter.begin(task, 1), observation.action) == observation
            boundaries.append(Boundary(claim.assumption, observation, predicted, None,
                                       reproduced, task.source_cluster))
    score = adapter.evaluate(claim, task, task.split)
    behavior = tuple(predict(claim.program, x) for x in caps["input_space"])
    evidence = EvidenceBundle(
        Provenance(adapter.domain_id, task.split, "real-execution", generation,
                   scientist_snapshot.content_hash, task.seed, adapter.verifier_version,
                   scientist_snapshot.parent_hash),
        task, tuple(observations), claim, claim, score, tuple(boundaries), digest(behavior),
        Usage(experiments=budget + len(boundaries), predictions=4 * score.cost,
              search=len(scientist.programs)))
    if not adapter.audit(evidence):
        raise RuntimeError("independent evidence audit failed")
    return evidence


def evaluate_pair(adapter, tasks, scientist, challenger, generation, search_budget, archive, label):
    evidence = tuple(episode(adapter, t, scientist, challenger, generation, search_budget) for t in tasks)
    for item in evidence:
        archive.append("episode", {"label": label, "scientist": scientist,
                                   "challenger": challenger, "evidence": item})
    return evidence


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
            value = float(novel and any(b.reproduced for b in e.boundaries)) / max(1, e.usage.experiments)
        allocated_cost = 2 * len(e.observations) + e.score.cost + 512
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
                       and all(b.reproduced for b in new.boundaries)
                       and not old.boundaries)
        result.append((new.task.source_cluster, float(success), new.score.valid,
                       2 * len(new.observations) + new.score.cost + 512, new.task.task_id))
    return tuple(result)
