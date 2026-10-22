"""Reexecute deterministic interactions, not just compare archive hashes."""
from __future__ import annotations

from .archive import read_verified
from .contracts import (ExperimentProgram, MetaPolicy, ModelState, Snapshot, ChallengeProposal,
                        plain)
from .interaction import episode
from ..runtime.policy_runtime import freeze


def snapshot_from(data):
    return Snapshot(data["role"], ModelState(**data["model"]),
                    MetaPolicy(**{**data["meta"], "operators": tuple(data["meta"]["operators"])}),
                    tuple(data["representation"]), ExperimentProgram(**data["experiment"]),
                    data["generation"], data["parent_hash"], tuple(data["knowledge"]))


def replay(path, adapter):
    from .contracts import TaskRef
    from .controller import implementation_hash
    records = read_verified(path)
    if not records or records[0][0] != "preregistration":
        raise ValueError("missing preregistration")
    prereg = records[0][1]
    if prereg["implementation_hash"] != implementation_hash():
        raise ValueError("implementation changed; use the archived implementation version")
    if prereg["capabilities"]["domain_id"] != adapter.domain_id:
        raise ValueError("replay domain mismatch")
    count = 0
    frozen = False
    final_seen = set()
    for kind, data in records:
        if kind == "world_attempt":
            proposal = data["proposal"]
            result = adapter.compile_challenge(ChallengeProposal(proposal["domain_id"], freeze(proposal["program"]), proposal["track"]))
            if plain(result) != data["certification"]:
                raise ValueError("world certification replay mismatch")
        elif kind == "freeze":
            frozen = True
            adapter.unseal_for_final()
        elif kind == "episode":
            item = data["evidence"]
            task = TaskRef(**item["task"])
            if task.split == "test":
                if not frozen or task.task_id in final_seen or data["label"] != "sealed-final-test":
                    raise ValueError("invalid/repeated test access")
                final_seen.add(task.task_id)
            regenerated = episode(adapter, task, snapshot_from(data["scientist"]),
                                  snapshot_from(data["challenger"]), item["provenance"]["generation"],
                                  prereg["config"]["search_budget"])
            if plain(regenerated) != item:
                raise ValueError("deterministic evidence replay mismatch")
            count += 1
    return {"verified": True, "episodes_reexecuted": count, "records": len(records),
            "note": "model responses are archived; no claim of stochastic API regeneration"}
