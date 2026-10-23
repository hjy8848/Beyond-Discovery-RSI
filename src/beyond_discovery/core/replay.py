"""Reexecute deterministic interactions, not just compare archive hashes."""
from __future__ import annotations

from dataclasses import replace

from .archive import read_verified
from .contracts import (ExperimentProgram, MetaPolicy, ModelState, Snapshot, ChallengeProposal,
                        plain, digest)
from .interaction import episode
from .protocol import CapabilityUnsupported, artifact_from, negotiate
from ..runtime.research import resolve_runtime
from ..runtime.policy_runtime import freeze


def snapshot_from(data):
    state=ModelState(**data["model"])
    if state.delta_checkpoint:
        from ..runtime.models import validate_checkpoint
        manifest=validate_checkpoint(state.delta_checkpoint,state.role,state.base_model)
        if manifest['training_data_hash']!=state.training_data_hash: raise ValueError('checkpoint provenance changed')
    meta_data = {**data["meta"], "operators": tuple(data["meta"]["operators"])}
    if isinstance(meta_data.get("program"), dict): meta_data["program"] = artifact_from(meta_data["program"])
    return Snapshot(data["role"], state,
                    MetaPolicy(**meta_data),
                    tuple(data["representation"]), ExperimentProgram(**data["experiment"]),
                    data["generation"], data["parent_hash"], tuple(data["knowledge"]),
                    tuple(artifact_from(a) for a in data.get("artifacts", ())))


def replay(path, adapter):
    from ..runtime.sandbox import ResearchSandbox
    if "research-expression-v1" in adapter.describe().artifact_languages:
        with ResearchSandbox(): return _replay(path,adapter)
    return _replay(path,adapter)


def _replay(path, adapter):
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
    if not adapter.describe().deterministic and "archived-inference-replay" not in adapter.describe().features:
        raise CapabilityUnsupported("exact reexecution requires a deterministic adapter; stochastic audit-only replay is not implemented in M1")
    runtime = resolve_runtime(adapter, prereg.get("runtime_id"))
    if prereg.get('runtime_id')=='apps-coding-v1':
        from ..runtime.coding import CodingLimits
        runtime.limits=CodingLimits(**prereg['runtime_context']['coding_limits'])
        runtime.method=prereg['runtime_context']['method']
        runtime.model_identities=prereg['runtime_context']['model_identities']
        if runtime.context(adapter)!=prereg['runtime_context']:raise ValueError('replay model/judge/data configuration changed')
    negotiated = negotiate(adapter.describe(), runtime.languages, prereg["config"]["action_budget"],
                           prereg["config"].get("required_world_tracks", ()))
    if plain(negotiated) != prereg.get("negotiated_protocol"):
        raise ValueError("replay capability negotiation changed")
    count = 0
    frozen = False
    final_seen = set()
    for kind, data in records:
        if kind == "world_attempt":
            proposal = data["proposal"]
            program = proposal["program"]
            if isinstance(program, dict) and "language" in program:
                program = artifact_from(program)
            else:
                program = freeze(program)
            result = adapter.compile_challenge(ChallengeProposal(proposal["domain_id"], program, proposal["track"]),
                                               **({"action_budget":prereg["config"]["action_budget"]} if "world-budget-certification" in adapter.describe().features else {}))
            if plain(result) != data["certification"]:
                raise ValueError("world certification replay mismatch")
        elif kind == "freeze":
            frozen = True
            if adapter.domain_id=="apps":adapter.authorize_final(digest(data))
            if prereg["config"].get("final_evaluation",True):adapter.unseal_for_final()
        elif kind == "episode":
            item = data["evidence"]
            task = TaskRef(**item["task"])
            if task.split == "test":
                if not frozen or task.task_id in final_seen or data["label"] != "sealed-final-test":
                    raise ValueError("invalid/repeated test access")
                final_seen.add(task.task_id)
            scientist=snapshot_from(data["scientist"]);challenger=snapshot_from(data["challenger"])
            if "frozen-challenge-binding" in adapter.describe().features and data['label']=='train':adapter.bind_frozen_scientist(scientist.content_hash)
            if hasattr(runtime,'replay_episode'):
                from .protocol import usage_within,ProtocolError
                regenerated=runtime.replay_episode(item,adapter,task,scientist,challenger,item['provenance']['generation'],prereg['config']['search_budget'])
                allocation=adapter.estimate_episode(min(scientist.experiment.action_budget,challenger.experiment.action_budget),prereg['config']['search_budget'])
                if not usage_within(regenerated.usage,allocation):raise ProtocolError('replay reservation exceeded')
                regenerated=replace(regenerated,allocation=allocation)
            else:
                regenerated=episode(adapter,task,scientist,challenger,item['provenance']['generation'],prereg['config']['search_budget'],runtime=runtime)
            if plain(regenerated) != item:
                raise ValueError("deterministic evidence replay mismatch")
            count += 1
    return {"verified": True, "episodes_reexecuted": count, "records": len(records),
            "note": "model responses are archived; no claim of stochastic API regeneration"}
