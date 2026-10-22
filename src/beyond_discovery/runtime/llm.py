"""Optional inference boundary, independent of any vendor SDK or credentials.

Inject a callable returning JSON text. The framework passes TRAIN evidence only,
records exact request/response and usage, and accepts only bounded policy data.
The default runnable system uses deterministic program search and makes no API
calls. This module does not claim to train model weights.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import json
from typing import Callable
from ..core.contracts import ExperimentProgram, MetaPolicy, RepresentationPatch, canonical, digest, plain
from .policy_runtime import PolicyError, validate_meta


@dataclass(frozen=True)
class Completion:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    truncated: bool = False
    behavior_logprob: float = None
    policy_id: str = None
    prompt_ids: tuple = ()
    completion_ids: tuple = ()


class JSONPolicyProposer:
    def __init__(self, complete: Callable[[str, int], Completion], model: str, token_limit: int = 4096):
        self.complete = complete
        self.model = model
        self.token_limit = token_limit
        self.total_tokens = 0
        self.calls = []

    def propose(self, parent, diagnosis, evidence, capabilities, limit, mode):
        if mode != "full":
            raise ValueError("external proposer currently supports full mode only")
        if parent.model.base_model != self.model:
            raise ValueError("shared base-model identity mismatch")
        if any(e.task.split != "train" for e in evidence):
            raise PermissionError("TRAIN-only generation")
        request = canonical({"contract": "Return JSON list of policies; keys: representation, experiment, meta. No Python.",
                             "role": parent.role, "parent": parent, "diagnosis": diagnosis,
                             "train_evidence": [e.agent_view() for e in evidence],
                             "capabilities": capabilities, "max_candidates": limit,
                             "allowed_operators": parent.meta.operators,
                             "meta_keys": ["operators", "required_conflicts", "required_stagnation", "search_order"]})
        result = self.complete(request, self.token_limit)
        # Record all responses including malformed and truncated ones. Caller
        # archives these before raising; no fabricated replacement candidate.
        self.calls.append({"request": request, "response": plain(result)})
        if min(result.input_tokens, result.output_tokens) < 0 or result.output_tokens > self.token_limit:
            raise PolicyError("MODEL_USAGE_INVALID")
        self.total_tokens += result.input_tokens + result.output_tokens
        if result.model != self.model or result.truncated:
            raise PolicyError("MODEL_TRUNCATED_OR_IDENTITY_MISMATCH")
        if len(result.text) > 100000:
            raise PolicyError("MODEL_RESPONSE_SIZE")
        data = json.loads(result.text)
        if not isinstance(data, list) or len(data) > limit:
            raise PolicyError("CODE_INVALID: candidate count")
        attempts = []
        for item in data:
            if not isinstance(item, dict) or set(item) != {"representation", "experiment", "meta"}:
                raise PolicyError("CODE_INVALID: candidate keys")
            rep = tuple(item["representation"])
            if not rep or len(set(rep)) != len(rep) or not set(rep) <= set(capabilities["expression_ops"]):
                raise PolicyError("CODE_INVALID: representation")
            if set(item["experiment"]) != {"strategy", "action_budget"}:
                raise PolicyError("CODE_INVALID: experiment keys")
            exp = ExperimentProgram(**item["experiment"])
            if exp.action_budget != parent.experiment.action_budget or exp.strategy not in ("sequential", "disagreement"):
                raise PolicyError("CODE_INVALID: experiment/budget")
            if set(item["meta"]) != {"operators", "required_conflicts", "required_stagnation", "search_order"}:
                raise PolicyError("CODE_INVALID: meta keys")
            meta = MetaPolicy(**{**item["meta"], "operators": tuple(item["meta"]["operators"]), "revision": parent.meta.revision + 1})
            validate_meta(meta)
            child = replace(parent, representation=rep, experiment=exp, meta=meta,
                            generation=parent.generation + 1, parent_hash=parent.content_hash)
            patch = RepresentationPatch("MODEL_PROPOSAL", parent.representation, rep, digest(request))
            attempts.append((child, patch))
        return tuple(attempts)
