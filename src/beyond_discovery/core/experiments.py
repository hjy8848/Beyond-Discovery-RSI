"""Bounded active experiment synthesis over executable competing hypotheses."""
from dataclasses import dataclass
from .contracts import canonical, digest, plain
from .protocol import Action, JSONValue, ProtocolError, ResearchArtifact, Schema
from ..runtime.expression import evaluate


def context_for(action, history, initial):
    if hasattr(action,"action"): action=action.action
    return {"input": action.arguments.value, "previous": plain(history[-1].outcome) if history else initial,
            "time": len(history)}


def rollout(program, actions, initial):
    previous = initial
    output = []
    for index, action in enumerate(actions):
        previous = evaluate(program, {"input": action.arguments.value, "previous": previous, "time": index})
        output.append(previous)
    return tuple(output)


def design_experiment(hypotheses, actions, initial, horizon, budget, provenance):
    if provenance.split != "train": raise PermissionError("experiment invention is train-only")
    if not hypotheses or not actions or horizon < 1 or budget < 1: raise ProtocolError("invalid experiment search")
    frontier = [()]
    best = None
    best_key = (-1, 0, "")
    examined = 0
    for depth in range(1, horizon + 1):
        next_frontier = []
        for prefix in frontier:
            for action in actions:
                if examined >= budget: break
                sequence = prefix + (action,)
                examined += 1
                outcomes = tuple(digest(rollout(model, sequence, initial)) for model in hypotheses)
                groups = [outcomes.count(value) for value in set(outcomes)]
                distinguishable = len(hypotheses) ** 2 - sum(n*n for n in groups)
                key = (distinguishable / len(sequence), -len(sequence), digest(sequence))
                if key > best_key: best, best_key = sequence, key
                next_frontier.append(sequence)
            if examined >= budget: break
        frontier = next_frontier
        if examined >= budget: break
    if best is None: raise ProtocolError("no executable experiment")
    artifact = ResearchArtifact("experiment", "research-protocol-v1", "1",
                                canonical({"actions": best, "hypotheses": tuple(digest(p) for p in hypotheses),
                                           "search_cost": examined}), Schema.of({"type": "object"}),
                                Schema.of({"type": "array"}), provenance)
    return artifact


def protocol_actions(artifact):
    import json
    if artifact.kind != "experiment" or artifact.language != "research-protocol-v1":
        raise ProtocolError("expected executable experiment protocol")
    data = json.loads(artifact.source)
    if set(data) != {"actions", "hypotheses", "search_cost"} or not isinstance(data["actions"], list) or len(data["actions"]) > 64:
        raise ProtocolError("invalid experiment protocol")
    return tuple(Action.create(x["operation"], x["arguments"]) for x in data["actions"])


def execute_protocol(artifact, call, budget):
    actions = protocol_actions(artifact)
    if len(actions) > budget: raise ProtocolError("experiment execution budget exceeded")
    result = []
    for action in actions:
        observation = call(action)
        result.append(observation)
        if observation.terminal: break
    return tuple(result)
