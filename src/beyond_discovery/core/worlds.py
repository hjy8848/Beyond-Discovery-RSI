"""Trusted world certification and evidence-conditioned challenge search."""
from dataclasses import dataclass
from itertools import product
import json
from .contracts import ChallengeProposal, Status, canonical, digest
from .protocol import ProtocolError, ResearchArtifact, Schema
from .experiments import context_for, rollout
from .synthesis import enumerate_programs
from ..runtime.expression import evaluate, freeze, node_size, typecheck


def reference_expression(node, context):
    """Independent tree semantics; never calls the candidate expression VM."""
    tag, *children = node
    if tag == "lit": return children[0]
    if tag == "ref":
        result = context
        for name in children[0].split("."): result = result[name]
        return result
    values = [reference_expression(c, context) for c in children]
    if tag == "add": return sum(values)
    if tag == "sub": return values[0] - values[1]
    if tag == "mul":
        result = 1
        for v in values: result *= v
        return result
    if tag == "xor": return values[0] ^ values[1]
    if tag == "max": return sorted(values)[-1]
    if tag == "concat": return "".join(values)
    if tag == "reverse": return "".join(reversed(values[0]))
    if tag == "length": return sum(1 for _ in values[0])
    if tag == "eq": return values[0] == values[1]
    if tag == "not": return not values[0]
    if tag == "and": return all(values)
    if tag == "or": return any(values)
    raise ProtocolError("reference semantics unavailable")


def expression_from(artifact):
    if not isinstance(artifact, ResearchArtifact) or artifact.language != "research-expression-v1":
        raise ProtocolError("expected executable expression artifact")
    data = json.loads(artifact.source)
    if not isinstance(data, dict) or "expression" not in data or set(data) - {"expression", "operator", "evidence", "models", "scientist_hash"}:
        raise ProtocolError("invalid expression artifact fields; no author-supplied answers permitted")
    return freeze(data["expression"])


def behavior(program, actions, initial, horizon):
    return tuple(rollout(program, sequence, initial) for size in range(1, horizon + 1) for sequence in product(actions, repeat=size))


@dataclass(frozen=True)
class WorldAudit:
    status: Status
    reason: str
    fingerprint: str = ""
    witness: tuple = ()
    evaluations: int = 0
    semantics_version: str = "independent-expression-world-v1"
    identification: object = None


class WorldCertifier:
    def __init__(self, grammar, actions, initial, horizon=2):
        self.grammar, self.actions, self.initial, self.horizon = grammar, actions, initial, horizon
        self.seen = set()

    def certify(self, artifact, track, frozen_models, resource_budget, action_budget=8):
        if track not in ("A", "B"): return WorldAudit(Status.INVALID, "unknown world track")
        if not frozen_models: return WorldAudit(Status.PENDING, "no executable frozen hypothesis for discrimination")
        count = 0
        try:
            node = expression_from(artifact)
            if artifact.kind != "world" or typecheck(node, dict(self.grammar.variables)) != self.grammar.output_type:
                raise ProtocolError("world interface/type mismatch")
            if node_size(node) > 63: raise ProtocolError("world source resource bound")
            variable_bounds = {}
            for name,kind in self.grammar.variables:
                if name.startswith("input."):
                    values = [a.arguments.value[name.split(".")[1]] for a in self.actions]
                    variable_bounds[name] = max(len(v) if type(v) is str else abs(v) for v in values)
                elif name == "time": variable_bounds[name] = 12
            previous_bound = len(self.initial) if type(self.initial) is str else abs(self.initial)
            def bound(n):
                if n[0] == "lit": return len(n[1]) if type(n[1]) is str else abs(n[1])
                if n[0] == "ref": return previous_bound if n[1] == "previous" else variable_bounds[n[1]]
                values = [bound(c) for c in n[1:]]
                if n[0] in ("add","sub","concat"): return sum(values)
                if n[0] == "xor": return sum(values)+1
                if n[0] == "mul": return values[0]*values[1]
                if n[0] in ("reverse","length"): return values[0]
                if n[0] == "max": return max(values)
                return 1
            for _ in range(12):
                previous_bound = bound(node)
                limit = 65536 if self.grammar.output_type == "string" else 1000000
                if previous_bound > limit: raise ProtocolError("world exceeds rollout resource bounds")
            if track == "A":
                if node_size(node)>self.grammar.max_nodes: raise ProtocolError("compositional track grammar limit")
                def registered(n):
                    if n[0]=='ref': return n[1] in dict(self.grammar.variables)
                    if n[0]=='lit': return any(type(n[1]) is type(c) and n[1]==c for c in self.grammar.constants)
                    return n[0] in self.grammar.operators and all(registered(c) for c in n[1:])
                if not registered(node): raise ProtocolError("compositional track unregistered primitive")
            outputs = []
            witness = ()
            for size in range(1, self.horizon + 1):
                for sequence in product(self.actions, repeat=size):
                    previous = self.initial
                    trace = []
                    for i, action in enumerate(sequence):
                        if count + 2 > resource_budget:
                            return WorldAudit(Status.PENDING, "certification resource budget", evaluations=count)
                        context = {"input": action.arguments.value, "previous": previous, "time": i}
                        trusted = reference_expression(node, context)
                        candidate = evaluate(node, context)
                        count += 2
                        if trusted != candidate: raise ProtocolError("independent interpreters disagree")
                        previous = trusted; trace.append(trusted)
                    outputs.append(tuple(trace))
                    if any(rollout(m, sequence, self.initial) != tuple(trace) for m in frozen_models): witness = sequence
            fingerprint = digest(outputs)
            if len({digest(trace[-1]) for trace in outputs}) < 2: raise ProtocolError("trivial world")
            if not witness: raise ProtocolError("no distinguishing experiment against frozen hypotheses")
            if fingerprint in self.seen: raise ProtocolError("behavioral duplicate")
            from .identification import identify_world
            identification=identify_world(node,self.grammar,self.actions,self.initial,action_budget,resource_budget-count,reference_expression)
            count+=identification.evaluations
            if identification.status!=Status.VERIFIED:
                return WorldAudit(Status.PENDING,"identifiability not established within registered scope/budget",fingerprint,witness,count,identification=identification)
            self.seen.add(fingerprint)
            return WorldAudit(Status.VERIFIED, "typed executable world; independent semantics; nontrivial; bounded identifiability certificate",
                              fingerprint, witness, count,identification=identification)
        except (ProtocolError, ValueError, TypeError, KeyError, IndexError) as exc:
            return WorldAudit(Status.INVALID, str(exc), evaluations=count)


class ChallengerInventor:
    def propose(self, grammar, actions, initial, old_models, training, provenance, scientist_hash, budget=256, track="A", novel_beyond=0, selection="full", seed=0):
        if provenance.split != "train" or any(e.task.split != "train" for e in training):
            raise PermissionError("challenge generation is train-only")
        if not old_models: raise ProtocolError("challenger requires current executable hypotheses")
        members = enumerate_programs(grammar, budget).programs
        sequences = tuple((a,) for a in actions) + tuple(product(actions, repeat=2))
        excluded = set()
        if novel_beyond:
            from dataclasses import replace
            for p in enumerate_programs(replace(grammar,max_nodes=novel_beyond,features=()),2048).programs:
                if typecheck(p,dict(grammar.variables)) == grammar.output_type:
                    excluded.add(digest(tuple(rollout(p,s,initial) for s in sequences)))
        scored = []
        witness_contexts = []
        for evidence in training:
            if any(getattr(c.kind, "value", c.kind) == "SPACE_REFUTED" for c in evidence.boundary_certificates):
                witness_contexts.extend((context_for(o, evidence.observations[:i], initial), o.outcome)
                                        for i, o in enumerate(evidence.observations))
        for program in members:
            if typecheck(program, dict(grammar.variables)) != grammar.output_type: continue
            predictions = tuple(rollout(program, s, initial) for s in sequences)
            if digest(predictions) in excluded: continue
            if len({digest(x[-1]) for x in predictions}) < 2: continue
            signatures = tuple(tuple(rollout(m, s, initial) for s in sequences) for m in old_models)
            # Distinguish entire competing explanations over multiple conditions,
            # not a difficulty score against a single answer bit.
            eliminated = sum(any(a != b for a,b in zip(predictions, signature)) for signature in signatures)
            conflicts = sum(sum(a != b for a,b in zip(predictions, signature)) for signature in signatures)
            support = sum(evaluate(program, context) == getattr(outcome, "value", outcome) for context, outcome in witness_contexts)
            scored.append(((eliminated, support, conflicts / max(1, node_size(program)), digest(program)), program))
        if not scored: raise ProtocolError("no discriminative challenge found within search budget")
        if selection=="random_challenger":
            import random
            node=random.Random(seed).choice(sorted((p for _,p in scored),key=digest))
        elif selection=="hardness_first":
            node=max(scored,key=lambda item:(item[0][2]*node_size(item[1]),digest(item[1])))[1]
        else:
            node = max(scored)[1]
        source = canonical({"expression": node, "models": old_models,
                            "scientist_hash": scientist_hash,
                            "evidence": tuple(digest(e) for e in training)})
        artifact = ResearchArtifact("world", "research-expression-v1", "1", source,
                                    Schema.of({"type": "object"}), Schema.of({"type": grammar.output_type}), provenance)
        return ChallengeProposal(provenance.domain_id, artifact, track)
