"""Evidence-driven typed program synthesis and executable feature invention.

Enumerative CEGIS is a genuine local search backend, not a simulated model.
The grammar and compute cap are explicit; no domain oracle enters generation.
"""
from dataclasses import dataclass, replace
from functools import lru_cache
from itertools import product
import json
from .contracts import Provenance, digest, canonical
from .protocol import ProtocolError, ResearchArtifact, Schema
from ..runtime.expression import SIGNATURES, evaluate, freeze, node_size, typecheck, value_type


@dataclass(frozen=True)
class Grammar:
    variables: tuple
    constants: tuple
    operators: tuple
    output_type: str
    max_nodes: int = 3
    features: tuple = ()

    def __post_init__(self):
        if not 1 <= self.max_nodes <= 15 or not set(self.operators) <= set(SIGNATURES):
            raise ProtocolError("invalid synthesis grammar")
        if len(dict(self.variables)) != len(self.variables): raise ProtocolError("duplicate grammar variables")
        for feature in self.features: typecheck(feature, dict(self.variables))


@dataclass(frozen=True)
class Example:
    context: object
    outcome: object
    evidence_hash: str
    split: str = "train"


@dataclass(frozen=True)
class Enumeration:
    programs: tuple
    complete: bool


@lru_cache(maxsize=256)
def enumerate_programs(grammar, budget):
    if type(budget) is not int or budget < 1: raise ProtocolError("invalid synthesis budget")
    layers = {}
    seen = set()
    programs = []
    seeds = tuple(("ref", name) for name, kind in grammar.variables) + tuple(("lit", v) for v in grammar.constants) + grammar.features
    for node in seeds:
        key = digest(node)
        if key in seen: continue
        if len(programs) == budget: return Enumeration(tuple(programs), False)
        kind = typecheck(node, dict(grammar.variables))
        layers.setdefault((1, kind), []).append(node)
        programs.append(node); seen.add(key)
    for cost in range(2, grammar.max_nodes + 1):
        for op in grammar.operators:
            args, output = SIGNATURES[op]
            allocations = [(cost - 1,)] if len(args) == 1 else [(i, cost - 1 - i) for i in range(1, cost - 1)]
            for allocation in allocations:
                choices = [layers.get((size, kind), ()) for size, kind in zip(allocation, args)]
                for children in product(*choices):
                    node = (op,) + tuple(children)
                    if node_size(node) > 63: continue
                    key = digest(node)
                    if key in seen: continue
                    if len(programs) == budget: return Enumeration(tuple(programs), False)
                    layers.setdefault((cost, output), []).append(node)
                    programs.append(node); seen.add(key)
    return Enumeration(tuple(programs), True)


def errors(program, examples):
    try: return sum(evaluate(program, e.context) != e.outcome for e in examples)
    except ProtocolError: return len(examples) + 1


def fit(grammar, examples, budget):
    enumeration = enumerate_programs(grammar, budget)
    programs = [p for p in enumeration.programs if typecheck(p, dict(grammar.variables)) == grammar.output_type]
    return tuple(sorted(programs, key=lambda p: (errors(p, examples), node_size(p), digest(p)))), enumeration.complete


def subtrees(node):
    yield node
    if node[0] not in ("ref", "lit"):
        for child in node[1:]: yield from subtrees(child)


def substitute(node, mapping):
    if node[0] == "ref": return ("ref", mapping.get(node[1], node[1]))
    if node[0] == "lit": return node
    return (node[0],) + tuple(substitute(x, mapping) for x in node[1:])


@dataclass(frozen=True)
class Invention:
    artifact: ResearchArtifact
    operator: str
    parent_hash: str
    evidence_hashes: tuple
    behavior_hash: str
    errors_before: int
    errors_after: int


class ArtifactInventor:
    def __init__(self, grammar, budget):
        self.grammar, self.budget = grammar, budget

    def invent(self, operator, parent_programs, examples, provenance, verified_assets=()):
        if any(e.split != "train" for e in examples) or provenance.split != "train":
            raise PermissionError("artifact invention consumes train evidence only")
        if not examples: return ()
        before = min((errors(p, examples) for p in parent_programs), default=len(examples))
        grammar = self.grammar
        proposals = []
        if operator in ("RELAX_ASSUMPTION", "REFRAME"):
            expanded = replace(grammar, max_nodes=min(15, grammar.max_nodes + 2))
            proposals, _ = fit(expanded, examples, self.budget)
        elif operator == "RECOMBINE":
            pieces = tuple(dict.fromkeys(t for p in parent_programs for t in subtrees(p)))
            for op in grammar.operators:
                args, output = SIGNATURES[op]
                for children in product(pieces, repeat=len(args)):
                    node = (op,) + children
                    try:
                        if typecheck(node, dict(grammar.variables)) == grammar.output_type: proposals.append(node)
                    except ProtocolError: continue
                    if len(proposals) >= self.budget: break
                if len(proposals) >= self.budget: break
        elif operator == "ANALOGIZE":
            # Assets are (executable source, source variable/type mapping,
            # audited train evidence hashes); no asserted natural-language mapping.
            for source, source_vars, evidence in verified_assets:
                if not evidence: continue
                names = [name for name, kind in source_vars]
                choices = [tuple(name for name, kind in grammar.variables if kind == k) for _, k in source_vars]
                for targets in product(*choices):
                    proposals.append(substitute(source, dict(zip(names, targets))))
                    if len(proposals) >= self.budget: break
        else:
            raise ProtocolError("use experiment synthesis for INVENT_EXPERIMENT")
        seen = set()
        results = []
        parent_behaviors = {digest(tuple(evaluate(p, e.context) for e in examples)) for p in parent_programs}
        for node in sorted(proposals, key=lambda p: (errors(p, examples), node_size(p), digest(p))):
            after = errors(node, examples)
            if after >= before: continue
            behavior = digest(tuple(evaluate(node, e.context) for e in examples))
            if behavior in seen or behavior in parent_behaviors: continue
            seen.add(behavior)
            source = canonical({"expression": node, "operator": operator, "evidence": tuple(e.evidence_hash for e in examples)})
            artifact = ResearchArtifact("representation" if operator == "REFRAME" else "hypothesis",
                                        "research-expression-v1", "1", source,
                                        Schema.of({"type": "object"}), Schema.of({"type": grammar.output_type}), provenance)
            results.append(Invention(artifact, operator, digest(parent_programs), tuple(e.evidence_hash for e in examples), behavior, before, after))
        return tuple(results)
