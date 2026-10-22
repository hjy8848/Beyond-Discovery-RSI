"""Creative operators dispatch into executable representation/experiment patches.

These small operators are explicit baseline heuristics, not an LLM or a claim of
autonomous scientific creativity. All domain-independent choices live here.
"""
from dataclasses import replace
from .contracts import RepresentationPatch


def invent(snapshot, diagnosis, capabilities, operator):
    available = tuple(x for x in capabilities["expression_ops"] if x not in snapshot.representation)
    representation = snapshot.representation
    experiment = snapshot.experiment
    if operator == "RECOMBINE":
        # Combine one additional sanctioned semantic primitive with the grammar.
        representation += available[:1]
    elif operator == "REFRAME":
        representation += available
    elif operator == "RELAX_ASSUMPTION":
        representation += available[-1:]
    elif operator == "ANALOGIZE":
        # Transfer an operator mapping recorded in verified TRAIN assets only.
        mappings = [x for x in available if any(("operator:" + x) == k for k in snapshot.knowledge)]
        representation += tuple(mappings[:1])
    elif operator == "INVENT_EXPERIMENT":
        experiment = replace(experiment, strategy="disagreement")
    else:
        raise ValueError("unknown invention operator")
    patch = RepresentationPatch(operator, snapshot.representation, representation,
                                "train conflict/plateau diagnosis: " + diagnosis.action)
    return replace(snapshot, representation=representation, experiment=experiment), patch
