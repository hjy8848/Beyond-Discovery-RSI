"""Explicit compatibility backend for the pre-M1 finite baseline.

All finite-language assumptions stay here, outside the generic controller and
interaction protocol. This is retained legacy behavior, not M3/M4 invention.
"""
from dataclasses import replace
from ..agents.scientist import Scientist
from ..agents.challenger import Challenger
from .policy_runtime import predict
from ..core.contracts import (Boundary, EvidenceBundle, Provenance, Usage, digest,
                              Snapshot, ModelState, MetaPolicy, ExperimentProgram, RepresentationPatch)


def finite_episode(adapter, task, scientist_snapshot, challenger_snapshot, generation, search_budget=256):
    before_verification = adapter.meter().verification_outputs
    caps = adapter.capabilities()
    scientist = Scientist(scientist_snapshot, caps, search_budget)
    experiment_driver = challenger_snapshot
    if scientist_snapshot.experiment.strategy == "disagreement":
        experiment_driver = replace(challenger_snapshot, experiment=replace(challenger_snapshot.experiment, strategy="disagreement"))
    challenger = Challenger(experiment_driver)
    budget = min(scientist_snapshot.experiment.action_budget, challenger_snapshot.experiment.action_budget)
    session = adapter.begin(task, budget)
    try:
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
                reproduction = adapter.begin(task, 1)
                try:
                    reproduced = adapter.step(reproduction, observation.action) == observation
                finally:
                    adapter.close(reproduction)
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
        return replace(evidence, usage=replace(evidence.usage, verification_outputs=adapter.meter().verification_outputs - before_verification))
    finally:
        adapter.close(session)


def finite_invent(snapshot, diagnosis, capabilities, operator):
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


class FiniteResearchRuntime:
    runtime_id = "finite-baseline-v1"
    languages = ("finite-expression-v1",)

    def initial_pair(self, config):
        c_meta = MetaPolicy(operators=("INVENT_EXPERIMENT", "RECOMBINE", "REFRAME", "RELAX_ASSUMPTION", "ANALOGIZE"))
        return tuple(Snapshot(role, ModelState(base_model=config.base_model, role=role),
                              c_meta if role == "challenger" else MetaPolicy(), ("add",),
                              ExperimentProgram(action_budget=config.action_budget))
                     for role in ("challenger", "scientist"))

    def validate_mode(self, mode):
        if mode not in ("full", "fixed", "program_evolution", "no_restructuring", "no_meta"):
            raise ValueError("unsupported finite baseline mode")

    def context(self, adapter):
        return adapter.capabilities()

    def episode(self, adapter, task, scientist, challenger, generation, search_budget):
        return finite_episode(adapter, task, scientist, challenger, generation, search_budget)

    def propose(self, public_context, challenger, scientist, training, generation):
        return Challenger(challenger).propose(public_context, generation)

    def candidates(self, parent, diagnosis, training, public_context, limit, mode):
        from ..core.evolution import candidates
        return candidates(parent, diagnosis, training, public_context, limit, mode)

    def adapt(self, parent, diagnosis, public_context, meta):
        return finite_invent(parent, diagnosis, public_context, meta.operators[0])
