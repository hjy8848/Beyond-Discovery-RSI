"""Fixed coevolution barriers and preregistered selection, independent of domain."""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from statistics import mean

from .archive import Archive
from .contracts import (Decision, ExperimentProgram, MetaPolicy, ModelState, Snapshot,
                        Status, Usage, digest, plain)
from .diagnosis import diagnose
from .evolution import candidates
from .gate import GateConfig, paired_gate
from .interaction import evaluate_pair, invention_rows, rows
from .invention import invent
from ..agents.challenger import Challenger


@dataclass(frozen=True)
class RunConfig:
    generations: int = 2
    candidate_limit: int = 3
    action_budget: int = 8
    search_budget: int = 256
    max_experiments: int = 100000
    max_predictions: int = 10000000
    max_search: int = 1000000
    max_verification_outputs: int = 10000000
    stagnation_patience: int = 2
    seed: int = 42
    mode: str = "full"
    base_model: str = "bounded-enumerator-v1"
    max_model_tokens: int = 100000
    gate: GateConfig = GateConfig()

    def validate(self):
        if not 1 <= self.generations <= 100 or not 1 <= self.candidate_limit <= 5:
            raise ValueError("generation/candidate bounds")
        if not 1 <= self.action_budget <= 16 or not 8 <= self.search_budget <= 512:
            raise ValueError("execution budget bounds")
        if min(self.max_experiments, self.max_predictions, self.max_search, self.max_verification_outputs, self.stagnation_patience, self.max_model_tokens) < 1:
            raise ValueError("invalid stop budget")
        if self.mode not in ("full", "fixed", "program_evolution", "no_restructuring", "no_meta"):
            raise ValueError("unsupported baseline/ablation")
        if not 0 < self.gate.confidence < 1 or self.gate.min_clusters < 2 or self.gate.bootstrap_samples < 100:
            raise ValueError("invalid statistical preregistration")
        if self.gate.minimum_gain < 0 or self.gate.max_regression < 0:
            raise ValueError("invalid gain/regression threshold")


def initial_pair(config):
    c_meta = MetaPolicy(operators=("INVENT_EXPERIMENT", "RECOMBINE", "REFRAME", "RELAX_ASSUMPTION", "ANALOGIZE"))
    return tuple(Snapshot(role, ModelState(base_model=config.base_model, role=role), c_meta if role == "challenger" else MetaPolicy(), ("add",),
                          ExperimentProgram(action_budget=config.action_budget))
                 for role in ("challenger", "scientist"))


def implementation_hash():
    root = Path(__file__).resolve().parents[1]
    return digest({str(p.relative_to(root)): p.read_text() for p in sorted(root.rglob("*.py"))})


def run_rsi(adapter, out, config=None, proposer=None):
    config = config or RunConfig()
    config.validate()
    caps = adapter.capabilities()
    if not {"A", "B-bounded-DSL"} <= set(caps.get("world_creation", ())):
        raise ValueError("CAPABILITY_UNSUPPORTED: both certification tracks required")
    challenger, scientist = initial_pair(config)
    archive = Archive(out)
    # Freeze configuration, exact task membership and implementation before runs.
    train = adapter.sample("train", config.seed)
    dev = adapter.sample("dev", config.seed + 1)
    prereg = {"config": config, "capabilities": caps, "implementation_hash": implementation_hash(),
              "train": train, "dev": dev, "initial_challenger": challenger,
              "initial_scientist": scientist, "joint_fallback": "rollback-both",
              "selection_order": "first-ACCEPT-in-predeclared-candidate-order",
              "meta_protocol": "one operator dispatch, fixed parent representation and fixed dev opportunities",
              "evidence_status": "synthetic program-only engineering run"}
    archive.marker("preregistration.json", prereg)
    archive.append("preregistration", prereg)
    usage = Usage()
    history = []
    method_signatures = set()
    stale = 0

    def evaluate(tasks, s, c, generation, label):
        nonlocal usage
        # Reserve worst-case work BEFORE touching tasks, including reproduction.
        budget = min(s.experiment.action_budget, c.experiment.action_budget)
        reserve = Usage(experiments=2 * budget * len(tasks),
                        predictions=4 * len(caps["input_space"]) * len(tasks),
                        search=config.search_budget * len(tasks),
                        verification_outputs=(2 * len(caps["input_space"]) + 4 * budget) * len(tasks))
        bound = usage + reserve
        if (bound.experiments > config.max_experiments or bound.predictions > config.max_predictions
                or bound.search > config.max_search or bound.verification_outputs > config.max_verification_outputs):
            archive.append("budget_stop", {"usage": usage, "reservation": reserve, "label": label})
            raise BudgetStop("preregistered execution budget exhausted")
        before_verification = adapter.verification_outputs
        evidence = evaluate_pair(adapter, tasks, s, c, generation, config.search_budget, archive, label)
        for e in evidence:
            usage = usage + e.usage
        usage = usage + Usage(verification_outputs=adapter.verification_outputs - before_verification)
        return evidence

    try:
        for generation in range(config.generations):
            # Save these parents; no assignment below can change frozen opponents.
            c_parent, s_parent = challenger, scientist
            training = evaluate(train, s_parent, c_parent, generation, "train")
            proposal = Challenger(c_parent).propose(caps, generation)
            if usage.verification_outputs + adapter.certification_budget() > config.max_verification_outputs:
                raise BudgetStop("world certification budget exhausted")
            before_verification = adapter.verification_outputs
            certificate = adapter.compile_challenge(proposal)
            usage = usage + Usage(verification_outputs=adapter.verification_outputs - before_verification)
            archive.append("world_attempt", {"proposal": proposal, "certification": certificate})
            if certificate.status == Status.VERIFIED:
                training += evaluate((certificate.task,), s_parent, c_parent, generation, "generated-train")
            diagnosis = diagnose(s_parent.meta, training)
            if proposer is None:
                c_attempts = candidates(c_parent, diagnosis, training, caps, config.candidate_limit, config.mode)
                s_attempts = candidates(s_parent, diagnosis, training, caps, config.candidate_limit, config.mode)
            else:
                before_tokens = proposer.total_tokens
                before_calls = len(proposer.calls)
                try:
                    c_attempts = proposer.propose(c_parent, diagnosis, training, caps, config.candidate_limit, config.mode)
                    s_attempts = proposer.propose(s_parent, diagnosis, training, caps, config.candidate_limit, config.mode)
                except Exception as exc:
                    archive.append("model_generation_failure", {"type": type(exc).__name__, "reason": str(exc)})
                    raise
                finally:
                    for call in proposer.calls[before_calls:]:
                        archive.append("model_call", call)
                    usage = usage + Usage(model_tokens=proposer.total_tokens - before_tokens)
                if usage.model_tokens > config.max_model_tokens:
                    raise BudgetStop("reported model-token budget exceeded; no candidate selected")
            # Stage barrier: both candidate lists persisted before any private dev.
            archive.append("candidate_barrier", {"generation": generation, "parents": (c_parent, s_parent),
                                                  "diagnosis": diagnosis, "challenger": c_attempts, "scientist": s_attempts})
            baseline = evaluate(dev, s_parent, c_parent, generation, "dev-parent")
            selected = {"challenger": c_parent, "scientist": s_parent}
            decisions = []
            for role, parent, attempts in (("challenger", c_parent, c_attempts), ("scientist", s_parent, s_attempts)):
                for index, (child, patch) in enumerate(attempts):
                    method_signature = None
                    s_eval = child if role == "scientist" else s_parent
                    c_eval = child if role == "challenger" else c_parent
                    observed = evaluate(dev, s_eval, c_eval, generation, "dev-%s-%s" % (role, index))
                    gate = paired_gate(rows(baseline, role), rows(observed, role), config.gate,
                                       config.seed + generation, len(attempts))
                    # Refit each candidate on TRAIN only for controlled behavioral
                    # novelty and improvement. Dev never becomes invention input.
                    novelty = False
                    if role == "scientist" and gate.decision == Decision.ACCEPT:
                        method_signature = digest(tuple((e.task.task_id, e.behavior_hash) for e in observed))
                        diagnostic = evaluate(train, child, c_parent, generation, "train-novelty-%s" % index)
                        novelty = any(old.score.value < new.score.value and old.behavior_hash != new.behavior_hash
                                      for old, new in zip(training[:len(train)], diagnostic))
                        if not novelty:
                            gate = replace(gate, decision=Decision.INCONCLUSIVE, reason="no independently effective behavioral novelty on old failures")
                    # Meta attribution: only replace M, hold task program/weights
                    # and adaptation budget fixed. Compare inventions it produces.
                    meta_gate = None
                    if child.meta != parent.meta and gate.decision == Decision.ACCEPT:
                        old_adapt, _ = invent(parent, diagnosis, caps, parent.meta.operators[0])
                        new_adapt, _ = invent(parent, diagnosis, caps, child.meta.operators[0])
                        old_e = evaluate(dev, old_adapt if role == "scientist" else s_parent,
                                         old_adapt if role == "challenger" else c_parent, generation, "meta-parent")
                        new_e = evaluate(dev, new_adapt if role == "scientist" else s_parent,
                                         new_adapt if role == "challenger" else c_parent, generation, "meta-candidate")
                        meta_gate = paired_gate(invention_rows(baseline, old_e, role), invention_rows(baseline, new_e, role), config.gate,
                                                config.seed + generation, len(attempts))
                        if meta_gate.decision != Decision.ACCEPT:
                            child = replace(child, meta=parent.meta)
                            # Restoring M may alter search/execution behavior.
                            # Recheck the exact version that can be inherited.
                            restored_e = evaluate(dev, child if role == "scientist" else s_parent,
                                                  child if role == "challenger" else c_parent,
                                                  generation, "dev-restored-meta-%s-%s" % (role, index))
                            gate = paired_gate(rows(baseline, role), rows(restored_e, role), config.gate,
                                               config.seed + generation, len(attempts))
                            if role == "scientist" and gate.decision == Decision.ACCEPT:
                                method_signature = digest(tuple((e.task.task_id, e.behavior_hash) for e in restored_e))
                                restored_train = evaluate(train, child, c_parent, generation, "train-restored-meta-novelty-%s" % index)
                                novelty = any(old.score.value < new.score.value and old.behavior_hash != new.behavior_hash
                                              for old, new in zip(training[:len(train)], restored_train))
                                if not novelty:
                                    gate = replace(gate, decision=Decision.INCONCLUSIVE, reason="restored meta lost effective behavioral novelty")
                    effective_new = novelty and gate.decision == Decision.ACCEPT and method_signature not in method_signatures
                    if effective_new:
                        method_signatures.add(method_signature)
                    archive.append("gate", {"generation": generation, "role": role, "index": index,
                                            "candidate": child, "patch": patch, "task_gate": gate,
                                            "meta_gate": meta_gate, "method_behavior_signature": method_signature,
                                            "effective_new_method": effective_new})
                    decisions.append((role, plain(gate), plain(meta_gate)))
                    if gate.decision == Decision.ACCEPT and selected[role] == parent:
                        selected[role] = child
            c_next, s_next = selected["challenger"], selected["scientist"]
            # Periodic four-way cross-pairing, fixed common audit pool, no updates.
            c_only = evaluate(dev, s_parent, c_next, generation, "joint-c-only")
            s_only = evaluate(dev, s_next, c_parent, generation, "joint-s-only")
            combined = evaluate(dev, s_next, c_next, generation, "joint-both")
            def performance(es): return mean(e.score.value for e in es)
            interaction = performance(combined) - performance(c_only) - performance(s_only) + performance(baseline)
            # Predeclared compatibility: quality may not regress versus parent;
            # accepted C boundary yield need not persist after S fixes that boundary.
            compatible = all(e.score.valid for e in combined) and all(
                new.score.value >= old.score.value - config.gate.max_regression
                for old, new in zip(baseline, combined))
            challenger, scientist = (c_next, s_next) if compatible else (c_parent, s_parent)
            changed = (challenger != c_parent or scientist != s_parent)
            stale = 0 if changed else stale + 1
            record = {"generation": generation, "challenger_hash": challenger.content_hash,
                      "scientist_hash": scientist.content_hash, "joint_compatible": compatible,
                      "interaction": interaction, "decisions": decisions, "usage": usage}
            archive.append("commit", {**record, "challenger": challenger, "scientist": scientist})
            history.append(plain(record))
            if stale >= config.stagnation_patience:
                break
        stop_reason = "stagnation" if stale >= config.stagnation_patience else "generation limit"
    except BudgetStop as exc:
        stop_reason = str(exc)

    frozen = {"challenger": challenger, "scientist": scientist, "implementation_hash": implementation_hash(),
              "stop_reason": stop_reason, "journal_head": archive.head}
    archive.marker("frozen.json", frozen)
    archive.append("freeze", frozen)
    # Marker written BEFORE unsealing. Interrupted final evaluation cannot retry
    # in the same output directory; training/selection never uses this result.
    archive.marker("final_started.json", {"frozen_hash": digest(frozen)})
    adapter.unseal_for_final()
    test = adapter.sample("test", config.seed + 2)
    try:
        final = evaluate(test, scientist, challenger, -1, "sealed-final-test")
        final_score = mean(e.score.value for e in final)
        final_status = "COMPLETE"
    except BudgetStop:
        final_score = None
        final_status = "BUDGET_EXHAUSTED"
    summary = {"domain": adapter.domain_id, "status": final_status, "stop_reason": stop_reason,
               "test_score": final_score, "generations": history, "usage": usage,
               "challenger": challenger, "scientist": scientist,
               "limitations": ["synthetic bounded DSL and deterministic enumerative agents",
                               "no API calls, weight training, GPU measurement or external benchmark result",
                               "bootstrap with few clusters is an engineering gate, not paper-level evidence"]}
    archive.append("summary", summary)
    archive.marker("summary.json", summary)
    return plain(summary)


class BudgetStop(RuntimeError):
    pass
