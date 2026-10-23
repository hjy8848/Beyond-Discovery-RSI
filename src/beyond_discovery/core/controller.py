"""Fixed coevolution barriers and preregistered selection, independent of domain."""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from statistics import mean

from .archive import Archive
from .contracts import Decision, Status, Usage, digest, plain
from .diagnosis import diagnose
from .gate import GateConfig, paired_gate
from .interaction import evaluate_pair, invention_rows, rows
from .protocol import CapabilityUnsupported, ProtocolError, checked_sample, negotiate, usage_within
from ..runtime.research import resolve_runtime


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
    max_training_steps: int = 1000
    gate: GateConfig = GateConfig()
    required_world_tracks: tuple = ("A", "B")

    def validate(self):
        if not 1 <= self.generations <= 100 or not 1 <= self.candidate_limit <= 5:
            raise ValueError("generation/candidate bounds")
        if type(self.action_budget) is not int or self.action_budget < 1 or type(self.search_budget) is not int or self.search_budget < 1:
            raise ValueError("execution budget bounds")
        if min(self.max_experiments, self.max_predictions, self.max_search, self.max_verification_outputs, self.stagnation_patience, self.max_model_tokens, self.max_training_steps) < 1:
            raise ValueError("invalid stop budget")
        if self.mode not in ("full", "fixed", "fixed_challenger", "program_evolution", "no_restructuring", "no_meta", "no_diagnosis", "no_falsification", "reflection", "random_challenger", "hardness_first"):
            raise ValueError("unsupported baseline/ablation")
        if not 0 < self.gate.confidence < 1 or self.gate.min_clusters < 2 or self.gate.bootstrap_samples < 100:
            raise ValueError("invalid statistical preregistration")
        if self.gate.minimum_gain < 0 or self.gate.max_regression < 0:
            raise ValueError("invalid gain/regression threshold")


def initial_pair(config):
    """Compatibility entry point for existing finite-baseline callers."""
    from ..runtime.finite_research import FiniteResearchRuntime
    return FiniteResearchRuntime().initial_pair(config)


def implementation_hash():
    root = Path(__file__).resolve().parents[1]
    return digest({str(p.relative_to(root)): p.read_text() for p in sorted(root.rglob("*.py"))})


def run_rsi(adapter, out, config=None, proposer=None, runtime=None):
    from ..runtime.sandbox import ResearchSandbox
    runtime=runtime or resolve_runtime(adapter)
    if "research-expression-v1" in runtime.languages:
        with ResearchSandbox():
            return _run_rsi(adapter,out,config,proposer,runtime)
    return _run_rsi(adapter,out,config,proposer,runtime)


def _run_rsi(adapter, out, config=None, proposer=None, runtime=None):
    config = config or RunConfig()
    config.validate()
    caps = adapter.describe()
    runtime = runtime or resolve_runtime(adapter)
    negotiated = negotiate(caps, runtime.languages, config.action_budget, config.required_world_tracks)
    runtime.validate_mode(config.mode)
    if getattr(runtime,"total_model_tokens",0)>config.max_model_tokens:
        raise ValueError("model ledger already exceeds preregistered budget")
    for client in getattr(runtime,"clients",{}).values():
        if client.ledger.limit>config.max_model_tokens: raise ValueError("model ledger exceeds controller cap")
    if getattr(runtime,"training_config",None) and runtime.training_config['max_steps']>config.max_training_steps:
        raise ValueError("training backend budget exceeds controller cap")
    challenger, scientist = runtime.initial_pair(config)
    public_context = runtime.context(adapter)
    # Freeze configuration, exact task membership and implementation before runs.
    train = checked_sample(adapter, "train", config.seed)
    dev = checked_sample(adapter, "dev", config.seed + 1)
    if {t.task_id for t in train} & {t.task_id for t in dev}:
        raise ProtocolError("train/dev task partitions overlap")
    archive = Archive(out)
    from ..runtime.sandbox import executor
    sandbox=executor.get()
    sandbox_limits=None if sandbox is None else {'memory_bytes':sandbox.memory,'memory_mode':sandbox.memory_mode,
                                                 'cpu_seconds':sandbox.cpu,'wall_seconds':sandbox.wall,'call_timeout':sandbox.timeout}
    prereg = {"sandbox_limits":sandbox_limits,"config": config, "capabilities": caps, "negotiated_protocol": negotiated,
              "runtime_id": runtime.runtime_id, "implementation_hash": implementation_hash(),
              "train": train, "dev": dev, "initial_challenger": challenger,
              "initial_scientist": scientist, "joint_fallback": "rollback-both",
              "selection_order": "first-ACCEPT-in-predeclared-candidate-order",
              "meta_protocol": "one operator dispatch, fixed parent representation and fixed dev opportunities",
              "evidence_status": "engineering protocol execution; no scientific result implied"}
    archive.marker("preregistration.json", prereg)
    archive.append("preregistration", prereg)
    usage = Usage()
    history = []
    method_signatures = set()
    stale = 0

    runtime_cost = Usage()
    event_offsets = {"events": 0, "model_events": 0}

    def flush_runtime():
        nonlocal usage, runtime_cost
        current = Usage(search=getattr(runtime,"total_search",0),
                        model_tokens=getattr(runtime,"total_model_tokens",0),
                        training_steps=getattr(runtime,"total_training_steps",0))
        delta = Usage(*(getattr(current,n)-getattr(runtime_cost,n) for n in current.__dataclass_fields__))
        usage = usage + delta
        runtime_cost = current
        for name in event_offsets:
            events=getattr(runtime,name,())
            for event in events[event_offsets[name]:]: archive.append("runtime_"+name,event)
            event_offsets[name]=len(events)
        if usage.search>config.max_search or usage.model_tokens>config.max_model_tokens or usage.training_steps>config.max_training_steps:
            raise BudgetStop("generation resource budget exhausted")

    def charge_prediction():
        nonlocal usage
        if usage.predictions>=config.max_predictions: raise BudgetStop("generation prediction budget exhausted")
        usage=usage+Usage(predictions=1)

    def generate(call):
        from ..runtime.expression import prediction_meter
        token=prediction_meter.set(charge_prediction)
        try: return call()
        except Exception as exc:
            archive.append("generation_failure",{"error_type":type(exc).__name__})
            raise
        finally:
            prediction_meter.reset(token)
            flush_runtime()

    def reserve_search(amount):
        if usage.search+getattr(runtime,"total_search",0)-runtime_cost.search+amount>config.max_search: raise BudgetStop("generation search budget exhausted")
    runtime.reserve_search=reserve_search

    def evaluate(tasks, s, c, generation, label):
        nonlocal usage
        if digest(adapter.describe()) != negotiated.capabilities_hash:
            raise ProtocolError("domain capabilities changed after preregistration")
        # Reserve worst-case work BEFORE touching tasks, including reproduction.
        budget = min(s.experiment.action_budget, c.experiment.action_budget)
        one = adapter.estimate_episode(budget, config.search_budget)
        reserve = Usage(*(getattr(one, name) * len(tasks) for name in one.__dataclass_fields__))
        limit = Usage(config.max_experiments, config.max_predictions, config.max_search,
                      config.max_model_tokens, config.max_training_steps, config.max_verification_outputs)
        if not usage_within(usage + reserve, limit):
            archive.append("budget_stop", {"usage": usage, "reservation": reserve, "label": label})
            raise BudgetStop("preregistered execution budget exhausted")
        from ..runtime.expression import prediction_meter
        token=prediction_meter.set(None)
        try:
            evidence = evaluate_pair(adapter, tasks, s, c, generation, config.search_budget, archive, label, runtime=runtime)
        finally: prediction_meter.reset(token)
        for e in evidence:
            usage = usage + e.usage
        return evidence

    try:
        for generation in range(config.generations):
            # Save these parents; no assignment below can change frozen opponents.
            c_parent, s_parent = challenger, scientist
            if "frozen-challenge-binding" in caps.features:adapter.bind_frozen_scientist(s_parent.content_hash)
            training = evaluate(train, s_parent, c_parent, generation, "train")
            runtime.mode=config.mode
            runtime.seed=config.seed
            proposal = generate(lambda: runtime.propose(public_context, c_parent, s_parent, training, generation))
            if proposal is not None:
                reserved = adapter.estimate_certification()
                limit = Usage(config.max_experiments, config.max_predictions, config.max_search,
                              config.max_model_tokens, config.max_training_steps, config.max_verification_outputs)
                if not usage_within(usage + reserved, limit):
                    raise BudgetStop("world certification budget exhausted")
                before = adapter.meter()
                certificate = adapter.compile_challenge(proposal, **({"action_budget":config.action_budget} if "world-budget-certification" in caps.features else {}))
                after = adapter.meter()
                cost = Usage(*(getattr(after, name) - getattr(before, name) for name in before.__dataclass_fields__))
                if not usage_within(cost, reserved):
                    raise ProtocolError("certification exceeded declared reservation")
                usage = usage + cost
                archive.append("world_attempt", {"proposal": proposal, "certification": certificate, "usage": cost})
                if certificate.status == Status.VERIFIED:
                    if certificate.task is None or certificate.task.split != "train" or certificate.task.domain_id != adapter.domain_id:
                        raise ProtocolError("certified generated worlds must remain in the train partition")
                    training += evaluate((certificate.task,), s_parent, c_parent, generation, "generated-train")
            elif config.required_world_tracks and not getattr(runtime,"supports_world_creation",False):
                raise CapabilityUnsupported("runtime did not implement required world creation")
            else:
                archive.append("world_search_exhausted", {"feature": "world-creation", "status": "NO_VALID_PROPOSAL"})
            diagnosis = diagnose(s_parent.meta, training)
            if proposer is None:
                c_attempts = generate(lambda: runtime.candidates(c_parent, diagnosis, training, public_context, config.candidate_limit, config.mode))
                s_attempts = generate(lambda: runtime.candidates(s_parent, diagnosis, training, public_context, config.candidate_limit, config.mode))
            else:
                before_tokens = proposer.total_tokens
                before_calls = len(proposer.calls)
                try:
                    c_attempts = proposer.propose(c_parent, diagnosis, training, public_context, config.candidate_limit, config.mode)
                    s_attempts = proposer.propose(s_parent, diagnosis, training, public_context, config.candidate_limit, config.mode)
                except Exception as exc:
                    archive.append("model_generation_failure", {"type": type(exc).__name__, "reason": str(exc)})
                    raise
                finally:
                    for call in proposer.calls[before_calls:]:
                        archive.append("model_call", call)
                    usage = usage + Usage(model_tokens=proposer.total_tokens - before_tokens)
                if usage.model_tokens > config.max_model_tokens:
                    raise BudgetStop("reported model-token budget exceeded; no candidate selected")
            if hasattr(runtime,"trained_candidates"):
                c_attempts=generate(lambda: runtime.trained_candidates(c_parent,c_attempts,training,
                    lambda child: evaluate(train,s_parent,child,generation,"train-model-reward-c"), Path(out)/"checkpoints"/str(generation)))
                s_attempts=generate(lambda: runtime.trained_candidates(s_parent,s_attempts,training,
                    lambda child: evaluate(train,child,c_parent,generation,"train-model-reward-s"), Path(out)/"checkpoints"/str(generation)))
            # Stage barrier: both candidate lists persisted before any private dev.
            archive.append("candidate_barrier", {"generation": generation, "parents": (c_parent, s_parent),
                                                  "diagnosis": diagnosis, "challenger": c_attempts, "scientist": s_attempts})
            baseline = evaluate(dev, s_parent, c_parent, generation, "dev-parent")
            selected = {"challenger": c_parent, "scientist": s_parent}
            decisions = []
            for role, parent, attempts in (("challenger", c_parent, c_attempts), ("scientist", s_parent, s_attempts)):
                for index, (child, patch) in enumerate(attempts):
                    proposed_meta=child.meta
                    method_signature = None
                    s_eval = child if role == "scientist" else s_parent
                    c_eval = child if role == "challenger" else c_parent
                    observed = evaluate(dev, s_eval, c_eval, generation, "dev-%s-%s" % (role, index))
                    control=patch.get('weight_control') if isinstance(patch,dict) else None
                    gate_baseline=baseline
                    if control is not None:
                        gate_baseline=evaluate(dev,control if role=="scientist" else s_parent,
                                               control if role=="challenger" else c_parent,generation,"dev-fixed-program-weight-control")
                    gate = paired_gate(rows(gate_baseline, role), rows(observed, role), config.gate,
                                       config.seed + generation, len(attempts))
                    # Refit each candidate on TRAIN only for controlled behavioral
                    # novelty and improvement. Dev never becomes invention input.
                    novelty = False
                    if role == "scientist" and gate.decision == Decision.ACCEPT:
                        method_signature = digest(tuple((e.task.task_id, e.behavior_hash) for e in observed))
                        diagnostic = evaluate(train, child, c_parent, generation, "train-novelty-%s" % index)
                        novelty_baseline=training[:len(train)] if control is None else evaluate(train,control,c_parent,generation,"train-fixed-program-weight-control")
                        novelty = any(old.score.value < new.score.value and old.behavior_hash != new.behavior_hash
                                      for old, new in zip(novelty_baseline, diagnostic))
                        if not novelty:
                            gate = replace(gate, decision=Decision.INCONCLUSIVE, reason="no independently effective behavioral novelty on old failures")
                    # Meta attribution: only replace M, hold task program/weights
                    # and adaptation budget fixed. Compare inventions it produces.
                    meta_gate = None
                    if child.meta != parent.meta and (gate.decision == Decision.ACCEPT or getattr(runtime,"separate_meta",False)):
                        old_adapt, _ = generate(lambda: runtime.adapt(parent, diagnosis, public_context, parent.meta))
                        new_adapt, _ = generate(lambda: runtime.adapt(parent, diagnosis, public_context, child.meta))
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
                    from .meta import source_diff
                    archive.append("gate", {"generation": generation, "role": role, "index": index,
                                            "candidate": child, "patch": patch, "task_gate": gate,
                                            "meta_gate": meta_gate, "method_behavior_signature": method_signature,
                                            "proposed_meta":proposed_meta,"meta_rollback":proposed_meta!=child.meta,
                                            "meta_source_diff":source_diff(parent.meta.program,proposed_meta.program),
                                            "effective_new_method": effective_new})
                    decisions.append((role, plain(gate), plain(meta_gate)))
                    if gate.decision == Decision.ACCEPT and selected[role] == parent:
                        selected[role] = child
                    elif meta_gate is not None and meta_gate.decision == Decision.ACCEPT and selected[role] == parent:
                        selected[role] = replace(parent,meta=child.meta,generation=parent.generation+1,parent_hash=parent.content_hash)
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
    test = checked_sample(adapter, "test", config.seed + 2)
    if {t.task_id for t in test} & {t.task_id for t in train + dev}:
        raise ProtocolError("sealed test overlaps training/development tasks")
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
               "protocol": negotiated, "runtime_id": runtime.runtime_id,
               "limitations": list(caps.limitations) + [
                   "H1-H4 require separate preregistered scientific experiments",
                   "bootstrap with few clusters is an engineering gate, not paper-level evidence"]}
    archive.append("summary", summary)
    archive.marker("summary.json", summary)
    return plain(summary)


class BudgetStop(RuntimeError):
    pass
