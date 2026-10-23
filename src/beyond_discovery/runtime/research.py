"""Executable structured research programs; no domain-specific problem solver.

M1 implements a bounded dataflow language (calls to negotiated primitives and
returns with references to earlier observations). Invention of these programs
belongs to M3/M4. Supplied programs are explicitly authored assets, never dummy
model output. Unsupported generation modes fail before a run is created.
"""
from __future__ import annotations

from dataclasses import replace
import json
from ..core.contracts import (Claim, EvidenceBundle, ExperimentProgram, MetaPolicy,
                              ModelState, Provenance, Snapshot, Usage, digest, plain)
from ..core.protocol import (Action, CapabilityUnsupported, JSONValue, ProtocolError,
                             ResearchArtifact, Schema)


LANGUAGE = "research-workflow-v1"


def _strict_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result: raise ProtocolError("duplicate source key")
        result[key] = value
    return result


def compile_workflow(artifact, instruction_budget):
    if artifact.language != LANGUAGE: raise CapabilityUnsupported("unknown artifact language")
    try:
        program = json.loads(artifact.source, object_pairs_hook=_strict_object)
    except (ValueError, RecursionError) as exc:
        raise ProtocolError("invalid artifact source: " + str(exc)) from exc
    JSONValue.of(program)
    if not isinstance(program, list) or not 1 <= len(program) <= min(512, instruction_budget):
        raise ProtocolError("artifact instruction budget exceeded")
    for instruction in program:
        if not isinstance(instruction, dict) or len(instruction) != 1 or not set(instruction) <= {"call", "return"}:
            raise ProtocolError("unsupported artifact instruction")
        if "call" in instruction:
            action = instruction["call"]
            if not isinstance(action, dict) or set(action) != {"operation", "arguments"} or type(action["operation"]) is not str:
                raise ProtocolError("invalid primitive call")
    if set(program[-1]) != {"return"} or any("return" in x for x in program[:-1]):
        raise ProtocolError("program must return exactly once at the end")
    return program


def _resolve(value, context, depth=0):
    if depth > 32: raise ProtocolError("artifact expression nesting exceeded")
    if isinstance(value, dict):
        if "$ref" in value:
            if set(value) != {"$ref"} or type(value["$ref"]) is not str:
                raise ProtocolError("invalid data reference")
            parts = value["$ref"].split(".")
            if parts[0] not in {"initial", "last", "history", "task"}:
                raise ProtocolError("reference outside agent-visible context")
            result = context
            try:
                for part in parts:
                    if isinstance(result, list): result = result[int(part)]
                    elif isinstance(result, dict): result = result[part]
                    else: raise KeyError(part)
            except (KeyError, IndexError, ValueError) as exc:
                raise ProtocolError("unresolved observation reference") from exc
            return JSONValue.of(result).value
        return {k: _resolve(v, context, depth + 1) for k, v in value.items()}
    if isinstance(value, list): return [_resolve(v, context, depth + 1) for v in value]
    return value


def execute_workflow(artifact, context, call, instruction_budget=256):
    artifact.input_schema.validate(context)
    program = compile_workflow(artifact, instruction_budget)
    context = JSONValue.of(context).value
    for instruction in program:
        if "call" in instruction:
            raw = _resolve(instruction["call"], context)
            result = call(Action.create(raw["operation"], raw["arguments"]))
            context["last"] = plain(result.outcome)
            context.setdefault("history", []).append(plain(result.outcome))
        else:
            result = JSONValue.of(_resolve(instruction["return"], context))
            artifact.output_schema.validate(result)
            return result, len(program)
    raise ProtocolError("artifact did not return")


def execute_result(artifact):
    if artifact.kind != "result": raise ProtocolError("expected result artifact")
    def no_tools(action): raise ProtocolError("result artifact has no experiment tool authority")
    result, _ = execute_workflow(artifact, {}, no_tools)
    return result


class ArtifactResearchRuntime:
    runtime_id = "artifact-dataflow-v1"
    languages = (LANGUAGE,)

    def __init__(self, experiment=None):
        self.experiment = experiment

    def validate_mode(self, mode):
        if mode != "fixed":
            raise CapabilityUnsupported("artifact invention not yet implemented in M1; use explicitly authored fixed programs or an invention backend")

    def context(self, adapter):
        return plain(adapter.describe())

    def initial_pair(self, config):
        if self.experiment is None: raise CapabilityUnsupported("initial executable research artifact required")
        if self.experiment.provenance.split != "train":
            raise PermissionError("initial agent artifact must originate from train")
        return tuple(Snapshot(role, ModelState(config.base_model, role), MetaPolicy(), (),
                              ExperimentProgram("artifact", config.action_budget), artifacts=(self.experiment,))
                     for role in ("challenger", "scientist"))

    def propose(self, public_context, challenger, scientist, training, generation):
        # Explicit fixed-program mode does not invent worlds. Negotiation must
        # have recorded a scope limitation; full RSI requirements reject it.
        return None

    def candidates(self, parent, diagnosis, training, public_context, limit, mode):
        if any(e.task.split != "train" for e in training): raise PermissionError("train evidence required")
        self.validate_mode(mode)
        return ()

    def adapt(self, parent, diagnosis, public_context, meta):
        raise CapabilityUnsupported("M3/M5 artifact invention required for meta adaptation")

    def episode(self, adapter, task, scientist, challenger, generation, search_budget):
        if not scientist.artifacts: raise CapabilityUnsupported("snapshot has no executable experiment artifact")
        artifact = scientist.artifacts[0]
        if artifact.kind != "experiment" or artifact.provenance.domain_id != adapter.domain_id:
            raise ProtocolError("artifact kind/domain mismatch")
        if artifact.provenance.split != "train": raise PermissionError("dev/test artifact cannot enter an agent")
        caps = adapter.describe()
        budget = min(scientist.experiment.action_budget, challenger.experiment.action_budget)
        before = adapter.meter()
        session = adapter.begin(task, budget)
        observations = []
        try:
            initial = adapter.initial_observation(session)
            caps.observation_schema.validate(initial)
            def call(action):
                if len(observations) >= budget: raise ProtocolError("experiment action budget exceeded")
                if observations and observations[-1].terminal: raise ProtocolError("action after terminal observation")
                caps.action_schema.validate(action)
                observation = adapter.step(session, action)
                if observation.action != action or observation.sequence != len(observations) + 1:
                    raise ProtocolError("adapter returned a mismatched/out-of-order observation")
                if type(observation.terminal) is not bool:
                    raise ProtocolError("adapter terminal flag must be boolean")
                caps.observation_schema.validate(observation.outcome)
                observations.append(observation)
                return observation
            result, instructions = execute_workflow(artifact, {"task": plain(task), "initial": plain(initial)}, call, search_budget)
            caps.result_schema.validate(result)
        finally:
            adapter.close(session)
        provenance = Provenance(adapter.domain_id, task.split, "artifact-execution", generation,
                                scientist.content_hash, task.seed, adapter.verifier_version, scientist.parent_hash)
        submission = ResearchArtifact("result", LANGUAGE, artifact.version,
                                     json.dumps([{"return": result.value}]), Schema.of({"type": "object"}),
                                     artifact.output_schema, provenance)
        claim = Claim(submission, "executable research artifact " + artifact.content_hash)
        score = adapter.evaluate(claim, task, task.split)
        evidence = EvidenceBundle(provenance, task, tuple(observations), claim, claim, score, (),
                                  digest(result), Usage())
        # Audit is trusted, never delegated to a candidate or its supplied output.
        if not adapter.audit(evidence): raise ProtocolError("independent evidence audit failed")
        after = adapter.meter()
        delta = Usage(*(getattr(after, name) - getattr(before, name) for name in before.__dataclass_fields__))
        if any(getattr(delta, name) < 0 for name in delta.__dataclass_fields__): raise ProtocolError("adapter resource meter moved backwards")
        return replace(evidence, usage=delta + Usage(search=instructions))


def resolve_runtime(adapter, runtime_id=None):
    from .finite_research import FiniteResearchRuntime
    from .synthesis import SynthesisRuntime
    from .coding import CodingResearchRuntime
    available = (FiniteResearchRuntime, ArtifactResearchRuntime, SynthesisRuntime,CodingResearchRuntime)
    caps = adapter.describe()
    for cls in available:
        if (runtime_id is None or cls.runtime_id == runtime_id) and set(cls.languages) & set(caps.artifact_languages):
            if cls is CodingResearchRuntime:return cls(replay_only=True)
            return cls(adapter.public_context()) if cls is SynthesisRuntime else cls()
    raise CapabilityUnsupported("no registered research runtime for domain artifact languages")
