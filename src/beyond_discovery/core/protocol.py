"""Domain Protocol 1.1: typed wire values, executable artifacts and negotiation.

No domain grammar, enumerated action universe or numeric observation assumption
is part of this module. Schemas use an explicitly supported JSON Schema subset;
unknown constraints are rejected rather than silently ignored.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Any, Protocol, Tuple
from .contracts import Provenance, canonical, digest, plain


PROTOCOL_VERSION = "1.1"
MAX_WIRE_BYTES = 1_000_000


class ProtocolError(ValueError):
    pass


class CapabilityUnsupported(ProtocolError):
    def __init__(self, reason):
        super().__init__("CAPABILITY_UNSUPPORTED: " + reason)


def _json_check(value, depth=0):
    if depth > 32:
        raise ProtocolError("wire nesting exceeds 32")
    if value is None or type(value) in (bool, str, int):
        return
    if type(value) is float and math.isfinite(value):
        return
    if isinstance(value, (list, tuple)):
        for item in value: _json_check(item, depth + 1)
        return
    if isinstance(value, dict) and all(type(k) is str for k in value):
        for item in value.values(): _json_check(item, depth + 1)
        return
    raise ProtocolError("wire data must contain finite JSON values and string keys")


@dataclass(frozen=True)
class JSONValue:
    encoded: str

    def __post_init__(self):
        if type(self.encoded) is not str:
            raise ProtocolError("wire encoding must be text")
        try:
            if len(self.encoded.encode()) > MAX_WIRE_BYTES:
                raise ProtocolError("wire payload too large")
            value = json.loads(self.encoded)
            _json_check(value)
            if canonical(value) != self.encoded:
                raise ProtocolError("wire value must use canonical encoding")
        except (TypeError, ValueError, RecursionError) as exc:
            raise ProtocolError(str(exc)) from exc

    @classmethod
    def of(cls, value):
        _json_check(value)
        return cls(canonical(value))

    @property
    def value(self):
        return json.loads(self.encoded)

    def to_json(self):
        return self.value


_KEYWORDS = {"type", "properties", "required", "additionalProperties", "items", "enum",
             "minimum", "maximum", "minItems", "maxItems", "minLength", "maxLength", "oneOf"}
_TYPES = {"object", "array", "string", "integer", "number", "boolean", "null"}


def _schema_check(schema, depth=0):
    if depth > 16 or not isinstance(schema, dict) or set(schema) - _KEYWORDS:
        raise ProtocolError("unsupported schema constraint or excessive nesting")
    if "oneOf" in schema:
        if set(schema) != {"oneOf"} or not isinstance(schema["oneOf"], list) or not schema["oneOf"]:
            raise ProtocolError("oneOf must be a nonempty standalone schema union")
        for child in schema["oneOf"]: _schema_check(child, depth + 1)
        return
    if schema.get("type") not in _TYPES:
        raise ProtocolError("schema requires a supported type")
    kind = schema["type"]
    allowed = {"type", "enum"}
    if kind == "object":
        allowed |= {"properties", "required", "additionalProperties"}
        props = schema.get("properties", {})
        required = schema.get("required", [])
        if (not isinstance(props, dict) or not isinstance(required, list)
                or any(type(k) is not str for k in required) or not set(required) <= set(props)
                or type(schema.get("additionalProperties", True)) is not bool):
            raise ProtocolError("invalid object schema")
        for child in props.values(): _schema_check(child, depth + 1)
    if kind == "array":
        allowed |= {"items", "minItems", "maxItems"}
        if "items" in schema: _schema_check(schema["items"], depth + 1)
    if kind == "string": allowed |= {"minLength", "maxLength"}
    if kind in ("number", "integer"): allowed |= {"minimum", "maximum"}
    if set(schema) - allowed:
        raise ProtocolError("schema constraint does not apply to its type")
    for low, high in (("minItems", "maxItems"), ("minLength", "maxLength"), ("minimum", "maximum")):
        for key in (low, high):
            if key in schema and (type(schema[key]) not in (int, float) or not math.isfinite(schema[key])):
                raise ProtocolError("invalid schema bound")
            if key in schema and key not in ("minimum", "maximum") and (type(schema[key]) is not int or schema[key] < 0):
                raise ProtocolError("length bounds must be nonnegative integers")
        if low in schema and high in schema and schema[low] > schema[high]:
            raise ProtocolError("contradictory schema bounds")
    if "enum" in schema and (not isinstance(schema["enum"], list) or not schema["enum"]):
        raise ProtocolError("empty/invalid enum")


def _validate(schema, value, path="$"):
    if "oneOf" in schema:
        successes = 0
        for child in schema["oneOf"]:
            try: _validate(child, value, path)
            except ProtocolError: continue
            successes += 1
        if successes != 1: raise ProtocolError(path + ": expected exactly one matching schema")
        return
    kind = schema["type"]
    matches = {"object": isinstance(value, dict), "array": isinstance(value, (list, tuple)),
               "string": type(value) is str, "integer": type(value) is int,
               "number": type(value) in (int, float), "boolean": type(value) is bool,
               "null": value is None}
    if not matches[kind]: raise ProtocolError(path + ": expected " + kind)
    if "enum" in schema and canonical(value) not in [canonical(x) for x in schema["enum"]]:
        raise ProtocolError(path + ": value outside enum")
    if kind == "object":
        props = schema.get("properties", {})
        if not set(schema.get("required", ())) <= set(value): raise ProtocolError(path + ": missing required field")
        if not schema.get("additionalProperties", True) and set(value) - set(props):
            raise ProtocolError(path + ": unknown field")
        for key, item in value.items():
            if key in props: _validate(props[key], item, path + "." + key)
    if kind == "array" and "items" in schema:
        for index, item in enumerate(value): _validate(schema["items"], item, path + "[%s]" % index)
    for low, high, quantity in (("minItems", "maxItems", len(value) if kind == "array" else None),
                                ("minLength", "maxLength", len(value) if kind == "string" else None),
                                ("minimum", "maximum", value if kind in ("integer", "number") else None)):
        if quantity is not None and ((low in schema and quantity < schema[low]) or (high in schema and quantity > schema[high])):
            raise ProtocolError(path + ": bound violation")


@dataclass(frozen=True)
class Schema:
    document: JSONValue

    def __post_init__(self):
        _schema_check(self.document.value)

    @classmethod
    def of(cls, value): return cls(JSONValue.of(value))

    def validate(self, value):
        value = plain(value)
        _json_check(value)
        if len(canonical(value).encode()) > MAX_WIRE_BYTES: raise ProtocolError("wire payload too large")
        _validate(self.document.value, value)


@dataclass(frozen=True)
class Action:
    operation: str
    arguments: JSONValue

    @classmethod
    def create(cls, operation, arguments):
        if type(operation) is not str or not operation: raise ProtocolError("action name required")
        return cls(operation, JSONValue.of(arguments))


@dataclass(frozen=True)
class DomainCapabilities:
    domain_id: str
    action_schema: Schema
    observation_schema: Schema
    result_schema: Schema
    artifact_languages: Tuple[str, ...]
    world_tracks: Tuple[str, ...]
    max_action_budget: int
    features: Tuple[str, ...] = ("audit", "sealed-test")
    deterministic: bool = True
    score_definition: str = "normalized objective; higher is better"
    limitations: Tuple[str, ...] = ()
    protocol_version: str = PROTOCOL_VERSION

    def __post_init__(self):
        if type(self.domain_id) is not str or not self.domain_id or type(self.max_action_budget) is not int or self.max_action_budget < 1:
            raise ProtocolError("invalid domain identity or action budget")
        if any(not isinstance(x, Schema) for x in (self.action_schema, self.observation_schema, self.result_schema)):
            raise ProtocolError("capability wire schemas must be Schema objects")
        if type(self.deterministic) is not bool or type(self.score_definition) is not str or not self.score_definition:
            raise ProtocolError("evaluation semantics required")
        for values in (self.artifact_languages, self.world_tracks, self.features, self.limitations):
            if not isinstance(values, tuple) or any(type(x) is not str for x in values) or len(set(values)) != len(values):
                raise ProtocolError("capability sets must be immutable unique string tuples")


@dataclass(frozen=True)
class NegotiatedProtocol:
    domain_id: str
    protocol_version: str
    language: str
    action_budget: int
    world_tracks: Tuple[str, ...]
    capabilities_hash: str


def negotiate(capabilities, languages, action_budget, world_tracks=(), required_features=("audit", "sealed-test")):
    if not isinstance(capabilities, DomainCapabilities): raise CapabilityUnsupported("typed domain descriptor required")
    if capabilities.protocol_version != PROTOCOL_VERSION: raise CapabilityUnsupported("domain protocol version mismatch")
    if not {"audit", "sealed-test", *required_features} <= set(capabilities.features):
        raise CapabilityUnsupported("required domain features unavailable")
    if type(action_budget) is not int or not 1 <= action_budget <= capabilities.max_action_budget:
        raise CapabilityUnsupported("requested action budget exceeds domain constraints")
    common = [x for x in languages if x in capabilities.artifact_languages]
    if not common: raise CapabilityUnsupported("no mutually executable artifact language")
    if not set(world_tracks) <= set(capabilities.world_tracks): raise CapabilityUnsupported("required world creation tracks unavailable")
    return NegotiatedProtocol(capabilities.domain_id, PROTOCOL_VERSION, common[0], action_budget, tuple(world_tracks), digest(capabilities))


@dataclass(frozen=True)
class ResearchArtifact:
    kind: str
    language: str
    version: str
    source: str
    input_schema: Schema
    output_schema: Schema
    provenance: Provenance
    entrypoint: str = "main"

    def __post_init__(self):
        if self.kind not in ("experiment", "result", "hypothesis", "representation", "meta", "world"):
            raise ProtocolError("invalid artifact kind")
        if (type(self.version) is not str or not self.version or type(self.language) is not str
                or not self.language or self.entrypoint != "main"):
            raise ProtocolError("artifact version/language/entrypoint unsupported")
        if type(self.source) is not str or len(self.source.encode()) > MAX_WIRE_BYTES:
            raise ProtocolError("artifact source must be bounded text")
        if self.provenance.split not in ("train", "dev", "test"):
            raise ProtocolError("invalid artifact provenance split")

    @property
    def content_hash(self): return digest(self)


def artifact_from(data):
    return ResearchArtifact(**{**data, "input_schema": Schema.of(data["input_schema"]["document"]),
                               "output_schema": Schema.of(data["output_schema"]["document"]),
                               "provenance": Provenance(**data["provenance"])})


def usage_within(actual, reserved):
    for name in actual.__dataclass_fields__:
        value, limit = getattr(actual, name), getattr(reserved, name)
        if type(value) is not int or value < 0:
            return False
        if limit is None:
            continue
        if type(limit) is not int or limit < 0 or value > limit:
            return False
    return True


def checked_sample(adapter, split, seed):
    """Validate membership metadata without exposing evaluator-private answers."""
    from .contracts import TaskRef
    tasks = adapter.sample(split, seed)
    if not isinstance(tasks, tuple) or not tasks:
        raise ProtocolError("sampler must return a nonempty immutable task tuple")
    if any(not isinstance(t, TaskRef) or t.domain_id != adapter.domain_id or t.split != split for t in tasks):
        raise ProtocolError("task sampler returned foreign or mispartitioned tasks")
    if len({t.task_id for t in tasks}) != len(tasks):
        raise ProtocolError("task sampler returned duplicate task refs")
    return tasks


class ResearchRuntime(Protocol):
    """Agent execution/invention extension; independent of trusted world semantics.

    Generation receives only public domain context and TRAIN evidence. Adapter
    access belongs to the trusted execution path, not candidate program code.
    """
    runtime_id: str
    languages: Tuple[str, ...]

    def validate_mode(self, mode: str) -> None: ...
    def context(self, adapter: Any) -> Any: ...
    def initial_pair(self, config: Any) -> tuple: ...
    def episode(self, adapter: Any, task: Any, scientist: Any, challenger: Any,
                generation: int, search_budget: int) -> Any: ...
    def propose(self, public_context: Any, challenger: Any, scientist: Any,
                training: tuple, generation: int) -> Any: ...
    def candidates(self, parent: Any, diagnosis: Any, training: tuple,
                   public_context: Any, limit: int, mode: str) -> tuple: ...
    def adapt(self, parent: Any, diagnosis: Any, public_context: Any, meta: Any) -> tuple: ...
