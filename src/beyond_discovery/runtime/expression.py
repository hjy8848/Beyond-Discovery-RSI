"""Bounded typed functional language for executable research artifacts.

The VM has no host calls, file/network access, reflection or dynamic evaluation.
Public primitive semantics are distinct from domain-specific hidden oracles.
"""
from functools import lru_cache
from ..core.contracts import canonical, plain
from ..core.protocol import JSONValue, ProtocolError


SIGNATURES = {
    "add": (("integer", "integer"), "integer"),
    "sub": (("integer", "integer"), "integer"),
    "mul": (("integer", "integer"), "integer"),
    "xor": (("integer", "integer"), "integer"),
    "max": (("integer", "integer"), "integer"),
    "concat": (("string", "string"), "string"),
    "reverse": (("string",), "string"),
    "length": (("string",), "integer"),
    "eq": (("integer", "integer"), "boolean"),
    "not": (("boolean",), "boolean"),
    "and": (("boolean", "boolean"), "boolean"),
    "or": (("boolean", "boolean"), "boolean"),
}


def freeze(value):
    return tuple(freeze(x) for x in value) if isinstance(value, (list, tuple)) else value


def value_type(value):
    if type(value) is bool: return "boolean"
    if type(value) is int: return "integer"
    if type(value) is str: return "string"
    raise ProtocolError("functional language supports integer/string/boolean values")


def node_size(node):
    return 1 if node[0] in ("lit", "ref") else 1 + sum(node_size(x) for x in node[1:])


def typecheck(node, variables, depth=0):
    if depth > 8 or not isinstance(node, tuple) or not node: raise ProtocolError("invalid/deep expression")
    if node[0] == "ref":
        if len(node) != 2 or node[1] not in variables: raise ProtocolError("unbound expression reference")
        return variables[node[1]]
    if node[0] == "lit":
        if len(node) != 2: raise ProtocolError("invalid literal")
        JSONValue.of(node[1])
        return value_type(node[1])
    if node[0] not in SIGNATURES: raise ProtocolError("unsupported primitive")
    args, output = SIGNATURES[node[0]]
    if len(args) != len(node) - 1: raise ProtocolError("primitive arity mismatch")
    if tuple(typecheck(x, variables, depth + 1) for x in node[1:]) != args:
        raise ProtocolError("primitive argument type mismatch")
    return output


def apply_primitive(op, args):
    if op == "add": return args[0] + args[1]
    if op == "sub": return args[0] - args[1]
    if op == "mul": return args[0] * args[1]
    if op == "xor": return args[0] ^ args[1]
    if op == "max": return max(args)
    if op == "concat": return args[0] + args[1]
    if op == "reverse": return args[0][::-1]
    if op == "length": return len(args[0])
    if op == "eq": return args[0] == args[1]
    if op == "not": return not args[0]
    if op == "and": return args[0] and args[1]
    if op == "or": return args[0] or args[1]
    raise ProtocolError("unsupported primitive")


def evaluate(node, context, fuel=256):
    remaining = [fuel]
    def walk(n, depth):
        remaining[0] -= 1
        if remaining[0] < 0 or depth > 8: raise ProtocolError("expression execution budget")
        if not isinstance(n, tuple) or not n: raise ProtocolError("invalid expression")
        if n[0] == "lit":
            value = n[1]
        elif n[0] == "ref":
            value = context
            try:
                for part in n[1].split("."):
                    value = value[part]
            except (KeyError, TypeError): raise ProtocolError("unbound public input")
        else:
            args = tuple(walk(x, depth + 1) for x in n[1:])
            signature = SIGNATURES.get(n[0])
            if signature is None or tuple(value_type(x) for x in args) != signature[0]:
                raise ProtocolError("invalid primitive execution")
            value = apply_primitive(n[0], args)
        if type(value) is int and abs(value) > 1_000_000: raise ProtocolError("numeric resource bound")
        if type(value) is str and len(value) > 65536: raise ProtocolError("string resource bound")
        return value
    return walk(node, 0)
