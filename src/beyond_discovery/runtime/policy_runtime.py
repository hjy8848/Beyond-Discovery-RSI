"""Capability-free expression VM; generated values are data, never Python code.

The language has no I/O, calls, imports, loops, attributes or reflection. Depth,
node count, input size, numeric range and instruction count are bounded. A
separate reference interpreter is used by certification, not this stack VM.
"""
from __future__ import annotations

OPS = {"add", "mul", "xor", "max", "min", "sub"}
LIMIT = 1_000_000


class PolicyError(ValueError):
    pass


def freeze(value):
    return tuple(freeze(x) for x in value) if isinstance(value, (list, tuple)) else value


def compile_program(program: tuple, width: int) -> tuple:
    code = []

    def walk(node, depth):
        if depth > 6 or len(code) > 63:
            raise PolicyError("PROGRAM_BUDGET: depth/nodes exceeded")
        if not isinstance(node, tuple) or not node or not isinstance(node[0], str):
            raise PolicyError("CODE_INVALID: expression must be a tuple")
        op = node[0]
        if op == "x":
            if len(node) != 2 or type(node[1]) is not int or not 0 <= node[1] < width:
                raise PolicyError("CODE_INVALID: input index")
            code.append(node)
        elif op == "const":
            if len(node) != 2 or type(node[1]) is not int or abs(node[1]) > 8:
                raise PolicyError("CODE_INVALID: constant")
            code.append(node)
        elif op in OPS and len(node) == 3:
            walk(node[1], depth + 1)
            walk(node[2], depth + 1)
            code.append((op,))
        else:
            raise PolicyError("CODE_INVALID: unsupported instruction")

    if not 1 <= width <= 16:
        raise PolicyError("input width out of bounds")
    walk(program, 0)
    if len(code) > 63:
        raise PolicyError("PROGRAM_BUDGET: node count exceeded")
    return tuple(code)


def execute(code: tuple, inputs: tuple) -> int:
    if len(inputs) > 16 or any(type(x) is not int or abs(x) > 8 for x in inputs):
        raise PolicyError("invalid input")
    stack = []
    for instruction in code:
        op = instruction[0]
        if op == "x":
            stack.append(inputs[instruction[1]])
        elif op == "const":
            stack.append(instruction[1])
        else:
            right, left = stack.pop(), stack.pop()
            if op == "add": value = left + right
            elif op == "sub": value = left - right
            elif op == "mul": value = left * right
            elif op == "xor": value = left ^ right
            elif op == "max": value = max(left, right)
            elif op == "min": value = min(left, right)
            else: raise PolicyError("invalid bytecode")
            if abs(value) > LIMIT:
                raise PolicyError("PROGRAM_BUDGET: numeric range exceeded")
            stack.append(value)
    if len(stack) != 1:
        raise PolicyError("invalid bytecode stack")
    return stack[0]


def predict(program: tuple, inputs: tuple) -> int:
    return execute(compile_program(program, len(inputs)), inputs)


def validate_meta(meta) -> None:
    allowed = {"RECOMBINE", "ANALOGIZE", "RELAX_ASSUMPTION", "REFRAME", "INVENT_EXPERIMENT"}
    if not meta.operators or len(meta.operators) > 5 or len(set(meta.operators)) != len(meta.operators):
        raise PolicyError("invalid operator dispatch")
    if not set(meta.operators) <= allowed:
        raise PolicyError("invalid creative operator")
    if not 1 <= meta.required_conflicts <= 32 or not 1 <= meta.required_stagnation <= 32:
        raise PolicyError("invalid diagnosis thresholds")
    if meta.search_order not in ("forward", "reverse"):
        raise PolicyError("invalid search order")
