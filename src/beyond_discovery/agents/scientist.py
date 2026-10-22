"""Domain-neutral bounded program induction; no domain expert solver."""
from __future__ import annotations

from functools import lru_cache
from ..core.contracts import Claim, digest
from ..runtime.policy_runtime import compile_program, execute


@lru_cache(maxsize=128)
def program_library(width, operators, budget, order="forward"):
    terminals = tuple(("x", i) for i in range(width)) + (("const", 0), ("const", 1))
    programs = list(terminals)
    # A deliberately small baseline language. Expanding it is explicit policy state.
    for op in operators:
        for left in terminals:
            for right in terminals:
                if op in ("add", "mul", "max", "min", "xor") and repr(left) > repr(right):
                    continue
                programs.append((op, left, right))
    if order == "reverse":
        programs.reverse()
    return tuple(programs[:budget])


@lru_cache(maxsize=4096)
def bytecode(program, width):
    return compile_program(program, width)


class Scientist:
    def __init__(self, snapshot, capabilities, search_budget=256):
        self.snapshot = snapshot
        self.capabilities = capabilities
        self.programs = program_library(capabilities["width"], snapshot.representation,
                                        search_budget, snapshot.meta.search_order)

    def hypothesize(self, observations):
        width = self.capabilities["width"]
        def key(program):
            code = bytecode(program, width)
            errors = sum(execute(code, x.action) != x.outcome for x in observations)
            return errors, len(code), digest(program)
        best = min(self.programs, key=key)
        return Claim(best, "observable mechanism expressible using " + ",".join(self.snapshot.representation))

    def compatible(self, observations):
        width = self.capabilities["width"]
        result = [p for p in self.programs if all(execute(bytecode(p, width), x.action) == x.outcome for x in observations)]
        return tuple(result) or self.programs
