from __future__ import annotations

from ..core.contracts import ChallengeProposal
from ..runtime.policy_runtime import execute
from .scientist import bytecode


class Challenger:
    def __init__(self, snapshot):
        self.snapshot = snapshot

    def discriminate(self, capabilities, scientist, observations):
        used = {o.action for o in observations}
        space = tuple(x for x in capabilities["input_space"] if x not in used)
        if not space:
            raise ValueError("input space exhausted")
        if self.snapshot.experiment.strategy == "sequential":
            return space[0]
        # Version-space disagreement, not failure-rate maximization.
        models = scientist.compatible(observations)
        width = capabilities["width"]
        # A fixed public computation budget independent of task outcomes.
        count = min(32, len(space))
        choices = tuple(space[i * (len(space) - 1) // max(1, count - 1)] for i in range(count))
        def score(action):
            counts = {}
            for program in models:
                value = execute(bytecode(program, width), action)
                counts[value] = counts.get(value, 0) + 1
            disagreement = len(models) ** 2 - sum(n * n for n in counts.values())
            # When the old family agrees, actively probe uncovered conditions
            # rather than repeatedly confirming its common assumption.
            coverage = min((sum(a != b for a, b in zip(action, old)) for old in used), default=0)
            return disagreement, coverage, sum(abs(x) for x in action)
        return max(choices, key=score)

    def propose(self, capabilities, generation):
        ops = capabilities["expression_ops"]
        op = ops[generation % len(ops)]
        other = ops[(generation + 1) % len(ops)]
        return ChallengeProposal(capabilities["domain_id"],
                                 (op, ("x", 0), (other, ("x", 1), ("x", 2))),
                                 "A" if generation % 2 == 0 else "B")
