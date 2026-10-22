"""Fixed outer selector: paired clustered intervals, hard constraints first."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
import random
from statistics import mean
from .contracts import Decision, GateResult


@dataclass(frozen=True)
class GateConfig:
    minimum_gain: float = 0.01
    max_regression: float = 0.15
    min_clusters: int = 4
    bootstrap_samples: int = 500
    confidence: float = 0.95


def paired_gate(parent, candidate, config, seed=0, comparisons=1):
    # Rows are (source cluster, score, valid, resource budget). Exact paired keys
    # are checked by the controller before passing these immutable ordered rows.
    if len(parent) != len(candidate) or not parent:
        return GateResult(Decision.INCONCLUSIVE, 0.0, (0.0, 0.0), 0, "unpaired/empty samples")
    grouped = defaultdict(list)
    for old, new in zip(parent, candidate):
        if old[0] != new[0] or (len(old) > 4 and old[4] != new[4]):
            return GateResult(Decision.REJECT, 0.0, (0.0, 0.0), 0, "source pairing mismatch")
        if not old[2] or not new[2] or new[3] > old[3] or not all(math.isfinite(x) for x in (old[1], new[1])):
            return GateResult(Decision.REJECT, 0.0, (0.0, 0.0), 0, "invalid evidence or budget violation")
        grouped[old[0]].append(new[1] - old[1])
    deltas = [mean(values) for values in grouped.values()]
    delta = mean(deltas)
    if min(deltas) < -config.max_regression:
        return GateResult(Decision.REJECT, delta, (min(deltas), max(deltas)), len(deltas), "cluster regression constraint")
    if len(deltas) < config.min_clusters:
        return GateResult(Decision.INCONCLUSIVE, delta, (min(deltas), max(deltas)), len(deltas), "insufficient independent clusters")
    rng = random.Random(seed)
    draws = sorted(mean(rng.choices(deltas, k=len(deltas))) for _ in range(config.bootstrap_samples))
    # Family-wise correction over all candidates generated before dev evaluation.
    alpha = (1.0 - config.confidence) / (2 * max(1, comparisons))
    lo = draws[int(alpha * (len(draws) - 1))]
    hi = draws[int((1 - alpha) * (len(draws) - 1))]
    decision = Decision.ACCEPT if lo > config.minimum_gain else Decision.INCONCLUSIVE
    return GateResult(decision, delta, (lo, hi), len(deltas),
                      "paired clustered gain" if decision == Decision.ACCEPT else "gain not established")
