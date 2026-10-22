from __future__ import annotations

import argparse
import json
from .core.controller import RunConfig, run_rsi
from .core.replay import replay
from .adapters.causal import CausalDiscoveryAdapter
from .adapters.vector import VectorProgramAdapter


def main():
    parser = argparse.ArgumentParser(description="Beyond-Discovery bounded program-only research core")
    parser.add_argument("command", choices=("run", "replay"))
    parser.add_argument("--domain", choices=("causal", "vector"), default="causal")
    parser.add_argument("--out", required=True, help="new run directory for run, existing directory for replay")
    parser.add_argument("--generations", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=("full", "fixed", "program_evolution", "no_restructuring", "no_meta"), default="full")
    args = parser.parse_args()
    adapter = CausalDiscoveryAdapter() if args.domain == "causal" else VectorProgramAdapter()
    if args.command == "replay":
        result = replay(args.out, adapter)
    else:
        result = run_rsi(adapter, args.out, RunConfig(generations=args.generations, seed=args.seed, mode=args.mode))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
