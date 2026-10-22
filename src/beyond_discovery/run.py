from __future__ import annotations

import argparse
import json
from pathlib import Path
from .core.controller import RunConfig, run_rsi
from .core.contracts import plain
from .core.conformance import check_conformance
from .core.protocol import artifact_from
from .core.replay import replay
from .adapters.causal import CausalDiscoveryAdapter
from .adapters.document import DocumentAdapter
from .adapters.vector import VectorProgramAdapter
from .runtime.research import ArtifactResearchRuntime


def main():
    parser = argparse.ArgumentParser(description="Beyond-Discovery versioned domain protocol and RSI controller")
    parser.add_argument("command", choices=("run", "replay", "conformance"))
    parser.add_argument("--domain", choices=("causal", "vector", "document"), default="causal")
    parser.add_argument("--out", help="new run directory for run, existing directory for replay")
    parser.add_argument("--experiment", help="versioned ResearchArtifact JSON; required for an authored document experiment")
    parser.add_argument("--action-budget", type=int, default=8)
    parser.add_argument("--search-budget", type=int, default=256)
    parser.add_argument("--scope", choices=("full", "protocol-only"), default="full",
                        help="protocol-only explicitly allows missing world creation; does not claim full RSI")
    parser.add_argument("--generations", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=("full", "fixed", "program_evolution", "no_restructuring", "no_meta"), default="full")
    args = parser.parse_args()
    adapter = {"causal": CausalDiscoveryAdapter, "vector": VectorProgramAdapter, "document": DocumentAdapter}[args.domain]()
    experiment = artifact_from(json.loads(Path(args.experiment).read_text())) if args.experiment else None
    runtime = ArtifactResearchRuntime(experiment) if experiment else None
    if args.command in ("run", "replay") and not args.out:
        parser.error("--out is required for run/replay")
    if args.command == "replay":
        result = replay(args.out, adapter)
    elif args.command == "conformance":
        result = plain(check_conformance(adapter, experiment, adapter.conformance_actions()))
    else:
        result = run_rsi(adapter, args.out, RunConfig(generations=args.generations, seed=args.seed, mode=args.mode,
                                                    action_budget=args.action_budget, search_budget=args.search_budget,
                                                    required_world_tracks=("A", "B") if args.scope == "full" else ()), runtime=runtime)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
