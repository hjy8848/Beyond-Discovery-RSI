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
from .adapters.transition import TransitionAdapter, StringTransitionAdapter
from .runtime.research import ArtifactResearchRuntime


def main():
    parser = argparse.ArgumentParser(description="Beyond-Discovery versioned domain protocol and RSI controller")
    parser.add_argument("command", choices=("run", "replay", "conformance"))
    parser.add_argument("--domain", choices=("causal", "vector", "document", "stateful-scm", "string-transducer"), default="causal")
    parser.add_argument("--out", help="new run directory for run, existing directory for replay")
    parser.add_argument("--experiment", help="versioned ResearchArtifact JSON; required for an authored document experiment")
    parser.add_argument("--action-budget", type=int, default=8)
    parser.add_argument("--search-budget", type=int, default=256)
    parser.add_argument("--scope", choices=("full", "protocol-only"), default="full",
                        help="protocol-only explicitly allows missing world creation; does not claim full RSI")
    parser.add_argument("--generations", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=("full", "fixed", "program_evolution", "no_restructuring", "no_meta"), default="full")
    parser.add_argument("--model-client", choices=("search","http","local"), default="search")
    parser.add_argument("--base-model", default="bounded-enumerator-v1", help="provider model ID or provisioned local HF directory")
    parser.add_argument("--endpoint", default="https://api.openai.com/v1")
    parser.add_argument("--key-env", default="OPENAI_API_KEY")
    parser.add_argument("--proxy", help="explicit proxy URL; do not embed credentials")
    parser.add_argument("--prompt-token-bound", type=int, help="provider-certified upper bound including prompt framing")
    parser.add_argument("--max-model-tokens", type=int, default=100000)
    parser.add_argument("--training", choices=("none","sft","rl"), default="none")
    parser.add_argument("--training-steps", type=int, default=1)
    parser.add_argument("--max-training-steps", type=int, default=100)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    adapter = {"causal": CausalDiscoveryAdapter, "vector": VectorProgramAdapter, "document": DocumentAdapter, "stateful-scm": TransitionAdapter, "string-transducer": StringTransitionAdapter}[args.domain]()
    experiment = artifact_from(json.loads(Path(args.experiment).read_text())) if args.experiment else None
    runtime = ArtifactResearchRuntime(experiment) if experiment else None
    if args.command=="run" and args.model_client!="search":
        from .runtime.models import TokenLedger, HTTPModelClient, LocalHFClient
        from .runtime.model_research import ModelResearchRuntime
        ledger=TokenLedger(args.max_model_tokens)
        if args.model_client=="http":
            if not args.prompt_token_bound: parser.error("HTTP requires --prompt-token-bound")
            if args.training!="none": parser.error("local LoRA requires --model-client local")
            clients={r:HTTPModelClient(args.base_model,args.endpoint,args.key_env,ledger,args.prompt_token_bound,args.proxy) for r in ("scientist","challenger")}
        else:
            clients={r:LocalHFClient(args.base_model,r,ledger,device=args.device) for r in ("scientist","challenger")}
        training=None if args.training=="none" else {"algorithm":args.training,"steps":args.training_steps,"max_steps":args.max_training_steps,"device":args.device}
        runtime=ModelResearchRuntime(adapter.public_context(),clients,training)
    elif args.training!="none": parser.error("training requires a real local model client")
    if args.command in ("run", "replay") and not args.out:
        parser.error("--out is required for run/replay")
    if args.command == "replay":
        result = replay(args.out, adapter)
    elif args.command == "conformance":
        result = plain(check_conformance(adapter, experiment, adapter.conformance_actions()))
    else:
        result = run_rsi(adapter, args.out, RunConfig(generations=args.generations, seed=args.seed, mode=args.mode,
                                                    action_budget=args.action_budget, search_budget=args.search_budget,
                                                    base_model=args.base_model,max_model_tokens=args.max_model_tokens,max_training_steps=args.max_training_steps,
                                                    required_world_tracks=("A", "B") if args.scope == "full" else ()), runtime=runtime)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
