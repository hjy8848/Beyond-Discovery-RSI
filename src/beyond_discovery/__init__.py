"""Beyond-Discovery RSI: an auditable research implementation, not a result."""
from .core.controller import RunConfig, run_rsi
from .core.protocol import Action, DomainCapabilities, ResearchArtifact, Schema

__all__ = ["RunConfig", "run_rsi", "Action", "DomainCapabilities", "ResearchArtifact", "Schema"]
