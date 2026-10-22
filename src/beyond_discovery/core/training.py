"""Role-isolated optional training contract. No fake LoRA/RL implementation."""
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol
from .contracts import ModelState, digest


@dataclass(frozen=True)
class TrainingBatch:
    role: str
    evidence_hashes: tuple
    base_model: str


class TrainingBackend(Protocol):
    def fit(self, state: ModelState, batch: TrainingBatch, output_directory: Path) -> ModelState: ...


def training_batch(snapshot, evidence):
    if any(e.task.split != "train" or not e.score.valid or e.failure is not None for e in evidence):
        raise PermissionError("training requires independently verified train evidence")
    return TrainingBatch(snapshot.role, tuple(digest(e) for e in evidence), snapshot.model.base_model)


def train_role(snapshot, evidence, backend, root):
    batch = training_batch(snapshot, evidence)
    output = Path(root).resolve() / snapshot.role / digest(batch)
    output.mkdir(parents=True, exist_ok=False)
    state = backend.fit(snapshot.model, batch, output)
    if state.role != snapshot.role or state.base_model != snapshot.model.base_model:
        raise ValueError("training changed role/shared base model")
    if not state.delta_checkpoint or not Path(state.delta_checkpoint).resolve().is_relative_to(output):
        raise ValueError("checkpoint must stay in role-isolated output directory")
    if not Path(state.delta_checkpoint).is_file():
        raise ValueError("checkpoint not produced")
    if state.training_data_hash != digest(batch):
        raise ValueError("training provenance mismatch")
    # Returns a candidate only. Caller MUST evaluate it through normal gates.
    return replace(snapshot, model=state, parent_hash=snapshot.content_hash,
                   generation=snapshot.generation + 1)
