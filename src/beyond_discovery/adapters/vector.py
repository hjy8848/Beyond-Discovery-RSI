from .finite import FiniteAdapter


class VectorProgramAdapter(FiniteAdapter):
    """Integer vector function synthesis, evaluated exhaustively on a CPU.

    This adapter exercises a second semantics, not GPU optimization. Instruction
    counts are reported only as counts, never as GPU speedups.
    """
    domain_id = "integer-vector"
    values = (-1, 0, 1)
    operators = ("add", "mul", "max")
