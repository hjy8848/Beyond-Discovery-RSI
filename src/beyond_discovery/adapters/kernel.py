class CapabilityUnsupported(RuntimeError):
    pass


class KernelBenchAdapter:
    def __init__(self, *args, **kwargs):
        raise CapabilityUnsupported(
            "CAPABILITY_UNSUPPORTED: KernelBench requires an isolated GPU worker, "
            "independent correctness reference, frozen tasks and timing protocol. "
            "VectorProgramAdapter is a CPU protocol fixture, not KernelBench."
        )
