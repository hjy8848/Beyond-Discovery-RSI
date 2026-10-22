"""Legacy call compatibility; general artifact invention is the M3 milestone."""


def invent(snapshot, diagnosis, capabilities, operator):
    from ..runtime.finite_research import finite_invent
    return finite_invent(snapshot, diagnosis, capabilities, operator)
