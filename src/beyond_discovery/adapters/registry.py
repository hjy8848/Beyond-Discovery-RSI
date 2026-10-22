"""Trusted adapter extensions; candidate code cannot import plugins."""
from importlib.metadata import entry_points
from ..core.protocol import CapabilityUnsupported


def external_adapter(name):
    entries=entry_points()
    entries=entries.select(group='beyond_discovery.adapters') if hasattr(entries,'select') else entries.get('beyond_discovery.adapters',())
    matches=[e for e in entries if e.name==name]
    if len(matches)!=1:
        raise CapabilityUnsupported('CAPABILITY_UNSUPPORTED: no uniquely installed real adapter for '+name)
    try: return matches[0].load()()
    except (ImportError,FileNotFoundError) as exc:
        raise CapabilityUnsupported('CAPABILITY_UNSUPPORTED: adapter dependency or benchmark assets unavailable') from exc
