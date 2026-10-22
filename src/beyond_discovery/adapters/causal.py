from .finite import FiniteAdapter


class CausalDiscoveryAdapter(FiniteAdapter):
    """Synthetic deterministic SCM: intervene on six Boolean causes; observe Y.

    No hidden-state discovery or CausalGame integration is claimed. Compositional
    and bounded open DSL worlds have independently checked finite semantics.
    """
    domain_id = "synthetic-causal"
    operators = ("add", "xor")
