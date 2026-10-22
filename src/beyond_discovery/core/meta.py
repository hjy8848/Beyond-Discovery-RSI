"""Executable meta-policy synthesis and fixed-opportunity causal evaluation."""
from dataclasses import dataclass
import difflib
import json
from .contracts import Decision, canonical, digest
from .gate import paired_gate
from .protocol import ProtocolError, ResearchArtifact, Schema
from .synthesis import Grammar, enumerate_programs
from ..runtime.expression import evaluate, freeze, node_size


META_VARIABLES = (("errors","integer"),("complexity","integer"),("cost","integer"),
                  ("novelty","integer"),("coverage","integer"))


def meta_artifact(node, provenance, version="1"):
    return ResearchArtifact("meta","research-expression-v1",version,canonical({"expression":node}),
                            Schema.of({"type":"object"}),Schema.of({"type":"integer"}),provenance)


def score(meta, facts):
    if set(facts) != {name for name,kind in META_VARIABLES}: raise ProtocolError("meta facts contract mismatch")
    if any(type(v) is not int for v in facts.values()): raise ProtocolError("meta facts must be integers")
    if meta.kind!="meta" or meta.language!="research-expression-v1": raise ProtocolError("invalid meta program")
    node=freeze(json.loads(meta.source)["expression"])
    return evaluate(node,facts)


def synthesize_meta(parent, probes, provenance, budget=64):
    if provenance.split!="train": raise PermissionError("meta synthesis is train-only")
    grammar=Grammar(META_VARIABLES,(0,1),("add","sub","mul"),"integer",3)
    before=tuple(score(parent,p) for p in probes)
    rankings=lambda values: tuple(sorted(range(len(values)),key=lambda i:(-values[i],i)))
    signature=rankings(before)
    seen={signature}
    result=[]
    for node in enumerate_programs(grammar,budget).programs:
        candidate=meta_artifact(node,provenance,str(int(parent.version)+1))
        values=tuple(score(candidate,p) for p in probes)
        behavior=rankings(values)
        if behavior in seen: continue
        seen.add(behavior); result.append(candidate)
    return tuple(result)


def source_diff(parent,candidate):
    if parent is None or candidate is None: return ""
    return "\n".join(difflib.unified_diff(parent.source.splitlines(),candidate.source.splitlines(),
                                         fromfile=parent.content_hash,tofile=candidate.content_hash,lineterm=""))


@dataclass(frozen=True)
class AdaptationOpportunity:
    opportunity_id: str
    source_cluster: str
    parent_program_hash: str
    frozen_opponent_hash: str
    training: tuple
    budget: int

    def __post_init__(self):
        if self.budget<1 or any(getattr(e,"split",getattr(getattr(e,"task",None),"split",None))!="train" for e in self.training):
            raise PermissionError("adaptation opportunities contain train examples only")


def evaluate_meta_pair(parent,candidate,opportunities,adapt,verify,gate_config,seed=0):
    """Only adapt sees train. Independent verify sees private validation closure.

    adapt(meta, opportunity) -> (artifact, actual search cost). Validation never
    feeds back to generation and all opportunities are fixed before either arm.
    """
    rows=[[],[]]; artifacts=[[],[]]
    for opportunity in opportunities:
        for arm,meta in enumerate((parent,candidate)):
            artifact,cost=adapt(meta,opportunity)
            if not 0<=cost<=opportunity.budget: raise ProtocolError("meta adaptation exceeded matched budget")
            valid,success=verify(artifact,opportunity)
            rows[arm].append((opportunity.source_cluster,float(success),valid,opportunity.budget,opportunity.opportunity_id))
            artifacts[arm].append(artifact.content_hash)
    gate=paired_gate(tuple(rows[0]),tuple(rows[1]),gate_config,seed)
    if artifacts[0]==artifacts[1] and gate.decision==Decision.ACCEPT:
        raise ProtocolError("identical inventions cannot establish meta improvement")
    return gate,{"parent":parent,"candidate":candidate,"source_diff":source_diff(parent,candidate),
                 "opportunities":opportunities,"inventions":tuple(tuple(a) for a in artifacts),"rows":tuple(tuple(r) for r in rows)}
