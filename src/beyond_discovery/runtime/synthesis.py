"""Generic evidence-driven Scientist/Challenger execution and invention backend.

No hidden world program, label pool or dev/test feedback enters generation. The
same implementation handles integer state machines and string transducers.
"""
from dataclasses import asdict, replace
import json
from ..core.contracts import (Boundary, Claim, EvidenceBundle, ExperimentProgram, MetaPolicy,
                              ModelState, Provenance, Snapshot, Usage, canonical, digest, plain)
from ..core.protocol import Action, JSONValue, ProtocolError, ResearchArtifact, Schema
from ..core.synthesis import ArtifactInventor, Example, Grammar, enumerate_programs, errors, fit
from ..core.boundary import BoundaryKind, HypothesisFamily, certify_boundary
from ..core.experiments import context_for, design_experiment, protocol_actions
from ..core.meta import meta_artifact, score as meta_score, synthesize_meta
from ..core.worlds import ChallengerInventor, expression_from
from .expression import evaluate, freeze, node_size, typecheck


def grammar_from(data):
    return Grammar(tuple(tuple(x) for x in data["variables"]),tuple(data["constants"]),tuple(data["operators"]),
                   data["output_type"],data.get("max_nodes",1),tuple(freeze(x) for x in data.get("features",())))


class SynthesisRuntime:
    runtime_id="evidence-synthesis-v1"
    languages=("research-expression-v1",)
    supports_world_creation=True
    separate_meta=True

    def __init__(self,context=None):
        self.public=context
        self.training=()
        self.search_budget=256
        self.events=[]
        self.total_search=0

    def context(self,adapter):
        self.public=JSONValue.of(adapter.public_context()).value
        return self.public

    def validate_mode(self,mode):
        if mode not in ("full","fixed","program_evolution","no_restructuring","no_meta","no_diagnosis","no_falsification","reflection","random_challenger","hardness_first"):
            raise ProtocolError("unsupported synthesis baseline")

    def provenance(self,parent=None,role="scientist",domain=None):
        return Provenance(domain or self.public["domain_id"],"train","evidence-program-synthesis",
                          parent.generation+1 if parent else 0,parent.content_hash if parent else role,
                          0,self.public["verifier_version"],parent.content_hash if parent else None)

    def artifact(self,kind,data,provenance,output_type="object"):
        return ResearchArtifact(kind,"research-expression-v1","1",canonical(data),Schema.of({"type":"object"}),
                                Schema.of({"type":output_type}),provenance)

    def initial_pair(self,config):
        if self.public is None: raise ProtocolError("public domain context required")
        self.search_budget=config.search_budget
        grammar=replace(grammar_from(self.public["grammar"]),max_nodes=1,features=())
        snapshots=[]
        for role in ("challenger","scientist"):
            provenance=self.provenance(role=role)
            representation=self.artifact("representation",{"grammar":asdict(grammar)},provenance)
            meta=MetaPolicy(program=meta_artifact(("sub",("lit",0),("ref","errors")),provenance))
            snapshots.append(Snapshot(role,ModelState(config.base_model,role),meta,(),
                                      ExperimentProgram("synthesized",config.action_budget),artifacts=(representation,)))
        return tuple(snapshots)

    def grammar(self,snapshot):
        return grammar_from(json.loads(snapshot.artifacts[0].source)["grammar"])

    def models(self,training):
        result=[]
        for e in training:
            p=expression_from(e.new_claim.program)
            if p not in result: result.append(p)
        return tuple(result)

    def examples(self,evidence):
        return tuple(Example(context_for(o,evidence.observations[:i],self.public["initial"]),plain(o.outcome),digest(evidence),evidence.task.split)
                     for i,o in enumerate(evidence.observations))

    def hypothesis(self,node,provenance):
        return self.artifact("hypothesis",{"expression":node},provenance,self.public["grammar"]["output_type"])

    def episode(self,adapter,task,scientist,challenger,generation,search_budget):
        self.context(adapter)
        budget=min(scientist.experiment.action_budget,challenger.experiment.action_budget)
        grammar=self.grammar(scientist)
        enumeration=enumerate_programs(grammar,search_budget)
        programs=tuple(p for p in enumeration.programs if typecheck(p,dict(grammar.variables))==grammar.output_type)
        if not programs: raise ProtocolError("no type-correct hypotheses within allocation")
        actions=tuple(Action.create(a["operation"],a["arguments"]) for a in self.public["actions"])
        protocols=[]
        for snapshot in (challenger,scientist):
            for a in snapshot.artifacts[1:]:
                if a.kind=="experiment": protocols.append(protocol_actions(a))
        schedule=tuple(p[i] for i in range(max((len(p) for p in protocols),default=0)) for p in protocols if i<len(p))
        before=adapter.meter()
        session=adapter.begin(task,budget)
        observations=[]
        try:
            for i in range(budget):
                action=schedule[i%len(schedule)] if schedule else actions[i%len(actions)]
                observations.append(adapter.step(session,action))
        finally: adapter.close(session)
        examples=tuple(Example(context_for(o,observations[:i],self.public["initial"]),plain(o.outcome),digest((task,observations))) for i,o in enumerate(observations))
        best=min(programs,key=lambda p:(errors(p,examples),node_size(p),digest(p)))
        provenance=Provenance(adapter.domain_id,task.split,"independent-interaction",generation,scientist.content_hash,
                              task.seed,adapter.verifier_version,scientist.parent_hash)
        artifact=self.hypothesis(best,provenance)
        claim=Claim(artifact,"registered typed program family "+digest(grammar),"finite grammar with declared derivation budget")
        boundaries=[]
        for index,o in enumerate(observations):
            predicted=evaluate(best,examples[index].context)
            if predicted!=plain(o.outcome):
                reproduction=adapter.begin(task,index+1)
                try:
                    actual=tuple(adapter.step(reproduction,x.action) for x in observations[:index+1])
                    reproduced=actual==tuple(observations[:index+1])
                finally: adapter.close(reproduction)
                boundaries.append(Boundary(claim.assumption,o,predicted,None,reproduced,task.source_cluster))
                break
        value=adapter.evaluate(claim,task,task.split)
        evidence=EvidenceBundle(provenance,task,tuple(observations),claim,claim,value,tuple(boundaries),adapter.fingerprint(best),Usage())
        family=HypothesisFamily("typed-executable-family","1",tuple(self.hypothesis(p,provenance) for p in programs),
                                "registered finite typed grammar "+digest(grammar),enumeration.complete)
        certificate=certify_boundary(adapter,evidence,family,
                                     lambda m,a,h:evaluate(expression_from(m),context_for(a,h,self.public["initial"])),search_budget)
        after=adapter.meter()
        usage=Usage(*(getattr(after,n)-getattr(before,n) for n in before.__dataclass_fields__))
        usage=usage+Usage(search=len(enumeration.programs),predictions=len(programs)*len(examples)+certificate.evaluations)
        return replace(evidence,boundary_certificates=(certificate,),usage=usage)

    def propose(self,public_context,challenger,scientist,training,generation):
        self.public=public_context; self.training=training
        grammar=replace(grammar_from(public_context["grammar"]),max_nodes=5)
        provenance=self.provenance(challenger,role="challenger")
        actions=tuple(Action.create(a["operation"],a["arguments"]) for a in public_context["actions"])
        try:
            self.total_search+=self.search_budget
            models=self._rank_models(self.models(training),training,challenger.meta)[:4]
            return ChallengerInventor().propose(grammar,actions,public_context["initial"],models,training,
                                                provenance,scientist.content_hash,self.search_budget,
                                                "A" if generation%2==0 else "B",novel_beyond=3)
        except ProtocolError as exc:
            self.events.append({"type":"challenge-search-failure","reason":str(exc),"generation":generation})
            return None

    def _rank_models(self,models,training,meta):
        def key(program):
            datasets=tuple(self.examples(e) for e in training)
            losses=tuple(errors(program,examples) for examples in datasets)
            facts={"errors":sum(losses),"complexity":node_size(program),"cost":node_size(program)*len(datasets),
                   "novelty":int(program not in self.models(training)),"coverage":sum(loss==0 for loss in losses)}
            return -meta_score(meta.program,facts),digest(program)
        return tuple(sorted(models,key=key))

    def _facts(self,invention):
        return {"errors":invention.errors_after,"complexity":node_size(expression_from(invention.artifact)),
                "cost":len(invention.evidence_hashes),"novelty":invention.errors_before-invention.errors_after,
                "coverage":len(set(invention.evidence_hashes))}

    def _generate(self,parent,diagnosis,training,public_context,limit,mode,meta):
        if any(e.task.split!="train" for e in training): raise PermissionError("candidate generation is train-only")
        if mode=="fixed": return ()
        self.public=public_context
        grammar=self.grammar(parent)
        enumeration=enumerate_programs(grammar,self.search_budget)
        parent_programs=tuple(p for p in enumeration.programs if typecheck(p,dict(grammar.variables))==grammar.output_type)
        actions=tuple(Action.create(a["operation"],a["arguments"]) for a in public_context["actions"])
        provenance=self.provenance(parent,role=parent.role)
        results=[]
        if parent.role=="scientist" and mode!="no_restructuring":
            pool=[]
            engine=ArtifactInventor(grammar,self.search_budget)
            # The same structural operators work in both domains. No adapter
            # chooses a representation, solves a task or supplies a correct model.
            for operator in meta.operators:
                if operator=="INVENT_EXPERIMENT": continue
                if mode=="reflection" and operator!="RECOMBINE": continue
                for e in training:
                    if mode not in ("program_evolution","no_diagnosis","reflection") and not any(c.kind==BoundaryKind.SPACE_REFUTED for c in e.boundary_certificates): continue
                    self.total_search+=self.search_budget
                    assets=tuple((expression_from(x.new_claim.program),grammar.variables,(digest(x),)) for x in training if x.score.valid and x.score.value==1.0)
                    pool.extend(engine.invent(operator,parent_programs,self.examples(e),provenance,assets))
            if meta.program is not None:
                pool.sort(key=lambda inv:(-meta_score(meta.program,self._facts(inv)),inv.artifact.content_hash))
            signatures=set()
            for invention in pool:
                node=expression_from(invention.artifact)
                signature=(node,invention.operator=="RELAX_ASSUMPTION")
                if signature in signatures: continue
                signatures.add(signature)
                updated=replace(grammar,features=grammar.features+(node,),
                                max_nodes=min(7,grammar.max_nodes+2) if invention.operator=="RELAX_ASSUMPTION" else grammar.max_nodes)
                code=self.artifact("representation",{"grammar":asdict(updated)},provenance)
                child=replace(parent,artifacts=(code,invention.artifact)+tuple(a for a in parent.artifacts[1:] if a.kind=="experiment"),
                              generation=parent.generation+1,parent_hash=parent.content_hash,
                              knowledge=parent.knowledge+invention.evidence_hashes)
                results.append((child,invention))
                if len(results)>=limit: break
        if len(results)<limit and mode!="no_falsification":
            old_models=self.models(training)
            alternatives=[]
            expanded=replace(grammar,max_nodes=min(7,grammar.max_nodes+2))
            for e in training:
                fitted,_=fit(expanded,self.examples(e),self.search_budget)
                self.total_search+=self.search_budget
                if fitted and fitted[0] not in alternatives: alternatives.append(fitted[0])
            hypotheses=tuple(dict.fromkeys(old_models+tuple(alternatives)))
            if parent.role=="challenger" and meta.program is not None:
                hypotheses=self._rank_models(hypotheses,training,meta)[:4]
            if len(hypotheses)>1:
                experiment=design_experiment(hypotheses,actions,public_context["initial"],min(4,parent.experiment.action_budget),self.search_budget,provenance)
                before=tuple(a.source for a in parent.artifacts if a.kind=="experiment")
                if experiment.source not in before:
                    child=replace(parent,artifacts=tuple(a for a in parent.artifacts if a.kind!="experiment")+(experiment,),
                                  generation=parent.generation+1,parent_hash=parent.content_hash)
                    results.append((child,experiment))
        return tuple(results[:limit])

    def candidates(self,parent,diagnosis,training,public_context,limit,mode):
        self.training=training
        attempts=self._generate(parent,diagnosis,training,public_context,limit,mode,parent.meta)
        if mode=="no_meta" or not attempts: return attempts
        facts=[]
        for child,patch in attempts:
            if hasattr(patch,"errors_after"): facts.append(self._facts(patch))
            else: facts.append({"errors":0,"complexity":len(child.artifacts),"cost":1,"novelty":1,"coverage":len(training)})
        # Add actual parent-asset alternatives to expose behavior changes in
        # generator ranking, rather than count source-string differences.
        for e in training:
            facts.append({"errors":len(e.boundaries),"complexity":node_size(expression_from(e.new_claim.program)),
                          "cost":e.usage.search,"novelty":0,"coverage":1})
        programs=synthesize_meta(parent.meta.program,tuple(facts),self.provenance(parent),min(128,self.search_budget))
        if not programs: return attempts
        return tuple((replace(child,meta=replace(parent.meta,program=programs[i%len(programs)],revision=parent.meta.revision+1)),patch)
                     for i,(child,patch) in enumerate(attempts))

    def adapt(self,parent,diagnosis,public_context,meta):
        # Fixed parent task artifacts, same frozen train opportunity set, same
        # adaptation allocation. No private gate feedback is retained here.
        attempts=self._generate(parent,diagnosis,self.training,public_context,1,"full",meta)
        if not attempts: return parent,None
        return attempts[0]
