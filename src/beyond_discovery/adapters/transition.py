"""Two real executable integration domains: stateful SCM and string machines.

Only primitive semantics, task generation, sessions and independent evaluation
live here. Model fitting, diagnosis, experiment design and invention live in Core.
These are controlled research environments, not external benchmark results.
"""
from dataclasses import dataclass
from itertools import product
import json
import random
from ..core.contracts import (Certification, Observation, Score, Status, TaskRef, Usage,
                              digest, plain)
from ..core.protocol import Action, DomainCapabilities, JSONValue, ProtocolError, Schema
from ..core.synthesis import Grammar, enumerate_programs
from ..core.worlds import WorldCertifier, behavior, expression_from, reference_expression
from ..core.experiments import rollout
from ..core.experiments import context_for
from ..runtime.expression import evaluate, node_size, typecheck


@dataclass
class _Session:
    task: TaskRef
    remaining: int
    state: object
    index: int = 0


class TransitionAdapter:
    verifier_version = "independent-transition-v1"
    domain_id = "stateful-scm"
    variables = (("input.x", "integer"), ("input.y", "integer"), ("previous", "integer"))
    constants = (0, 1)
    operators = ("xor", "add")
    output_type = "integer"
    initial = 0

    def __init__(self):
        self._usage = Usage()
        self._sealed = True
        self._sessions = {}
        self._refs, self._worlds = {}, {}
        self.actions = self._actions()
        self.grammar = Grammar(self.variables, self.constants, self.operators, self.output_type, 3)
        self._certifier = WorldCertifier(Grammar(self.variables,self.constants,self.operators,self.output_type,7),self.actions,self.initial)
        seen = set()
        worlds = []
        for p in enumerate_programs(self.grammar, 2048).programs:
            if typecheck(p, dict(self.variables)) != self.output_type: continue
            fingerprint = digest(behavior(p, self.actions, self.initial, 2))
            if fingerprint in seen: continue
            seen.add(fingerprint); worlds.append(p)
        # Public low-complexity source family is excluded from generated sources;
        # this property does not query any private dev/test task labels.
        self._certifier.seen.update(seen)
        for i,p in enumerate(worlds):
            split = ("train","dev","test")[i % 3]
            task = TaskRef(digest((self.domain_id,split,i)),self.domain_id,split,"structure/"+digest(p),i)
            self._refs[task.task_id] = task; self._worlds[task.task_id] = p

    def _actions(self):
        return tuple(Action.create("intervene",{"x":x,"y":y}) for x,y in product((0,1),repeat=2))

    def public_context(self):
        return {"grammar": {"variables":self.variables,"constants":self.constants,"operators":self.operators,
                            "output_type":self.output_type,"max_nodes":3},
                "actions":plain(self.actions),"initial":self.initial,"domain_id":self.domain_id,
                "verifier_version":self.verifier_version}

    def describe(self):
        properties = {name.split(".")[-1]:{"type":kind} for name,kind in self.variables if name.startswith("input.")}
        action = Schema.of({"type":"object","properties":{"operation":{"type":"string","enum":[self.actions[0].operation]},
                            "arguments":{"type":"object","properties":properties,"required":list(properties),"additionalProperties":False}},
                            "required":["operation","arguments"],"additionalProperties":False})
        result = Schema.of({"type":self.output_type})
        return DomainCapabilities(self.domain_id,action,result,result,("research-expression-v1",),("A","B"),8,
                                  features=("audit","sealed-test","stateful","structured-observations","world-budget-certification"),
                                  limitations=("controlled finite primitive-action probes; not an external benchmark",))

    def sample(self,split,seed):
        if split not in ("train","dev","test"): raise ProtocolError("unknown split")
        if split == "test" and self._sealed: raise PermissionError("test sealed")
        tasks = [t for t in self._refs.values() if t.split==split and not t.source_cluster.startswith("generated/")]
        random.Random(seed).shuffle(tasks)
        return tuple(tasks[:6])

    def _check(self,task):
        if self._refs.get(task.task_id)!=task: raise ProtocolError("unregistered/forged task")
        if task.split=="test" and self._sealed: raise PermissionError("test sealed")

    def unseal_for_final(self):
        if not self._sealed: raise PermissionError("test already opened")
        self._sealed=False

    def begin(self,task,budget):
        self._check(task)
        if type(budget) is not int or not 1<=budget<=8: raise ProtocolError("session budget")
        s=_Session(task,budget,self.initial); self._sessions[id(s)]=s
        return s

    def initial_observation(self,s):
        if self._sessions.get(id(s)) is not s: raise ProtocolError("foreign/closed session")
        return JSONValue.of(s.state)

    def _oracle(self,p,context):
        self._usage=self._usage+Usage(verification_outputs=1)
        value=reference_expression(p,context)
        if type(value) is int and abs(value)>1000000: raise ProtocolError("world numeric resource limit")
        if type(value) is str and len(value)>65536: raise ProtocolError("world string resource limit")
        return value

    def step(self,s,action):
        self.initial_observation(s); self._check(s.task)
        self.describe().action_schema.validate(action)
        if s.remaining<=0: raise ProtocolError("session budget exhausted")
        if action not in self.actions: raise ProtocolError("action outside declared primitive affordances")
        s.state=self._oracle(self._worlds[s.task.task_id],{"input":action.arguments.value,"previous":s.state,"time":s.index})
        s.index+=1; s.remaining-=1
        self._usage=self._usage+Usage(experiments=1)
        return Observation(action,JSONValue.of(s.state),s.index)

    def close(self,s): self._sessions.pop(id(s),None)
    def meter(self): return self._usage
    def conformance_actions(self): return self.actions[:2]

    def estimate_episode(self,action_budget,search_budget):
        return Usage(experiments=2*action_budget,predictions=1024+search_budget*action_budget*2,
                     search=search_budget,verification_outputs=2048+action_budget*10)

    def estimate_certification(self): return Usage(verification_outputs=200000)

    def fingerprint(self,program):
        # Public diagnostic predictions, no oracle answers.
        result=behavior(program,self.actions,self.initial,2)
        self._usage=self._usage+Usage(predictions=sum(len(s) for s in result))
        return digest(result)

    def evaluate(self,submission,task,split):
        self._check(task)
        if task.split!=split: raise PermissionError("evaluation split mismatch")
        try:
            p=expression_from(submission.program)
            if typecheck(p,dict(self.variables))!=self.output_type: raise ProtocolError("submission type")
            rng=random.Random(991+task.seed)
            correct=0; total=0
            # All scored trajectories exceed the discovery action budget, with
            # independent hidden sequences. No teacher-forced true state enters prediction.
            for _ in range(8):
                predicted=actual=self.initial
                for index in range(12):
                    action=rng.choice(self.actions)
                    predicted=evaluate(p,{"input":action.arguments.value,"previous":predicted,"time":index})
                    self._usage=self._usage+Usage(predictions=1)
                    actual=self._oracle(self._worlds[task.task_id],{"input":action.arguments.value,"previous":actual,"time":index})
                    total+=1; correct+=int(predicted==actual)
            return Score(correct/total,True,total)
        except (ProtocolError,TypeError,ValueError,KeyError) as exc:
            return Score(0.0,False,96,str(exc))

    def audit(self,e):
        self._check(e.task)
        if e.provenance.domain_id!=self.domain_id or e.provenance.split!=e.task.split or e.provenance.verifier_version!=self.verifier_version: return False
        previous=self.initial
        try:
            for index,o in enumerate(e.observations):
                self.describe().action_schema.validate(o.action)
                if o.action not in self.actions or o.sequence!=index+1: return False
                previous=self._oracle(self._worlds[e.task.task_id],{"input":o.action.arguments.value,"previous":previous,"time":index})
                if previous!=plain(o.outcome): return False
            node=expression_from(e.new_claim.program)
            for b in e.boundaries:
                if b.witness not in e.observations or not b.reproduced: return False
                index=e.observations.index(b.witness)
                predicted=evaluate(expression_from(e.old_claim.program),context_for(b.witness,e.observations[:index],self.initial))
                if predicted!=b.old_prediction or predicted==plain(b.witness.outcome): return False
            if self.fingerprint(node)!=e.behavior_hash: return False
            return self.evaluate(e.new_claim,e.task,e.task.split)==e.score
        except (ProtocolError,TypeError,ValueError,KeyError): return False

    def compile_challenge(self,proposal,action_budget=8):
        if proposal.domain_id!=self.domain_id: raise ProtocolError("world domain mismatch")
        artifact=proposal.program
        try:
            node=expression_from(artifact)
            data=json.loads(artifact.source)
            # Author claims are code, never trusted labels. Final reward requires
            # actual interaction with the controller's frozen Scientist.
            models=tuple(freeze_expression(x) for x in data.get("models",()))
            audit=self._certifier.certify(artifact,proposal.track,models,200000,action_budget)
        except (ProtocolError,ValueError,TypeError,KeyError,IndexError):
            return Certification(Status.INVALID,None,"",digest(artifact),self.verifier_version,"malformed executable world",())
        self._usage=self._usage+Usage(verification_outputs=audit.evaluations)
        task=None
        if audit.status==Status.VERIFIED:
            task=TaskRef(digest((self.domain_id,"generated",artifact.content_hash)),self.domain_id,"train","generated/"+audit.fingerprint,0)
            self._refs[task.task_id]=task; self._worlds[task.task_id]=node
        return Certification(audit.status,task,digest(node),artifact.content_hash,audit.semantics_version,audit.reason,
                             (audit.fingerprint,plain(audit.witness),audit.evaluations,plain(audit.identification)))


def freeze_expression(value):
    from ..runtime.expression import freeze
    return freeze(value)


class StringTransitionAdapter(TransitionAdapter):
    domain_id="string-transducer"
    variables=(("input.text","string"),("previous","string"))
    constants=("","a")
    operators=("reverse","concat")
    output_type="string"
    initial=""

    def _actions(self):
        return tuple(Action.create("emit",{"text":text}) for text in ("","a","bc"))
