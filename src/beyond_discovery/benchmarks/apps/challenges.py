"""Independent single-program counterexamples and restricted APPS input creation.

An operator-reviewed validator defines legal inputs. Independently provisioned
reference programs must pass official tests and agree on a new output. Neither
answers nor reference code supplied by an Agent have certification authority.
"""
from dataclasses import dataclass
import json
from ...core.contracts import (Boundary,Observation,Certification,Status,TaskRef,digest,plain)
from ...core.protocol import Action,JSONValue,ProtocolError,CapabilityUnsupported
from .semantics import compare

@dataclass(frozen=True)
class SolutionCertificate:
    kind: str
    program_hash: str
    evidence_hash: str
    scope: str
    witness_hash: str
    repetitions: int
    verifier_version: str='apps-solution-refutation-v1'


def solution_boundary(adapter,evidence):
    """Reproduce a PUBLIC counterexample to one exact submitted program twice."""
    public=adapter.public_task(evidence.task);old=evidence.old_claim.program
    candidates=[]
    for observation in evidence.observations:
        args=observation.action.arguments.value
        if args['code']!=old.source:continue
        indexes=args.get('indexes',list(range(len(public['public_cases']))))
        feedback=observation.outcome.value
        for index,verdict in zip(indexes,feedback['verdicts']):
            if verdict is False:candidates.append(index)
    if not candidates:return (),()
    index=min(candidates);case=public['public_cases'][index];results=[]
    for _ in range(2):
        result=adapter._judge(old.source,[case],public['fn_name'],'independent-public-counterexample')
        if result.verdicts!=(False,) or len(result.outputs)!=1:return (),()
        results.append(result.outputs[0])
    if results[0]!=results[1]:return (),()
    action=Action.create('public_test',{'code':old.source,'indexes':[index]})
    witness=Observation(action,JSONValue.of({'expected':case['output']}),1)
    boundary=Boundary('exact Python artifact '+old.content_hash,witness,JSONValue.of(results[0]),JSONValue.of(case['output']),True,evidence.task.source_cluster)
    cert=SolutionCertificate('SOLUTION_REFUTED',old.content_hash,
        digest((evidence.provenance,evidence.task,evidence.observations,evidence.old_claim)),
        'one exact program; no claim about an algorithm or representation space',digest(case),2)
    return (boundary,),(cert,)


class RestrictedOracle:
    def __init__(self,configuration):
        # Configuration is operator-owned, private, immutable and never agent input.
        if configuration.get('format')!='apps-reviewed-oracles-v1':raise ProtocolError('Oracle configuration format')
        self.entries=configuration['tasks'];self.identity=digest(configuration)
        for entry in self.entries.values():
            if set(entry)!={'validator','references','review_id'} or not entry['review_id'] or not 2<=len(entry['references'])<=4:
                raise ProtocolError('Oracle requires reviewed validator and 2-4 independent references')
            if len(set(entry['references']))!=len(entry['references']):raise ProtocolError('duplicate reference programs are not independent')
    def certify(self,adapter,proposal):
        data=json.loads(proposal.program.source);world=digest((proposal,self.identity));generator=digest(proposal)
        def result(status,reason,task=None,certificate=()):
            return Certification(status,task,world,generator,adapter.verifier_version,reason,certificate)
        if proposal.track!='A' or proposal.domain_id!='apps' or proposal.program.language!='apps-challenge-v1':return result(Status.INVALID,'restricted APPS world schema')
        if not isinstance(data,dict) or set(data)!={'parent_task_id','old_artifact_hash','input'}:return result(Status.INVALID,'Agent cannot supply answers, reference programs or legality code')
        found=adapter.store.refs.get(data['parent_task_id'])
        if found is None or found[0].split!='train':return result(Status.INVALID,'generated tasks must derive from registered train tasks')
        parent=found[0];entry=self.entries.get(parent.task_id)
        if entry is None:return result(Status.PENDING,'CAPABILITY_UNSUPPORTED: no reviewed Oracle for this train task')
        old=adapter._submissions.get((parent.task_id,data['old_artifact_hash']))
        if old is None or old.provenance.agent_version!=adapter.frozen_scientist:return result(Status.INVALID,'old solution not bound to the frozen Scientist')
        private=adapter._data(parent);fn=private['fn_name'];new_input=data['input']
        if len(json.dumps(new_input).encode())>65536:return result(Status.INVALID,'generated input budget')
        legality=adapter._judge(entry['validator'],[{'input':[new_input],'output':True}],'validate','oracle-legality')
        if not legality.accepted:return result(Status.INVALID,'input rejected by independently reviewed legality validator')
        outputs=[]
        for code in entry['references']:
            checked=adapter._judge(code,private['private_cases'],fn,'oracle-official-conformance')
            if not checked.accepted:return result(Status.PENDING,'CAPABILITY_UNSUPPORTED: reference failed official conformance')
            # Output is independently computed, never taken from proposal/Agent.
            probe=adapter._judge(code,[{'input':new_input,'output':None}],fn,'oracle-new-input')
            if probe.status!='COMPLETE' or len(probe.outputs)!=1:return result(Status.PENDING,'Oracle execution did not complete')
            outputs.append(probe.outputs[0])
        if any(o!=outputs[0] for o in outputs[1:]):return result(Status.INVALID,'independent Oracles disagree')
        expected=outputs[0] if fn else '\n'.join(outputs[0])
        case={'input':new_input,'output':expected}
        if any(c['input']==new_input for c in private['private_cases']+private['public_cases']):return result(Status.INVALID,'duplicate input is not a new challenge')
        for _ in range(2):
            failure=adapter._judge(old.source,[case],fn,'oracle-distinguishing-reproduction')
            if failure.verdicts!=(False,):return result(Status.INVALID,'not a reproducible distinguishing wrong answer')
        task=TaskRef(world,'apps','train',parent.source_cluster,parent.seed)
        generated={**private,'private_cases':[case],'public_cases':[case],'solutions':[]}
        adapter._generated[world]={'data':generated,'certificate':{'oracle':self.identity,'parent':parent,'old_hash':old.content_hash}}
        return result(Status.VERIFIED,'legal, new, nontrivial, Oracle-agreed training counterexample; same parent problem scope',task,
                      (('oracle',self.identity),('parent',parent.task_id),('old',old.content_hash),('input',digest(new_input))))
