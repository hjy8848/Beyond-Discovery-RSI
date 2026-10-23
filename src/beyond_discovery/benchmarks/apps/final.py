"""One-shot official evaluation of previously frozen policies; no adaptation."""
from statistics import mean
from pathlib import Path
from ...core.archive import Archive,read_verified,create_json
from ...core.contracts import Usage,digest,plain
from ...core.controller import implementation_hash
from ...core.replay import snapshot_from
from ...core.interaction import episode
from ...core.protocol import ProtocolError,usage_within


def validate_frozen(path,adapter,limits):
    records=read_verified(path)
    if not records or records[0][0]!='preregistration':raise ProtocolError('missing frozen preregistration')
    prereg=records[0][1];freezes=[v for k,v in records if k=='freeze'];summaries=[v for k,v in records if k=='summary']
    if len(freezes)!=1 or not summaries or summaries[-1]['status']!='SMOKE_COMPLETE' or summaries[-1].get('test_unsealed'):
        raise ProtocolError('final requires a completed smoke with sealed test')
    if prereg['implementation_hash']!=implementation_hash():raise ProtocolError('frozen implementation changed')
    from ...runtime.coding import CodingResearchRuntime
    check=CodingResearchRuntime(limits=limits,method=prereg['runtime_context']['method'],replay_only=True)
    check.model_identities=prereg['runtime_context']['model_identities']
    if check.context(adapter)!=prereg['runtime_context']:raise ProtocolError('frozen data/judge/model budgets changed')
    if (Path(path)/'final_consumed.json').exists():raise ProtocolError('frozen run already used for one official final attempt')
    return prereg,freezes[0],digest(records),Path(path)


def run_final(source,adapter,runtime,out,max_tokens,max_launches,max_cases):
    prereg,frozen,source_hash,source_path=source
    c=snapshot_from(frozen['challenger']);s=snapshot_from(frozen['scientist'])
    if runtime.model_identities!=prereg['runtime_context']['model_identities'] or any(client.model!=s.model.base_model for client in runtime.clients.values()):raise ProtocolError('frozen model changed')
    runtime.validate_mode('full');runtime.context(adapter)
    caps=prereg['config']
    if (max_tokens,max_launches,max_cases)!=(caps['max_model_tokens'],caps['max_experiments'],caps['max_verification_outputs']):raise ProtocolError('final budget must equal frozen allocation')
    archive=Archive(out)
    create_json(source_path/'final_consumed.json',{'frozen_source_hash':source_hash,'final_output':str(Path(out).resolve())})
    final_prereg={**prereg,'config':{**prereg['config'],'final_evaluation':True},'train':[],'dev':[],
        'initial_challenger':c,'initial_scientist':s,'frozen_source_hash':source_hash,'selection':'DISABLED'}
    archive.append('preregistration',final_prereg);archive.marker('preregistration.json',final_prereg)
    archive.append('freeze',frozen);archive.marker('frozen.json',frozen)
    archive.marker('final_started.json',{'frozen_hash':digest(frozen)})
    adapter.authorize_final(digest(frozen))
    adapter.unseal_for_final();tasks=adapter.sample('test',prereg['config']['seed']+2)
    per=adapter.estimate_episode(min(s.experiment.action_budget,c.experiment.action_budget),prereg['config']['search_budget'])
    reserve=Usage(*(getattr(per,k)*len(tasks) for k in per.__dataclass_fields__))
    if not usage_within(reserve,Usage(experiments=max_launches,model_tokens=max_tokens,verification_outputs=max_cases)):
        raise ProtocolError('final worst-case reservation exceeds budget; no test execution')
    evidence=[]
    try:
        for task in tasks:
            item=episode(adapter,task,s,c,-1,prereg['config']['search_budget'],runtime=runtime)
            archive.append('episode',{'label':'sealed-final-test','scientist':s,'challenger':c,'evidence':item});evidence.append(item)
    finally:
        for event in runtime.model_events:archive.append('runtime_model_events',event)
        for event in adapter.judge_events:archive.append('runtime_judge_events',event)
        archive.append('final_cost',{'model_tokens':runtime.total_model_tokens,'judge':adapter.meter()})
    result={'status':'COMPLETE','test_score':mean(e.score.value for e in evidence),'episodes':len(evidence),
        'usage':plain(adapter.meter()+Usage(model_tokens=runtime.total_model_tokens)),'test_unsealed':True,'source':source_hash}
    archive.append('summary',result);archive.marker('summary.json',result)
    return result
