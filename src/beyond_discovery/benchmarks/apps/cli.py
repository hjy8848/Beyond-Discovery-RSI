"""Explicit APPS operations; smoke/suite never unseal official test."""
import argparse,json,os
from pathlib import Path
from dataclasses import replace
from ...core.contracts import plain
from ...core.controller import RunConfig,run_rsi
from ...core.protocol import CapabilityUnsupported,ProtocolError,Action
from ...core.replay import replay
from ...core.archive import read_verified,create_json
from ...core.conformance import check_conformance
from ...runtime.coding import CodingResearchRuntime,CodingLimits
from ...runtime.models import HTTPModelClient,LocalHFClient,TokenLedger
from .adapter import APPSAdapter
from .data import import_archive,APPSStore
from .judge import DockerJudge
from .challenges import RestrictedOracle

METHODS=('direct','self-repair','fixed-meta-rsi','evolved-meta-rsi')


def judge(args):
    if not args.judge_image:raise CapabilityUnsupported('reviewed digest-pinned APPS_JUDGE_IMAGE not provisioned')
    return DockerJudge(args.judge_image,args.engine)


def adapter(args,limits):
    oracle=RestrictedOracle(json.loads(Path(args.oracle).read_text())) if args.oracle else None
    return APPSAdapter(APPSStore(args.store),judge(args),limits.tokens,args.task_limit,oracle=oracle)


def runtime(args,limits,method):
    if not args.model:raise CapabilityUnsupported('explicit code model ID/local model directory required')
    ledger=TokenLedger(args.max_model_tokens)
    if args.provider=='http':
        if not os.environ.get(args.key_env):raise CapabilityUnsupported('missing model credential environment variable '+args.key_env)
        clients={role:HTTPModelClient(args.model,args.endpoint,args.key_env,ledger,limits.input_tokens,args.proxy) for role in ('scientist','challenger')}
    else:
        clients={role:LocalHFClient(args.model,role,ledger,device=args.device,seed=args.seed) for role in ('scientist','challenger')}
    return CodingResearchRuntime(clients,limits,method)


def audit_report(out,summary):
    records=read_verified(out);episodes=[data['evidence'] for kind,data in records if kind=='episode']
    valid={(e['task']['task_id'],c['program_hash'],c['witness_hash']) for e in episodes for c in e['boundary_certificates'] if c['kind']=='SOLUTION_REFUTED'}
    return {'usage':summary['usage'],'episodes':len(episodes),'valid_counterexamples':len(valid),
        'verified_worlds':sum(k=='world_attempt' and v['certification']['status']=='VERIFIED' for k,v in records),
        'gate_results':[v for k,v in records if k=='gate'],
        'behavior_changes':[{'role':v['role'],'effective_new_method':v['effective_new_method'],'meta_rollback':v['meta_rollback']} for k,v in records if k=='gate'],
        'replay':{'status':'NOT_RUN'},'scientific_status':'H1-H4 UNVERIFIED; public APPS may be in pretraining'}


def execute(args,method,out):
    limits=CodingLimits(args.model_calls,args.prompt_token_bound,args.output_tokens)
    a=adapter(args,limits);r=runtime(args,limits,method)
    config=RunConfig(generations=args.generations,candidate_limit=args.candidates,action_budget=args.action_budget,
        required_world_tracks=('A',),final_evaluation=False,seed=args.seed,base_model=args.model,
        max_model_tokens=args.max_model_tokens,max_experiments=args.max_judge_launches,max_verification_outputs=args.max_judge_cases)
    summary=run_rsi(a,out,config,runtime=r);report=audit_report(out,summary)
    if args.verify_replay:
        fresh=adapter(args,limits);report['replay']=replay(out,fresh);report['replay']['additional_judge_usage']=plain(fresh.meter())
    create_json(Path(out)/'apps_audit.json',report)
    return {'method':method,'summary':summary,'audit':report}


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('import','check','smoke','suite','replay','conformance','official-parity','final'))
    p.add_argument('--archive');p.add_argument('--sha256');p.add_argument('--out')
    p.add_argument('--store');p.add_argument('--judge-image',default=os.environ.get('APPS_JUDGE_IMAGE'));p.add_argument('--engine',default='docker')
    p.add_argument('--oracle',help='private operator-reviewed Oracle configuration, never Agent supplied')
    p.add_argument('--provider',choices=('http','hf'),default='http');p.add_argument('--model',default=os.environ.get('APPS_MODEL'))
    p.add_argument('--endpoint',default='https://api.openai.com/v1');p.add_argument('--key-env',default='APPS_API_KEY');p.add_argument('--proxy')
    p.add_argument('--device',default='cpu');p.add_argument('--prompt-token-bound',type=int,default=32768)
    p.add_argument('--output-tokens',type=int,default=2048);p.add_argument('--model-calls',type=int,default=2)
    p.add_argument('--max-model-tokens',type=int,default=2000000);p.add_argument('--max-judge-launches',type=int,default=100000);p.add_argument('--max-judge-cases',type=int,default=10000000)
    p.add_argument('--task-limit',type=int,default=4);p.add_argument('--generations',type=int,default=1);p.add_argument('--candidates',type=int,default=1)
    p.add_argument('--action-budget',type=int,default=8);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--method',choices=METHODS,default='evolved-meta-rsi');p.add_argument('--verify-replay',action='store_true')
    p.add_argument('--frozen-run',action='append',help='completed frozen smoke run(s); all validated before any official test unseal')
    return p


def main(argv=None):
    p=parser();args=p.parse_args(argv)
    try:
        if args.command=='import':
            if not args.archive or not args.sha256 or not args.out:p.error('import requires --archive --sha256 --out')
            manifest=import_archive(args.archive,args.out,args.sha256)
            from collections import Counter
            counts=Counter(row['split'] for row in manifest['tasks'].values())
            result={'status':'IMPORTED','manifest_hash':manifest['manifest_hash'],'tasks':len(manifest['tasks']),'split_counts':dict(counts),'unsupported':sum(row['execution_support']!='SUPPORTED' for row in manifest['tasks'].values())}
        elif args.command=='official-parity':
            from .parity import check_official_parity
            result=check_official_parity(judge(args))
        else:
            if not args.store:p.error('--store required')
            if args.task_limit is not None and args.task_limit<1:p.error('positive --task-limit required')
            limits=CodingLimits(args.model_calls,args.prompt_token_bound,args.output_tokens)
            if args.command=='check':
                a=adapter(args,limits)
                result={'status':'AVAILABLE','context':a.public_context(),'train':len(a.sample('train',args.seed)),'dev':len(a.sample('dev',args.seed+1)),'official_test':'SEALED'}
            elif args.command=='conformance':
                a=adapter(args,limits);r=runtime(args,limits,args.method)
                result=plain(check_conformance(a,runtime=r,config=RunConfig(base_model=args.model),actions=(Action.create('public_test',{'code':'pass'}),)))
                result['usage']={'model_tokens':r.total_model_tokens,'judge':plain(a.meter())}
            elif args.command=='replay':
                if not args.out:p.error('replay requires existing --out')
                result=replay(args.out,adapter(args,limits))
            elif args.command=='final':
                if not args.out or not args.frozen_run:p.error('final requires new --out and --frozen-run for every policy to evaluate')
                from .final import validate_frozen,run_final
                # Validate EVERY frozen strategy before accessing ANY test data.
                sources=[validate_frozen(path,adapter(args,limits),limits) for path in args.frozen_run]
                Path(args.out).mkdir(parents=True,exist_ok=False,mode=0o700)
                result=[run_final(source,adapter(args,limits),runtime(args,limits,source[0]['runtime_context']['method']),Path(args.out)/str(i),args.max_model_tokens,args.max_judge_launches,args.max_judge_cases) for i,source in enumerate(sources)]
            elif args.command=='smoke':
                if not args.out:p.error('smoke requires new --out')
                result=execute(args,args.method,args.out)
            else:
                if not args.out:p.error('suite requires new --out')
                Path(args.out).mkdir(parents=True,exist_ok=False,mode=0o700)
                runs=[execute(args,method,Path(args.out)/method) for method in METHODS]
                result={'allocation_matched':True,'actual_costs_may_differ':True,'test_unsealed':False,'runs':runs}
                create_json(Path(args.out)/'comparison.json',result)
        print(json.dumps(plain(result),ensure_ascii=False));return 0
    except CapabilityUnsupported as exc:
        print(json.dumps({'status':'CAPABILITY_UNSUPPORTED','reason':str(exc),'real_model_smoke':'NOT_VERIFIED'}));return 2
    except (ProtocolError,ValueError,OSError,RuntimeError) as exc:
        print(json.dumps({'status':'FAILED','error_type':type(exc).__name__,'reason':str(exc)}));return 1

if __name__=='__main__':raise SystemExit(main())
