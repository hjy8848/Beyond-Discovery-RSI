"""Explicit APPS operations; smoke/suite never unseal official test."""
import argparse,json,os
import hashlib
from pathlib import Path
from dataclasses import replace
from ...core.contracts import plain,Usage
from ...core.controller import RunConfig,run_rsi,implementation_hash
from ...core.protocol import CapabilityUnsupported,ProtocolError,Action,negotiate,usage_within
from ...core.replay import replay
from ...core.archive import Archive,read_verified,create_json
from ...core.interaction import evaluate_pair
from ...core.conformance import check_conformance
from ...runtime.coding import CodingResearchRuntime,CodingLimits
from ...runtime.models import HTTPModelClient,LocalHFClient,TokenLedger
from .adapter import APPSAdapter
from .data import import_archive,APPSStore
from .judge import DockerJudge
from .challenges import RestrictedOracle

METHODS=('direct','self-repair','fixed-meta-rsi','evolved-meta-rsi')


def audit_self_repair(records,observations,control_trace,old_claim,new_claim):
    if len(records)!=2:raise ProtocolError('self-repair must archive exactly two model generations')
    first_prompt=json.loads(records[0]['request']['prompt'])
    second_prompt=json.loads(records[1]['request']['prompt'])
    public_observations=plain(observations)
    feedback=second_prompt.get('public_feedback')
    second_generation=next((i for i,row in enumerate(control_trace)
        if row.get('role')=='scientist' and row.get('name')=='code_generate' and row.get('arguments')==[True]
        and sum(1 for earlier in control_trace[:i] if earlier.get('role')=='scientist' and
            earlier.get('name')=='code_generate' and earlier.get('arguments')==[True])==1),None)
    prior_tests=sum(1 for row in control_trace[:second_generation or 0]
        if row.get('role')=='scientist' and row.get('name')=='code_test')
    expected_feedback=public_observations[:prior_tests]
    prior_feedback=(isinstance(feedback,list) and bool(feedback) and
        feedback==expected_feedback and len(expected_feedback)==prior_tests)
    feedback_is_public=prior_feedback and all(
        item.get('action',{}).get('operation')=='public_test' and
        item.get('outcome',{}).get('test_scope')=='declared-public-tests' and
        item.get('outcome',{}).get('status')!='NO_PUBLIC_TESTS'
        for item in feedback)
    result={'first_code_hash':old_claim.program.content_hash,
        'second_code_hash':new_claim.program.content_hash,
        'code_source_changed':old_claim.program.content_hash!=new_claim.program.content_hash,
        'first_and_second_versions_distinct':old_claim.program.version!=new_claim.program.version,
        'first_prompt_had_no_prior_code':first_prompt.get('prior_code') is None,
        'first_prompt_had_no_feedback':first_prompt.get('public_feedback')==[],
        'second_prompt_prior_code_matches_first':second_prompt.get('prior_code')==old_claim.program.source,
        'second_prompt_feedback_matches_available_public_observations':bool(prior_feedback),
        'feedback_test_scope':[item.get('outcome',{}).get('test_scope') for item in (feedback if isinstance(feedback,list) else [])],
        'feedback_status':[item.get('outcome',{}).get('status') for item in (feedback if isinstance(feedback,list) else [])],
        'feedback_verdicts':[item.get('outcome',{}).get('verdicts') for item in (feedback if isinstance(feedback,list) else [])],
        'prompt_uses_feedback':bool(feedback_is_public and
            second_prompt.get('prior_code')==old_claim.program.source)}
    if not result['prompt_uses_feedback']:
        raise ProtocolError('second Self-Repair prompt did not consume prior legal public-test feedback')
    return result


def judge(args):
    if not args.judge_image:raise CapabilityUnsupported('reviewed digest-pinned APPS_JUDGE_IMAGE not provisioned')
    return DockerJudge(args.judge_image,args.engine)


def adapter(args,limits):
    oracle=RestrictedOracle(json.loads(Path(args.oracle).read_text())) if args.oracle else None
    return APPSAdapter(APPSStore(args.store),judge(args),limits.tokens,args.task_limit,oracle=oracle,public_sidecar=args.public_sidecar)


def runtime(args,limits,method):
    if not args.model:raise CapabilityUnsupported('explicit code model ID/local model directory required')
    ledger=TokenLedger(args.max_model_tokens)
    if args.provider=='http':
        if not os.environ.get(args.key_env):raise CapabilityUnsupported('missing model credential environment variable '+args.key_env)
        clients={role:HTTPModelClient(args.model,args.endpoint,args.key_env,ledger,limits.input_tokens,args.proxy,args.allow_private_http) for role in ('scientist','challenger')}
    else:
        clients={role:LocalHFClient(args.model,role,ledger,device=args.device,seed=args.seed) for role in ('scientist','challenger')}
    return CodingResearchRuntime(clients,limits,method)


def optional_token_limit(value):
    """Parse a finite token cap or the explicit no-aggregate-cap setting."""
    if value is None or value.strip().lower() in ('unlimited','none','no-limit'):
        return None
    try: result=int(value)
    except (TypeError,ValueError) as exc: raise argparse.ArgumentTypeError('expected a positive integer or unlimited') from exc
    if result<1: raise argparse.ArgumentTypeError('token cap must be positive or unlimited')
    return result


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


def execute_train_only_smoke(args,out):
    """One explicit TRAIN episode with archived model calls and no DEV access."""
    if args.method!='self-repair' or args.model_calls!=2:
        raise ProtocolError('train-only smoke requires self-repair with exactly two model calls')
    if not args.task_id:raise ProtocolError('train-only-smoke requires --task-id')
    limits=CodingLimits(args.model_calls,args.prompt_token_bound,args.output_tokens)
    a=adapter(args,limits);r=runtime(args,limits,'self-repair')
    selected=a.store.refs.get(args.task_id)
    if selected is None:raise CapabilityUnsupported('task id is absent from the locked APPS task list')
    task,key,row=selected
    if task.split!='train':raise CapabilityUnsupported('train-only smoke accepts only locked TRAIN tasks')
    if row.get('execution_support')!='SUPPORTED':raise CapabilityUnsupported(row['execution_support'])
    public=a.public_task(task)
    if not public['public_cases']:raise CapabilityUnsupported('train-only self-repair requires a legal declared-public test')
    if not out:raise ProtocolError('train-only-smoke requires a new --out directory')
    config=RunConfig(generations=1,candidate_limit=1,action_budget=args.action_budget,seed=args.seed,
        mode='no_meta',base_model=args.model,max_model_tokens=args.max_model_tokens,
        max_experiments=args.max_judge_launches,max_verification_outputs=args.max_judge_cases,
        required_world_tracks=('A',),final_evaluation=False)
    config.validate();r.validate_mode(config.mode)
    if config.max_model_tokens is not None and (r.total_model_tokens>config.max_model_tokens or
            any(c.ledger.limit is not None and c.ledger.limit>config.max_model_tokens for c in r.clients.values())):
        raise ProtocolError('model token ledger exceeds train-only smoke budget')
    negotiated=negotiate(a.describe(),r.languages,config.action_budget,config.required_world_tracks)
    reservation=a.estimate_episode(config.action_budget,config.search_budget)
    limit=Usage(
        experiments=config.max_experiments,predictions=config.max_predictions,search=config.max_search,
        model_tokens=config.max_model_tokens,training_steps=config.max_training_steps,
        verification_outputs=config.max_verification_outputs)
    if not usage_within(reservation,limit):raise CapabilityUnsupported('train-only episode exceeds its preregistered resource budget')
    challenger,scientist=r.initial_pair(config)
    Path(out).parent.mkdir(parents=True,exist_ok=True)
    archive=Archive(out)
    prereg={'sandbox_limits':None,'config':config,'capabilities':a.describe(),'negotiated_protocol':negotiated,
        'runtime_id':r.runtime_id,'runtime_context':r.context(a),'implementation_hash':implementation_hash(),
        'train':(task,),'dev':(),'initial_challenger':challenger,'initial_scientist':scientist,
        'joint_fallback':'not_applicable_train_only_smoke','selection_order':'single explicit train task; no Gate selection',
        'meta_protocol':'not evaluated; isolated Self-Repair workflow execution',
        'evidence_status':'engineering smoke only; DEV and official TEST were not sampled'}
    archive.marker('preregistration.json',prereg);archive.append('preregistration',prereg)
    r.mode=config.mode;r.seed=config.seed
    try:
        evidence=evaluate_pair(a,(task,),scientist,challenger,0,config.search_budget,archive,'train-only-smoke',runtime=r)[0]
    except Exception as exc:
        # Failed provider calls can still reserve budget. Retain exact request
        # metadata and the Judge meter without exposing credentials or error bodies.
        archive.append('runtime_model_events',r.model_events)
        archive.append('runtime_judge_events',a.judge_events)
        failure={'status':'FAILED','error_type':type(exc).__name__,
            'model_events':r.model_events,'provider_reported_tokens':None,
            'model_budget_used':r.total_model_tokens,'judge_usage':plain(a.meter()),
            'judge_events':a.judge_events,'implementation_hash':implementation_hash()}
        archive.append('train_only_failure',failure)
        archive.marker('failure.json',failure)
        raise
    records=evidence.execution_records[0]['records']
    control_trace=next(item['trace'] for item in evidence.execution_records if item['kind']=='control-trace')
    repair_audit=audit_self_repair(records,evidence.observations,control_trace,evidence.old_claim,evidence.new_claim)
    archive.append('runtime_model_events',r.model_events)
    archive.append('runtime_judge_events',a.judge_events)
    frozen={'scientist':scientist,'challenger':challenger,'implementation_hash':implementation_hash(),
        'stop_reason':'single train-only interaction; no policy selection or meta-evolution'}
    archive.marker('frozen.json',frozen);archive.append('freeze',frozen)
    fresh=adapter(args,limits)
    replay_result=replay(out,fresh)
    model_rows=[{'role':row['request']['role'],'model':row['completion']['model'],
        'input_tokens':row['completion']['input_tokens'],'output_tokens':row['completion']['output_tokens'],
        'prompt_sha256':hashlib.sha256(row['request']['prompt'].encode()).hexdigest(),
        'completion_sha256':hashlib.sha256(row['completion']['text'].encode()).hexdigest()}
        for row in records]
    report={'status':'TRAIN_ONLY_SMOKE_COMPLETED','task':{'task_id':task.task_id,'archive_key':key,
        'split':task.split,'support':row.get('execution_support'),'dataset_manifest_hash':a.store.manifest['manifest_hash'],
        'question_sha256':row['files']['question.txt']},
        'difficulty_provenance':a.public_sidecar.get('selection_provenance',{}),
        'public_sidecar_sha256':a.public_sidecar['sha256'],'judge_identity':a.judge.identity,
        'privacy':{'dev_tasks_sampled':0,'official_test_unsealed':False,'agent_feedback_scope':'declared public question examples only'},
        'code_version':implementation_hash(),'model':{'id':args.model,'endpoint':args.endpoint,
        'requests':len(model_rows),'actual_tokens':sum(x['input_tokens']+x['output_tokens'] for x in model_rows),
        'calls':model_rows},'self_repair':repair_audit,
        'judge':{'initial_usage':plain(a.meter()),'initial_events':a.judge_events,
        'replay_usage':plain(fresh.meter()),'replay_events':fresh.judge_events},
        'replay':replay_result,'gate':'NOT_RUN: train-only smoke intentionally excludes development evaluation',
        'scientific_status':'H1-H4 UNVERIFIED; this single engineering smoke is not a benchmark result'}
    create_json(Path(out)/'apps_audit.json',report)
    return report


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('import','check','smoke','train-only-smoke','suite','replay','conformance','official-parity','final'))
    p.add_argument('--archive');p.add_argument('--sha256');p.add_argument('--out')
    p.add_argument('--store');p.add_argument('--judge-image',default=os.environ.get('APPS_JUDGE_IMAGE'));p.add_argument('--engine',default='docker')
    p.add_argument('--oracle',help='private operator-reviewed Oracle configuration, never Agent supplied')
    p.add_argument('--provider',choices=('http','hf'),default='http');p.add_argument('--model',default=os.environ.get('APPS_MODEL'))
    p.add_argument('--endpoint',default='https://api.openai.com/v1');p.add_argument('--key-env',default='APPS_API_KEY');p.add_argument('--proxy');p.add_argument('--allow-private-http',action='store_true')
    p.add_argument('--public-sidecar',help='operator-reviewed examples from locked question statements; SHA256 bound per train task')
    p.add_argument('--task-id',help='explicit locked task id for the dev-isolated train-only smoke')
    p.add_argument('--device',default='cpu');p.add_argument('--prompt-token-bound',type=int,default=32768)
    p.add_argument('--output-tokens',type=int,default=2048);p.add_argument('--model-calls',type=int,default=2)
    p.add_argument('--max-model-tokens',type=optional_token_limit,default=2000000,
                   help='aggregate model token cap; use unlimited for accounting without a total token stop')
    p.add_argument('--max-judge-launches',type=int,default=100000);p.add_argument('--max-judge-cases',type=int,default=10000000)
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
            elif args.command=='train-only-smoke':
                result=execute_train_only_smoke(args,args.out)
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
