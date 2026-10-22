"""Preregistered budget-matched method runs; no aggregate across domain scores."""
from dataclasses import replace
from pathlib import Path
from ..core.controller import run_rsi
from ..core.contracts import plain

METHODS=('full','fixed','fixed_challenger','random_challenger','hardness_first',
         'program_evolution','reflection','no_diagnosis','no_restructuring','no_falsification','no_meta')


def comparison_suite(factory,root,config,methods=METHODS):
    if not methods or not set(methods)<=set(METHODS) or len(set(methods))!=len(methods):
        raise ValueError('invalid methods')
    root=Path(root); root.mkdir(parents=True,exist_ok=False)
    results={}
    for method in methods:
        # New adapter, sealed pool and role states per run. Shared preregistered
        # ceilings/opportunities; actual expenditures reported separately.
        result=run_rsi(factory(),root/method,replace(config,mode=method))
        results[method]={'status':result['status'],'test_score':result['test_score'],'usage':result['usage'],
                         'stop_reason':result['stop_reason']}
    return {'domain':factory.domain_id,'config':plain(config),'results':results,
            'interpretation':'engineering controls only; multiple seeds and statistical power required for H1-H4'}
