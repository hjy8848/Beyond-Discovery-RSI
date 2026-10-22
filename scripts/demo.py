#!/usr/bin/env python3
"""Reproduce both complete local loops and deterministic replay; no provider calls."""
import argparse
import json
from pathlib import Path
import time
from beyond_discovery.adapters.transition import TransitionAdapter,StringTransitionAdapter
from beyond_discovery.core.controller import RunConfig,run_rsi,implementation_hash
from beyond_discovery.core.replay import replay
from beyond_discovery.core.archive import read_verified
from beyond_discovery.core.contracts import digest


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',required=True,help='new output directory')
    parser.add_argument('--generations',type=int,default=2)
    args=parser.parse_args()
    root=Path(args.out); root.mkdir(parents=True,exist_ok=False)
    report={'implementation_hash':implementation_hash(),'domains':{},'scientific_claims':'H1-H4 unverified'}
    for cls in (TransitionAdapter,StringTransitionAdapter):
        start=time.perf_counter(); out=root/cls.domain_id
        summary=run_rsi(cls(),out,RunConfig(generations=args.generations,candidate_limit=2))
        audited=replay(out,cls()); records=read_verified(out)
        gates=[d for k,d in records if k=='gate']
        worlds=[d['certification'] for k,d in records if k=='world_attempt']
        report['domains'][cls.domain_id]={'status':summary['status'],'test_score':summary['test_score'],'usage':summary['usage'],
            'elapsed_seconds_including_replay':time.perf_counter()-start,'replay':audited,
            'world_statuses':[w['status'] for w in worlds],
            'task_decisions':[d['task_gate']['decision'] for d in gates],
            'meta_decisions':[d['meta_gate']['decision'] if d['meta_gate'] else None for d in gates],
            'meta_rollbacks':sum(d['meta_rollback'] for d in gates),
            'journal_sha256':__import__('hashlib').sha256((out/'journal.jsonl').read_bytes()).hexdigest(),
            'summary_hash':digest(summary)}
    (root/'demo-report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
