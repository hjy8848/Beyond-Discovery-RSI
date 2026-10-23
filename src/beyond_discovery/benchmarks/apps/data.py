"""Immutable APPS archive import and sealed task store. No automatic test sampling."""
from pathlib import Path,PurePosixPath
import hashlib,json,re,tarfile
from ...core.contracts import canonical,digest,TaskRef
from ...core.protocol import ProtocolError,CapabilityUnsupported
from .semantics import REVISION,VENDOR_SHA256

SOURCE_URL='https://people.eecs.berkeley.edu/~hendrycks/APPS.tar.gz'
LOCKED_ARCHIVE_SHA256='6ef8e98ecca10b0159df0da4b524ecc1ca782a3b9473c57fc547ebccbbc2d0ca'
LOCKED_MANIFEST_SHA256='cbc0ebbdd1d6fb262d88c6d45cdd46b8f8c5ac1a7daa0869531a9382ee191328'
FILES={'question.txt','input_output.json','solutions.json','starter_code.py','public_tests.json'}


def file_hash(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''): h.update(chunk)
    return h.hexdigest()


def import_archive(archive,out,expected_sha256):
    if not re.fullmatch('[0-9a-f]{64}',expected_sha256) or file_hash(archive)!=expected_sha256:
        raise ProtocolError('APPS archive checksum mismatch; data version must be frozen before import')
    out=Path(out);out.mkdir(parents=True,exist_ok=False,mode=0o700);(out/'blobs').mkdir(mode=0o700)
    tasks={};total=0
    with tarfile.open(archive,'r|*') as source:
        for member in source:
            parts=PurePosixPath(member.name).parts
            if '..' in parts or member.issym() or member.islnk(): raise ProtocolError('unsafe APPS archive member')
            if not member.isfile() or len(parts)<3 or parts[-1] not in FILES: continue
            split,number,name=parts[-3:]
            if split not in ('train','test') or not number.isdigit(): continue
            if member.size>134217728: raise ProtocolError('APPS member resource limit')
            total+=member.size
            if total>4294967296: raise ProtocolError('APPS archive resource limit')
            staging=out/'blobs'/'.importing';hasher=hashlib.sha256()
            with source.extractfile(member) as original,staging.open('xb') as destination:
                for chunk in iter(lambda:original.read(1048576),b''):
                    destination.write(chunk);hasher.update(chunk)
            sha=hasher.hexdigest();path=out/'blobs'/sha
            if path.exists():staging.unlink()
            else:staging.rename(path);path.chmod(0o600)
            row=tasks.setdefault(split+'/'+number,{'official_split':split,'files':{}})
            if name in row['files']: raise ProtocolError('duplicate APPS task file')
            row['files'][name]=sha
    if not tasks: raise ProtocolError('archive contains no APPS tasks')
    def blob(sha):return (out/'blobs'/sha).read_text()
    for key,row in tasks.items():
        if not {'question.txt','input_output.json'}<=set(row['files']):
            row['fn_name']=None;row['case_count']=0;row['public_case_count']=0
            row['cluster']=digest(('unsupported missing APPS files',key))
            row['split']='test' if row['official_split']=='test' else ('dev' if int(row['cluster'],16)%5==0 else 'train')
            row['execution_support']='CAPABILITY_UNSUPPORTED: official task is missing question or input/output data'
            continue
        tests=json.loads(blob(row['files']['input_output.json']))
        valid_pairs=isinstance(tests,dict) and isinstance(tests.get('inputs'),list) and isinstance(tests.get('outputs'),list) and bool(tests['inputs']) and len(tests['inputs'])==len(tests['outputs'])
        row['fn_name']=tests.get('fn_name') if isinstance(tests,dict) else None
        row['case_count']=len(tests.get('inputs',[])) if isinstance(tests,dict) else 0
        row['execution_support']=('CAPABILITY_UNSUPPORTED: official task has empty/mismatched test pairs' if not valid_pairs else
            ('CAPABILITY_UNSUPPORTED: task JSON exceeds 64 MiB runtime bound' if (out/'blobs'/row['files']['input_output.json']).stat().st_size>67108864 else
            ('CAPABILITY_UNSUPPORTED: more than 10000 cases' if row['case_count']>10000 else 'SUPPORTED')))
        question=blob(row['files']['question.txt'])
        row['cluster']=digest(' '.join(question.split()))
        row['split']='test' if row['official_split']=='test' else ('dev' if int(row['cluster'],16)%5==0 else 'train')
        public=json.loads(blob(row['files']['public_tests.json'])) if 'public_tests.json' in row['files'] else []
        if not isinstance(public,list) or any(set(t)!={'input','output'} for t in public): raise ProtocolError('public test sidecar schema')
        row['public_case_count']=len(public)
    test_clusters={r['cluster'] for r in tasks.values() if r['split']=='test'}
    for row in tasks.values():
        if row['official_split']=='train' and row['cluster'] in test_clusters: row['split']='excluded-overlap'
    manifest={'format':'apps-manifest-v1','archive_sha256':expected_sha256,'source_url':SOURCE_URL,
              'evaluator_revision':REVISION,'evaluator_sha256':VENDOR_SHA256,
              'dev_rule':'official-train question-cluster SHA256 modulo 5 == 0',
              'public_test_rule':'explicit operator-reviewed sidecar only; never inferred from hidden tests',
              'tasks':dict(sorted(tasks.items()))}
    manifest['manifest_hash']=digest(manifest)
    (out/'manifest.json').write_text(canonical(manifest));(out/'manifest.json').chmod(0o600)
    return manifest


class APPSStore:
    def __init__(self,root):
        self.root=Path(root).resolve();self.manifest=json.loads((self.root/'manifest.json').read_text())
        data={k:v for k,v in self.manifest.items() if k!='manifest_hash'}
        if digest(data)!=self.manifest['manifest_hash'] or data['evaluator_sha256']!=VENDOR_SHA256 or data['evaluator_revision']!=REVISION:
            raise ProtocolError('APPS manifest/evaluator version mismatch')
        self.sealed=True;self.loaded=[]
        self.refs={}
        for key,row in self.manifest['tasks'].items():
            if row['split'] not in ('train','dev','test'):continue
            ref=TaskRef(digest((self.manifest['manifest_hash'],key)),'apps',row['split'],row['cluster'],0)
            self.refs[ref.task_id]=(ref,key,row)
    def sample(self,split,seed):
        if split=='test' and self.sealed: raise PermissionError('official APPS test remains sealed')
        if split not in ('train','dev','test'):raise ProtocolError('unknown APPS split')
        return tuple(sorted((r[0] for r in self.refs.values() if r[0].split==split),key=lambda r:digest((seed,r.task_id))))
    def load(self,task):
        item=self.refs.get(task.task_id)
        if item is None or item[0]!=task:raise PermissionError('forged APPS task')
        if task.split=='test' and self.sealed:raise PermissionError('official test data sealed')
        row=item[2]
        if row.get('execution_support','SUPPORTED')!='SUPPORTED':raise CapabilityUnsupported(row['execution_support'])
        self.loaded.append(item[1]);data={}
        for name,sha in row['files'].items():
            if name=='solutions.json':continue  # references are never needed for inference or candidate generation
            if not re.fullmatch('[0-9a-f]{64}',sha):raise ProtocolError('invalid blob identifier')
            content=(self.root/'blobs'/sha).read_bytes()
            if hashlib.sha256(content).hexdigest()!=sha:raise ProtocolError('APPS task blob corruption')
            data[name]=content.decode()
        tests=json.loads(data['input_output.json'])
        return {'question':data['question.txt'],'starter':data.get('starter_code.py',''),'fn_name':tests.get('fn_name'),
                'private_cases':[{'input':x,'output':y} for x,y in zip(tests['inputs'],tests['outputs'])],
                'public_cases':json.loads(data.get('public_tests.json','[]')),'solutions':[]}


def load_public_sidecar(path,store):
    """Load operator-provided examples extracted from a locked question statement.

    This sidecar is separate from the archive lock. Only locked, supported train
    tasks are accepted, and each entry is bound to the exact question blob hash.
    """
    raw=Path(path).read_bytes()
    if len(raw)>4*1024*1024: raise ProtocolError('APPS public sidecar exceeds size bound')
    def reject_constant(value): raise ValueError('non-finite JSON constant')
    try: data=json.loads(raw,parse_constant=reject_constant)
    except (ValueError,UnicodeDecodeError): raise ProtocolError('invalid APPS public sidecar JSON')
    if not isinstance(data,dict) or set(data)-{'format','dataset_manifest_hash','entries','selection_provenance'} or not {'format','dataset_manifest_hash','entries'}<=set(data):
        raise ProtocolError('APPS public sidecar schema')
    if data['format']!='apps-public-sidecar-v1' or data['dataset_manifest_hash']!=store.manifest['manifest_hash']:
        raise ProtocolError('APPS public sidecar dataset lock mismatch')
    if 'selection_provenance' in data and not isinstance(data['selection_provenance'],dict):
        raise ProtocolError('APPS sidecar selection provenance schema')
    entries=data['entries']
    if not isinstance(entries,list) or len(entries)>1000: raise ProtocolError('APPS public sidecar entry bound')
    cases_by_task={}
    for entry in entries:
        required={'task_id','archive_key','question_sha256','source','cases'}
        if not isinstance(entry,dict) or set(entry)!=required: raise ProtocolError('APPS public sidecar entry schema')
        item=store.refs.get(entry['task_id'])
        if item is None or item[1]!=entry['archive_key'] or item[0].split!='train':
            raise ProtocolError('APPS public sidecar must target a locked train task')
        row=item[2]
        if row.get('execution_support')!='SUPPORTED': raise CapabilityUnsupported('APPS sidecar task is unsupported')
        if entry['source']!='official-question-example' or entry['question_sha256']!=row.get('files',{}).get('question.txt'):
            raise ProtocolError('APPS sidecar source/question hash mismatch')
        cases=entry['cases']
        if not isinstance(cases,list) or not 1<=len(cases)<=100 or entry['task_id'] in cases_by_task:
            raise ProtocolError('APPS public sidecar case bound/duplicate task')
        for case in cases:
            if not isinstance(case,dict) or set(case)!={'input','output'}:
                raise ProtocolError('APPS public sidecar case schema')
            if len(json.dumps(case,ensure_ascii=False,allow_nan=False).encode())>1000000:
                raise ProtocolError('APPS public sidecar case size')
        cases_by_task[entry['task_id']]=cases
    return {'cases':cases_by_task,'sha256':hashlib.sha256(raw).hexdigest(),
            'selection_provenance':data.get('selection_provenance',{})}
