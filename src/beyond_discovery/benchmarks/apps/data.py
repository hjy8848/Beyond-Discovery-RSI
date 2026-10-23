"""Immutable APPS archive import and sealed task store. No automatic test sampling."""
from pathlib import Path,PurePosixPath
import hashlib,json,re,tarfile
from ...core.contracts import canonical,digest,TaskRef
from ...core.protocol import ProtocolError
from .semantics import REVISION,VENDOR_SHA256

SOURCE_URL='https://people.eecs.berkeley.edu/~hendrycks/APPS.tar.gz'
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
            if member.size>67108864: raise ProtocolError('APPS member resource limit')
            total+=member.size
            if total>4294967296: raise ProtocolError('APPS archive resource limit')
            content=source.extractfile(member).read();sha=hashlib.sha256(content).hexdigest()
            path=out/'blobs'/sha
            if not path.exists():
                with path.open('xb') as f:f.write(content)
                path.chmod(0o600)
            row=tasks.setdefault(split+'/'+number,{'official_split':split,'files':{}})
            if name in row['files']: raise ProtocolError('duplicate APPS task file')
            row['files'][name]=sha
    if not tasks: raise ProtocolError('archive contains no APPS tasks')
    def blob(sha):return (out/'blobs'/sha).read_text()
    for key,row in tasks.items():
        if not {'question.txt','input_output.json'}<=set(row['files']): raise ProtocolError('incomplete APPS task')
        tests=json.loads(blob(row['files']['input_output.json']))
        if not tests.get('inputs') or len(tests['inputs'])!=len(tests.get('outputs',[])): raise ProtocolError('invalid official APPS test pairs')
        row['fn_name']=tests.get('fn_name');row['case_count']=len(tests['inputs'])
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
        self.loaded.append(item[1]);row=item[2];data={}
        for name,sha in row['files'].items():
            if not re.fullmatch('[0-9a-f]{64}',sha):raise ProtocolError('invalid blob identifier')
            content=(self.root/'blobs'/sha).read_bytes()
            if hashlib.sha256(content).hexdigest()!=sha:raise ProtocolError('APPS task blob corruption')
            data[name]=content.decode()
        tests=json.loads(data['input_output.json'])
        return {'question':data['question.txt'],'starter':data.get('starter_code.py',''),'fn_name':tests.get('fn_name'),
                'private_cases':[{'input':x,'output':y} for x,y in zip(tests['inputs'],tests['outputs'])],
                'public_cases':json.loads(data.get('public_tests.json','[]')),'solutions':json.loads(data.get('solutions.json','[]'))}
