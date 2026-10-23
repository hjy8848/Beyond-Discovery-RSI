import ast,hashlib,json,shutil,tempfile,unittest,tarfile,io
from pathlib import Path
from unittest.mock import patch
from apps_fixtures import fixture_store,ProtocolJudge
from beyond_discovery.benchmarks.apps.data import APPSStore
from beyond_discovery.benchmarks.apps.adapter import APPSAdapter
from beyond_discovery.benchmarks.apps.judge import DockerJudge,ContainerFailure
from beyond_discovery.benchmarks.apps.semantics import compare,prepare,official,VENDOR,VENDOR_SHA256
from beyond_discovery.benchmarks.apps.wire import encode,decode
from beyond_discovery.core.protocol import Action,CapabilityUnsupported,ProtocolError


class APPSAdapterTests(unittest.TestCase):
    def test_pinned_official_semantics_golden_cases(self):
        self.assertEqual(hashlib.sha256(VENDOR.read_bytes()).hexdigest(),VENDOR_SHA256)
        for output,expected,answer in [([' 3 '],'3',True),(['1.000001'],'1',True),(['1.1'],'1',False),(['2','1'],'1\n2',True),(['wrong'],'3',False)]:
            self.assertEqual(compare(output,expected,None),answer)
        self.assertTrue(compare((1,2),[[1,2]],'solve'))
        self.assertFalse(compare([2,1],[1,2],'solve'))
        self.assertIn('def code():',prepare('print(input())',None))
    def test_manifest_partitions_sealing_and_public_projection(self):
        with tempfile.TemporaryDirectory() as root:
            store=fixture_store(root);adapter=APPSAdapter(store,ProtocolJudge(),testing=True)
            with self.assertRaises(PermissionError):adapter.sample('test',0)
            self.assertTrue(adapter.sample('train',0));self.assertTrue(adapter.sample('dev',0))
            task=adapter.sample('train',0)[0];public=adapter.public_task(task)
            self.assertNotIn('PRIVATE_REFERENCE_SENTINEL',json.dumps(public))
            self.assertNotIn('[2, 3]',json.dumps(public))
            self.assertTrue(all(not p.startswith('test/') for p in store.loaded))
            session=adapter.begin(task,1);observation=adapter.step(session,Action.create('public_test',{'code':'def add(a,b): return a+b'}))
            self.assertEqual(observation.outcome.value['verdicts'],[True])
            with self.assertRaises(ProtocolError):adapter.step(session,observation.action)
            adapter.close(session)
            with self.assertRaises(PermissionError):adapter.initial_observation(session)
    def test_missing_container_is_unsupported_not_host_fallback(self):
        with patch('shutil.which',return_value=None):
            with self.assertRaises(CapabilityUnsupported):DockerJudge('reviewed/apps@sha256:'+'a'*64)
        with self.assertRaises(ValueError):DockerJudge('python:latest')
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(TypeError):APPSAdapter(fixture_store(root),ProtocolJudge())
    def test_container_command_has_no_host_mount_or_privileges(self):
        with patch.object(DockerJudge,'preflight'),patch('shutil.which',return_value='/usr/bin/docker'):
            judge=DockerJudge('reviewed/apps@sha256:'+'a'*64)
        cmd=judge.command('test-container',3)
        for flag in ('--network=none','--read-only','--cap-drop=ALL','--pids-limit=32','--pull=never','--user=65534:65534'):self.assertIn(flag,cmd)
        self.assertNotIn('--privileged',cmd);self.assertNotIn('-v',cmd)
    def test_data_only_wire_cannot_unpickle_and_retains_integer_keys(self):
        value={1:(2,3)};self.assertEqual(decode(encode(value)),value)
        with self.assertRaises(ValueError):decode({'__reduce__':'os.system'})
    def test_blob_tamper_and_forged_refs_rejected(self):
        from dataclasses import replace
        with tempfile.TemporaryDirectory() as root:
            store=fixture_store(root);task=store.sample('train',0)[0]
            with self.assertRaises(PermissionError):store.load(replace(task,split='dev'))
            sha=store.refs[task.task_id][2]['files']['question.txt'];(store.root/'blobs'/sha).write_text('tampered')
            with self.assertRaises(ProtocolError):store.load(task)

    def test_timeout_and_output_limit_are_failures_not_acceptances(self):
        with patch.object(DockerJudge,'preflight'),patch('shutil.which',return_value='/usr/bin/docker'):
            judge=DockerJudge('reviewed/apps@sha256:'+'a'*64)
        for reason in ('TIMEOUT','OUTPUT_LIMIT','WORKER_EXIT'):
            with patch('beyond_discovery.benchmarks.apps.judge.DockerSession') as cls:
                session=cls.return_value.__enter__.return_value;session.ready='READY'
                session.exchange.side_effect=ContainerFailure(reason)
                result=judge.grade('while True: pass',[{'input':'','output':'0'}])
            self.assertFalse(result.accepted);self.assertEqual(result.status,reason)
            self.assertEqual(result.launches,1);self.assertEqual(result.cases,1)
        with self.assertRaises(ProtocolError):judge.grade('x'*65537,[{'input':'','output':''}])
    def test_official_v1_task_lock_and_private_payload_hash_list(self):
        from pathlib import Path
        lockpath=Path(__file__).parents[1]/'src/beyond_discovery/benchmarks/apps/dataset-lock.json'
        lock=json.loads(lockpath.read_text());tasklist=lockpath.with_name(lock['task_list'])
        self.assertEqual(hashlib.sha256(tasklist.read_bytes()).hexdigest(),lock['task_list_sha256'])
        rows=[json.loads(line) for line in tasklist.read_text().splitlines()]
        self.assertEqual(len(rows),lock['official_train_tasks']+lock['official_test_tasks'])
        self.assertEqual(sum(x['split']=='test' for x in rows),lock['official_test_tasks'])
        self.assertGreater(lock['unsupported_tasks'],0)
        self.assertNotIn('PRIVATE_REFERENCE_SENTINEL',tasklist.read_text())
        self.assertNotIn('"inputs"',tasklist.read_text())
    def test_official_task_with_unusable_test_manifest_stays_unsupported(self):
        with tempfile.TemporaryDirectory() as root:
            archive=Path(root)/'apps.tar';out=Path(root)/'store'
            with tarfile.open(archive,'w') as tar:
                for name,value in [('question.txt','task'),('input_output.json','{"inputs":[],"outputs":[]}')]:
                    raw=value.encode();info=tarfile.TarInfo('APPS/train/0/'+name);info.size=len(raw);tar.addfile(info,io.BytesIO(raw))
            sha=hashlib.sha256(archive.read_bytes()).hexdigest();from beyond_discovery.benchmarks.apps.data import import_archive
            import_archive(archive,out,sha);store=APPSStore(out)
            with self.assertRaises(CapabilityUnsupported):store.load(store.sample('train',1)[0])
