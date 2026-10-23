"""Protocol fixtures test transport/failure behavior, never real model quality."""
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import patch
from urllib.error import HTTPError
from beyond_discovery.runtime.models import HTTPModelClient, TokenLedger, validate_checkpoint
from beyond_discovery.runtime.model_research import strict_json, ModelResearchRuntime
from beyond_discovery.runtime.trainer import TrainingRecord, LoRABackend, validate_records
from beyond_discovery.core.training import TrainingBatch
from beyond_discovery.core.protocol import CapabilityUnsupported, ProtocolError
from beyond_discovery.core.contracts import ModelState


class ModelTrainingTests(unittest.TestCase):
    def test_private_http_requires_explicit_rfc1918_opt_in(self):
        ledger=TokenLedger(100)
        with self.assertRaises(ValueError):HTTPModelClient('private','http://10.130.138.46:8010/v1','TEST_RSI_KEY',ledger,20)
        client=HTTPModelClient('private','http://10.130.138.46:8010/v1','TEST_RSI_KEY',ledger,20,allow_private_http=True)
        self.assertEqual(client.url,'http://10.130.138.46:8010/v1/chat/completions')
        for url in ('http://8.8.8.8/v1','http://gateway.internal/v1'):
            with self.assertRaises(ValueError):HTTPModelClient('private',url,'TEST_RSI_KEY',ledger,20,allow_private_http=True)

    def test_atomic_budget_and_ambiguous_failure_charges(self):
        ledger=TokenLedger(10)
        self.assertEqual(ledger.reserve(8),8)
        with self.assertRaises(ProtocolError): ledger.reserve(3)
        ledger.settle(8,5); self.assertEqual(ledger.used,5)
        with self.assertRaises(ProtocolError): ledger.settle(2,3)

    def test_actual_http_protocol_usage_truncation_and_no_secret_log(self):
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args): pass
            def do_POST(self):
                data=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                self.server.request=data
                payload=json.dumps({'choices':[{'message':{'content':'[]'},'finish_reason':'length'}],
                                    'usage':{'prompt_tokens':5,'completion_tokens':3}}).encode()
                self.send_response(200); self.end_headers(); self.wfile.write(payload)
        server=HTTPServer(('127.0.0.1',0),Handler)
        thread=threading.Thread(target=server.serve_forever,daemon=True); thread.start()
        try:
            ledger=TokenLedger(100)
            client=HTTPModelClient('protocol-fixture','http://127.0.0.1:'+str(server.server_port)+'/v1','TEST_RSI_KEY',ledger,20)
            with patch.dict(os.environ,{'TEST_RSI_KEY':'test-secret'}): result=client('train prompt',10)
            self.assertEqual(ledger.used,8); self.assertTrue(result.truncated)
            self.assertEqual(server.request['max_completion_tokens'],10)
            self.assertNotIn('test-secret',json.dumps(client.calls))
            with patch.dict(os.environ,{},clear=True):
                with self.assertRaises(CapabilityUnsupported): client('x',1)
        finally: server.shutdown();server.server_close();thread.join()

    def test_http_error_archives_status_without_body_or_secret(self):
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args): pass
            def do_POST(self):
                self.rfile.read(int(self.headers['Content-Length']))
                self.send_response(404);self.end_headers();self.wfile.write(b'{"error":"model missing"}')
        server=HTTPServer(('127.0.0.1',0),Handler)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            ledger=TokenLedger(100)
            client=HTTPModelClient('missing','http://127.0.0.1:'+str(server.server_port)+'/v1','TEST_RSI_KEY',ledger,20)
            with patch.dict(os.environ,{'TEST_RSI_KEY':'test-secret'}):
                with self.assertRaises(HTTPError):client('public prompt',10)
            self.assertEqual(client.calls[0]['http_status'],404)
            self.assertEqual(client.calls[0]['error_type'],'HTTPError')
            self.assertNotIn('model missing',json.dumps(client.calls))
            self.assertNotIn('test-secret',json.dumps(client.calls))
            self.assertEqual(ledger.used,30)  # ambiguous provider failures retain the reservation
        finally: server.shutdown();server.server_close();thread.join()

    def test_checkpoint_identity_and_tamper(self):
        with tempfile.TemporaryDirectory() as root:
            p=Path(root)/'manifest.json'
            p.write_text(json.dumps({'role':'scientist','base_model':'base','files':{'../escape':'hash'}}))
            with self.assertRaises(ProtocolError): validate_checkpoint(p,'challenger','base')
            with self.assertRaises(ProtocolError): validate_checkpoint(p,'scientist','base')

    def test_training_leak_and_rl_provenance_fail_closed(self):
        batch=TrainingBatch('scientist',('train-hash',),'base')
        for record in (TrainingRecord('p','c',('dev-hash',),1),TrainingRecord('p','c',('train-hash',),1,'test')):
            with self.assertRaises(PermissionError): validate_records((record,),batch,'sft')
        record=TrainingRecord('p','c',('train-hash',),1)
        with self.assertRaises(ProtocolError): validate_records((record,),batch,'rl')
        with tempfile.TemporaryDirectory() as root:
            # Dependency check is an explicit unsupported outcome in this stdlib CI.
            import importlib.util
            if importlib.util.find_spec('peft') is None:
                with self.assertRaises(CapabilityUnsupported): LoRABackend((record,)).fit(ModelState('base','scientist'),batch,Path(root))
                self.assertEqual(list(Path(root).iterdir()),[])

    def test_model_json_rejects_ambiguity_and_nonfinite(self):
        for text in ('{"grammar":1,"grammar":2}','[NaN]'):
            with self.assertRaises(ProtocolError): strict_json(text)

    def test_model_generated_code_enters_real_controller_and_replay(self):
        from beyond_discovery.adapters.transition import StringTransitionAdapter
        from beyond_discovery.core.controller import RunConfig, run_rsi
        from beyond_discovery.core.replay import replay
        from beyond_discovery.core.archive import read_verified
        from beyond_discovery.runtime.llm import Completion
        adapter=StringTransitionAdapter(); ledger=TokenLedger(100000)
        class Client:
            model='transport-fixture'
            def __init__(self): self.ledger=ledger; self.calls=[]
            def __call__(self,prompt,limit):
                request=json.loads(prompt)
                if request['task']=='world': data={'expression':['concat',['ref','input.text'],['ref','previous']],'track':'A'}
                else: data=[{'grammar':request['public_context']['grammar']}]
                self.ledger.reserve(10)
                result=Completion(json.dumps(data),self.model,5,5)
                self.calls.append({'request':prompt,'response':result.__dict__})
                return result
        runtime=ModelResearchRuntime(adapter.public_context(),{r:Client() for r in ('scientist','challenger')})
        with tempfile.TemporaryDirectory() as root:
            out=Path(root)/'run'
            result=run_rsi(adapter,out,RunConfig(generations=1,candidate_limit=1,base_model='transport-fixture'),runtime=runtime)
            self.assertEqual(result['status'],'COMPLETE')
            self.assertEqual(result['usage']['model_tokens'],30)
            self.assertEqual(len([x for k,x in read_verified(out) if k=='runtime_model_events']),3)
            self.assertTrue(replay(out,StringTransitionAdapter())['verified'])

    def test_model_meta_code_changes_candidate_behavior_on_real_train_evidence(self):
        from dataclasses import replace
        from beyond_discovery.adapters.transition import StringTransitionAdapter
        from beyond_discovery.core.controller import RunConfig
        from beyond_discovery.core.meta import meta_artifact
        from beyond_discovery.core.interaction import episode
        from beyond_discovery.runtime.llm import Completion
        adapter=StringTransitionAdapter()
        grammar=adapter.public_context()['grammar']
        class Client:
            model='fixture'
            def __init__(self): self.ledger=TokenLedger(10000); self.calls=[]
            def __call__(self,prompt,limit):
                result=Completion(json.dumps([{'grammar':{**grammar,'max_nodes':1}},{'grammar':grammar}]),self.model,1,1)
                self.calls.append({'request':prompt,'response':result.__dict__})
                return result
        runtime=ModelResearchRuntime(adapter.public_context(),{r:Client() for r in ('scientist','challenger')})
        c,s=runtime.initial_pair(RunConfig(base_model='fixture',candidate_limit=2))
        train=tuple(episode(adapter,t,s,c,0,runtime=runtime) for t in adapter.sample('train',0))
        good=runtime.candidates(s,None,train,adapter.public_context(),2,'full')
        reverse_meta=replace(s.meta,program=meta_artifact(('ref','errors'),runtime.provenance(s)))
        bad=runtime.candidates(replace(s,meta=reverse_meta),None,train,adapter.public_context(),2,'full')
        self.assertEqual(runtime.grammar(good[0][0]).max_nodes,3)
        self.assertEqual(runtime.grammar(bad[0][0]).max_nodes,1)
