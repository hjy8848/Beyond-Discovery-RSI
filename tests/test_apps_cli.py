import argparse,json
import unittest
from types import SimpleNamespace

from beyond_discovery.benchmarks.apps.cli import audit_self_repair,optional_token_limit
from beyond_discovery.core.protocol import ProtocolError


def claim(source,version,content_hash):
    return SimpleNamespace(program=SimpleNamespace(source=source,version=version,content_hash=content_hash))


def observation(sequence,verdict):
    return {'action':{'operation':'public_test'},'outcome':{
        'status':'COMPLETE','test_scope':'declared-public-tests','verdicts':[verdict]},
        'sequence':sequence,'terminal':False}


def record(prompt):
    return {'request':{'prompt':json.dumps(prompt)}}


class APPSCLITests(unittest.TestCase):
    def test_unlimited_model_token_option_is_explicit_and_rejects_invalid_caps(self):
        self.assertIsNone(optional_token_limit('unlimited'))
        self.assertIsNone(optional_token_limit('none'))
        self.assertEqual(optional_token_limit('4096'),4096)
        with self.assertRaises(argparse.ArgumentTypeError):optional_token_limit('0')
        with self.assertRaises(argparse.ArgumentTypeError):optional_token_limit('lots')

    def setUp(self):
        self.observations=[observation(1,False),observation(2,True)]
        self.trace=[
            {'role':'scientist','name':'code_generate','arguments':[True]},
            {'role':'scientist','name':'code_test','arguments':[0]},
            {'role':'scientist','name':'code_submit','arguments':[0]},
            {'role':'scientist','name':'code_generate','arguments':[True]},
            {'role':'scientist','name':'code_test','arguments':[1]},
            {'role':'scientist','name':'code_submit','arguments':[1]},
        ]
        self.first='print(1)'
        self.second='print(2)'

    def records(self,feedback):
        return [record({'prior_code':None,'public_feedback':[]}),
            record({'prior_code':self.first,'public_feedback':feedback})]

    def test_second_prompt_uses_only_prior_legal_public_feedback(self):
        result=audit_self_repair(self.records(self.observations[:1]),self.observations,self.trace,
            claim(self.first,'1','first-hash'),claim(self.second,'2','second-hash'))
        self.assertTrue(result['prompt_uses_feedback'])
        self.assertEqual(result['feedback_verdicts'],[[False]])
        self.assertTrue(result['code_source_changed'])

    def test_second_prompt_cannot_claim_future_or_private_feedback(self):
        bad_feedback=[self.observations[:],
            [{**self.observations[0],'outcome':{**self.observations[0]['outcome'],'test_scope':'private'}}],
            []]
        for feedback in bad_feedback:
            with self.subTest(feedback=feedback):
                with self.assertRaises(ProtocolError):
                    audit_self_repair(self.records(feedback),self.observations,self.trace,
                        claim(self.first,'1','first-hash'),claim(self.second,'2','second-hash'))


if __name__=='__main__':
    unittest.main()
