"""Optional actual OCI tests. Never execute submitted Python on the host."""
import os,unittest
from beyond_discovery.benchmarks.apps.judge import DockerJudge
from beyond_discovery.benchmarks.apps.parity import check_official_parity

@unittest.skipUnless(os.environ.get('APPS_JUDGE_IMAGE'),'no reviewed APPS_JUDGE_IMAGE; actual OCI execution UNVERIFIED')
class LiveContainerTests(unittest.TestCase):
    def setUp(self):self.judge=DockerJudge(os.environ['APPS_JUDGE_IMAGE'])
    def test_known_correct_wrong_class_and_runtime_errors_match_official(self):
        self.assertEqual(check_official_parity(self.judge)['status'],'VERIFIED')
    def test_timeout_and_private_host_files_are_inaccessible(self):
        result=self.judge.grade('def solve():\n while True: pass',[{'input':[],'output':0}],'solve')
        self.assertFalse(result.accepted)
        source="def solve():\n try:\n  open('/etc/rsi-private-sentinel','w').write('x')\n except (PermissionError,OSError):\n  return 1\n return 0"
        self.assertTrue(self.judge.grade(source,[{'input':[],'output':1}],'solve').accepted)
