from dataclasses import replace
import unittest
from beyond_discovery.adapters.causal import CausalDiscoveryAdapter
from beyond_discovery.core.boundary import BoundaryKind, HypothesisFamily, certify_boundary, verify_certificate
from beyond_discovery.core.controller import RunConfig, initial_pair
from beyond_discovery.core.interaction import episode
from beyond_discovery.core.protocol import ProtocolError
from beyond_discovery.runtime.policy_runtime import predict


class BoundaryTests(unittest.TestCase):
    def setUp(self):
        self.a = CausalDiscoveryAdapter()
        c, s = initial_pair(RunConfig())
        self.e = episode(self.a, self.a.sample("train", 0)[0], s, c, 0)
        self.predict = lambda p, a, h: predict(p, a)

    def family(self, members): return HypothesisFamily("test-declared", "1", tuple(members), "test grammar only")

    def test_single_prediction_error_is_search_failure_when_consistent_member_exists(self):
        correct = self.e.old_claim.program
        e = replace(self.e, old_claim=replace(self.e.old_claim, program=("const", 8)))
        cert = certify_boundary(self.a, e, self.family([("const", 8), correct]), self.predict, 2)
        self.assertEqual(cert.kind, BoundaryKind.SEARCH_FAILURE)

    def test_exhaustive_refutation_and_independent_verification(self):
        family = self.family([("const", 7), ("const", 8)])
        cert = certify_boundary(self.a, self.e, family, self.predict, 2)
        self.assertEqual(cert.kind, BoundaryKind.SPACE_REFUTED)
        self.assertTrue(verify_certificate(cert, self.a, self.e, family, self.predict))
        self.assertFalse(verify_certificate(replace(cert, total=999), self.a, self.e, family, self.predict))

    def test_partial_search_is_unknown_not_refutation(self):
        cert = certify_boundary(self.a, self.e, self.family([("const", 7), ("const", 8)]), self.predict, 1)
        self.assertEqual(cert.kind, BoundaryKind.UNDETERMINED)

    def test_fake_observation_cannot_be_certified(self):
        e = replace(self.e, observations=(replace(self.e.observations[0], outcome=100),))
        with self.assertRaises(ProtocolError): certify_boundary(self.a, e, self.family([("const", 7)]), self.predict, 1)
