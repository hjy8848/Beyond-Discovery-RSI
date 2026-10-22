from dataclasses import replace
import unittest
from beyond_discovery.core.contracts import Provenance, Status, canonical
from beyond_discovery.core.protocol import Action, ResearchArtifact, Schema
from beyond_discovery.core.synthesis import Grammar
from beyond_discovery.core.worlds import WorldCertifier, ChallengerInventor


class WorldTests(unittest.TestCase):
    def setUp(self):
        self.g = Grammar((("input.x","integer"),("previous","integer")), (0,1), ("xor","add"), "integer", 3)
        self.actions = tuple(Action.create("probe", {"x": x}) for x in (0,1))
        self.p = Provenance("test", "train", "synthesis", 0, "c", 0, "reference")
        self.certifier = WorldCertifier(self.g, self.actions, 0)

    def world(self, node):
        return ResearchArtifact("world","research-expression-v1","1",canonical({"expression":node}),
                                Schema.of({"type":"object"}),Schema.of({"type":"integer"}),self.p)

    def test_both_tracks_independent_certification_and_duplicate_rejection(self):
        for track in ("A","B"):
            verifier = WorldCertifier(self.g,self.actions,0)
            world = self.world(("xor",("ref","input.x"),("ref","previous")))
            result = verifier.certify(world,track,(("ref","input.x"),),10000)
            self.assertEqual(result.identification.status,Status.VERIFIED)
            self.assertEqual(result.identification.remaining_classes,1)
            self.assertEqual(result.status,Status.VERIFIED)
            self.assertGreater(len(result.witness),1)
            self.assertEqual(verifier.certify(world,track,(("ref","input.x"),),100).status,Status.INVALID)

    def test_resource_insufficiency_is_pending_and_author_answers_are_rejected(self):
        world = self.world(("ref","input.x"))
        self.assertEqual(self.certifier.certify(world,"B",(("lit",0),),1).status,Status.PENDING)
        bad = replace(world,source=canonical({"expression":("ref","input.x"),"answer":1}))
        self.assertEqual(self.certifier.certify(bad,"B",(("lit",0),),100).status,Status.INVALID)

    def test_trivial_and_undiscriminating_worlds_never_verify(self):
        self.assertEqual(self.certifier.certify(self.world(("lit",0)),"A",(("lit",1),),100).status,Status.INVALID)
        self.assertEqual(self.certifier.certify(self.world(("ref","input.x")),"A",(("ref","input.x"),),100).status,Status.INVALID)

    def test_challenge_is_conditioned_on_frozen_scientist_and_evidence_provenance(self):
        engine = ChallengerInventor()
        a = engine.propose(self.g,self.actions,0,(("lit",0),),(),self.p,"old-a",128)
        b = engine.propose(self.g,self.actions,0,(("ref","input.x"),),(),self.p,"old-b",128)
        self.assertNotEqual(a.program.source,b.program.source)
        with self.assertRaises(PermissionError): engine.propose(self.g,self.actions,0,(("lit",0),),(),replace(self.p,split="dev"),"x",128)

    def test_independent_identifiability_is_scoped_and_requires_action_budget(self):
        world=self.world(('xor',('ref','input.x'),('ref','previous')))
        result=self.certifier.certify(world,'A',(('ref','input.x'),),10000,action_budget=1)
        self.assertEqual(result.status,Status.PENDING)
        result=self.certifier.certify(world,'A',(('ref','input.x'),),10000,action_budget=8)
        self.assertEqual(result.status,Status.VERIFIED)
        self.assertTrue(result.identification.family_hash)
        self.assertLessEqual(len(result.identification.experiment),8)

    def test_restricted_track_cannot_smuggle_unregistered_primitive(self):
        world=self.world(('max',('ref','input.x'),('ref','previous')))
        result=self.certifier.certify(world,'A',(('lit',0),),10000)
        self.assertEqual(result.status,Status.INVALID)
        self.assertIn('unregistered',result.reason)
