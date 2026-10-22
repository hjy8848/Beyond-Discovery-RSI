from dataclasses import replace
import unittest
from beyond_discovery.core.contracts import Provenance, Decision
from beyond_discovery.core.gate import GateConfig
from beyond_discovery.core.meta import AdaptationOpportunity, evaluate_meta_pair, meta_artifact, score, synthesize_meta
from beyond_discovery.core.synthesis import Example
from beyond_discovery.core.protocol import ProtocolError


class MetaTests(unittest.TestCase):
    def setUp(self):
        self.p=Provenance("meta","train","meta-search",0,"s",0,"test")
        self.old=meta_artifact(("sub",("lit",0),("ref","errors")),self.p)
        self.facts=({"errors":0,"complexity":5,"cost":9,"novelty":2,"coverage":4},
                    {"errors":1,"complexity":1,"cost":1,"novelty":1,"coverage":1})

    def test_executable_policy_changes_ranking_and_textual_noops_do_not_count(self):
        candidates=synthesize_meta(self.old,self.facts,self.p,64)
        self.assertTrue(candidates)
        self.assertTrue(any(score(c,self.facts[0])<score(c,self.facts[1]) for c in candidates))
        with self.assertRaises(PermissionError): synthesize_meta(self.old,self.facts,replace(self.p,split="dev"))

    def test_opportunities_refuse_private_adaptation_inputs(self):
        with self.assertRaises(PermissionError): AdaptationOpportunity("id","cluster","program","opponent",(Example({},0,"e","test"),),10)

    def test_fixed_parent_opponent_budget_and_independent_validation(self):
        opportunities=tuple(AdaptationOpportunity(str(i),str(i),"fixed-task","fixed-opponent",(Example({},0,"train"),),8) for i in range(5))
        new=meta_artifact(("ref","novelty"),self.p)
        def adapt(meta,opportunity):
            self.assertEqual(opportunity.parent_program_hash,"fixed-task")
            self.assertEqual(opportunity.frozen_opponent_hash,"fixed-opponent")
            # Actual executable products of different generator programs.
            expression=("lit",score(meta,self.facts[0]))
            return meta_artifact(expression,self.p),4
        def verify(artifact,opportunity):
            return True, score(artifact,self.facts[0])>0
        gate,record=evaluate_meta_pair(self.old,new,opportunities,adapt,verify,GateConfig())
        self.assertEqual(gate.decision,Decision.ACCEPT)
        self.assertTrue(record["source_diff"])
        self.assertNotEqual(record["inventions"][0],record["inventions"][1])

    def test_budget_violation_is_not_an_improvement(self):
        opportunity=AdaptationOpportunity("id","c","p","o",(),1)
        with self.assertRaises(ProtocolError): evaluate_meta_pair(self.old,self.old,(opportunity,),lambda m,o:(m,2),lambda a,o:(True,True),GateConfig())
