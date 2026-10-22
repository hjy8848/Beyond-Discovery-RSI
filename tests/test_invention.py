from dataclasses import replace
import json
import unittest
from beyond_discovery.core.contracts import Provenance
from beyond_discovery.core.synthesis import ArtifactInventor, Example, Grammar, enumerate_programs, fit
from beyond_discovery.core.protocol import ProtocolError
from beyond_discovery.runtime.expression import evaluate, freeze
from beyond_discovery.core.experiments import design_experiment, protocol_actions, rollout
from beyond_discovery.core.protocol import Action


class InventionTests(unittest.TestCase):
    def setUp(self):
        self.g = Grammar((("input.a", "integer"), ("input.b", "integer")), (0, 1), ("add", "mul"), "integer", 1)
        self.examples = tuple(Example({"input": {"a": a, "b": b}}, a + b, str((a,b))) for a,b in ((1,2),(2,3),(3,4)))
        self.prov = Provenance("unit", "train", "synthesis", 0, "s", 0, "test")

    def test_recombination_generates_executable_code_improving_actual_examples(self):
        parent = (("ref", "input.a"), ("ref", "input.b"))
        result = ArtifactInventor(self.g, 256).invent("RECOMBINE", parent, self.examples, self.prov)
        self.assertTrue(result)
        self.assertEqual(result[0].errors_after, 0)
        node = freeze(json.loads(result[0].artifact.source)["expression"])
        self.assertEqual(evaluate(node, {"input": {"a": 10, "b": 11}}), 21)

    def test_relaxation_and_reframing_change_executable_representation(self):
        for op in ("RELAX_ASSUMPTION", "REFRAME"):
            results = ArtifactInventor(self.g, 256).invent(op, (("ref", "input.a"),), self.examples, self.prov)
            self.assertTrue(results)
            self.assertEqual(results[0].errors_after, 0)

    def test_analogy_requires_verified_asset_and_type_preserving_mapping(self):
        engine = ArtifactInventor(self.g, 256)
        self.assertFalse(engine.invent("ANALOGIZE", (("ref", "input.a"),), self.examples, self.prov))
        assets = ((("add", ("ref", "x"), ("ref", "y")), (("x", "integer"), ("y", "integer")), ("verified-train-asset",)),)
        result = engine.invent("ANALOGIZE", (("ref", "input.a"),), self.examples, self.prov, assets)
        self.assertTrue(result)
        self.assertEqual(result[0].errors_after, 0)

    def test_structurally_new_but_behaviorally_identical_is_not_invention(self):
        examples = tuple(replace(e, outcome=e.context["input"]["a"]) for e in self.examples)
        self.assertFalse(ArtifactInventor(self.g, 128).invent("REFRAME", (("ref", "input.a"),), examples, self.prov))

    def test_generation_rejects_private_data(self):
        with self.assertRaises(PermissionError): ArtifactInventor(self.g, 128).invent("REFRAME", (), (replace(self.examples[0], split="dev"),), self.prov)

    def test_string_composition_and_resource_exhaustion(self):
        g = Grammar((("input.text", "string"),), ("",), ("reverse", "concat"), "string", 3)
        examples = tuple(Example({"input": {"text": s}}, s[::-1], s) for s in ("abc", "de"))
        programs, complete = fit(g, examples, 100)
        self.assertEqual(evaluate(programs[0], {"input": {"text": "world"}}), "dlrow")
        self.assertFalse(enumerate_programs(g, 1).complete)
        with self.assertRaises(ProtocolError): evaluate(("__import__", ("lit", "os")), {})

    def test_experiment_invention_finds_multistep_distinguishing_program(self):
        old = ("ref", "input.x")
        new = ("xor", ("ref", "input.x"), ("ref", "previous"))
        actions = tuple(Action.create("probe", {"x": n}) for n in (0, 1))
        self.assertTrue(all(rollout(old, (a,), 0) == rollout(new, (a,), 0) for a in actions))
        experiment = design_experiment((old, new), actions, 0, 3, 32, self.prov)
        sequence = protocol_actions(experiment)
        self.assertNotEqual(rollout(old, sequence, 0), rollout(new, sequence, 0))
        self.assertGreater(len(sequence), 1)
