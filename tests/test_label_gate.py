"""The optional `label_logprobs` verb and the engine's gate behind it.

These are the engine's half of the contract (CONTRACT.md, Optional verbs);
whether a package that really scores labels is exact is the package's own
test. `mockup` carries the verb but declares no capabilities -- the shape of
a package whose manifest never offered label scoring -- and `mockup-thinking`
declares the capability and scripts its values (`scripted_label_logprobs`),
so a test can pin what comes back.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from footless.engine import Engine
from footless.sdk import LABEL_LOGPROBS, ContractError

MODEL = Path(__file__).resolve().parent / "mockup"
MODEL_THINKING = Path(__file__).resolve().parent / "mockup-thinking"


class LabelLogprobsGateTest(unittest.TestCase):
    """What the engine does with the capability, and without it.

    The gate is on the capability, never on the package: `mockup` carries the
    very verb being asked for and is still refused, and `mockup-thinking`
    declaring the capability without a working verb is a contradiction, not a
    silent no-op.
    """

    def test_a_package_without_the_capability_is_refused_with_a_reason(self):
        engine = Engine.open(MODEL)
        self.addCleanup(engine.close)
        self.assertNotIn(LABEL_LOGPROBS, engine.facts.capabilities)
        self.assertTrue(callable(engine.runtime.label_logprobs),
                        "the mock carries the verb; the gate is the capability")
        with self.assertRaises(ContractError) as caught:
            engine.label_logprobs(engine.runtime.encode("hi"), ["A"])
        message = str(caught.exception)
        self.assertIn("label_logprobs", message)
        self.assertIn("it declares: nothing", message)

    def test_a_declared_capability_with_no_verb_is_a_contradiction(self):
        """Reported, not swallowed -- the manifest and the code must agree."""
        engine = Engine.open(MODEL_THINKING)
        self.addCleanup(engine.close)
        self.assertIn(LABEL_LOGPROBS, engine.facts.capabilities)
        engine.runtime.label_logprobs = None  # manifest and code disagree now
        with self.assertRaises(ContractError) as caught:
            engine.label_logprobs(engine.runtime.encode("hi"), ["A"])
        self.assertIn("no label_logprobs verb", str(caught.exception))


class LabelLogprobsScoringTest(unittest.TestCase):
    """The verb on the package that offers it: scripted, ordered, exact."""

    def setUp(self) -> None:
        self.engine = Engine.open(MODEL_THINKING)
        self.addCleanup(self.engine.close)

    def test_values_are_scripted_and_the_order_follows_the_labels(self):
        runtime = self.engine.runtime
        prompt = runtime.encode("Question: pick one?\n")
        runtime.scripted_label_logprobs = {"A": -1.5, "B": -0.25}
        self.assertEqual(self.engine.label_logprobs(prompt, ["B", "A"]),
                         [(ord("B"), -0.25), (ord("A"), -1.5)])
        self.assertEqual(self.engine.label_logprobs(prompt, ["A", "B"]),
                         [(ord("A"), -1.5), (ord("B"), -0.25)])

    def test_unscripted_labels_get_a_stable_default(self):
        prompt = self.engine.runtime.encode("Question: pick one?\n")
        self.assertEqual(self.engine.label_logprobs(prompt, ["A", "B", "C"]),
                         [(ord("A"), -1.0), (ord("B"), -2.0), (ord("C"), -3.0)])

    def test_a_yes_or_no_label_is_one_token_the_way_a_vocabulary_has_it(self):
        """`yes` and `no` are one token each in a real vocabulary and several
        bytes in this one; the mock routes them to one id each (LABEL_TOKENS),
        which is what a yes or no decision question needs."""
        prompt = self.engine.runtime.encode("Question: true?\n")
        self.assertEqual(self.engine.label_logprobs(prompt, ["yes", "no"]),
                         [(300, -1.0), (301, -2.0)])

    def test_a_label_that_is_not_one_token_is_refused_by_name(self):
        """These mocks are byte tokens: "é" adds two of them, and a label the
        answer position cannot hold as one token is the caller's mistake."""
        prompt = self.engine.runtime.encode("Question: pick one?\n")
        with self.assertRaises(ValueError) as caught:
            self.engine.label_logprobs(prompt, ["A", "é"])
        self.assertIn("é", str(caught.exception))

    def test_labels_that_collide_on_one_token_are_refused_by_name(self):
        """Every label must claim its own token; two labels on one id would
        score the same candidate twice under two names."""
        prompt = self.engine.runtime.encode("Question: pick one?\n")
        with self.assertRaises(ValueError) as caught:
            self.engine.label_logprobs(prompt, ["A", "A"])
        self.assertIn("'A'", str(caught.exception))

    def test_prompt_text_that_does_not_round_trip_is_refused(self):
        """Ids whose text does not re-encode to them have no answer position
        to score at, and the verb says so rather than scoring the wrong one. A
        chat-rendered prompt round-trips here -- markers decode to text and
        `encode` takes them back -- so the lossy case is an id the vocabulary
        has no text for."""
        tokens = self.engine.runtime.encode("hi") + [9999]
        with self.assertRaises(ValueError) as caught:
            self.engine.label_logprobs(tokens, ["A"])
        self.assertIn("round-trip", str(caught.exception))

    def test_scoring_changes_nothing_the_model_generates(self):
        """An optional verb must not change what the model returns. Here that
        is testable and tested: the same decode before and after scoring is
        the same text."""
        before = self.engine.generate("hello", 64).text
        prompt = self.engine.runtime.encode("Question: pick one?\n")
        self.engine.label_logprobs(prompt, ["A", "B"])
        self.assertEqual(self.engine.generate("hello", 64).text, before)


if __name__ == "__main__":
    unittest.main()
