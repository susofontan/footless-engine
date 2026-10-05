"""Typed decision semantics: prompt wording v1, label alphabets, and the math."""

from __future__ import annotations

import math
import unittest

from footless.decisions import (
    PROMPT_FORMAT_VERSION,
    QuestionView,
    check_option_names,
    choice_confidence,
    default_labels,
    label_mass,
    pair_labels,
    probabilities,
    render_question,
    render_text,
    score_confidence,
)

TEXT = '{"ticket":"Refund please","tags":["billing"]}'


class RenderTest(unittest.TestCase):
    def test_the_version_is_pinned(self):
        # A wording change needs a new PROMPT_FORMAT_VERSION and its own fixture.
        self.assertEqual(PROMPT_FORMAT_VERSION, 1)

    def test_render_text(self):
        self.assertEqual(render_text(None), "")
        self.assertEqual(render_text("Mood?"), "Mood?")
        self.assertEqual(render_text({"k": 1}), '{"k":1}')
        self.assertEqual(render_text(["billing"]), '["billing"]')
        self.assertEqual(render_text({"tags": ["b"], "n": {"z": 2}}),
                         '{"tags":["b"],"n":{"z":2}}')

    def test_prompt_text(self):
        # The wording of PROMPT_FORMAT_VERSION, one fixture per question kind.
        cases = {
            "choice": (
                QuestionView("choice", {"question": "Which team?"},
                             ["billing", "sales", "other"],
                             ["Payments", None, {"k": 1}]),
                ["A", "B", "C"],
                ['Question: {"question":"Which team?"}',
                 "A: billing - Payments",
                 "B: sales",
                 'C: other - {"k":1}',
                 "Answer with the letter of one option only."],
            ),
            "score": (
                QuestionView("score", "Mood?", ["0", "1"], ["Calm", "Angry"]),
                ["0", "1"],
                ["Question: Mood?",
                 "0: Calm",
                 "1: Angry",
                 "Answer with the number of one level only."],
            ),
            "yes_no": (
                QuestionView("yes_no", "Urgent", ["yes", "no"], [None, "Can wait"]),
                ["yes", "no"],
                ["Is the following true? Urgent",
                 "no: Can wait",
                 "Answer with yes or no only."],
            ),
        }
        for kind, (view, labels, lines) in cases.items():
            with self.subTest(kind):
                self.assertEqual(render_question(TEXT, view, labels),
                                 "\n".join([TEXT, "", *lines]))

    def test_prompt_text_without_question_text(self):
        # Renderings only blank questions reach: no question line, criteria-only
        # yes or no questions, and two-letter labels beyond 26 options.
        cases = {
            "choice": (
                QuestionView("choice", None, ["billing", "sales"],
                             ["Payments", None]),
                ["AA", "AB"],
                ["AA: billing - Payments",
                 "AB: sales",
                 "Answer with the letter of one option only."],
            ),
            "score": (
                QuestionView("score", " ", ["0", "1"], ["Calm", "Angry"]),
                ["0", "1"],
                ["0: Calm",
                 "1: Angry",
                 "Answer with the number of one level only."],
            ),
            "yes_no": (
                QuestionView("yes_no", None, ["yes", "no"],
                             ["Reply today", "Can wait"]),
                ["yes", "no"],
                ["Is the following true?",
                 "yes: Reply today",
                 "no: Can wait",
                 "Answer with yes or no only."],
            ),
        }
        for kind, (view, labels, lines) in cases.items():
            with self.subTest(kind):
                self.assertEqual(render_question("s", view, labels),
                                 "\n".join(["s", "", *lines]))

    def test_an_empty_description_makes_no_description_line(self):
        view = QuestionView("choice", "Q", ["a", "b", "c"], [None, "", {}])
        rendered = render_question("s", view, ["A", "B", "C"])
        self.assertEqual(rendered.splitlines()[2:], [
            "Question: Q",
            "A: a",
            "B: b",
            "C: c - {}",
            "Answer with the letter of one option only.",
        ])


class LabelTest(unittest.TestCase):
    def test_default_labels_follow_the_label_alphabets(self):
        choice = QuestionView("choice", "Q", ["any"] * 2, [])
        self.assertEqual(default_labels(choice), ["A", "B"])
        full = QuestionView("choice", "Q", ["any"] * 26, [])
        self.assertEqual(default_labels(full), list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
        score = QuestionView("score", "Q", [str(i) for i in range(10)], [])
        self.assertEqual(default_labels(score), list("0123456789"))
        yes_no = QuestionView("yes_no", "Q", ["yes", "no"], [])
        self.assertEqual(default_labels(yes_no), ["yes", "no"])

    def test_pair_labels_are_fixed(self):
        labels = pair_labels()
        self.assertEqual(labels[:3], ["AA", "AB", "AC"])
        self.assertEqual(labels[25:27], ["AZ", "BA"])
        self.assertEqual(labels[-1], "ZZ")
        self.assertEqual(len(labels), 26 * 26)

    def test_option_names_must_be_unambiguous(self):
        check_option_names(["billing", "sales", "other"])
        check_option_names(["st", "ß"])  # casefold reads ß as ss
        refused = {
            "blank": (["a", " "], "option names must be nonempty"),
            "empty": ([""], "option names must be nonempty"),
            "repeats after strip": (["a", "A "],
                                    "option name 'A ' repeats another option"),
            "repeats after casefold": (["a", "A"],
                                       "option name 'A' repeats another option"),
            "repeats after casefold of ß": (["ss", "ß", "SS"],
                                            "option name 'ß' repeats another option"),
        }
        for name in ("a\nB", "a\tb", "a b", "a b", "\x00x"):
            refused[f"control or line break in {name!r}"] = (
                [name, "c"],
                f"option name {name!r} must not contain control or line break "
                "characters")
        for case, (names, message) in refused.items():
            with self.subTest(case), self.assertRaises(ValueError) as caught:
                check_option_names(names)
            self.assertEqual(str(caught.exception), message)


class MathTest(unittest.TestCase):
    def test_label_mass(self):
        self.assertAlmostEqual(label_mass([math.log(0.1), math.log(0.2)]), 0.3)
        self.assertAlmostEqual(label_mass([math.log(0.25)] * 4), 1.0)
        self.assertAlmostEqual(label_mass([0.0, math.log(3.0)]), 4.0)

    def test_probabilities(self):
        # exp([0, log 3]) / 4, and the same at temperature 2: 1 and sqrt(3).
        for got, want in zip(probabilities([0.0, math.log(3.0)]), [0.25, 0.75]):
            self.assertAlmostEqual(got, want)
        root = math.sqrt(3.0)
        for got, want in zip(probabilities([0.0, math.log(3.0)], 2.0),
                             [1 / (1 + root), root / (1 + root)]):
            self.assertAlmostEqual(got, want)
        for got, want in zip(
                probabilities([math.log(1.0), math.log(2.0), math.log(3.0)]),
                [1 / 6, 1 / 3, 1 / 2]):
            self.assertAlmostEqual(got, want)

    def test_probabilities_sum_to_one_at_any_temperature(self):
        logprobs = [0.0, math.log(3.0), math.log(0.5)]
        for temperature in (0.25, 1.0, 5.0):
            with self.subTest(temperature):
                self.assertAlmostEqual(sum(probabilities(logprobs, temperature)), 1.0)

    def test_label_mass_ignores_temperature(self):
        # The mass is the labels' own probability sum; temperature never enters.
        logprobs = [0.0, math.log(3.0)]
        scored = [probabilities(logprobs, t) for t in (0.5, 1.0, 2.0)]
        self.assertNotEqual(scored[0], scored[-1])
        self.assertAlmostEqual(label_mass(logprobs), 4.0)

    def test_probabilities_are_centered(self):
        # A max-subtracted softmax: no overflow or underflow at the extremes.
        self.assertEqual(probabilities([-1000.0, -1000.0]), [0.5, 0.5])
        self.assertEqual(probabilities([0.0, -1000.0]), [1.0, 0.0])
        self.assertEqual(probabilities([-1000.0, 0.0], 0.1), [0.0, 1.0])


class ConfidenceTest(unittest.TestCase):
    def test_choice_confidence(self):
        self.assertAlmostEqual(choice_confidence([0.25] * 4), 0.0)  # uniform
        self.assertAlmostEqual(choice_confidence([0.1, 0.9]), 0.8)
        self.assertEqual(choice_confidence([0.0, 1.0]), 1.0)  # one-hot
        self.assertEqual(choice_confidence([1.0]), 1.0)  # one candidate
        self.assertEqual(choice_confidence([1.2, 0.0]), 1.0)  # clamped to 1
        self.assertEqual(choice_confidence([0.3, 0.3, 0.3]), 0.0)  # clamped to 0

    def test_score_confidence(self):
        self.assertAlmostEqual(score_confidence([0, 0.14, 0.86, 0, 0]),
                               1 - 0.14 / 1.2)
        self.assertAlmostEqual(score_confidence([0, 0, 0.48, 0.52]), 0.52)
        self.assertEqual(score_confidence([0.5, 0, 0, 0, 0.5]), 0.0)  # spread out
        self.assertAlmostEqual(score_confidence([1 / 3] * 3), 0.0)  # uniform
        self.assertEqual(score_confidence([0, 1, 0]), 1.0)  # one-hot
        self.assertEqual(score_confidence([1.0]), 1.0)  # one candidate


if __name__ == "__main__":
    unittest.main()
