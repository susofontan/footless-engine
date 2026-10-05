"""The chat loop's turn-block anchor: `_turn_block`.

A turn that extends the accumulated stream appends only the text the new
render puts AFTER the previous answer. That cut is anchored on the answer's
text, and the answer's text is NOT unique in the render: a short answer sits
inside the reasoning that drafted it, and a question may ask for exactly the
text the model answered with. These tests pin the two rules that keep the cut
honest -- walk the occurrences backward, and accept one only when what follows
still carries the question -- plus the fallback that keeps a lost anchor a
slow turn instead of a broken one.

    python3 -m pytest tests/test_chat_block.py -q
"""

from __future__ import annotations

import unittest

from footless.cli import _turn_block


def _render(reasoning: str, answer: str, question: str) -> str:
    """The shape the template renders: thinking block, answer, new turn."""
    return (
        "<think>\n" + reasoning + "\n</think>\n" + answer
        + "\n<|im_start|>user\n" + question + "\n<|im_start|>assistant\n"
    )


class TurnBlockTest(unittest.TestCase):
    def test_block_is_everything_after_the_answer(self):
        r = _render("think hard", "the answer is 42", "why?")
        self.assertEqual(
            _turn_block(r, "the answer is 42", "why?"),
            "\n<|im_start|>user\nwhy?\n<|im_start|>assistant\n",
        )

    def test_the_answer_inside_the_reasoning_keeps_the_real_cut(self):
        # "42" is drafted in the thinking AND given as the answer; the cut
        # after the thinking occurrence would splice the answer (and the new
        # turn) in twice
        r = _render("it must be 42. Let me verify.", "42", "why?")
        block = _turn_block(r, "42", "why?")
        self.assertTrue(block.endswith("<|im_start|>assistant\n"))
        self.assertIn("why?", block)
        self.assertNotIn("Let me verify", block)

    def test_the_answer_quoted_in_the_question_keeps_the_real_cut(self):
        # the question quotes the answer AFTER it, so the last occurrence is
        # the quote -- and the remainder after the quote no longer carries the
        # question, which walks the anchor back to the real answer
        r = _render("say it", "hola", 'di exactamente "hola" otra vez')
        block = _turn_block(r, "hola", 'di exactamente "hola" otra vez')
        self.assertIn('di exactamente "hola" otra vez', block)
        self.assertNotIn("say it", block)

    def test_a_question_that_is_the_answer_still_cuts_after_the_answer(self):
        r = _render("reproduce the question verbatim", "por qué?",
                    "por qué?")
        block = _turn_block(r, "por qué?", "por qué?")
        self.assertEqual(block.count("por qué?"), 1)

    def test_a_reasoning_anchor_is_usable_when_the_answer_is_empty(self):
        r = _render("only thinking this turn", "", "next?")
        block = _turn_block(r, "only thinking this turn", "next?")
        self.assertIn("next?", block)
        self.assertTrue(block.startswith("\n</think>\n"))

    def test_a_lost_anchor_falls_back_to_a_full_rerender(self):
        r = _render("think", "answer", "why?")
        self.assertIsNone(_turn_block(r, "something else entirely", "why?"))

    def test_an_empty_anchor_is_a_lost_anchor(self):
        self.assertIsNone(_turn_block("anything", "", "why?"))


if __name__ == "__main__":
    unittest.main()
