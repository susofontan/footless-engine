"""One generation's output assembly: spans, stop rules, deltas."""

from __future__ import annotations

import unittest

from footless.render import Output, ToolCalls
from footless.sdk import ContractError

OPEN, CLOSE = 1000, 1001


def plain(ids) -> str:
    return "".join(chr(token) for token in ids)


def marked(ids) -> str:
    pieces = {OPEN: "<think>", CLOSE: "</think>"}
    return "".join(pieces.get(token, chr(token)) for token in ids)


def feed(output: Output, text: str) -> list:
    deltas = []
    for char in text:
        output.push(ord(char))
        deltas.append(output.take())
    return deltas


class SpanTest(unittest.TestCase):
    def test_text_is_the_answer_outside_thinking_blocks(self):
        output = Output(marked, [OPEN, CLOSE])
        for token in [OPEN] + [ord(c) for c in "idea"] + [CLOSE] \
                + [ord(c) for c in "answer"]:
            output.push(token)
        output.close()
        self.assertEqual(output.reasoning, "idea")
        self.assertEqual(output.content, "answer")
        self.assertEqual(output.reasoning_tokens, 4)

    def test_the_markers_are_never_rendered(self):
        output = Output(marked, [OPEN, CLOSE], open_at_start=True)
        for token in [ord(c) for c in "idea"] + [CLOSE] + [ord(c) for c in "now"]:
            output.push(token)
        output.close()
        self.assertEqual(output.reasoning, "idea")
        self.assertEqual(output.content, "now")
        self.assertNotIn("<", output.reasoning + output.content)
        self.assertEqual(output.text, "ideanow")

    def test_a_block_may_open_again(self):
        output = Output(marked, [OPEN, CLOSE])
        for token in [OPEN] + [ord(c) for c in "a"] + [CLOSE] \
                + [ord(c) for c in "b"] + [OPEN] + [ord(c) for c in "c"] + [CLOSE]:
            output.push(token)
        output.close()
        self.assertEqual(output.reasoning, "ac")
        self.assertEqual(output.content, "b")


class StopTest(unittest.TestCase):
    def test_a_stop_string_trims_the_text_at_its_first_character(self):
        output = Output(plain, stop=["stop"])
        feed(output, "the stopped cat")
        output.close()
        self.assertEqual(output.text, "the ")
        self.assertEqual(output.stop_reason, "stop")

    def test_a_stop_string_can_stay_in_the_text(self):
        output = Output(plain, stop=["stop"], include_stop_str_in_output=True)
        feed(output, "the stopped cat")
        output.close()
        self.assertEqual(output.text, "the stop")

    def test_the_earliest_stop_string_wins(self):
        output = Output(plain, stop=["cat", "the"])
        feed(output, "the cat")
        output.close()
        self.assertEqual(output.text, "")
        self.assertEqual(output.stop_reason, "the")

    def test_held_back_text_is_never_streamed_before_it_is_safe(self):
        """`ab` may still complete while the stream ends in `a`."""
        output = Output(plain, stop=["ab"])
        deltas = feed(output, "axab")
        self.assertEqual(deltas[0], ([], []))  # the `a` could still be a stop
        self.assertEqual(deltas[1], ([("content", "ax")], []))
        output.close()
        self.assertEqual(output.text, "ax")

    def test_a_stop_string_may_cross_a_thinking_boundary(self):
        output = Output(marked, [OPEN, CLOSE], stop=["aA"])
        for token in [OPEN] + [ord(c) for c in "xxa"] + [CLOSE] \
                + [ord(c) for c in "Ayy"]:
            output.push(token)
        output.close()
        self.assertEqual(output.text, "xx")
        self.assertEqual(output.reasoning, "xx")
        self.assertEqual(output.content, "")

    def test_a_stop_token_ends_the_output_without_being_rendered(self):
        output = Output(plain)
        feed(output, "hey")
        output.stop(256)
        pieces, _ = output.close()
        self.assertEqual(output.text, "hey")
        self.assertEqual(output.stop_reason, 256)
        self.assertEqual(output.tokens, [ord(c) for c in "hey"])


class DeltaTest(unittest.TestCase):
    def test_every_token_leaves_once_it_is_safe(self):
        output = Output(plain)
        deltas = feed(output, "hello")
        for (pieces, _), char in zip(deltas, "hello"):
            self.assertEqual(pieces, [("content", char)])

    def test_deltas_carry_their_kind(self):
        output = Output(marked, [OPEN, CLOSE], open_at_start=True)
        deltas = []
        for token in [ord(c) for c in "ab"] + [CLOSE] + [ord(c) for c in "c"]:
            output.push(token)
            deltas.append(output.take())
        self.assertEqual(deltas[0], ([("reasoning", "a")], []))
        self.assertEqual(deltas[1], ([("reasoning", "b")], []))
        self.assertEqual(deltas[2], ([], []))  # a marker renders nothing
        self.assertEqual(deltas[3], ([("content", "c")], []))

    def test_close_returns_what_the_holdback_kept(self):
        output = Output(plain, stop=["yx"])  # the trailing `y` may still grow
        deltas = feed(output, "hey")
        self.assertEqual("".join(t for p, _ in deltas for _, t in p), "he")
        pieces, _ = output.close()
        self.assertEqual("".join(text for _, text in pieces), "y")


class LogprobTest(unittest.TestCase):
    def test_rows_follow_the_tokens_that_are_kept(self):
        output = Output(plain, with_logprobs=True)
        for char in "ab":
            output.push(ord(char), [(ord(char), -0.5), (0, -2.0)])
        pieces, rows = output.close()
        self.assertEqual([row.text for row in rows], ["a", "b"])
        self.assertEqual(rows[0].logprob, -0.5)
        self.assertEqual([(row.start, row.end) for row in rows], [(0, 1), (1, 2)])
        self.assertEqual(rows[0].top[0], (ord("a"), "a", -0.5))

    def test_rows_past_a_stop_string_are_dropped_with_their_text(self):
        output = Output(plain, with_logprobs=True, stop=["b"])
        for char in "abc":
            output.push(ord(char), [(ord(char), -0.5)])
        pieces, rows = output.close()
        self.assertEqual(output.text, "a")
        self.assertEqual([row.text for row in rows], ["a"])

    def test_an_unreported_logprob_is_none(self):
        output = Output(plain, with_logprobs=True)
        output.push(ord("a"))
        _, rows = output.close()
        self.assertIsNone(rows[0].logprob)
        self.assertEqual(rows[0].top, [])

    def test_no_rows_without_logprobs(self):
        output = Output(plain)
        output.push(ord("a"), [(ord("a"), -0.5)])
        _, rows = output.close()
        self.assertEqual(rows, [])


class SplitCharacterTest(unittest.TestCase):
    """A character over several tokens: byte tokens, as a byte-level BPE
    splits an emoji or an accent it has no single token for."""

    @staticmethod
    def utf8(ids) -> str:
        return bytes(t for t in ids if t < 256).decode("utf-8", "replace")

    def run_bytes(self, text: str, with_logprobs: bool = False):
        output = Output(self.utf8, with_logprobs=with_logprobs)
        streamed = ""
        for byte in text.encode("utf-8"):
            output.push(byte, [(byte, -0.5)])
            streamed += "".join(t for _, t in output.take()[0])
        streamed += "".join(t for _, t in output.close()[0])
        return output, streamed

    def test_no_replacement_characters_reach_the_text(self):
        text = "Puntuación: 🏆 ⏸️ niño"
        output, streamed = self.run_bytes(text)
        self.assertEqual(streamed, text)
        self.assertEqual(output.content, text)

    def test_rows_still_cover_every_token(self):
        output, _ = self.run_bytes("é🙂", with_logprobs=True)
        self.assertEqual(len(output._rows), len("é🙂".encode()))
        self.assertEqual(output._rows[-1].end, len("é🙂"))

    def test_bytes_that_never_complete_still_leave(self):
        output = Output(self.utf8)
        for byte in b"ok\xf0\x9f":
            output.push(byte)
        self.assertEqual("".join(t for _, t in output.close()[0]), "ok\ufffd")


class ToolCallsTest(unittest.TestCase):
    """The calls' news: a name once, then the arguments as they grow."""

    @staticmethod
    def reader(*reads):
        """A package whose successive reads are `reads` (content, calls)."""
        it = iter(reads)
        return ToolCalls(lambda text, final: next(it))

    def test_a_partial_call_sends_its_name_then_its_arguments(self):
        calls = self.reader(
            ("", [{"name": "f", "arguments": "{", "partial": True}]),
            ("", [{"name": "f", "arguments": '{"a": 1', "partial": True}]),
            ("", [{"name": "f", "arguments": '{"a": 1}'}]))
        self.assertEqual(calls.push("x")[1], [(0, "f", ""), (0, None, "{")])
        self.assertEqual(calls.calls, [])            # not complete yet
        self.assertEqual(calls.push("x")[1], [(0, None, '"a": 1')])
        self.assertEqual(calls.close()[1], [(0, None, "}")])
        self.assertEqual(calls.calls, [{"name": "f", "arguments": '{"a": 1}'}])

    def test_arguments_that_do_not_extend_what_was_sent_are_refused(self):
        calls = self.reader(("", [{"name": "f", "arguments": '{"a"', "partial": True}]),
                            ("", [{"name": "f", "arguments": '{"b": 2}'}]))
        calls.push("x")
        with self.assertRaises(ContractError):
            calls.close()

    def test_a_partial_call_in_a_final_read_is_refused(self):
        calls = self.reader(("", [{"name": "f", "arguments": "{", "partial": True}]))
        with self.assertRaises(ContractError):
            calls.close()


if __name__ == "__main__":
    unittest.main()
