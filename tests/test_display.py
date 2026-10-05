"""The display mutes thinking blocks and reports per-phase rates at the end."""

from __future__ import annotations

import io
import re
import unittest

from footless.display import GRAY, RESET, StreamDisplay, _stats
from footless.engine import Rates, Timing


def _decode(ids: list[int]) -> str:
    return "".join(f"<{t}>" for t in ids)


class Screen:
    """What a user would see: a terminal that wraps, scrolls and moves a cursor.

    Only the handful of sequences the display emits are understood; anything
    else is ignored. A line feed is a logical newline (the tty's onlcr turns
    it into a carriage return and a line feed), which scrolls at the bottom
    row — the behaviour a real terminal gives.
    """

    def __init__(self, rows: int = 10, cols: int = 40) -> None:
        self.rows, self.cols = rows, cols
        self.grid = [[" "] * cols for _ in range(rows)]
        self.row = self.col = 0

    def feed(self, text: str) -> None:
        index = 0
        while index < len(text):
            char = text[index]
            if char == "\x1b":
                index = self._escape(text, index)
                continue
            index += 1
            self._print(char)

    def _print(self, char: str) -> None:
        if char == "\n":
            self.row, self.col = self.row + 1, 0
            self._scroll_if_needed()
        elif char == "\r":
            self.col = 0
        elif char == "\t":
            self.col = (self.col // 8 + 1) * 8
        else:
            self.grid[self.row][self.col] = char
            self.col += 1
            if self.col >= self.cols:  # wrap, scrolling at the bottom row
                self.row, self.col = self.row + 1, 0
                self._scroll_if_needed()

    def _scroll_if_needed(self) -> None:
        while self.row >= self.rows:
            self.grid.pop(0)
            self.grid.append([" "] * self.cols)
            self.row -= 1

    def _escape(self, text: str, index: int) -> int:
        end = index + 2
        while end < len(text) and not ("@" <= text[end] <= "~"):
            end += 1
        return end + 1  # the display only colours; the text is what matters

    def text(self) -> str:
        return "\n".join("".join(row).rstrip() for row in self.grid).rstrip()


class _Feed:
    """A file-like that feeds a Screen, keeping writes and flushes apart."""

    def __init__(self, screen: Screen) -> None:
        self.screen = screen

    def write(self, text: str) -> int:
        self.screen.feed(text)
        return len(text)

    def flush(self) -> None:
        pass


class DisplayTest(unittest.TestCase):
    def test_thinking_blocks_are_muted_and_markers_hidden(self):
        out = io.StringIO()
        display = StreamDisplay(out, _decode, thinking_markers=[1, 2])
        timing = Timing(prefill_tokens=3, prefill_seconds=0.5, decode_tokens=4, decode_seconds=1.0)
        for token in [1, 7, 8, 2, 9]:
            display.token(token, timing)
        display.close(timing)
        text = out.getvalue()
        self.assertIn("<7><8>", text)
        self.assertNotIn("<1>", text)
        self.assertNotIn("<2>", text)
        self.assertLess(text.index(GRAY), text.index("<7><8>"))
        self.assertLess(text.index("<7><8>"), text.index(RESET))

    def test_close_reports_prefill_and_decode_rates(self):
        out = io.StringIO()
        display = StreamDisplay(out, _decode, thinking_markers=[1, 2])
        timing = Timing(prefill_tokens=4, prefill_seconds=0.5, decode_tokens=8, decode_seconds=1.0)
        display.token(9, timing)
        display.close(timing)
        text = out.getvalue()
        self.assertIn("tokens", text)  # the table's header
        prefill, decode = self.rows(text)
        self.assertIn("4", prefill)
        self.assertTrue(prefill.rstrip().endswith("8.0"))
        self.assertIn("8", decode)
        self.assertTrue(decode.rstrip().endswith("8.0"))

    def test_close_reports_each_phase_and_the_two_durations(self):
        """Each phase closes with its tokens and its own mean, plus when the
        turn started and how long it took.

        The rate is the PHASE's mean -- `tokens / seconds` -- and not the mean
        of its per-step rates. A model that verifies a draft hands some tokens
        out on steps that do no forward at all, so the mean of the steps
        understates what the turn delivered, and this row is what a reader
        compares between turns.
        """
        out = io.StringIO()
        display = StreamDisplay(out, _decode, thinking_markers=[1, 2])
        timing = Timing(prefill_tokens=4, prefill_seconds=0.5,
                        decode_tokens=3, decode_seconds=0.3,
                        first_token_seconds=0.52, total_seconds=0.94,
                        prefill_rates=Rates([8.0]),
                        decode_rates=Rates([10.0, 20.0, 5.0]))
        display.token(9, timing)
        display.close(timing)
        prefill, decode = self.rows(out.getvalue())
        # 4 tokens in 0.5 s is 8.0 t/s, and 3 in 0.3 s is 10.0 -- NOT the 11.7
        # mean of the three per-step rates, which is a different population
        for cell in ("4", "8.0"):
            self.assertIn(cell, prefill)
        for cell in ("3", "10.0"):
            self.assertIn(cell, decode)
        self.assertNotIn("11.7", decode, "the per-step mean leaked back in")
        plain = re.sub(r"\x1b\[[0-9;]*m", "", out.getvalue())
        self.assertIn("0.5", plain, "time to the first token is missing")
        self.assertIn("0.9", plain, "the turn's total time is missing")

    @staticmethod
    def rows(text: str) -> tuple[str, str]:
        """The table's two phase rows, colours aside."""
        plain = re.sub(r"\x1b\[[0-9;]*m", "", text)
        found = [line for line in plain.splitlines()
                 if line.startswith("  prefill") or line.startswith("  decode")]
        assert len(found) == 2, plain
        return found[0], found[1]

    def test_unclosed_thinking_is_terminated_on_close(self):
        out = io.StringIO()
        display = StreamDisplay(out, _decode, thinking_markers=[1, 2])
        display.token(1, Timing())
        display.token(7, Timing())
        display.close(Timing())
        self.assertTrue(
            out.getvalue().endswith(f"{GRAY}{_stats(Timing())}{RESET}\n")
        )

    def test_the_rates_appear_only_at_the_end(self):
        """While the text streams, nothing else shares its rows."""
        out = io.StringIO()
        display = StreamDisplay(out, _decode, thinking_markers=[1, 2])
        timing = Timing(prefill_tokens=4, prefill_seconds=0.5, decode_tokens=2, decode_seconds=0.5)
        for token in [7, 8, 9]:
            display.token(token, timing)
            display.flush(timing)
        self.assertNotIn("prefill", out.getvalue())
        display.close(timing)
        self.assertEqual(out.getvalue().count("prefill"), 1)

    def test_open_at_start_mutes_the_first_segment(self):
        out = io.StringIO()
        display = StreamDisplay(out, _decode, thinking_markers=[1, 2],
                                open_at_start=True)
        timing = Timing()
        display.token(7, timing)  # first token: the span opens before the paint
        display.token(8, timing)
        display.token(2, timing)  # the closing marker arrives in the stream
        display.token(9, timing)
        display.close(timing)
        self.assertIn(f"{GRAY}<7><8>{RESET}\n<9>", out.getvalue())

    def test_every_token_paints_as_it_arrives(self):
        """One write per token, so the answer reads as a stream.

        It used to batch behind a 0.1 s interval, which is a rate limit with
        no timer behind it: a token arriving early had nothing to wake it and
        waited for the NEXT one, so the output read as chunks. The interval
        was there to stop a terminal holding text in its line buffer, and
        `test_every_paint_flushes_the_stream` is what actually fixes that.
        """
        out = io.StringIO()
        display = StreamDisplay(out, _decode, thinking_markers=[1, 2])
        timing = Timing()
        display.token(7, timing)
        self.assertGreater(len(out.getvalue()), 0)
        # no sleeping, no waiting: each token is on the stream before the next
        for _ in range(50):
            before = len(out.getvalue())
            display.token(8, timing)
            self.assertGreater(len(out.getvalue()), before,
                               "a token arrived and nothing was painted")
        self.assertEqual(out.getvalue().count("<8>"), 50)
        display.close(timing)

    def test_every_paint_flushes_the_stream(self):
        """A terminal buffers lines: without a flush a paint stays inside
        the program and the text arrives in one lump at the end."""

        class _Counting(io.StringIO):
            def __init__(self) -> None:
                super().__init__()
                self.flushes = 0

            def flush(self) -> None:
                self.flushes += 1

        out = _Counting()
        display = StreamDisplay(out, _decode, thinking_markers=[1, 2])
        timing = Timing()
        display.token(7, timing)  # the first token paints immediately
        self.assertGreaterEqual(out.flushes, 1)


class ScreenTest(unittest.TestCase):
    """What the terminal shows is the text, and the rates below it.

    Each test paints onto a Screen and compares what is left with what the
    same text renders as on its own: anything the machinery leaves in the
    text's rows — a stray line, a drifted piece — breaks the equality.
    """

    def shown(self, tokens, decode, timing, **kwargs) -> str:
        display = StreamDisplay(io.StringIO(), decode, thinking_markers=[1, 2],
                                **kwargs)
        screen = Screen()
        display.out = _Feed(screen)
        for token in tokens:
            display.token(token, timing)
            display.flush(timing)  # a slow stream paints as often as it can
        display.close(timing)
        return screen.text()

    def golden(self, text: str, timing) -> str:
        screen = Screen()
        screen.feed(f"{text}\n{_stats(timing)}\n")
        return screen.text()

    def test_a_long_generation_lays_out_as_just_its_text(self):
        timing = Timing(prefill_tokens=8, prefill_seconds=1.0,
                        decode_tokens=610, decode_seconds=60.0)
        text = "the sea is wide and the wind is cold upon the northern shore. " * 10
        tokens = [ord(c) for c in text]
        shown = self.shown(tokens, lambda ids: "".join(map(chr, ids)), timing)
        self.assertEqual(shown, self.golden(text, timing))

    def test_a_span_break_puts_its_texts_on_their_own_rows(self):
        timing = Timing(prefill_tokens=4, prefill_seconds=1.0,
                        decode_tokens=10, decode_seconds=2.0)
        pieces = {1: "", 2: ""}  # the markers render as nothing here
        tokens = [1] + [ord(c) for c in "think"] + [2] + [ord(c) for c in "answer"]
        shown = self.shown(tokens, lambda ids: "".join(pieces.get(t, chr(t)) for t in ids),
                           timing)
        self.assertEqual(shown, self.golden("\nthink\nanswer", timing))

    def test_the_rates_are_the_last_line_under_the_response(self):
        timing = Timing(prefill_tokens=4, prefill_seconds=1.0,
                        decode_tokens=3, decode_seconds=1.0)
        shown = self.shown([ord(c) for c in "abc"], lambda ids: "".join(map(chr, ids)),
                           timing)
        self.assertEqual(shown, self.golden("abc", timing))
        self.assertEqual(" ".join(shown.split()),
                         " ".join(("abc\n" + _stats(timing)).split()))


if __name__ == "__main__":
    unittest.main()
