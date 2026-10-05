"""The benchmark sweeps frontiers and reports the ds4-bench CSV columns."""

from __future__ import annotations

import io
import unittest
from pathlib import Path

from footless.bench import frontiers, run
from footless.engine import Engine
from footless.sdk import ContractError

MODEL = Path(__file__).resolve().parent / "mockup"


class BenchTest(unittest.TestCase):
    def test_frontier_sweeps(self):
        self.assertEqual(frontiers(8, 24, 8, 1.0), [8, 16, 24])
        self.assertEqual(frontiers(4, 64, 0, 2.0), [4, 8, 16, 32, 64])

    def test_rows_follow_the_bench_schema(self):
        engine = Engine.open(MODEL)
        self.addCleanup(engine.close)
        tokens = engine.runtime.encode("0123456789" * 4)
        out = io.StringIO()
        run(engine, tokens, [8, 16], gen_tokens=4, show_output=False, out=out)
        lines = out.getvalue().strip().splitlines()
        self.assertTrue(lines[0].startswith("ctx_tokens,prefill_tokens,prefill_tps"))
        row = lines[1].split(",")
        self.assertEqual(row[0], "8")  # ctx_tokens
        self.assertEqual(row[1], "7")  # prefill_tokens: the last token goes to cycle 1
        self.assertEqual(row[3], "4")  # gen_tokens
        self.assertEqual(row[6], "3")  # gen_steady_tokens
        self.assertGreater(int(row[8]), 0)  # kvcache_bytes
        self.assertEqual(lines[2].split(",")[0], "16")
        self.assertEqual(lines[2].split(",")[1], "7")

    def test_prefill_only_feeds_the_whole_interval(self):
        engine = Engine.open(MODEL)
        self.addCleanup(engine.close)
        tokens = engine.runtime.encode("0123456789" * 4)
        out = io.StringIO()
        run(engine, tokens, [8], gen_tokens=0, show_output=False, out=out)
        row = out.getvalue().strip().splitlines()[1].split(",")
        self.assertEqual(row[1], "8")
        self.assertEqual(row[3], "0")

    def test_frontier_past_the_prompt_is_an_error(self):
        engine = Engine.open(MODEL)
        self.addCleanup(engine.close)
        self.assertRaises(
            ContractError, run, engine, [1, 2, 3], [8], 4, False, io.StringIO()
        )


if __name__ == "__main__":
    unittest.main()
