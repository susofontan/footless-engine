"""tools/check_package.py against the mock packages: each keeps the contract.

The checker is how a new model package is told it conforms; the mocks are the
smallest packages that do. If a contract change breaks either, this says so.
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("check_package", ROOT / "tools/check_package.py")
check_package = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check_package)


class CheckPackageTest(unittest.TestCase):
    def run_on(self, name: str) -> str:
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            status = check_package.main(["--model", str(ROOT / "tests" / name)])
        self.assertEqual(status, 0, out.getvalue())
        return out.getvalue()

    def test_every_mock_package_conforms(self):
        for name in ("mockup", "mockup-thinking", "mockup-tools"):
            with self.subTest(package=name):
                report = self.run_on(name)
                self.assertNotIn("FAIL", report)

    def test_declared_capabilities_are_checked_and_undeclared_skipped(self):
        report = self.run_on("mockup-thinking")
        self.assertRegex(report, r"PASS\s+label_logprobs")
        self.assertRegex(report, r"SKIP\s+tool_calls\s+not declared")
        self.assertRegex(self.run_on("mockup-tools"), r"PASS\s+tool_calls")

    def test_a_package_that_breaks_a_rule_fails(self):
        """A mock whose clone forgets the state is caught: the check has teeth."""
        engine_open = check_package.Engine.open

        def broken_open(*args, **kwargs):
            engine = engine_open(*args, **kwargs)
            rt = engine.runtime
            rt.clone_state = lambda state: rt.new_state()   # a copy of nothing
            return engine

        check_package.Engine.open = broken_open
        try:
            out = io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
                status = check_package.main(["--model", str(ROOT / "tests/mockup")])
        finally:
            check_package.Engine.open = engine_open
        self.assertEqual(status, 1)
        self.assertRegex(out.getvalue(), r"FAIL\s+clone_state")
