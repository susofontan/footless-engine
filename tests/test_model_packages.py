"""Model-package tests live with their packages; this runs them with the suite.

A package keeps its own tests in `<model>/footless/tests/`. They load weights
and want the package's hardware, so they run only when asked for, by package
directory:

    FOOTLESS_PACKAGES=models/<model> python3 -m unittest discover -s tests -t .

Several packages are separated by the path separator (`:` on POSIX). Without
the variable this module adds nothing, and the engine's suite stays stdlib only.
"""
from __future__ import annotations

import importlib.util
import os
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _package_tests() -> list[Path]:
    raw = os.environ.get("FOOTLESS_PACKAGES", "")
    out = []
    for entry in filter(None, raw.split(os.pathsep)):
        path = Path(entry)
        if not path.is_absolute():
            path = ROOT / path
        out.append(path / "footless" / "tests")
    return out


def _module_name(folder: Path, path: Path) -> str:
    """A unique, importable name: package directory names hold dots and dashes."""
    package = re.sub(r"\W", "_", folder.parent.parent.name)
    return f"model_test_{package}_{path.stem}"


def _skipped(path: Path, reason: str) -> type:
    """A module that cannot load here (a missing dependency, no device) reports one skip."""
    def test(self):
        self.skipTest(reason)  # noqa: ARG001

    return type(f"Skipped_{path.stem}", (unittest.TestCase,), {"test_skipped": test})


def load_tests(loader, tests, standard):
    suite = unittest.TestSuite()
    for folder in _package_tests():
        if not folder.is_dir():
            suite.addTest(loader.loadTestsFromTestCase(
                _skipped(folder, f"no package tests at {folder}")))
            continue
        for path in sorted(folder.glob("test_*.py")):
            name = _module_name(folder, path)
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            # Registered BEFORE it runs: unittest finds a module's
            # `tearDownModule` through sys.modules, so a module loaded by path
            # and never registered never gets its teardown -- and a package
            # suite that frees its device memory there would leave it held for
            # the next file.
            sys.modules[name] = module
            try:
                spec.loader.exec_module(module)
            except (ModuleNotFoundError, unittest.SkipTest) as exc:
                sys.modules.pop(name, None)
                suite.addTest(loader.loadTestsFromTestCase(_skipped(path, str(exc))))
                continue
            suite.addTests(loader.loadTestsFromModule(module))
    return suite
