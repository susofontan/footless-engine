"""templates/model-package: the starting point for a new model stays loadable.

Its manifest parses and matches its folders, its runtime loads through the
engine's own loader with every required verb present, and its facts agree with
its manifest -- so a package started from it begins inside the contract.
"""
from __future__ import annotations

import unittest
from pathlib import Path

from footless.engine import load_runtime
from footless.manifest import read, verify, verify_backends
from footless.sdk import RUNTIME_METHODS

TEMPLATE = Path(__file__).resolve().parent.parent / "templates" / "model-package" / "footless"


class TemplateTest(unittest.TestCase):
    def test_the_template_package_loads_and_agrees_with_itself(self):
        manifest = read(TEMPLATE)
        verify_backends(manifest, TEMPLATE)
        for backend in manifest.backends:
            runtime = load_runtime(TEMPLATE / backend / "runtime.py")
            for name in RUNTIME_METHODS:
                self.assertTrue(callable(getattr(runtime, name, None)), name)
            verify(manifest, runtime.facts())
