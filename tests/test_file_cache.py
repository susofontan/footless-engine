"""Saved states on disk: the prompt cache and the prefix cache.

A `FileCache` keeps `export_state` blobs in a directory of its own, under a
byte cap of its own: a whole prompt's state (the prompt cache) lets the same
prompt again skip its prefill, and any absorbed prefix's (the prefix cache)
lets a shared head skip its own -- both across runs. Least-recently-used
entries go when the cap is reached, and when the filesystem is full.

    python3 -m unittest tests.test_file_cache
"""

from __future__ import annotations

import errno
import tempfile
import unittest
import zlib
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from footless.cli import _caching, _size
from footless.engine import COMPRESS, Admit, Engine, FileCache, Want
from footless.sdk import ContractError

MODEL = Path(__file__).resolve().parent / "mockup"


def _state_of(runtime, tokens):
    """A mock state holding `tokens`, all fed (as a saved prompt's is)."""
    state = runtime.new_state()
    state.tokens.extend(tokens)
    state.fed = len(state.tokens)
    return state


class FileCacheTest(unittest.TestCase):
    """The disk tier alone: keys, limits, and what a full disk does."""

    def setUp(self) -> None:
        self.engine = Engine.open(MODEL)
        self.addCleanup(self.engine.close)
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)

    def _state(self, tokens):
        return _state_of(self.engine.runtime, tokens)

    def _disk_size(self, tokens) -> int:
        """What a saved state's blob costs on disk: the cap counts file
        bytes, and blobs are stored zlib-compressed."""
        return len(zlib.compress(
            self.engine.runtime.export_state(self._state(tokens)), COMPRESS))

    def test_lru_drops_the_oldest_at_the_cap(self):
        entry = self._disk_size([1])
        cache = FileCache(self.engine.runtime, self.dir.name, 2 * entry)
        keys = [[1], [2], [3]]
        for key in keys:
            cache.save(key, self._state(key))
        self.assertFalse(cache.has([1]))
        self.assertTrue(cache.has([2]) and cache.has([3]))
        self.assertEqual(len(list(Path(self.dir.name).glob("*.state"))), 2)

    def test_a_hit_moves_its_entry_off_the_front(self):
        entry = self._disk_size([1])
        cache = FileCache(self.engine.runtime, self.dir.name, 2 * entry)
        cache.save([1], self._state([1]))
        cache.save([2], self._state([2]))
        cache.load([1])                    # [2] is the oldest now
        cache.save([3], self._state([3]))
        self.assertTrue(cache.has([1]))
        self.assertFalse(cache.has([2]))

    def test_a_full_disk_evicts_until_the_write_fits(self):
        cache = FileCache(self.engine.runtime, self.dir.name, 1 << 20)
        cache.save([1], self._state([1]))
        real, writes = Path.write_bytes, []

        def flaky(path, data):
            writes.append(path)
            if len(writes) == 1:
                raise OSError(errno.ENOSPC, "No space left on device")
            return real(path, data)

        with mock.patch.object(Path, "write_bytes", flaky):
            cache.save([2], self._state([2]))
        self.assertTrue(cache.has([2]))
        self.assertFalse(cache.has([1]))   # the oldest made room

    def test_a_state_over_the_cap_is_not_saved(self):
        cache = FileCache(self.engine.runtime, self.dir.name, 4)
        cache.save([1, 2, 3, 4, 5], self._state([1, 2, 3, 4, 5]))
        self.assertFalse(cache.has([1, 2, 3, 4, 5]))
        self.assertEqual(list(Path(self.dir.name).glob("*.state")), [])

    def test_an_unreadable_entry_is_a_miss_and_is_dropped(self):
        cache = FileCache(self.engine.runtime, self.dir.name, 1 << 20)
        cache.save([1], self._state([1]))
        ident = next(Path(self.dir.name).glob("*.state")).stem
        (Path(self.dir.name) / f"{ident}.state").write_bytes(b"not a state")
        self.assertIsNone(cache.load([1]))
        self.assertFalse(cache.has([1]))

    def test_a_saved_state_round_trips_and_the_file_is_compressed(self):
        state = self._state([7] * 400)
        cache = FileCache(self.engine.runtime, self.dir.name, 1 << 20)
        cache.save([7] * 400, state)
        (blob,) = Path(self.dir.name).glob("*.state")
        self.assertLess(blob.stat().st_size,
                        len(self.engine.runtime.export_state(state)),
                        "a repetitive state stores smaller than its export")
        back = cache.load([7] * 400)
        self.assertIsNotNone(back)
        self.assertEqual(self.engine.runtime.export_state(back),
                         self.engine.runtime.export_state(state))


class SentinelTest(unittest.TestCase):
    """What a hit was cut back to, learned and then preferred."""

    def test_a_cut_point_is_saved_and_beats_a_longer_partial_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = Engine.open(MODEL, prefix_cache=(tmp, 1 << 20))
            self.addCleanup(engine.close)
            files = engine.cache.files
            # an entry the next request only shares its first five tokens
            # with: the hit is cut back to those five
            files.save([1, 2, 3, 4, 5, 6, 7, 8, 9],
                       _state_of(engine.runtime, [1, 2, 3, 4, 5, 6, 7, 8, 9]))
            gen = engine.sequence([1, 2, 3, 4, 5, 9, 9], 4)
            want = next(gen)
            self.assertIsInstance(want, Admit)
            while not isinstance(want, Want):
                want = next(gen)     # a disk load's Waits, then the first Want
            self.assertTrue(files.has([1, 2, 3, 4, 5]),
                            "the cut point was not saved as a sentinel")
            gen.close()
            # a later request sharing seven tokens with the LONG entry and
            # five with the sentinel: the clean sentinel wins (no truncate,
            # no replay) over the longer partial match
            got = files.match([1, 2, 3, 4, 5, 6, 7, 50], 0)
            self.assertIsNotNone(got)
            state, matched = got
            self.assertEqual(matched, 5)
            self.assertEqual(state.tokens, [1, 2, 3, 4, 5])


class AcrossRunsTest(unittest.TestCase):
    """What the caches are for: the same work twice, from disk."""

    def test_a_saved_prompt_skips_its_prefill(self):
        prompt = self.engine_prompt()
        with tempfile.TemporaryDirectory() as tmp:
            first = Engine.open(MODEL, prompt_cache=(tmp, 1 << 20))
            cold = first.generate_tokens(prompt, 8)
            first.close()
            self.assertTrue(list(Path(tmp).glob("*.state")))
            again = Engine.open(MODEL, prompt_cache=(tmp, 1 << 20))
            warm = again.generate_tokens(prompt, 8)
            again.close()
        self.assertEqual(warm.prompt_tokens_fed, 0)      # nothing re-fed
        self.assertEqual(warm.text, cold.text)

    def test_a_saved_prefix_serves_the_shared_head(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = Engine.open(MODEL, prefix_cache=(tmp, 1 << 20))
            head = first.runtime.encode("hello")
            tail = first.runtime.encode("!!!")
            first.generate_tokens(head, 4)
            first.close()
            again = Engine.open(MODEL, prefix_cache=(tmp, 1 << 20))
            warm = again.generate_tokens(head + tail, 4)
            again.close()
        self.assertEqual(warm.prompt_tokens_fed, len(tail))

    def test_without_the_flags_nothing_is_saved(self):
        engine = Engine.open(MODEL)
        try:
            self.assertIsNone(engine.prompt_cache)
            self.assertIsNone(engine.cache.files)
        finally:
            engine.close()

    @staticmethod
    def engine_prompt():
        engine = Engine.open(MODEL)
        try:
            return engine.runtime.encode("hello")
        finally:
            engine.close()


class FlagsTest(unittest.TestCase):
    """The serve flags: sizes, and each cache's pair of them."""

    def test_size(self):
        self.assertEqual(_size("512M"), 512 << 20)
        self.assertEqual(_size("2G"), 2 << 30)
        self.assertEqual(_size("1024"), 1024)
        for bad in ("", "G", "1.5G", "abc", "-1"):
            with self.assertRaises(ContractError):
                _size(bad)

    def test_a_cache_needs_its_directory_and_its_cap(self):
        self.assertIsNone(_caching(SimpleNamespace(), "prompt-cache"))
        self.assertEqual(_caching(SimpleNamespace(prompt_cache="d",
                                                  prompt_cache_max="2G"),
                                  "prompt-cache"), ("d", 2 << 30))
        for args in (SimpleNamespace(prompt_cache="d"),
                     SimpleNamespace(prompt_cache_max="2G")):
            with self.assertRaises(ContractError):
                _caching(args, "prompt-cache")


if __name__ == "__main__":
    unittest.main()
