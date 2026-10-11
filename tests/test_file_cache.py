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
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from footless.cli import _caching, _size
from footless.engine import Admit, Engine, FileCache, Wait, Want
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
        bytes, and a blob is the export's bytes as they are (compression was
        tried and dropped: ~1.1x on float states for gigabytes of transient
        RAM at save time)."""
        return len(self.engine.runtime.export_state(self._state(tokens)))

    def _cache(self, max_bytes: int, log=None) -> FileCache:
        cache = FileCache(self.engine.runtime, self.dir.name, max_bytes, log=log)
        self.addCleanup(cache.close)
        return cache

    def test_lru_drops_the_oldest_at_the_cap(self):
        entry = self._disk_size([1])
        cache = self._cache(2 * entry)
        keys = [[1], [2], [3]]
        for key in keys:
            cache.save(key, self._state(key))
        cache.flush()                      # saves land beside the model
        self.assertFalse(cache.has([1]))
        self.assertTrue(cache.has([2]) and cache.has([3]))
        self.assertEqual(len(list(Path(self.dir.name).glob("*.state"))), 2)

    def test_a_hit_moves_its_entry_off_the_front(self):
        entry = self._disk_size([1])
        cache = self._cache(2 * entry)
        cache.save([1], self._state([1]))
        cache.save([2], self._state([2]))
        cache.flush()
        cache.load([1])                    # [2] is the oldest now
        cache.save([3], self._state([3]))
        cache.flush()
        self.assertTrue(cache.has([1]))
        self.assertFalse(cache.has([2]))

    def test_a_full_disk_evicts_until_the_write_fits(self):
        cache = self._cache(1 << 20)
        cache.save([1], self._state([1]))
        cache.flush()                      # [1] is on disk before the full disk
        real, writes = Path.write_bytes, []

        def flaky(path, data):
            writes.append(path)
            if len(writes) == 1:
                raise OSError(errno.ENOSPC, "No space left on device")
            return real(path, data)

        with mock.patch.object(Path, "write_bytes", flaky):
            cache.save([2], self._state([2]))
            cache.flush()
        self.assertTrue(cache.has([2]))
        self.assertFalse(cache.has([1]))   # the oldest made room

    def test_a_state_over_the_cap_is_not_saved(self):
        cache = self._cache(4)
        cache.save([1, 2, 3, 4, 5], self._state([1, 2, 3, 4, 5]))
        cache.flush()
        self.assertFalse(cache.has([1, 2, 3, 4, 5]))
        self.assertEqual(list(Path(self.dir.name).glob("*.state")), [])

    def test_a_state_whose_export_would_not_fit_is_not_saved(self):
        # the peak of a save is the state with the export's copies beside it:
        # no room, and the state is not saved (here: nothing left at all)
        log = []
        cache = FileCache(self.engine.runtime, self.dir.name, 1 << 30,
                          log=lambda level, msg: log.append(msg), available=lambda: 4)
        self.addCleanup(cache.close)
        owned = self._state([1])
        cache.save([1], owned, owned=True)
        cache.flush()
        self.assertFalse(cache.has([1]))
        self.assertIsNone(owned.claim, "an owned state is freed when it is not saved")
        self.assertTrue(any("not saved" in m for m in log))

    def test_image_placeholder_keys_never_match_another_image(self):
        one = [1, 2, ("a" * 64, 0), 3]
        other = [1, 2, ("b" * 64, 0), 3]
        cache = self._cache(1 << 20)
        cache.save(one, self._state(one))
        cache.flush()
        self.assertIsNotNone(cache.load(one))
        self.assertIsNone(cache.load(other), "a state keyed on one image serves another")
        self.assertFalse(cache.has(other))
        hit = cache.match(other)
        self.assertIsNotNone(hit)
        self.assertEqual(hit[1], 2, "the shared part stops before the image")

    def test_packed_and_placeholder_keys_share_one_cache(self):
        cache = self._cache(1 << 20)
        cache.save([1, 2, 3], self._state([1, 2, 3]))
        cache.save([1, 2, ("a" * 64, 0)], self._state([1, 2, 3]))
        cache.flush()
        self.assertTrue(cache.has([1, 2, 3]))
        self.assertTrue(cache.has([1, 2, ("a" * 64, 0)]))
        self.assertIsNone(cache.match([9, 9]))


    def test_an_unreadable_entry_is_a_miss_and_is_dropped(self):
        cache = self._cache(1 << 20)
        cache.save([1], self._state([1]))
        cache.flush()
        ident = next(Path(self.dir.name).glob("*.state")).stem
        (Path(self.dir.name) / f"{ident}.state").write_bytes(b"not a state")
        self.assertIsNone(cache.load([1]))
        self.assertFalse(cache.has([1]))

    def test_a_saved_state_round_trips_and_the_file_is_its_bytes(self):
        state = self._state([7] * 400)
        cache = self._cache(1 << 20)
        cache.save([7] * 400, state)
        cache.flush()
        (blob,) = Path(self.dir.name).glob("*.state")
        self.assertEqual(blob.stat().st_size,
                         len(self.engine.runtime.export_state(state)),
                         "the file holds the export's bytes as they are")
        back = cache.load([7] * 400)
        self.assertIsNotNone(back)
        self.assertEqual(self.engine.runtime.export_state(back),
                         self.engine.runtime.export_state(state))

    def test_an_owned_save_is_written_and_its_state_freed_once(self):
        runtime = self.engine.runtime
        freed = []
        real_free = runtime.free_state
        runtime.free_state = lambda state: (freed.append(state), real_free(state))
        cache = self._cache(1 << 20)
        clone = runtime.clone_state(self._state([1, 2, 3]))
        cache.save([1, 2, 3], clone, owned=True)
        cache.flush()
        self.assertTrue(list(Path(self.dir.name).glob("*.state")))
        self.assertEqual([id(s) for s in freed], [id(clone)])
        second = runtime.clone_state(self._state([4, 5]))
        cache.save([4, 5], second, owned=True)
        cache.close()                      # and the same through close
        self.assertEqual([id(s) for s in freed], [id(clone), id(second)])
        self.assertEqual(len(list(Path(self.dir.name).glob("*.state"))), 2)

    def test_a_save_dropped_by_the_queue_bound_still_frees_its_state(self):
        runtime = self.engine.runtime
        freed, logs = [], []
        real_free, real_export = runtime.free_state, runtime.export_state
        runtime.free_state = lambda state: (freed.append(state), real_free(state))
        gate = threading.Event()

        def blocked_export(state):
            gate.wait()                    # the worker cannot finish one job
            return real_export(state)

        runtime.export_state = blocked_export
        cache = self._cache(1 << 20, log=lambda level, message: logs.append((level, message)))
        clones = [runtime.clone_state(self._state([i])) for i in range(10)]
        for i, clone in enumerate(clones):
            cache.save([i], clone, owned=True)
        gate.set()
        cache.close()
        self.assertEqual(len(freed), 10, "every owned state freed exactly once")
        self.assertEqual({id(s) for s in freed}, {id(c) for c in clones})
        self.assertLess(len(list(Path(self.dir.name).glob("*.state"))), 10,
                        "the save queue bound drops the oldest queued save")
        self.assertTrue(any(level == "info" and "dropped" in message
                            for level, message in logs), logs)


class SentinelTest(unittest.TestCase):
    """What a hit was cut back to, learned and then preferred."""

    def test_a_cut_point_is_saved_and_beats_a_longer_partial_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = Engine.open(MODEL, prefix_cache=(tmp, 1 << 20))
            try:
                files = engine.cache.files
                # an entry the next request only shares its first five tokens
                # with: the hit is cut back to those five
                files.save([1, 2, 3, 4, 5, 6, 7, 8, 9],
                           _state_of(engine.runtime, [1, 2, 3, 4, 5, 6, 7, 8, 9]))
                files.flush()
                gen = engine.sequence([1, 2, 3, 4, 5, 9, 9], 4)
                want = next(gen)
                self.assertIsInstance(want, Admit)
                while not isinstance(want, Want):
                    want = next(gen)  # a disk load's Waits, then the first Want
                files.flush()         # the sentinel's save lands beside the model
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
            finally:
                engine.close()        # before the directory goes away


class SupersedeTest(unittest.TestCase):
    def test_a_conversations_next_turn_replaces_its_saved_state(self):
        """The prefix cache's saved copy of an entry a later turn supersedes
        goes with the entry: one saved state a conversation."""
        with tempfile.TemporaryDirectory() as tmp:
            engine = Engine.open(MODEL, prefix_cache=(tmp, 1 << 20))
            try:
                rt = engine.runtime
                first = engine.generate_tokens(rt.encode("user: hi"), 4)
                engine.cache.files.flush()
                self.assertEqual(len(list(Path(tmp).glob("*.state"))), 1)
                turn2 = rt.encode("user: hi") + first.absorbed_tokens + rt.encode(" user: more")
                engine.generate_tokens(turn2, 4)
                engine.cache.files.flush()
                saved = [json.loads(p.read_text()) for p in Path(tmp).glob("*.key")]
                self.assertEqual(len(saved), 1, saved)
                self.assertEqual(saved[0][:len(turn2)], turn2)
            finally:
                engine.close()


class WaitTest(unittest.TestCase):
    """A disk hit waits for its read without stopping the engine."""

    SCRIPT_TEXT = "the mock model speaks"

    @staticmethod
    def _two_runs(tmp: str):
        """A first engine leaves its prefix on disk; a second one only
        finds it there (returns its head and tail token ids)."""
        first = Engine.open(MODEL, prefix_cache=(tmp, 1 << 20))
        head = first.runtime.encode("hello")
        tail = first.runtime.encode("!!!")
        first.generate_tokens(head, 64)
        first.close()
        return head, tail

    def test_a_disk_hit_waits_before_the_first_want_and_still_answers(self):
        with tempfile.TemporaryDirectory() as tmp:
            head, tail = self._two_runs(tmp)
            again = Engine.open(MODEL, prefix_cache=(tmp, 1 << 20))
            gate, real_read = threading.Event(), Path.read_bytes

            def slow_read(path):
                data = real_read(path)
                if path.suffix == ".state":
                    gate.wait()          # the read takes its time
                return data

            try:
                with mock.patch.object(Path, "read_bytes", slow_read):
                    gen = again.sequence(head + tail, 64)
                    self.assertIsInstance(next(gen), Admit)
                    self.assertIsInstance(next(gen), Wait,
                                          "the disk load must yield Wait")
                    gate.set()
                    want = next(gen)
                    while not isinstance(want, Want):
                        want = next(gen)  # more Waits, then the first Want
                    result = None
                    while True:          # step the items the sequence wants
                        step = again.runtime.step([want.item], want.ctx)
                        try:
                            want = gen.send((step[0], 0.0))
                        except StopIteration as stop:
                            result = stop.value
                            break
            finally:
                gate.set()
                again.close()            # before the directory goes away
            self.assertEqual(result.text, self.SCRIPT_TEXT)

    def test_the_batcher_waits_for_a_slow_read_instead_of_spinning(self):
        with tempfile.TemporaryDirectory() as tmp:
            head, tail = self._two_runs(tmp)
            again = Engine.open(MODEL, prefix_cache=(tmp, 1 << 20))
            real_read = Path.read_bytes

            def slow_read(path):
                data = real_read(path)
                if path.suffix == ".state":
                    time.sleep(0.1)      # the ticket completes ~100 ms later
                return data

            try:
                with mock.patch.object(Path, "read_bytes", slow_read):
                    result = again.generate_tokens(head + tail, 64)
            finally:
                again.close()            # before the directory goes away
            self.assertEqual(result.text, self.SCRIPT_TEXT)


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



class BudgetTest(unittest.TestCase):
    """The states' budget: down under pressure, back within its declaration."""

    def test_the_budget_shrinks_and_comes_back_but_never_past_declared(self):
        from footless.engine import Ledger

        ledger = Ledger(1000, lambda: False)
        ledger.set_budget(400)                  # pressure takes room away
        self.assertEqual(ledger.available(), 400)
        ledger.set_budget(2000)                 # calm gives it back, capped
        self.assertEqual(ledger.available(), 1000)


    def test_pressure_evicts_the_cache_but_never_refuses_a_live_request(self):
        """A budget lowered by memory pressure evicts cached states on the
        next claim; with nothing left to evict a live request still gets its
        room up to the ceiling (the most set_budget gave), and only past it
        is refused."""
        from footless.engine import Ledger
        from footless.sdk import BudgetRefused

        cached = []                             # two cached states' claims

        def evict():
            if not cached:
                return False
            ledger.release(cached.pop())
            return True
        ledger = Ledger(10_000, evict)
        ledger.set_budget(1000)                 # the runtime's room at open: the ceiling
        cached += [ledger.claim(100), ledger.claim(100)]
        ledger.set_budget(100)                  # critical pressure
        ledger.claim(600)                       # a live request: the cache goes, the claim stays
        self.assertEqual((cached, ledger.used), ([], 600))
        self.assertEqual(ledger.headroom(), 400)
        with self.assertRaises(BudgetRefused):
            ledger.claim(500)                   # past the ceiling
        ledger.set_budget(1000)
        self.assertEqual(ledger.limit(), 1000)

    def test_a_long_prompt_runs_under_pressure(self):
        """The whole-budget refusal at a request's start reads the ceiling:
        pressure is not a reason to refuse a conversation that fits it."""
        engine = Engine.open(MODEL)
        try:
            engine.services.accounting.set_budget(8 * 200)
            engine.services.accounting.set_budget(8 * 20)      # pressure
            result = engine.generate("x" * 100, 8)
            self.assertTrue(result.tokens)
        finally:
            engine.close()

if __name__ == "__main__":
    unittest.main()
