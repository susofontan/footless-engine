"""End-to-end tests of the engine against the scripted mock model."""

from __future__ import annotations

import random
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

from footless import manifest
from dataclasses import replace

from footless.cli import _assistant_message, _sampling
from footless import engine as engine_module
from footless.engine import Batcher, Engine, loop_cut, loop_cut_exact
from footless.manifest import Manifest
from footless.sdk import (
    BudgetRefused,
    ContractError,
    Facts,
    SamplingSpec,
    StepItem,
    StepResult,
)

MODEL = Path(__file__).resolve().parent / "mockup"
MODEL_THINKING = Path(__file__).resolve().parent / "mockup-thinking"


class _NotCancelled:
    def cancelled(self) -> bool:
        return False


class _Cancelled:
    def cancelled(self) -> bool:
        return True


class MockModelTest(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = Engine.open(MODEL)
        self.addCleanup(self.engine.close)

    def test_generate_produces_the_script(self):
        r = self.engine.generate("hello", 64)
        self.assertEqual(r.text, "the mock model speaks")
        self.assertIs(r.finished, True)

    def test_generate_stops_at_max_tokens(self):
        r = self.engine.generate("hello", 5)
        self.assertEqual(r.text, "the m")
        self.assertIs(r.finished, False)
        self.assertEqual(len(r.tokens), 5)

    def test_prefix_reuse_feeds_fewer_prompt_tokens(self):
        self.engine.generate("hello world", 64)
        r2 = self.engine.generate("hello there", 64)
        self.assertLess(
            r2.prompt_tokens_fed, len(self.engine.runtime.encode("hello there"))
        )

    def test_a_cached_state_sharing_a_few_tokens_is_not_taken(self):
        """A cached state is taken only when the shared prefix is worth its
        copy (HIT_MIN tokens, or half its prompt): a long prompt's state
        sharing three first tokens is left alone, the same prompt again is a
        whole hit."""
        enc = self.engine.runtime.encode
        self.engine.generate("abc" + "x" * 200, 8)
        r = self.engine.generate("abc" + "y" * 20, 8)
        self.assertEqual(r.prompt_tokens_fed, len(enc("abc" + "y" * 20)), "three shared tokens were taken")
        again = self.engine.generate("abc" + "y" * 20, 8)
        self.assertLess(again.prompt_tokens_fed, 2, "the same prompt again was not a hit")

    def test_a_replay_counts_as_prompt_fed(self):
        """A runtime whose truncate replays what it kept reports it in the
        first step's forwarded_tokens: those tokens were not saved."""
        runtime = self.engine.runtime
        self.engine.generate("hello world", 64)
        step = runtime.step
        replay = {"n": 0}

        def truncate(state, n):
            replay["n"] = n                   # this "model" forwards all it kept again
            type(runtime).truncate_state(runtime, state, n)

        def replaying(batch, ctx):
            results = step(batch, ctx)
            if replay["n"]:
                item, r = batch[0], results[0]
                r.forwarded_tokens = replay["n"] + max(len(item.input_tokens), 1)
                replay["n"] = 0
            return results

        runtime.truncate_state, runtime.step = truncate, replaying
        for prompt in ("hello there", "hello world"):    # a cut prompt, then the whole one again
            r = self.engine.generate(prompt, 4)
            self.assertEqual(r.prompt_tokens_fed, len(runtime.encode(prompt)), prompt)

    def test_state_verbs_roundtrip(self):
        runtime = self.engine.runtime
        st = runtime.new_state()
        runtime.step(
            [StepItem(st, runtime.encode("hello world"), None)], _NotCancelled()
        )
        self.assertEqual(runtime.state_nbytes(st), 8 * 11)
        clone = runtime.clone_state(st)
        runtime.truncate_state(clone, 5)
        self.assertEqual(runtime.state_nbytes(clone), 8 * 5)
        self.assertEqual(runtime.state_nbytes(st), 8 * 11)
        blob = runtime.export_state(st)
        back = runtime.import_state(blob)
        self.assertEqual(runtime.export_state(back), blob)

    def test_manifest_mismatch_is_hard_error(self):
        man = manifest.read(MODEL / "footless")
        bad = Facts(
            cache_bytes_per_token=999,
            max_batch=4,
            max_context=256,
            capabilities=[],
            stop_tokens=[256],
        )
        self.assertRaises(ContractError, manifest.verify, man, bad)

    def test_the_runtime_table_is_the_packages_own(self):
        head = ('name = "x"\nbackends = ["mock"]\nmax_batch = 1\nmax_context = 1\n'
                'cache_bytes_per_token = 1\ncapabilities = []\n')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest"
            path.write_text(head + '[runtime]\nkv_cache = "anything"\n')
            manifest.read(Path(tmp))           # not interpreted, only accepted
            path.write_text(head + 'runtime = "int8"\n')
            self.assertRaises(ContractError, manifest.read, Path(tmp))

    def test_backends_must_match_disk(self):
        man = Manifest(
            name="x",
            backends=["nope"],
            max_batch=4,
            max_context=256,
            cache_bytes_per_token=8,
            capabilities=[],
        )
        self.assertRaises(
            ContractError, manifest.verify_backends, man, MODEL / "footless"
        )

    def test_a_packages_own_tests_folder_is_not_a_backend(self):
        with tempfile.TemporaryDirectory() as tmp:
            footless_dir = Path(tmp) / "footless"
            (footless_dir / "mock").mkdir(parents=True)
            (footless_dir / "mock" / "runtime.py").write_text("")
            (footless_dir / "tests").mkdir()
            man = Manifest(
                name="x",
                backends=["mock"],
                max_batch=1,
                max_context=1,
                cache_bytes_per_token=1,
                capabilities=[],
            )
            manifest.verify_backends(man, footless_dir)

    def test_a_tooling_cache_folder_is_not_a_backend(self):
        """pytest leaves a cache beside the backends when run from the package."""
        with tempfile.TemporaryDirectory() as tmp:
            footless_dir = Path(tmp) / "footless"
            (footless_dir / "mock").mkdir(parents=True)
            (footless_dir / "mock" / "runtime.py").write_text("")
            (footless_dir / ".pytest_cache").mkdir()
            man = Manifest(
                name="x",
                backends=["mock"],
                max_batch=1,
                max_context=1,
                cache_bytes_per_token=1,
                capabilities=[],
            )
            manifest.verify_backends(man, footless_dir)

    def test_budget_refusal_is_a_hard_error(self):
        engine = Engine.open(MODEL, budget=8)  # 8 bytes = one cached token
        self.addCleanup(engine.close)
        self.assertRaises(BudgetRefused, engine.generate, "hello", 64)

    def test_a_finished_request_leaves_its_own_state_in_the_cache(self):
        """No copy at store: a copy needs the room of a second state, which a
        long conversation's state does not leave."""
        runtime = self.engine.runtime
        clones = []
        clone = runtime.clone_state
        runtime.clone_state = lambda st: clones.append(1) or clone(st)
        self.addCleanup(delattr, runtime, "clone_state")
        self.engine.generate("hello", 8)
        self.assertEqual(clones, [])
        self.assertEqual(len(self.engine.cache._entries), 1)

    def test_a_copy_that_does_not_fit_hands_the_entry_over(self):
        """The next turn continues the cached state itself when no copy fits
        beside it, instead of re-forwarding the whole prefix."""
        engine = self.engine
        first = engine.generate("hello there", 4)
        engine.cache.available = lambda: 0
        prompt = engine.runtime.encode("hello there") + first.absorbed_tokens + [65, 66]
        second = engine.generate_tokens(prompt, 4)
        self.assertEqual(second.timing.prefill_tokens, 2, "the prefix was re-forwarded")
        self.assertEqual(len(engine.cache._entries), 1)   # the continued state, stored back

    def test_a_continued_conversation_keeps_one_entry(self):
        """The next turn of a conversation supersedes the entry it started
        from (it read past that entry's prompt): one entry a conversation,
        not one a turn. A request that parts inside an entry's prompt (another
        conversation on the same system prompt) leaves the entry alone."""
        engine = self.engine
        rt = engine.runtime
        system = rt.encode("system: you are terse. ")
        first = engine.generate_tokens(system + rt.encode("user: hi"), 4)
        self.assertEqual(len(engine.cache._entries), 1)
        turn2 = system + rt.encode("user: hi") + first.absorbed_tokens + rt.encode(" user: more")
        engine.generate_tokens(turn2, 4)
        self.assertEqual(len(engine.cache._entries), 1, "the first turn's entry stayed")
        other = system + rt.encode("user: something else")
        engine.generate_tokens(other, 4)
        self.assertEqual(len(engine.cache._entries), 2, "a sibling conversation dropped the first")
        prompts = sorted(engine.cache._prompts.values())
        self.assertEqual(prompts, sorted([len(turn2), len(other)]))

    def test_a_failed_step_frees_the_request_state(self):
        runtime = self.engine.runtime
        made, freed = [], []
        new_state, free_state = runtime.new_state, runtime.free_state
        runtime.new_state = lambda: made.append(new_state()) or made[-1]
        runtime.free_state = lambda st: freed.append(st) or free_state(st)

        def failing(batch, ctx):
            raise RuntimeError("device fault")
        runtime.step = failing
        for name in ("new_state", "free_state", "step"):
            self.addCleanup(delattr, runtime, name)
        self.assertRaises(RuntimeError, self.engine.generate, "hello", 8)
        self.assertEqual(len(made), 1)
        self.assertIn(made[0], freed, "the request's state leaked")

    def test_a_prompt_whose_cache_cannot_fit_is_refused_before_any_step(self):
        engine = Engine.open(MODEL, budget=8)  # 8 bytes = one cached token
        self.addCleanup(engine.close)
        steps = []
        step = engine.runtime.step
        engine.runtime.step = lambda batch, ctx: steps.append(1) or step(batch, ctx)
        self.assertRaises(BudgetRefused, engine.generate, "hello", 4)
        self.assertEqual(steps, [], "the refusal came after work had started")

    def test_step_honours_cancellation(self):
        runtime = self.engine.runtime
        st = runtime.new_state()
        result = runtime.step([StepItem(st, [65], SamplingSpec())], _Cancelled())
        self.assertEqual(len(result), 1)
        self.assertIs(result[0].finished, True)
        self.assertIsNone(result[0].token)

    def test_streaming_reports_each_token_with_timing(self):
        seen = []
        r = self.engine.generate(
            "hello", 64, on_token=lambda token, timing: seen.append(token)
        )
        self.assertEqual(seen, r.tokens)
        self.assertEqual(r.timing.prefill_tokens, len(self.engine.runtime.encode("hello")))
        self.assertGreaterEqual(r.timing.prefill_seconds, 0.0)
        # the first token is sampled in the last prefill step, the rest decode
        self.assertEqual(r.timing.decode_tokens, len(r.tokens) - 1)
        self.assertGreaterEqual(r.timing.decode_seconds, 0.0)

    def test_timing_carries_the_step_speeds_of_the_turn(self):
        """One sample per prefill chunk and per producing decode step, so a
        report can give the turn's slowest, mean and fastest speed and not
        just its average."""
        r = self.engine.generate("hello", 64)
        self.assertEqual(len(r.timing.prefill_rates.values), 1)  # one chunk
        self.assertEqual(len(r.timing.decode_rates.values), r.timing.decode_tokens)
        self.assertTrue(all(v > 0 for v in r.timing.decode_rates.values))
        self.assertLessEqual(r.timing.decode_rates.minimum,
                             r.timing.decode_rates.mean)
        self.assertLessEqual(r.timing.decode_rates.mean,
                             r.timing.decode_rates.maximum)

    def test_the_timing_times_the_model_not_the_front_end(self):
        """A callback that takes its time must not read as a slow model."""
        def slow_painter(token, timing):
            time.sleep(0.02)

        r = self.engine.generate("hello", 8, on_token=slow_painter)
        self.assertGreater(len(r.tokens), 1)
        self.assertLess(r.timing.decode_seconds, 0.05,
                        "the callback's sleep was counted as decode time")

    def test_chat_tokens_render_the_conversation(self):
        toks = self.engine.chat_tokens(
            [{"role": "user", "content": "hi"}], thinking=False
        )
        self.assertEqual(toks, self.engine.runtime.encode("user: hi\nassistant: "))

    # ---------------------------------------------------------- seeding policy
    def test_unpinned_requests_get_different_draw_streams(self):
        """Mutation check for the constant default seed: two unpinned
        requests must not share a draw stream. The default seed used to be
        the constant 1234, so every request replayed the same stream and its
        first sampled token came from the same fixed mass-quantile. Let the
        constant default return and this fails on the seeds and the streams
        (the mock records the stream each request's spec drives)."""
        self.engine.generate("hello", 8)
        self.engine.generate("hello", 8)
        streams = self.engine.runtime.draw_streams
        self.assertEqual(len(streams), 2, "one draw stream per request")
        self.assertNotEqual(streams[0][:4], streams[1][:4],
                            "two unpinned requests drew the same stream")
        self.assertGreater(len(set(streams[0]) | set(streams[1])), 2)

    def test_an_explicit_seed_replays_one_per_request_stream(self):
        """A pinned seed is reproducible, and a seed identifies one
        PER-REQUEST stream: the same seed replays the same draws even when
        one spec object is shared across requests, and only the seed moves
        the stream."""
        shared = SamplingSpec(seed=1234)
        first = self.engine.generate("hello", 8, sampling=shared)
        second = self.engine.generate("hello", 8, sampling=shared)
        streams = self.engine.runtime.draw_streams
        self.assertEqual(len(streams), 2, "each request gets its own stream")
        self.assertEqual(streams[0], streams[1], "same seed, same draws")
        self.assertEqual(first.tokens, second.tokens, "same seed, same sequence")
        want = random.Random(1234)
        self.assertEqual(streams[0], [want.random() for _ in streams[0]],
                         "the seed's stream is random.Random(seed) advancing")
        self.engine.generate("hello", 8, sampling=SamplingSpec(seed=1234))
        self.assertEqual(self.engine.runtime.draw_streams[2], streams[0])
        self.engine.generate("hello", 8, sampling=SamplingSpec(seed=7))
        self.assertNotEqual(self.engine.runtime.draw_streams[3], streams[0],
                            "a different seed, a different stream")


class SpeculativeTest(unittest.TestCase):
    """The optional verb, and what the engine does without it.

    `mockup` declares no capabilities, which is the shape every package that
    cannot speculate has. These are the engine's half of the contract; whether
    a package that DOES declare it is exact is the package's own test.
    """

    def setUp(self) -> None:
        self.engine = Engine.open(MODEL)
        self.addCleanup(self.engine.close)

    def test_a_package_without_the_capability_is_refused_with_a_reason(self):
        self.assertNotIn("speculative", self.engine.facts.capabilities)
        with self.assertRaises(ContractError) as caught:
            self.engine.speculative(True)
        self.assertIn("speculative", str(caught.exception))

    def test_asking_for_nothing_is_not_an_error(self):
        self.assertIsNone(self.engine.speculative(False))

    def _declare_speculative(self) -> None:
        facts = self.engine.facts
        self.engine.facts = replace(facts, capabilities=facts.capabilities
                                    + ["speculative"])

    def test_a_declared_capability_with_no_verb_is_a_contradiction(self):
        """Reported, not swallowed — the manifest and the code must agree."""
        self._declare_speculative()
        with self.assertRaises(ContractError) as caught:
            self.engine.speculative(True)
        self.assertIn("set_speculative", str(caught.exception))

    def test_the_verb_is_called_when_the_capability_is_there(self):
        calls = []
        self._declare_speculative()
        self.engine.runtime.set_speculative = calls.append
        self.engine.speculative(True)
        self.assertEqual(calls, [True])

    def test_a_step_that_did_not_forward_is_not_a_decode_speed(self):
        """`forwarded=False` keeps a free step out of the per-step statistics.

        A model that verifies a draft hands the second token out on the next
        step with no forward at all. That step is a real token and real wall
        time -- both belong in the aggregate, which is what the display's
        `speed t/s` reports -- but it is not a sample of how fast a forward
        runs, and booking it as one reads in the hundreds of thousands and
        takes the turn's mean and max with it. Measured on the model package:
        a per-step mean of 68 701 t/s beside a real 15.0.

        The check is on the COUNT, one-sided, and not on the size: the mock's
        own forwards are instantaneous, so a magnitude would say nothing. Every
        recorded sample must be a step that forwarded AND produced; before the
        flag existed the free ones were recorded too, and the count came out
        over the number of forwards, which is what this catches.
        """
        engine = self.engine
        runtime = engine.runtime
        real_step = runtime.step
        stop = engine.facts.stop_tokens
        held = [None]
        forwards = [0]

        def fake(batch, ctx):
            if held[0] is None:
                held[0] = real_step(batch, ctx)
                last = held[0][-1]
                # a decode step, not the prefill: prefill samples are kept
                if not batch[0].input_tokens and last.token is not None \
                        and last.token not in stop:
                    forwards[0] += 1
                return held[0]
            done, held[0] = held[0], None
            token = done[-1].token
            if token is None or token in stop:
                return done
            return [StepResult(token, False, None, forwarded=False)]

        runtime.step = fake
        try:
            r = engine.generate_tokens(runtime.encode("hello"), 9,
                                       sampling=SamplingSpec(temperature=0.0,
                                                              top_k=1, seed=1))
        finally:
            runtime.step = real_step

        self.assertGreater(forwards[0], 0, "the fake never forwarded a step")
        self.assertLessEqual(len(r.timing.decode_rates.values), forwards[0],
                             "a step that forwarded nothing was booked as a "
                             "decode speed")
        # ...and the aggregate still counts every token and every second. It
        # is `len(tokens)` at most, not equal to it: a stop token is sampled
        # and ends the turn without joining the output, and that predates
        # anything here.
        self.assertGreater(r.timing.decode_tokens, 0)
        self.assertLessEqual(r.timing.decode_tokens, len(r.tokens))
        self.assertGreater(r.timing.decode_seconds, 0.0)

    def test_timing_reports_when_the_turn_started_and_how_long_it_took(self):
        """TTFT and the total are prefixes of one another, and both are real.

        These are the two numbers a reader feels rather than derives, and they
        were not recorded at all: a chat that answers in a sentence can be
        slower than one that answers in a page, and the per-phase rates do not
        say so.
        """
        engine = self.engine
        runtime = engine.runtime
        t = engine.generate_tokens(runtime.encode("hello"), 6,
                                   sampling=SamplingSpec(temperature=0.0,
                                                          top_k=1, seed=1))
        timing = t.timing
        self.assertGreater(timing.first_token_seconds, 0.0,
                           "time to the first token was never set")
        self.assertGreater(timing.total_seconds, timing.first_token_seconds,
                           "the turn did not outlast its own first token")
        # the phases account for the output, up to a stop token: it is
        # sampled and ends the turn without joining the text
        self.assertLessEqual(timing.decode_tokens, len(t.tokens))
        self.assertGreater(timing.decode_tokens, 0)
        # a turn that produced nothing has no first token and no duration
        empty = engine.generate_tokens(runtime.encode("hello"), 0)
        self.assertEqual(empty.timing.first_token_seconds, 0.0)
        self.assertEqual(empty.timing.decode_tokens, 0)

    def test_generation_is_unchanged_by_asking_and_declining(self):
        """The switch is the model's, and refusing it changes no output."""
        plain = self.engine.generate("hello", 16).text
        with self.assertRaises(ContractError):
            self.engine.speculative(True)
        self.assertEqual(self.engine.generate("hello", 16).text, plain)


class LoopGuardTest(unittest.TestCase):
    """The repetition rule, opt-in, and that a turn actually falls under it.

    The rule lives in `loop_cut`; the end-to-end half is here because the
    mock can script a collapse (`SCRIPT` patched below) where a real model
    produces one at random. Both halves matter: the rule alone would hide a
    wiring mistake that never applies it.
    """

    def setUp(self) -> None:
        self.engine = Engine.open(MODEL)
        self.addCleanup(self.engine.close)

    def test_flowing_text_is_never_cut(self):
        self.assertIsNone(loop_cut(list(range(500))))
        self.assertIsNone(loop_cut(list("the mock model speaks".encode())))

    def test_short_repeats_are_ordinary_text(self):
        """The rule must not fire on style: a refrain, ellipses, three
        identical short lines. It wants 512 tokens' worth of pure repetition
        past the first copy, and three copies of the span at least."""
        self.assertIsNone(loop_cut_exact([7] * 512))  # 511 repeated tokens < 512
        self.assertIsNone(loop_cut([1, 2, 3] * 2))  # a refrain: two copies
        self.assertIsNone(loop_cut([1, 2, 3, 4, 5, 6, 7, 8] * 3))  # three lines
        self.assertIsNone(loop_cut(list(range(80)) * 2))

    def test_repetitive_code_is_not_a_loop(self):
        """What valid code repeats: a zeros array (`0, ` x 60), a level map
        row, a 20 x 10 board written out row by row. The old bar of 24 cut
        each one mid-line and ended the answer there."""
        zeros = [15, 11, 220] * 60
        row = [256, 58, 15] + [11, 15] * 9 + [1125, 198]           # "  [0,0,...,0],\n"
        board = list(range(1000, 1040)) + row * 20
        for tokens in (zeros, row * 3, board):  # the default rule
            self.assertIsNone(loop_cut_exact(tokens))

    def test_a_runaway_tail_keeps_one_cycle(self):
        self.assertEqual(loop_cut([7] * 513), 1)  # 512 repeated tokens exactly
        self.assertEqual(loop_cut([7] * 2000), 1)
        self.assertEqual(loop_cut([1, 2, 3] * 200), 3)
        self.assertEqual(loop_cut(list(range(80)) * 8), 80)

    def test_a_short_cycle_inside_a_long_answer_is_caught(self):
        """The collapse this exists for: a few tokens of code repeated
        after a long varied prefix. A threshold over the WHOLE run hid this
        behind a hundred copies and let it stream to max_tokens -- it took
        96 tokens of the cycle to trip and then left the code half-written."""
        cycle = [207, 91, 17, 88, 240]  # e.g. '$this->jugadores[$j'
        tokens = list(range(1000, 2000)) + cycle * 120
        self.assertEqual(loop_cut(tokens), 1000 + len(cycle))

    def test_the_prefix_before_the_loop_is_kept(self):
        self.assertEqual(loop_cut([9] * 50 + [1, 2, 3] * 200), 53)

    def _script(self, tokens: list[int]) -> None:
        """What the mock emits next, in the module its `step` reads."""
        globals_ = type(self.engine.runtime).step.__globals__
        saved = globals_["SCRIPT"]
        globals_["SCRIPT"] = tokens
        self.addCleanup(globals_.__setitem__, "SCRIPT", saved)

    def test_a_repeating_generation_is_cut_short_when_asked(self):
        self._script([97, 98] * 300)  # "ab" forever
        r = self.engine.generate("hello", 4096, loop_guard=True)
        self.assertIs(r.looped, True)
        self.assertIs(r.finished, True)
        # the rewrite pass sees two equal 128-token blocks first and keeps one
        self.assertEqual(r.text, "ab" * 64)

    def test_the_default_guard_cuts_a_verbatim_runaway(self):
        """By default only the EXACT rule runs: a verbatim repeat is a
        runaway, cut to one cycle."""
        self._script([97, 98] * 300)  # 300 cycles of "ab", 600 tokens
        r = self.engine.generate("hello", 4096)
        self.assertIs(r.looped, True)
        self.assertEqual(r.text, "ab")

    def test_the_default_guard_leaves_near_copies_alone(self):
        """Near-copies (one token in eight differs) are what a table of
        iterations looks like to a rule over token ids, so the fuzzy passes
        run only when asked: `full` cuts this, the default does not."""
        rows = [[1, 2, 3, 4, 5, 6, 7, 10 + i] for i in range(40)]
        self._script([t for r in rows for t in r])
        r = self.engine.generate("hello", 4096)
        self.assertIs(r.looped, False, "the default guard cut near-copies")
        self.assertEqual(len(r.tokens), 320)
        r = self.engine.generate("hello", 4096, loop_guard="full")
        self.assertIs(r.looped, True, "the full guard should see near-copies")

    def test_the_guard_can_be_turned_off(self):
        """'off' (or False) lets a collapse run its course."""
        self._script([97, 98] * 300)
        for off in ("off", False):
            r = self.engine.generate("hello", 4096, loop_guard=off)
            self.assertIs(r.looped, False)
            self.assertEqual(r.text, "ab" * 300, "the collapse runs its course")

    def test_a_normal_turn_is_reported_as_not_looped(self):
        r = self.engine.generate("hello", 64)
        self.assertEqual(r.text, "the mock model speaks")
        self.assertIs(r.looped, False)


class AssistantMessageTest(unittest.TestCase):
    """What `chat` puts back into the conversation (`_assistant_message`).

    The template renders a history turn as `reasoning_content` inside a
    thinking block and `content` outside it. The whole stream as `content` --
    what this replaces -- shows the model its own reasoning chatter standing
    where the answer belongs, and it answers with more chatter.
    """

    def setUp(self) -> None:
        self.engine = Engine.open(MODEL_THINKING)
        self.addCleanup(self.engine.close)

    def _turn(self, thinking: bool) -> dict:
        tokens = self.engine.chat_tokens(
            [{"role": "user", "content": "hi"}], thinking
        )
        return _assistant_message(self.engine, self.engine.generate_tokens(tokens, 64),
                                  thinking)

    def test_thinking_and_answer_go_where_the_template_puts_them(self):
        message = self._turn(thinking=True)
        self.assertEqual(message["role"], "assistant")
        self.assertEqual(message["reasoning_content"], "thinking hard")
        self.assertEqual(message["content"], "the answer")

    def test_a_thinking_off_turn_is_all_answer(self):
        message = self._turn(thinking=False)
        self.assertEqual(message["reasoning_content"], "")
        self.assertEqual(message["content"], "the answer")


class ModelDefaultSamplingTest(unittest.TestCase):
    """What a request does not pin runs under the package's own defaults.

    `chat` and `serve` fill nothing in themselves: `Facts.default_sampling`
    is the checkpoint's generation config, a request's pins go over it, and
    the engine's own spec is the floor for a package that declares none.
    """

    def setUp(self) -> None:
        self.engine = Engine.open(MODEL)
        self.addCleanup(self.engine.close)

    def _ran_under(self) -> SamplingSpec:
        """The spec one generate ran under: the steps see it."""
        runtime = self.engine.runtime
        seen: list[SamplingSpec] = []
        real = runtime.step

        def spy(batch, ctx):
            seen.extend(i.sampling for i in batch if i.sampling is not None)
            return real(batch, ctx)

        runtime.step = spy
        try:
            self.engine.generate("hello", 4)
        finally:
            runtime.step = real
        self.assertTrue(seen, "no step sampled anything")
        return seen[0]

    def test_an_unpinned_request_runs_the_package_defaults(self):
        self.engine.facts = replace(
            self.engine.facts,
            default_sampling=SamplingSpec(temperature=1.0, top_k=20, top_p=0.95),
        )
        spec = self._ran_under()
        self.assertEqual((spec.temperature, spec.top_k, spec.top_p),
                         (1.0, 20, 0.95))
        self.assertIsNotNone(spec.seed, "the engine still owns randomness")

    def test_a_package_that_declares_none_falls_back_to_the_engine(self):
        self.assertIsNone(self.engine.facts.default_sampling)
        spec = self._ran_under()
        self.assertEqual(spec, replace(SamplingSpec(), seed=spec.seed))

    def test_cli_pins_go_over_the_package_defaults(self):
        """Pinning `--temperature` must not smuggle in the engine's `top_k`
        where the package declares its own."""
        defaults = SamplingSpec(temperature=1.0, top_k=20, top_p=0.95)
        args = SimpleNamespace(temperature=0.5, top_p=None, top_k=None,
                               min_p=None, seed=None)
        spec = _sampling(args, defaults)
        self.assertEqual((spec.temperature, spec.top_k, spec.top_p),
                         (0.5, 20, 0.95))

    def test_cli_pins_nothing_leaves_the_package_defaults_untouched(self):
        defaults = SamplingSpec(temperature=1.0, top_k=20, top_p=0.95)
        args = SimpleNamespace(temperature=None, top_p=None, top_k=None,
                               min_p=None, seed=None)
        self.assertIsNone(_sampling(args, defaults))


if __name__ == "__main__":
    unittest.main()


class _Flag:
    """A step context the test cancels when it wants."""

    def __init__(self) -> None:
        self.on = False

    def cancelled(self) -> bool:
        return self.on


class BatcherTest(unittest.TestCase):
    """Several requests stepped together (engine.Batcher), against the mock."""

    SCRIPT_TEXT = "the mock model speaks"

    def setUp(self) -> None:
        self.engine = Engine.open(MODEL)
        self.addCleanup(self.engine.close)
        self.rt = self.engine.runtime

    def seq(self, text: str, max_tokens: int = 64, **kwargs):
        return self.engine.sequence(self.rt.encode(text), max_tokens, **kwargs)

    def drain(self, batcher: Batcher) -> None:
        while batcher.busy():
            batcher.step()

    def patch(self, name: str, value) -> None:
        saved = getattr(engine_module, name)
        setattr(engine_module, name, value)
        self.addCleanup(setattr, engine_module, name, saved)

    def test_requests_step_together_and_each_gets_what_it_gets_alone(self):
        texts = ("one", "two two", "three")
        alone = [self.engine.generate(t, 64).text for t in texts]
        self.engine.cache.free_all()
        self.rt.batches.clear()
        batcher = Batcher(self.engine)
        tasks = [batcher.add(self.seq(t)) for t in texts]
        self.drain(batcher)
        self.assertEqual([t.result.text for t in tasks], alone)
        self.assertEqual(max(self.rt.batches), 3, "the three decodes share steps")
        self.assertLess(len(self.rt.batches), 3 * 23, "fewer steps than one at a time")

    def test_no_more_than_max_batch_at_once(self):
        batcher = Batcher(self.engine)
        tasks = [batcher.add(self.seq(f"request {i}")) for i in range(7)]
        self.drain(batcher)
        self.assertEqual(max(self.rt.batches), self.engine.facts.max_batch)
        self.assertEqual({t.result.text for t in tasks}, {self.SCRIPT_TEXT})
        self.assertTrue(all(t.error is None for t in tasks))

    def test_one_prefill_a_step_in_smaller_chunks_beside_decodes(self):
        self.patch("SHARED_CHUNK", 8)
        fed = []
        step = self.rt.step

        def recording(batch, ctx):
            fed.append([len(item.input_tokens) for item in batch])
            return step(batch, ctx)

        self.rt.step = recording
        batcher = Batcher(self.engine)
        first = batcher.add(self.seq("a short one"))
        batcher.step()                          # its prefill: it decodes from here
        second = batcher.add(self.seq("x" * 100))
        third = batcher.add(self.seq("y" * 30))
        self.drain(batcher)
        for step_fed in fed:
            self.assertLessEqual(sum(1 for n in step_fed if n), 1, step_fed)
        shared = [n for step_fed in fed[1:] for n in step_fed if n and len(step_fed) > 1]
        self.assertTrue(shared and max(shared) <= 8, shared)
        for task in (first, second, third):
            self.assertEqual(task.result.text, self.SCRIPT_TEXT)

    def test_a_request_waiting_on_a_load_never_holds_up_the_others(self):
        """One request waits for work outside the model (a saved state still
        loading): the others step on meanwhile, whichever was admitted first."""
        for waiter_first in (True, False):
            ticket = engine_module.Ticket()

            def waiting():
                yield engine_module.Admit(0)
                while not ticket.ready():
                    yield engine_module.Wait(ticket)

            batcher = Batcher(self.engine)
            if waiter_first:
                waiter = batcher.add(waiting())
                runner = batcher.add(self.seq("runs on"))
            else:
                runner = batcher.add(self.seq("runs on"))
                waiter = batcher.add(waiting())
            for _ in range(200):
                if runner.done:
                    break
                batcher.step()
            self.assertIsNone(runner.error, waiter_first)
            self.assertEqual(runner.result.text if runner.result else None,
                             self.SCRIPT_TEXT, waiter_first)
            self.assertFalse(waiter.done)
            ticket.finish(None)
            self.drain(batcher)

    def overlap(self, prefill_seconds: float):
        """The mock as a package that declares concurrent_prefill, its
        prefill steps taking a while: the steps it ran, with the thread
        each ran on and whether a prefill was running meanwhile."""
        import threading
        self.engine.facts.capabilities.append("concurrent_prefill")
        log, inside = [], []
        step = self.rt.step

        def slow(batch, ctx):
            prefill = any(item.input_tokens for item in batch)
            log.append((prefill, threading.current_thread().name, bool(inside)))
            if prefill:
                inside.append(1)
                time.sleep(prefill_seconds)
                try:
                    return step(batch, ctx)
                finally:
                    inside.pop()
            return step(batch, ctx)

        self.rt.step = slow
        return log

    def test_with_concurrent_prefill_decodes_go_on_beside_a_prefill(self):
        """A package that declares concurrent_prefill: a long prompt's chunks
        run on a thread of their own, one at a time, while the requests
        already decoding step on -- prefill and decode never share a step,
        and every answer is the one it gets alone."""
        self.patch("OVERLAP_CHUNK", 16)
        log = self.overlap(0.02)
        batcher = Batcher(self.engine)
        first = batcher.add(self.seq("short"))
        batcher.step()
        while not batcher._flight is None:
            batcher.step()
        second = batcher.add(self.seq("z" * 120))
        self.drain(batcher)
        self.assertEqual(first.result.text, self.SCRIPT_TEXT)
        self.assertEqual(second.result.text, self.SCRIPT_TEXT)
        beside = [entry for entry in log if not entry[0] and entry[2]]
        self.assertTrue(beside, "no decode step ran while a prefill was in flight")
        prefill_threads = {name for prefill, name, _ in log if prefill}
        self.assertEqual(prefill_threads, {"footless-prefill"})
        self.assertFalse(any(prefill and busy for prefill, _, busy in log),
                         "two prefill steps at once")

    def test_with_concurrent_prefill_a_refused_prefill_waits_for_room(self):
        """A prefill step the budget refuses beside a running request is
        preempted (it waits for room), not failed; both answers come out."""
        self.patch("ADMIT_HEADROOM", 0)
        self.patch("OVERLAP_CHUNK", 8)
        self.patch("SHARED_CHUNK", 8)
        engine = Engine.open(MODEL, budget=8 * 123)
        self.addCleanup(engine.close)
        engine.facts.capabilities.append("concurrent_prefill")
        logs = []
        engine.services.log = lambda level, message: logs.append(message)
        batcher = Batcher(engine)
        first = batcher.add(engine.sequence(engine.runtime.encode("x" * 10), 64))
        for _ in range(3):
            batcher.step()
        second = batcher.add(engine.sequence(engine.runtime.encode("y" * 100), 64))
        for _ in range(10000):
            if not batcher.busy():
                break
            batcher.step()
        self.assertIsNone(second.error)
        self.assertEqual(first.result.text, self.SCRIPT_TEXT)
        self.assertEqual(second.result.text, self.SCRIPT_TEXT)
        self.assertTrue(any("budget refused" in m for m in logs), logs)

    def grouped(self, decode_seconds: float, prefill_seconds: float = 0.0):
        """The mock as a package that declares concurrent_decode (and
        concurrent_prefill), its steps taking a while: each step's kind,
        thread, and how many decode steps ran at that moment (itself
        included), and whether a prefill was running."""
        import threading
        caps = self.engine.facts.capabilities
        caps.extend(c for c in ("concurrent_prefill", "concurrent_decode") if c not in caps)
        log, decoding, prefilling = [], [], []
        lock = threading.Lock()
        step = self.rt.step

        def slow(batch, ctx):
            prefill = any(item.input_tokens for item in batch)
            mine = prefilling if prefill else decoding
            with lock:
                mine.append(1)
                log.append((prefill, threading.current_thread().name, len(decoding), bool(prefilling)))
            try:
                time.sleep(prefill_seconds if prefill else decode_seconds)
                return step(batch, ctx)
            finally:
                with lock:
                    mine.pop()

        self.rt.step = slow
        return log

    def test_with_concurrent_decode_two_groups_go_at_their_own_pace(self):
        """A package that declares concurrent_decode: requests decoding
        together are stepped as two groups on threads of their own, at most
        two decode steps at once; every answer is the one it gets alone."""
        texts = ("one", "two two", "three", "four four")
        alone = [self.engine.generate(t, 64).text for t in texts]
        log = self.grouped(0.003)
        batcher = Batcher(self.engine)
        tasks = [batcher.add(self.seq(t)) for t in texts]
        self.drain(batcher)
        self.assertEqual([t.result.text for t in tasks], alone)
        decode_steps = [e for e in log if not e[0]]
        self.assertTrue(any(e[1] == "footless-decode" and e[2] == 2 for e in decode_steps),
                        "no two decode groups ran at once")
        self.assertLessEqual(max(e[2] for e in decode_steps), 2, "more than two decode steps at once")

    def test_with_concurrent_decode_a_prefill_brings_the_groups_together(self):
        """While a prompt is read (a prefill in flight) the decoding requests
        step together on the Batcher's thread, not as groups; the answers
        are the ones they get alone."""
        self.patch("OVERLAP_CHUNK", 16)
        self.patch("SHARED_CHUNK", 16)
        log = self.grouped(0.002, prefill_seconds=0.02)
        batcher = Batcher(self.engine)
        first = [batcher.add(self.seq(t)) for t in ("one", "two two")]
        for _ in range(6):
            batcher.step()
        late = batcher.add(self.seq("z" * 120))
        self.drain(batcher)
        for task in first + [late]:
            self.assertEqual(task.result.text, self.SCRIPT_TEXT)
        launched_beside = [e for e in log if not e[0] and e[1] == "footless-decode" and e[3]]
        # a group already flying when the prefill started may finish its step
        self.assertLessEqual(len(launched_beside), 2, "decode groups were launched beside a prefill")
        self.assertTrue(any(not e[0] and e[1] != "footless-decode" and e[3] for e in log),
                        "the decodes did not step together beside the prefill")

    def test_with_concurrent_decode_a_refused_group_moves_nothing(self):
        """A decode group's step the budget refuses moves nothing: the decodes
        step together again (the youngest of all yields, as without groups)
        until the live requests change -- no two groups preempting each
        other in turn. The answers are the ones the same requests get
        without groups."""
        import threading
        self.patch("ADMIT_HEADROOM", 0)

        def run(caps):
            engine = Engine.open(MODEL, budget=8 * 123)
            self.addCleanup(engine.close)
            engine.facts.capabilities.extend(caps)
            engine.services.log = lambda level, message: None
            refused, step = [], engine.runtime.step

            def recording(batch, ctx):
                try:
                    return step(batch, ctx)
                except BudgetRefused:
                    refused.append(threading.current_thread().name)
                    raise
            engine.runtime.step = recording
            batcher = Batcher(engine)
            tasks = [batcher.add(engine.sequence(engine.runtime.encode(c * 24), 64)) for c in "xywv"]
            for _ in range(2000):
                if not batcher.busy():
                    break
                batcher.step()
            for task in tasks:
                self.assertIsNone(task.error)
            return [t.result.text for t in tasks], refused

        want, _ = run(["concurrent_prefill"])
        got, refused = run(["concurrent_prefill", "concurrent_decode"])
        self.assertIn("footless-decode", refused, "no group's step was refused: the test has no teeth")
        self.assertLess(len(refused), 10)
        self.assertEqual(got, want)

    def test_a_short_prompt_is_not_held_behind_a_long_one(self):
        """Prefill goes to the request with the fewest prompt tokens left: a
        short prompt arriving while a long one is read is read first and
        starts answering before the long one is through, with or without
        concurrent_prefill; one passed over for PREFILL_AGING goes first (at
        zero they take turns)."""
        self.patch("CHUNK", 8)
        self.patch("SHARED_CHUNK", 8)
        self.patch("OVERLAP_CHUNK", 8)
        fed = []
        step = self.rt.step

        def recording(batch, ctx):
            fed.extend(chr(item.input_tokens[0]) for item in batch if item.input_tokens)
            return step(batch, ctx)

        self.rt.step = recording

        def run(overlap):
            if overlap and "concurrent_prefill" not in self.engine.facts.capabilities:
                self.engine.facts.capabilities.append("concurrent_prefill")
            self.engine.cache.free_all()            # the prompts are read again, not hit
            order = []
            batcher = Batcher(self.engine)
            long = batcher.add(self.seq("L" * 120, 2, on_token=lambda t, tm: order.append("long")))
            batcher.step()
            fed.clear()
            short = batcher.add(self.seq("S" * 40, 2, on_token=lambda t, tm: order.append("short")))
            self.drain(batcher)
            for task in (long, short):
                self.assertIsNone(task.error)
            return order, list(fed)

        for overlap in (False, True):
            order, chunks = run(overlap)
            self.assertEqual(order[0], "short", (overlap, order))
            self.assertEqual(chunks[:5], ["S"] * 5, "the short prompt's five chunks first")
        self.patch("PREFILL_AGING", 0.0)            # always starved: they take turns
        order, chunks = run(True)
        self.assertEqual(order[0], "short")
        self.assertIn("L", chunks[:5], "turns: the long prompt between the short one's chunks")

    def test_short_prompts_ride_along_in_one_prefill_step(self):
        """Prompts of at most PACK_TOKENS that fit whole join the chosen
        prefill's step; a longer one, or one cut in chunks, waits its turn."""
        fed = []
        step = self.rt.step

        def recording(batch, ctx):
            fed.append(sorted(len(item.input_tokens) for item in batch if item.input_tokens))
            return step(batch, ctx)

        self.rt.step = recording
        self.patch("PACK_TOKENS", 12)
        batcher = Batcher(self.engine)
        tasks = [batcher.add(self.seq(text)) for text in ("a" * 30, "b" * 10, "c" * 12, "d" * 5)]
        self.drain(batcher)
        self.assertEqual(fed[:2], [[5, 10, 12], [30]], "the shortest, the short ones beside it, then 30")
        self.assertEqual([t.result.text for t in tasks], [self.SCRIPT_TEXT] * 4)

    def test_requests_started_together_read_a_shared_head_once(self):
        """Two requests arriving together whose prompts share a long head:
        the second waits for the first's state at the shared point (a copy
        joins the prefix cache there) and reads only its own tail; both
        answer as alone."""
        self.patch("SHARE_MIN", 16)
        self.patch("CHUNK", 32)
        self.patch("SHARED_CHUNK", 32)
        head = "h" * 100
        batcher = Batcher(self.engine)
        a = batcher.add(self.seq(head + "first tail"))
        b = batcher.add(self.seq(head + "second, longer tail"))
        self.drain(batcher)
        self.assertEqual(a.result.text, self.SCRIPT_TEXT)
        self.assertEqual(b.result.text, self.SCRIPT_TEXT)
        self.assertEqual(a.result.prompt_tokens_fed, 110)
        self.assertEqual(b.result.prompt_tokens_fed, len("second, longer tail"))
        self.assertEqual(self.engine._producers, [])

    def test_a_shared_head_is_read_once_with_disk_caches(self):
        """With the disk caches on (a request's first lookups wait on the
        disk), requests arriving together still read their shared head once:
        the first to look registers as its reader before its lookups, and the
        others wait for its state there instead of each reading the head."""
        import tempfile
        self.patch("SHARE_MIN", 16)
        self.patch("CHUNK", 32)
        self.patch("SHARED_CHUNK", 32)
        self.patch("OVERLAP_CHUNK", 32)
        dirs = [tempfile.TemporaryDirectory() for _ in range(2)]
        for d in dirs:
            self.addCleanup(d.cleanup)
        engine = Engine.open(MODEL, prompt_cache=(dirs[0].name, 1 << 20), prefix_cache=(dirs[1].name, 1 << 20))
        self.addCleanup(engine.close)
        head = "h" * 100
        # an earlier conversation on disk sharing the prompts' first tokens
        # (as every chat prompt shares its template's): their lookups read it
        engine.generate_tokens(engine.runtime.encode("hhhh, an earlier conversation"), 4)
        engine.cache.files.flush()
        engine.cache.free_all()
        batcher = Batcher(engine)
        tails = ("first tail", "second, longer tail", "a third")
        tasks = [batcher.add(engine.sequence(engine.runtime.encode(head + t), 64)) for t in tails]
        while batcher.busy():
            batcher.step()
        for task in tasks:
            self.assertEqual(task.result.text, self.SCRIPT_TEXT)
        fed = sorted(t.result.prompt_tokens_fed for t in tasks)
        self.assertEqual(fed, sorted([len(head) + len(tails[0])] + [len(t) for t in tails[1:]]),
                         "the head was read more than once")
        self.assertEqual(engine._producers, [])

    def test_a_shared_head_cut_short_lets_the_waiting_request_go_on(self):
        """The request a second one waits for ends before the shared point
        (cancelled): the second reads its whole prompt itself."""
        self.patch("SHARE_MIN", 16)
        self.patch("CHUNK", 8)
        self.patch("SHARED_CHUNK", 8)
        flag = _Flag()
        head = "h" * 100
        batcher = Batcher(self.engine)
        a = batcher.add(self.seq(head + "first", ctx=flag))
        b = batcher.add(self.seq(head + "second"))
        batcher.step()
        batcher.step()
        flag.on = True
        self.drain(batcher)
        self.assertIsNone(b.error)
        self.assertEqual(b.result.text, self.SCRIPT_TEXT)
        self.assertEqual(self.engine._producers, [])

    def test_a_cancelled_request_ends_without_the_others(self):
        flag = _Flag()
        batcher = Batcher(self.engine)
        gone = batcher.add(self.seq("leaving", ctx=flag))
        stays = batcher.add(self.seq("staying"))
        for _ in range(4):
            batcher.step()
        flag.on = True
        self.drain(batcher)
        self.assertLess(len(gone.result.tokens), len(self.SCRIPT_TEXT))
        self.assertEqual(stays.result.text, self.SCRIPT_TEXT)

    def test_a_refused_step_preempts_the_youngest_request(self):
        """A step the budget refuses runs again without its youngest request;
        that one waits in the prefix cache and is admitted again later, and
        both answers come out whole."""
        self.patch("ADMIT_HEADROOM", 0)         # admit on the prompt alone
        self.patch("SHARED_CHUNK", 8)
        # 123 tokens of mock state: either request fits alone (the second
        # takes 122), and the two together run out while the second is still
        # reading its prompt. (Later, it would be preempted with output: the
        # engine then re-feeds that output after the prompt, and this mock,
        # whose script restarts at every prompt, would start its script over.)
        engine = Engine.open(MODEL, budget=8 * 123)
        self.addCleanup(engine.close)
        logs = []
        engine.services.log = lambda level, message: logs.append(message)
        batcher = Batcher(engine)
        first = batcher.add(engine.sequence(engine.runtime.encode("x" * 10), 64))
        batcher.step()
        second = batcher.add(engine.sequence(engine.runtime.encode("y" * 100), 64))
        self.drain(batcher)
        self.assertIsNone(second.error)
        self.assertEqual(first.result.text, self.SCRIPT_TEXT)
        self.assertEqual(second.result.text, self.SCRIPT_TEXT)
        self.assertTrue(any("youngest" in m for m in logs), logs)

    def test_alone_a_refusal_is_the_request_s_error(self):
        engine = Engine.open(MODEL, budget=8 * 12)
        self.addCleanup(engine.close)
        with self.assertRaises(BudgetRefused):
            engine.generate("x" * 10, 64)
        self.assertEqual(engine.services.accounting.used, 0)
