"""The engine: policy and bookkeeping only. It knows nothing about models."""

from __future__ import annotations

import array
import ctypes
import errno
import hashlib
import importlib.util
import json
import os
import random
import sys
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field, replace
from pathlib import Path

from .manifest import Manifest, read as read_manifest, verify, verify_backends
from .sdk import (
    CONCURRENT_PREFILL,
    LABEL_LOGPROBS,
    TOOL_CALLS,
    VISION,
    ImageSpan,
    RUNTIME_METHODS,
    BudgetRefused,
    Claim,
    ContractError,
    Facts,
    Runtime,
    SamplingSpec,
    State,
    StepItem,
    StepResult,
)

CHUNK = 4096  # prefill is fed in chunks; only the last one samples
# A prefill that shares its steps with other requests' decodes goes in chunks of
# this many tokens instead: each of their tokens waits for one chunk.
SHARED_CHUNK = 1024
# Beside other requests, a package that runs prefill steps beside decode steps
# (`concurrent_prefill`) takes chunks of this many: the decodes do not wait for
# them, but a request arriving meanwhile waits for the chunk in flight before
# its own prefill starts. SHARED_CHUNK again when another prefill is waiting.
OVERLAP_CHUNK = 2048
# The prefill a step runs is the one with the fewest prompt tokens left (a
# short request is not held behind a long prompt's every chunk), unless one has
# waited this long since its last chunk ran: the oldest such goes first. A
# safety net, not a time share: two long prompts at once are best served one
# after the other (the first is through at half the time, the second as late
# either way), and max_batch bounds how many short ones can keep a long one
# waiting.
PREFILL_AGING = 20.0
# Prompts this short ride along in the step of the prefill chosen (one step,
# their whole prompt): a forward costs its weights again, and a ride-along
# costs the chosen one its rows. Measured (two prompts of n tokens, one step
# against two): 32 -> 220 / 350 ms, 128 -> 446 / 569, 256 -> 760 / 864 -- the
# first one later by ~1.2 ms a token, the second sooner by ~120 ms: even at
# ~100-130 tokens, so only short ones join (two long prompts together would
# only make the first one later).
PACK_TOKENS = 128
# A request starting while another one reads a prompt that shares at least
# this many more tokens with its own than any cached state does waits for
# that one's state at the shared point instead of reading those tokens again
# (agents started together on one system prompt and tools).
SHARE_MIN = 256
DEFAULT_BUDGET = 1 << 34
DEFAULT_MAX_TOKENS = 16384
# The room a request must find to be admitted beside others (Batcher): its
# context's cache and this many tokens' worth more, for its state's fixed part
# and its growth. Alone, a request is always admitted.
ADMIT_HEADROOM = 8192
# Saved states in flight, bounded by BYTES and not by job count: one blob can
# be gigabytes. A save whose state, or whose runtime's export transient (the
# export copies the state into its blob), cannot fit the memory left is not
# saved: a cache never risks the machine it is speeding up. The queue's cap
# is what waits in RAM to be written; the transient check is what a save may
# briefly need beside its state.
SAVE_QUEUE_BYTES = 2 << 30
SAVE_QUEUE_MAX = 8  # queued saves kept; over that the oldest queued is dropped
SAVE_TRANSIENT = 3  # a save's peak, in multiples of the state's own bytes

# The repetition guard: how long a repeating tail may run before it is a loop.
# A model that collapses repeats one span of tokens verbatim until the budget
# runs out -- 16k tokens of "Let me structure the response..." -- and nothing
# in the model's own stop rules ever fires, because the loop emits no stop
# token. The threshold counts the REPEATED part, the copies past the first:
# counting the whole run instead hid a short cycle inside a long answer (a
# few tokens of code repeated) behind a hundred copies of it, and counting
# nothing would fire on style -- three identical lines, a refrain.
LOOP_PERIOD_MAX = 256  # longer spans than this repeating are not covered
LOOP_MIN_REPEATS = 3  # copies of the span that end the output
# Tokens in the copies past the first, at least. 24 cut valid code: a map row
# `[1,1,1,...` at 33 tokens, a zeros array, a table's identical rows -- a
# truncated file the model then took for a broken write. A runaway never ends,
# so a high bar only costs it ~500 more tokens, and a matrix of real runaways
# still fired at it.
LOOP_MIN_REPEATED_TOKENS = 512
LOOP_NOISY_PERIOD_MIN = 16  # below this the exact rule is the cheaper answer
# copies may differ by at most a third of their tokens and still be the same
# span: a runaway that re-names its variables between copies is still a
# runaway, and exact-only matching let "the same code block with different
# identifiers" run to the token cap all afternoon. Measured against the case
# that motivated this: 3 varying identifiers in a 20-token block (15%) -- an
# eighth (12.5%) missed it.
LOOP_MISMATCH_DEN = 3
# the fuzzy match is looser, so it demands one MORE copy than the exact rule:
# laxer evidence, stricter quorum. A real runaway hands over the 4th copy
# within one span anyway; a refrain of three does not.
LOOP_NOISY_MIN_REPEATS = 4


# What the repetition guard runs (`generate_tokens(loop_guard=...)`): nothing,
# the exact rule alone (the default), or all three passes. The exact rule cut
# every verbatim runaway of a 54-generation sampling matrix and nothing in 40k
# tokens of real prose and code; the fuzzy passes are the ones that cut healthy
# structured answers.
LOOP_GUARD_MODES = ("off", "exact", "full")


def loop_cut(tokens: list[int]) -> int | None:
    """Where a repeating tail says to keep the output, or None when it flows.

    Returns the length to keep -- the prefix plus ONE copy of the repeated
    span -- when the output ends with the same span of 1..LOOP_PERIOD_MAX
    tokens repeated LOOP_MIN_REPEATS times or more, those repeats past the
    first holding LOOP_MIN_REPEATED_TOKENS tokens or more. The repeats must
    reach the end of the output: a mid-text refrain is style, a runaway tail is
    the bug this exists for. Copies may be EXACT (the fast rule, 3 copies) or
    differ by up to a third of their tokens (the noisy rule, 4 copies,
    adjacent copy compared to adjacent copy so drift cannot accumulate past
    the bar).
    """
    return loop_cut_exact(tokens) or _noisy_loop_cut(tokens) or _rewrite_cut(tokens)


def loop_cut_exact(tokens: list[int]) -> int | None:
    """`loop_cut`'s first pass alone: the tail is the same span, verbatim,
    LOOP_MIN_REPEATS times or more."""
    n = len(tokens)
    for period in range(1, min(LOOP_PERIOD_MAX, n // LOOP_MIN_REPEATS) + 1):
        if tokens[n - 1] != tokens[n - 1 - period]:
            continue  # the tail cannot be copies of this period
        copies, at = 1, n - period
        while at - period >= 0 and tokens[at - period : at] == tokens[at : at + period]:
            copies += 1
            at -= period
        if copies >= LOOP_MIN_REPEATS \
                and (copies - 1) * period >= LOOP_MIN_REPEATED_TOKENS:
            return n - (copies - 1) * period
    return None


def _noisy_loop_cut(tokens: list[int]) -> int | None:
    """loop_cut's second pass: the same span with small variations between
    copies. Adjacent copies are compared to adjacent copies (not all against
    the last), so a drifting loop is caught at its beginning rather than only
    at the moment the drift happens to fit the bar. The mismatch bar aborts
    early, so ordinary prose costs a few compares per period and dies."""
    n = len(tokens)
    for period in range(LOOP_NOISY_PERIOD_MIN,
                        min(LOOP_PERIOD_MAX, n // LOOP_MIN_REPEATS) + 1):
        allow = max(period // LOOP_MISMATCH_DEN, 1)
        copies, at = 1, n - period
        while at - period >= 0:
            bad = 0
            for i in range(period):
                if tokens[at - period + i] != tokens[at + i]:
                    bad += 1
                    if bad > allow:
                        break
            if bad > allow:
                break
            copies += 1
            at -= period
        if copies >= LOOP_NOISY_MIN_REPEATS \
                and (copies - 1) * period >= LOOP_MIN_REPEATED_TOKENS:
            return n - (copies - 1) * period
    return None


# The rewrite rule's reach: the loop the two rules above cannot see. A model
# that REWRITES a long block (same implementation, renamed variables) produces
# periods of hundreds of tokens with only TWO copies before the budget dies
# mid-expression -- measured on the case that motivated this: key code lines
# appearing twice at 1200 tokens (~600-token period), no guard message, output
# cut off inside a variable name.
LOOP_REWRITE_PERIOD_MIN = 128
LOOP_REWRITE_PERIOD_MAX = 2048


def _rewrite_cut(tokens: list[int]) -> int | None:
    """loop_cut's third pass: the last two long blocks are the same block.

    Tail-anchored like the others: two near-duplicate spans of 128..2048 tokens
    closing the output mean the second one is a rewrite, and the output keeps
    the first. Runs every 8 tokens and pre-samples 16 positions per period --
    the per-token path must not pay for 2000 full span compares, and a runaway
    hands over another copy every period, so the delay is one span's worth of
    noise. The bar is the noisy rule's looser cousin: a quarter of the tokens
    may differ (renamed identifiers), and the pre-sample is only a cheap
    rejection of obviously-different spans."""
    n = len(tokens)
    if n % 8:
        return None
    for period in range(LOOP_REWRITE_PERIOD_MIN,
                       min(LOOP_REWRITE_PERIOD_MAX, n // 2) + 1):
        allow = period // 4
        bad = 0
        for s in range(16):  # cheap pre-sample: reject obvious non-matches
            i = s * period // 16
            if tokens[n - period + i] != tokens[n - 2 * period + i]:
                bad += 1
        if bad > 6:
            continue
        bad = 0
        for i in range(period):
            if tokens[n - period + i] != tokens[n - 2 * period + i]:
                bad += 1
                if bad > allow:
                    break
        if bad <= allow:
            return n - period
    return None


def supersedes(source, matched: int):
    """The prefix-cache entry a request's state makes redundant: the one it
    started from, when it read past that entry's prompt (PrefixCache)."""
    if source is None:
        return None
    key, prompt = source
    return key if matched >= prompt else None


def request_spec(sampling: SamplingSpec | None) -> SamplingSpec:
    """The request's sampling spec — one per-request draw stream.

    A `seed` identifies one per-request draw stream, advanced once per
    sampled token, so the spec is copied per request even when the caller
    pins every field: a shared spec object must not carry one request's
    stream into the next. An unpinned seed (None, the default) is drawn
    here from fresh entropy — the engine owns randomness, the model
    executes the spec where its logits live.
    """
    base = sampling if sampling is not None else SamplingSpec()
    seed = base.seed
    if seed is None:
        seed = random.SystemRandom().randrange(1 << 62)
    return replace(base, seed=seed)


class Ledger:
    """Byte accounting: the budget is the engine's to enforce."""

    def __init__(self, budget: int, evict, lock=None) -> None:
        self.declared = budget     # the opener's ceiling
        self.budget = budget
        self.used = 0
        self._evict = evict
        # claims come from every thread that runs the model (a prefill step
        # beside decode steps, a cache worker freeing a state): the prefix
        # cache's lock, since a claim may evict one of its entries
        self._lock = lock if lock is not None else threading.RLock()

    def claim(self, nbytes: int) -> Claim:
        if nbytes < 0:
            raise ContractError(f"negative claim: {nbytes}")
        with self._lock:
            while self.used + nbytes > self.budget:
                if not self._evict():
                    raise BudgetRefused(
                        f"budget {self.budget}B cannot admit {nbytes}B "
                        f"({self.used}B used, nothing left to evict)"
                    )
            self.used += nbytes
        return Claim(nbytes)

    def release(self, claim: Claim) -> None:
        with self._lock:
            self.used -= claim.nbytes

    def available(self) -> int:
        with self._lock:
            return self.budget - self.used

    def set_budget(self, budget: int) -> None:
        """Re-base the budget on the device's truth. Optional on the
        Accounting protocol: a runtime whose device is smaller than the
        declared budget clamps it here, so states and cached clones are
        admitted against what the card actually has rather than meeting
        cuMemAlloc mid-generation. Never GROWS past the opener's declaration,
        and may move within it: a runtime answering memory pressure takes
        the states' room away (the next claims evict) and gives it back when
        the pressure passes."""
        with self._lock:
            self.budget = min(self.declared, max(budget, 0))


class HostBuffers:
    def __init__(self, ledger: Ledger) -> None:
        self._ledger = ledger

    def host(self, nbytes: int):
        from .sdk import Buffer

        return Buffer(bytearray(nbytes), self._ledger.claim(nbytes))

    def free(self, buffer) -> None:
        self._ledger.release(buffer.claim)


class StepContext:
    """The step context service, checked at step boundaries."""

    def cancelled(self) -> bool:
        return False


class EngineServices:
    def __init__(self, ledger: Ledger) -> None:
        self.accounting = ledger
        self.buffers = HostBuffers(ledger)
        self.log = self._log

    @staticmethod
    def _log(level: str, message: str) -> None:
        print(f"[{level}] {message}", file=sys.stderr)


def _plain(key) -> list:
    """A key as json: tuples (an image's `(digest, index)` placeholders) as lists."""
    return [_plain(x) for x in key] if isinstance(key, tuple) else key


def _canon(raw) -> tuple:
    """A key as read back: lists as tuples, so it hashes."""
    return tuple(_canon(x) for x in raw) if isinstance(raw, list) else raw


def _pack(key):
    """A key for the indexes: the token ids as little-endian uint32 bytes
    (four per token, against the ~36 a Python int costs in a tuple), so a
    long session's key is a megabyte and not nine. A key with an image's
    `(digest, index)` placeholders stays a tuple: its digests are part of the
    key and no 32-bit word holds one. Both forms hash and compare exactly,
    and both go through `_common`. Packing is idempotent: a packed key
    passes through (a bytes iterable would otherwise be read as one int per
    byte)."""
    if isinstance(key, bytes):
        return key
    try:
        return array.array("I", key).tobytes()
    except (TypeError, ValueError, OverflowError):
        return tuple(_canon(key))


def _ident(key) -> str:
    packed = _pack(key)
    if not isinstance(packed, bytes):
        packed = json.dumps(_plain(packed), separators=(",", ":")).encode()
    return hashlib.sha256(packed).hexdigest()


def _keylen(key) -> int:
    """A packed key's tokens (its bytes are four per token)."""
    return len(key) // 4 if isinstance(key, bytes) else len(key)


def _key_json(key) -> list:
    """A key as its `.key` file holds it: the token ids as a list (a packed
    key unpacks; a key with image placeholders keeps them nested)."""
    if isinstance(key, bytes):
        return array.array("I", key).tolist()
    return _plain(key)


_CMP = ctypes.CDLL(None).memcmp
_CMP.restype = ctypes.c_int
_CMP.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]


def _common(a, b) -> int:
    """How many leading tokens two packed keys share. Packed bytes go
    through `memcmp` in C -- the Python walk was the engine's slowest code at
    cache scale (a second a lookup over long shared heads) -- and keys with
    image placeholders walk."""
    if isinstance(a, bytes) and isinstance(b, bytes):
        aa = ctypes.cast(ctypes.c_char_p(a), ctypes.c_void_p).value
        bb = ctypes.cast(ctypes.c_char_p(b), ctypes.c_void_p).value
        lo, hi = 0, min(len(a), len(b)) // 4  # equal for lo words, unknown to hi
        step = 4
        while lo < hi and step < hi - lo:
            if _CMP(aa + 4 * lo, bb + 4 * lo, 4 * step) == 0:
                lo += step
                step *= 2
            else:
                break
        hi = min(lo + step, hi)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if _CMP(aa, bb, 4 * mid) == 0:
                lo = mid
            else:
                hi = mid - 1
        return lo
    a, b = tuple(a), tuple(b)
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


class Ticket:
    """Work outside the model in flight: `result()` waits for it.

    `settle` maps what the worker produced to what the caller asked for and
    runs at `result()` time on the CALLER's thread: a state is imported and
    budget-claimed where the runtime already lives, while the worker only
    moves and compresses bytes. A ticket made with `waiting=` shares that
    ticket's readiness: it ends when the work behind it does. One caller
    consumes a ticket (the engine thread), so settling runs once.
    """

    __slots__ = ("_event", "_raw", "_settle", "source")

    def __init__(self, settle=None, waiting: "Ticket | None" = None) -> None:
        self._event = waiting._event if waiting is not None else threading.Event()
        self._raw = None
        self._settle = settle
        self.source = None      # a prefix-cache lookup: (entry key, its prompt length) it came from

    def ready(self) -> bool:
        return self._event.is_set()

    def wait(self, timeout: float | None = None) -> bool:
        return self._event.wait(timeout)

    def result(self):
        self._event.wait()
        if self._settle is not None:
            settle, self._settle = self._settle, None
            self._raw = settle(self._raw)
        return self._raw

    def finish(self, raw=None) -> None:
        """Whoever does the work completes the ticket with its outcome."""
        self._raw = raw
        self._event.set()


def _settled(value=None) -> Ticket:
    """A ticket for work already done (or never begun)."""
    ticket = Ticket()
    ticket.finish(value)
    return ticket


class FileCache:
    """Saved states on disk: one directory, a byte cap of its own, LRU.

    The engine's second tier under `PrefixCache`, and the one that survives
    restarts (CONTRACT.md: `export_state` / `import_state` move state between
    tiers and across runs). A request's whole prompt keys the PROMPT cache --
    an exact hit starts past its prefill -- and any absorbed prefix keys the
    PREFIX cache. Blobs are opaque states; keys are token ids (an image's
    placeholders are its digest, as in `PrefixCache`).

    Each entry is `<sha256>.key` (its tokens) beside `<sha256>.state` (the
    blob, written last and renamed into place, so a crash never leaves half
    an entry). Order in `self._entries` is least-recently-used first: a hit
    moves its entry to the end, and `save` drops the front while the cache is
    over `max_bytes` -- or while the filesystem has no room for the write. A
    cache never fails a request and never grows past its limit; a state that
    cannot fit even an empty cache -- or beside which its own export would
    not fit the memory left -- is simply not saved.

    Disk work runs on ONE worker thread of its own, reads before saves (a
    request waits for a read; nobody waits for a save), so the engine thread
    only ever runs the runtime's own calls. `save(..., owned=True)` hands
    over a private clone for the worker to export and free on every path;
    `load_async` / `begin_match` return a `Ticket` the request can wait on.
    """

    def __init__(self, runtime, path, max_bytes: int, log=None, available=None) -> None:
        self._runtime = runtime
        self._log = log or (lambda level, message: None)
        self.available = available    # the ledger's free bytes: room for transients
        self._path = Path(path)
        self._max_bytes = max(0, int(max_bytes))
        self._entries: "OrderedDict[tuple, tuple[str, int]]" = OrderedDict()
        self._total = 0
        self._queued_bytes = 0       # raw bytes of the saves queued or running
        self._lock = threading.RLock()
        self._cond = threading.Condition(self._lock)
        self._reads: deque = deque()   # what a request waits for, taken first
        self._saves: deque = deque()
        self._pending: set[tuple] = set()  # keys queued but not yet written
        self._saving = 0                   # saves queued or running
        self._worker: threading.Thread | None = None
        self._stopped = False
        self._load_index()

    # -- lookup -------------------------------------------------------------------------

    def has(self, key) -> bool:
        with self._lock:
            return _pack(key) in self._entries

    def load(self, key) -> State | None:
        """The state saved under this whole key (an exact hit), or None."""
        return self.load_async(key).result()

    def match(self, key, min_len: int = 0) -> tuple[State, int] | None:
        """The saved state sharing the best prefix with `key` -- more than
        `min_len` tokens of it -- fitted to it, as (state, matched).

        A CLEAN entry wins first (one whose whole key matches, so the fit
        needs no `truncate_state` at all) and only then the longest: cutting
        an imported state back re-forwards everything it keeps -- its blob
        carries no snapshots to cut back to (CONTRACT.md) -- so the replay a
        partial hit costs is unbounded and a hit that cannot cost one wins
        at any length. Within a class the longest match wins. (The RAM tier
        keeps longest-match: its truncates are snap-bounded.)"""
        return self.begin_match(key, min_len).result()

    def load_async(self, key) -> Ticket:
        """Begin reading the state saved under this whole key (an exact
        hit); the ticket's `result()` is the state or None."""
        key = _pack(key)
        with self._lock:
            entry = self._entries.get(key)
        if entry is None:
            return _settled()
        return self._read(key, entry, None, lambda state: state)

    def begin_match(self, key, min_len: int = 0) -> Ticket:
        """`match` begun: the file read runs beside the model and the
        ticket's `result()` is (state, matched) or None."""
        key = _pack(key)
        with self._lock:
            picked = self._pick(key, min_len)
        if picked is None:
            return _settled()
        entry_key, entry, matched, clean = picked
        return self._read(entry_key, entry, None if clean else matched,
                          lambda state: (state, matched))

    def match_len(self, key) -> int:
        """How many tokens of `key` the best saved state matches (no read)."""
        key = _pack(key)
        with self._lock:
            picked = self._pick(key, 0)
        return picked[2] if picked is not None else 0

    def _pick(self, key, min_len: int):
        """(entry key, entry, matched, clean) for `match`'s ranking (its
        docstring is the rule), or None. Called under the lock."""
        best, best_len, best_clean = None, 0, False
        for entry_key in self._entries:
            common = _common(entry_key, key)
            if common <= min_len:
                continue
            clean = _keylen(entry_key) == common
            if best is None or (clean, common) > (best_clean, best_len):
                best, best_len, best_clean = entry_key, common, clean
        if best is None:
            return None
        return best, self._entries[best], best_len, best_clean

    def _read(self, key: tuple, entry, cut, wrap) -> Ticket:
        """The read begun: the worker reads and decompresses, and the
        ticket's `result()` imports the state on this thread and truncates
        it to `cut` (None: no truncate -- a clean hit needs none)."""
        ident = entry[0]

        def settle(raw):
            kind, payload = raw
            if kind == "gone":
                return None
            if kind != "ok":
                self._log("warning", f"saved state unreadable, dropped: {payload}")
                self._drop_if(key, ident)
                return None
            try:
                state = self._runtime.import_state(payload)
            except BudgetRefused:
                self._log("info", "a saved state did not fit the budget: not used")
                return None
            except Exception as exc:  # noqa: BLE001 - an unreadable entry is a miss
                self._log("warning", f"saved state unreadable, dropped: {exc}")
                self._drop_if(key, ident)
                return None
            if cut is not None:
                self._runtime.truncate_state(state, cut)
            self._touch(key)
            return wrap(state)

        ticket = Ticket(settle)
        with self._cond:
            if self._stopped:
                ticket.finish(("gone", None))
                return ticket
            self._reads.append((key, ident, ticket))
            self._cond.notify_all()
        self._ensure_worker()
        return ticket

    # -- save --------------------------------------------------------------------------

    def save(self, key, state: State, owned: bool = False) -> Ticket:
        """Keep `state` under `key`, the write run beside the model (the
        ticket completes when the entry lands or is dropped). A key already
        saved holds that same state by construction: its entry is only
        refreshed as used. With `owned` the state is this cache's private
        copy (`clone_state`d for it alone) -- the worker exports and frees
        it on every path: written, dropped, refused. Without it the state is
        the caller's and may be reused imminently, so it is exported HERE
        and now and only its bytes move."""
        key = _pack(key)
        ticket = Ticket()
        with self._lock:
            known = key in self._entries or key in self._pending
            if key in self._entries:
                self._touch(key)
        if known:
            if owned:
                self._runtime.free_state(state)
            ticket.finish(None)
            return ticket
        # the peak of a save is the state with the export's copies of it
        # beside it: where that would not fit, the state is not saved -- a
        # cache never risks the machine it is speeding up
        size = self._runtime.state_nbytes(state)
        if self.available is not None and SAVE_TRANSIENT * size > self.available():
            self._log("info", f"a state of {size} B is not saved: its export wants "
                              f"{SAVE_TRANSIENT * size} B beside it and "
                              f"{self.available()} B are free")
            if owned:
                self._runtime.free_state(state)
            ticket.finish(None)
            return ticket
        blob = None
        if not owned:
            try:
                blob = self._runtime.export_state(state)
                size = len(blob)
            except Exception as exc:  # noqa: BLE001 - a cache never fails a request
                self._log("warning", f"state not saved: {exc}")
                ticket.finish(None)
                return ticket
        job = (key, blob, state, owned, ticket, size)
        dropped = []
        with self._cond:
            closed = self._stopped
            if not closed:
                self._pending.add(key)
                self._saves.append(job)
                self._saving += 1
                self._queued_bytes += size
                while len(self._saves) > SAVE_QUEUE_MAX \
                        or self._queued_bytes > SAVE_QUEUE_BYTES:
                    drop = self._saves.popleft()
                    dropped.append(drop)
                    self._pending.discard(drop[0])
                    self._saving -= 1
                    self._queued_bytes -= drop[5]
                self._cond.notify_all()
        if closed:
            if owned:
                self._runtime.free_state(state)
            ticket.finish(None)
            return ticket
        for _, _, drop_state, drop_owned, drop_ticket, _ in dropped:
            self._log("info", f"save queue full ({SAVE_QUEUE_MAX} waiting or "
                              f"{SAVE_QUEUE_BYTES} B): the oldest queued state was dropped")
            if drop_owned:
                self._runtime.free_state(drop_state)
            drop_ticket.finish(None)
        self._ensure_worker()
        return ticket

    def flush(self) -> None:
        """Wait for every queued save to land (or be dropped)."""
        with self._cond:
            while self._saving:
                self._cond.wait()

    def close(self) -> None:
        """Flush, then stop the worker after its last job."""
        self.flush()
        with self._cond:
            self._stopped = True
            self._cond.notify_all()
            worker = self._worker
        if worker is not None:
            worker.join()

    # -- the worker ----------------------------------------------------------------------

    def _ensure_worker(self) -> None:
        with self._lock:
            if self._worker is None and not self._stopped:
                self._worker = threading.Thread(
                    target=self._work, name="footless-file-cache", daemon=True)
                self._worker.start()

    def _work(self) -> None:
        """The one thread that touches the disk: reads first (a request
        waits for one), then saves, until `close` leaves nothing."""
        while True:
            with self._cond:
                job, read = None, False
                while job is None:
                    if self._reads:
                        job, read = self._reads.popleft(), True
                    elif self._saves:
                        job = self._saves.popleft()
                    elif self._stopped:
                        return
                    else:
                        self._cond.wait()
            try:
                if read:
                    self._run_read(*job)
                else:
                    self._run_save(*job)
            except Exception as exc:  # noqa: BLE001 - one job never kills the worker
                self._log("warning", f"cache worker: {type(exc).__name__}: {exc}")
            finally:
                with self._cond:
                    if not read:
                        self._saving -= 1
                        self._queued_bytes -= job[5]
                        self._pending.discard(job[0])
                    self._cond.notify_all()

    def _run_read(self, key: tuple, ident: str, ticket: Ticket) -> None:
        try:
            raw = ("ok", (self._path / f"{ident}.state").read_bytes())
        except Exception as exc:  # noqa: BLE001 - unreadable: the settle drops it
            raw = ("bad", exc)
        ticket.finish(raw)

    def _run_save(self, key: tuple, blob, state, owned: bool,
                  ticket: Ticket, size: int) -> None:
        """One queued save: export (an owned state), make room, write. An
        owned state is freed whatever happens."""
        try:
            if owned:
                try:
                    blob = self._runtime.export_state(state)
                except Exception as exc:  # noqa: BLE001 - a cache never fails a request
                    self._log("warning", f"state not saved: {exc}")
                    return
            size = len(blob)
            if size > self._max_bytes:
                self._log("info", f"a state of {size} B does not fit the cache's "
                                  f"{self._max_bytes} B: not saved")
                return
            ident = _ident(key)
            with self._lock:
                while self._total + size > self._max_bytes:
                    if not self._evict_lru():
                        break
            if self._write(ident, key, blob):
                with self._lock:
                    self._entries[key] = (ident, size)
                    self._total += size
        finally:
            if owned:
                self._runtime.free_state(state)
            ticket.finish(None)

    def _write(self, ident: str, key, blob: bytes) -> bool:
        """The key and then the blob, the blob renamed into place. A full
        disk evicts LRU until the write fits; when even an empty cache cannot
        hold it, the state is dropped with the reason logged."""
        kpath = self._path / f"{ident}.key"
        spath = self._path / f"{ident}.state"
        tmp = self._path / f"{ident}.state.tmp"
        while True:
            try:
                kpath.write_text(json.dumps(_key_json(key)))
                tmp.write_bytes(blob)
                os.replace(tmp, spath)
                return True
            except OSError as exc:
                tmp.unlink(missing_ok=True)
                if exc.errno != errno.ENOSPC or not self._evict_lru():
                    kpath.unlink(missing_ok=True)
                    self._log("warning", f"state not saved: {exc}")
                    return False

    # -- the index ---------------------------------------------------------------------

    def _drop_if(self, key: tuple, ident: str) -> None:
        """Drop `key`'s entry only if it is still the one that failed: a
        newer save of the same key must not go with an old read's error."""
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None and entry[0] == ident:
                self._drop(key)

    def _load_index(self) -> None:
        """The directory as a previous run left it, oldest first (LRU order
        is the files' age; every hit touches its file)."""
        self._path.mkdir(parents=True, exist_ok=True)
        found = []
        for kpath in self._path.glob("*.key"):
            spath = kpath.with_suffix(".state")
            try:
                found.append((kpath.stat().st_mtime, kpath.stem,
                              _pack(_canon(json.loads(kpath.read_text()))),
                              spath.stat().st_size))
            except (OSError, ValueError):
                continue
        for _, ident, key, size in sorted(found):
            self._entries[key] = (ident, size)
            self._total += size
        while self._total > self._max_bytes and self._evict_lru():
            pass

    def _touch(self, key: tuple) -> None:
        with self._lock:
            self._entries.move_to_end(key)
            ident = self._entries[key][0]
        try:
            os.utime(self._path / f"{ident}.state")
        except OSError:
            pass

    def _drop(self, key: tuple) -> None:
        with self._lock:
            entry = self._entries.pop(key, None)
            if entry is not None:
                self._total -= entry[1]
        if entry is not None:
            self._unlink(entry[0])

    def _evict_lru(self) -> bool:
        with self._lock:
            if not self._entries:
                return False
            _, (ident, size) = self._entries.popitem(last=False)
            self._total -= size
        self._unlink(ident)
        return True

    def _unlink(self, ident: str) -> None:
        for p in (self._path / f"{ident}.key", self._path / f"{ident}.state"):
            try:
                p.unlink()
            except OSError:
                pass


class PrefixCache:
    """Prefix reuse: which cached state to keep, key it, evict it.

    `files` (a `FileCache`) is the second tier: states saved on disk, which
    survive a restart and a budget eviction. Memory wins a tie, a longer
    match from the files wins outright.

    An entry remembers how much of its key was prompt. A request that
    started from an entry and read past that entry's prompt (the next turn
    of its conversation, which re-renders its answer, or the same prompt
    again) supersedes it: when it stores its own state the entry goes --
    that state holds everything of it a request can still ask for, and the
    entry's answer is the part its client did not keep. A request that
    parted inside the entry's prompt (another conversation sharing its
    system prompt) leaves it alone.
    """

    def __init__(self, runtime: Runtime, available=None,
                 files: "FileCache | None" = None) -> None:
        self._runtime = runtime
        self._entries: "OrderedDict[tuple[int, ...], State]" = OrderedDict()
        self._prompts: dict = {}     # an entry's key -> how much of it was prompt
        # the ledger's too (Engine.open): an eviction runs inside a claim, from
        # whichever thread claims. Held for the entries' bookkeeping (and the
        # frees an eviction does), never across a copy or a truncate
        self.lock = threading.RLock()
        # the ledger's free bytes (set by Engine.open): a copy that would not
        # fit beside the entry hands the entry itself over instead
        self.available = available
        self.files = files

    def lookup(self, tokens: list[int]) -> tuple[State, int]:
        """`begin_lookup` run to its result (synchronous callers)."""
        return self.begin_lookup(tokens).result()

    def begin_lookup(self, tokens: list[int]) -> Ticket:
        """This cache's answer for `tokens` begun -- (state, matched). The
        disk tier's read runs beside the model; the RAM side is taken when
        the ticket's `result()` is called."""
        key = _pack(tokens)
        best = None
        best_len = 0
        with self.lock:
            entries = list(self._entries)
        for entry in entries:
            common = _common(entry, key)
            # a partial match truncates the clone, and a truncate may queue the
            # kept ids for re-forwarding in the next step (CONTRACT.md,
            # `truncate_state`) -- invisible here, which is why the rates book
            # `StepResult.forwarded_tokens` instead of what the request fed.
            if common > best_len:
                best, best_len = entry, common

        def settle(files_hit):
            # a saved state may match longer than anything in memory -- the
            # machine restarted, or the budget evicted the entry
            if files_hit is not None:
                return files_hit
            if best is None:
                return self._runtime.new_state(), 0
            # out of the cache while it is copied: making room for the copy may
            # evict entries, and never this one. It may even be gone by now
            # (another request's step evicted it while the disk was read).
            with self.lock:
                entry = self._entries.pop(best, None)
                prompt = self._prompts.pop(best, _keylen(best))
            if entry is None:
                return self._runtime.new_state(), 0
            ticket.source = (best, prompt)
            fits = self.available is None or \
                self._runtime.state_nbytes(entry) <= self.available()
            state = None
            if fits:
                try:
                    state = self._runtime.clone_state(entry)
                except BudgetRefused:
                    state = None
            if state is None:
                # no room for a copy (a long conversation's state is most of
                # the card): the request takes the entry itself and continues it --
                # a fresh state would re-forward the whole prefix
                self._runtime.truncate_state(entry, best_len)
                return entry, best_len
            with self.lock:
                self._entries[best] = entry
                self._prompts[best] = prompt
            self._runtime.truncate_state(state, best_len)
            return state, best_len

        if self.files is not None and best_len < _keylen(key):
            inner = self.files.begin_match(key, best_len)
            ticket = Ticket(lambda _raw: settle(inner.result()), waiting=inner)
            return ticket
        ticket = Ticket(lambda _raw: settle(None))
        ticket.finish(None)
        return ticket

    def match_len(self, tokens) -> int:
        """How many tokens of `tokens` the best cached state (RAM or disk)
        would start with, without taking it."""
        key = _pack(tokens)
        with self.lock:
            entries = list(self._entries)
        best = max((_common(entry, key) for entry in entries), default=0)
        if self.files is not None:
            best = max(best, self.files.match_len(key))
        return best

    def nbytes(self) -> int:
        """What the entries hold: room an eviction can give back."""
        with self.lock:
            return sum(self._runtime.state_nbytes(state) for state in self._entries.values())

    def store(self, tokens: list[int], state: State, prompt: int | None = None,
              supersedes=None) -> None:
        """Keep `state` under `tokens`, the first `prompt` of them prompt
        (all, unsaid); `supersedes`: an entry key this state makes redundant
        (the class docstring), dropped."""
        key = _pack(tokens)
        if self.files is not None:
            # saved before it joins the entries: once there, a claim on
            # another thread may evict (free) it
            self.files.save(key, state)
        with self.lock:
            old = self._entries.pop(key, None)
            if old is not None:
                self._runtime.free_state(old)
            self._entries[key] = state
            self._prompts[key] = _keylen(key) if prompt is None else min(prompt, _keylen(key))
            if supersedes is not None and supersedes != key:
                gone = self._entries.pop(supersedes, None)
                self._prompts.pop(supersedes, None)
                if gone is not None:
                    self._runtime.free_state(gone)

    def evict_lru(self) -> bool:
        with self.lock:
            if not self._entries:
                return False
            key, state = self._entries.popitem(last=False)
            self._prompts.pop(key, None)
            self._runtime.free_state(state)
            return True

    def free_all(self) -> None:
        while self.evict_lru():
            pass


@dataclass
class Rates:
    """One phase's per-step speeds, in tokens per second.

    A prefill sample is one chunk, a decode sample is one step — for the
    models here, one token. A step is a sample only if it RAN: one that
    produced nothing did not run, and neither did one that returned a token it
    had already computed (`StepResult.forwarded`). Both are real tokens and
    both are real wall time — they are in the phase's token and second totals,
    which is what the `speed t/s` column reports — but neither is a sample of
    how fast a forward runs, and a microsecond sample drags the mean and the
    max three orders of magnitude off.
    """

    values: list[float] = field(default_factory=list)

    def add(self, tokens: int, seconds: float) -> None:
        if tokens > 0 and seconds > 0:
            self.values.append(tokens / seconds)

    @property
    def minimum(self) -> float | None:
        return min(self.values) if self.values else None

    @property
    def mean(self) -> float | None:
        return sum(self.values) / len(self.values) if self.values else None

    @property
    def maximum(self) -> float | None:
        return max(self.values) if self.values else None


@dataclass
class Timing:
    """Per-phase progress, filled live while a generation runs."""

    prefill_tokens: int = 0
    prefill_seconds: float = 0.0
    decode_tokens: int = 0
    decode_seconds: float = 0.0
    # How long the answer took to START, and how long the whole request took.
    # `first_token_seconds` is set when the first token joins the output, so it
    # is a prefix of `total_seconds` and covers the prefill plus one step: the
    # two numbers a reader actually feels, and the reason a chat that answers
    # in a sentence can be slower than one that answers in a page.
    first_token_seconds: float = 0.0
    total_seconds: float = 0.0
    prefill_rates: Rates = field(default_factory=Rates)
    decode_rates: Rates = field(default_factory=Rates)


class _Producer:
    """A request reading its prompt right now (`Engine.sequence`): its key,
    how far it has read, and the points other requests wait for -- each a
    ticket finished once the state there is in the prefix cache (or the
    request will not get there)."""

    __slots__ = ("key", "pos", "cuts")

    def __init__(self, key, pos: int) -> None:
        self.key, self.pos, self.cuts = key, pos, {}


@dataclass
class Prompt:
    """A conversation as the model reads it: tokens, and where its images sit."""

    tokens: list[int]
    images: list[ImageSpan]


def _images_in(message: dict) -> bool:
    content = message.get("content")
    return isinstance(content, list) and any(
        isinstance(p, dict) and p.get("type") == "image" for p in content)


@dataclass
class GenerationResult:
    text: str
    tokens: list[int]
    prompt_tokens_fed: int  # prompt tokens the model ran, a truncate's replay included
    finished: bool
    timing: Timing = field(default_factory=Timing)
    # Did the repetition guard cut the output? `tokens` then holds the prefix
    # plus one copy of the repeated span, while `timing` counts everything the
    # model actually generated before the cut.
    looped: bool = False
    # What the model ABSORBED this request, cut or not: the tail the state
    # holds and the cache is keyed on. `tokens` above is what the output shows
    # (the guard's trimmed view); a caller that extends the conversation as a
    # token stream -- the chat loop, so the prefix cache stays an exact hit --
    # must extend it with THESE, or every turn after a cut silently re-runs the
    # whole history (the key keeps the trimmed tail, the stream would not).
    absorbed_tokens: list[int] = field(default_factory=list)


class Engine:
    def __init__(
        self,
        model: Manifest,
        runtime: Runtime,
        facts: Facts,
        services: EngineServices,
        cache: PrefixCache,
    ) -> None:
        self.model = model
        self.runtime = runtime
        self.facts = facts
        self.services = services
        self.cache = cache
        # set by a Batcher before it resumes its sequences: the prefill chunk a
        # sequence cuts next (`sequence`; CHUNK alone, smaller beside others)
        self._chunk = CHUNK
        self.vision_on = False   # images read: asked for by the operator (Engine.vision)
        # the saved-prompt tier (a `FileCache`), when the operator asked for
        # one: a whole prompt seen before starts past its prefill
        self.prompt_cache: FileCache | None = None
        self._producers: list[_Producer] = []   # requests reading their prompt now

    @classmethod
    def open(
        cls,
        model_dir: str | Path,
        backend: str | None = None,
        budget: int = DEFAULT_BUDGET,
        prompt_cache: tuple[str, int] | None = None,
        prefix_cache: tuple[str, int] | None = None,
    ) -> "Engine":
        model_dir = Path(model_dir)
        footless_dir = model_dir / "footless"
        if not footless_dir.is_dir():
            raise ContractError(f"not a model package: {model_dir} (no footless/ folder)")
        model = read_manifest(footless_dir)
        verify_backends(model, footless_dir)
        if backend is None:
            if len(model.backends) != 1:
                raise ContractError(f"pick a backend, manifest has: {model.backends}")
            backend = model.backends[0]
        if backend not in model.backends:
            raise ContractError(f"backend {backend!r} not in manifest: {model.backends}")
        runtime = load_runtime(footless_dir / backend / "runtime.py")
        cache = PrefixCache(runtime)
        services = EngineServices(Ledger(budget, cache.evict_lru, cache.lock))
        cache.available = services.accounting.available
        facts = runtime.open(str(model_dir), services)
        verify(model, facts)
        if prefix_cache is not None:
            cache.files = FileCache(runtime, *prefix_cache, log=services.log,
                                    available=services.accounting.available)
        engine = cls(model, runtime, facts, services, cache)
        if prompt_cache is not None:
            engine.prompt_cache = FileCache(runtime, *prompt_cache, log=services.log,
                                            available=services.accounting.available)
        return engine

    def generate(
        self, prompt: str, max_tokens: int = DEFAULT_MAX_TOKENS, on_token=None,
        sampling: SamplingSpec | None = None, loop_guard: bool | str = "exact",
    ) -> GenerationResult:
        return self.generate_tokens(
            self.runtime.encode(prompt), max_tokens, on_token, sampling,
            loop_guard=loop_guard,
        )

    def vision(self, enabled: bool = True) -> None:
        """Turn image input on, if this package reads images (capability
        `vision`). Like `speculative`, the operator asks for it; a package that
        does not declare the capability is refused with the reason, and one
        that declares it without its verbs is a reported contradiction. What
        reading images loads is the package's business (it may claim from
        `accounting`)."""
        if not enabled:
            return
        if VISION not in self.facts.capabilities:
            raise ContractError(
                "image input was asked for, but this package declares no vision "
                f"capability (it declares: {self.facts.capabilities or 'nothing'})")
        for name in ("set_vision", "chat_images"):
            if not callable(getattr(self.runtime, name, None)):
                raise ContractError(
                    f"this package declares the vision capability but its runtime exposes no {name} verb")
        self.runtime.set_vision(True)
        self.vision_on = True

    def speculative(self, enabled: bool = True) -> None:
        """Turn speculative decoding on, if this package offers it.

        The engine does not know how a package speculates, or whether it
        speculates at all — it asks, and the package's own `capabilities`
        answer (rule 4). A package that does not declare `speculative` is
        refused here, with the reason, rather than quietly generating
        without it.

        `set_speculative` is an OPTIONAL runtime verb (CONTRACT.md, Runtime):
        it is deliberately not in `RUNTIME_METHODS`, so a package that does not
        implement one is not broken by this engine, and a package that declares
        the capability without the verb is a contradiction we report rather
        than a silent no-op.
        """
        if not enabled:
            return
        if "speculative" not in self.facts.capabilities:
            raise ContractError(
                "speculative decoding was asked for, but this package declares "
                f"no speculative capability (it declares: "
                f"{self.facts.capabilities or 'nothing'})"
            )
        verb = getattr(self.runtime, "set_speculative", None)
        if not callable(verb):
            raise ContractError(
                "this package declares the speculative capability but its "
                "runtime exposes no set_speculative verb"
            )
        verb(True)

    def label_logprobs(self, input_tokens: list[int],
                       labels: list[str]) -> list[tuple[int, float]]:
        """Score `labels` at the answer position after `input_tokens`, if this
        package offers it.

        The engine does not know how a package resolves or scores a label, or
        whether it scores labels at all — it asks, and the package's own
        `capabilities` answer (rule 4). A package that does not declare
        `label_logprobs` is refused here, with the reason, rather than quietly
        scoring without it.

        `label_logprobs` is an OPTIONAL runtime verb (CONTRACT.md, Optional
        verbs): it is deliberately not in `RUNTIME_METHODS`, so a package that
        does not implement one is not broken by this engine, and a package that
        declares the capability without the verb is a contradiction we report
        rather than a silent no-op.

        A `ValueError` from the verb — a label that is not exactly one token
        at the answer position, labels colliding on one token id, prompt text
        that does not round-trip through decode/encode — propagates unchanged:
        the caller's input at fault, which the serving layer maps to a client
        error.
        """
        if LABEL_LOGPROBS not in self.facts.capabilities:
            raise ContractError(
                "label scoring was asked for, but this package declares no "
                "label_logprobs capability (it declares: "
                f"{self.facts.capabilities or 'nothing'})"
            )
        verb = getattr(self.runtime, "label_logprobs", None)
        if not callable(verb):
            raise ContractError(
                "this package declares the label_logprobs capability but its "
                "runtime exposes no label_logprobs verb"
            )
        return verb(input_tokens, labels)

    def label_logprobs_batch(self, items: list, ctx: StepContext | None = None,
                             shared: int = 0) -> list[list[tuple[int, float]]] | None:
        """`label_logprobs` for several (input_tokens, labels) pairs, in order.

        Through the package's optional `label_logprobs_batch` when it has one
        (it may share the work the prompts have in common), else one
        `label_logprobs` call per pair -- the same gate either way, and a
        cancelled `ctx` stops between pairs (None is returned). A `ValueError`
        carries `.item`, the index of the pair at fault.

        `shared` names a head of that many tokens the caller expects later
        requests to repeat (a decision's template and input text), clamped to
        what every pair really shares short of its last token. With the batch
        verb the head goes through the prefix cache: looked up, the rest of it
        forwarded by `step`, the pairs scored from that state, and the state
        kept as the head's entry -- a repeated head is not forwarded again.
        """
        if LABEL_LOGPROBS not in self.facts.capabilities:
            self.label_logprobs([], [])        # raises the capability's ContractError
        batch = getattr(self.runtime, "label_logprobs_batch", None)
        if callable(batch):
            if shared > 0 and items:
                first = items[0][0]
                shared = min(shared, *(len(t) - 1 for t, _ in items))
                for t, _ in items[1:]:
                    n = 0
                    while n < shared and t[n] == first[n]:
                        n += 1
                    shared = n
            if shared > 0 and items:
                return self._label_from_cache(batch, items, list(first[:shared]), ctx)
            return batch(items)
        out = []
        for k, (input_tokens, labels) in enumerate(items):
            if ctx is not None and ctx.cancelled():
                return None
            try:
                out.append(self.label_logprobs(input_tokens, labels))
            except ValueError as exc:
                exc.item = k
                raise
        return out

    def _label_from_cache(self, batch, items: list, head: list[int],
                          ctx: StepContext | None):
        """`label_logprobs_batch` from the cached state of `head`."""
        ctx = ctx if ctx is not None else StepContext()
        state, matched = self.cache.lookup(head)
        try:
            rest = head[matched:]
            for i in range(0, len(rest), CHUNK):
                self.runtime.step([StepItem(state, rest[i:i + CHUNK], None)], ctx)
                if ctx.cancelled():
                    self.runtime.free_state(state)
                    return None
            if not rest:
                # settles what a partial match queued (a replay), forwards nothing otherwise
                self.runtime.step([StepItem(state, [], None)], ctx)
            result = batch(items, state=state)
        except ValueError:
            self.cache.store(head, state)  # the caller's labels at fault; the head is good
            raise
        except BaseException:
            self.runtime.free_state(state)
            raise
        # the verb leaves the state as it found it: the head's entry
        self.cache.store(head, state)
        return result

    def chat_prompt(self, messages: list[dict], thinking: bool = True,
                    level: str | None = None, tools: list[dict] | None = None) -> "Prompt":
        """`chat_tokens` for a conversation that may carry images (content
        parts {"type": "image", "image": sdk.Image}): the tokens and where
        each image sits among them, rendered by the package (`chat_images`,
        behind `vision`, which must be on)."""
        if not any(_images_in(m) for m in messages):
            return Prompt(self.chat_tokens(messages, thinking, level, tools), [])
        if not self.vision_on:
            raise ContractError("the conversation carries images, but image input is not on "
                                "(the operator turns it on: --vision)")
        kwargs = {}
        if level is not None:
            kwargs["level"] = level
        if tools:
            self._tool_gate()
            kwargs["tools"] = tools
        tokens, spans = self.runtime.chat_images(messages, thinking, **kwargs)
        return Prompt(list(tokens), list(spans))

    def chat_tokens(self, messages: list[dict], thinking: bool = True,
                    level: str | None = None, tools: list[dict] | None = None) -> list[str]:
        """The conversation as prompt tokens, via the package's own template.

        `level` is one of the package's `thinking_levels`; None leaves the
        package its own default. `tools` are function tools in OpenAI's
        shape, for a package that declares `tool_calls`. Each is passed only
        when set, so a runtime written before the parameter keeps serving
        requests that do not use it.
        """
        if any(_images_in(m) for m in messages):
            raise ValueError("a conversation with images becomes a prompt through chat_prompt")
        kwargs = {}
        if level is not None:
            kwargs["level"] = level
        if tools:
            self._tool_gate()
            kwargs["tools"] = tools
        return self.runtime.chat(messages, thinking, **kwargs)

    def tool_calls(self, text: str, tools: list[dict],
                   final: bool) -> tuple[str, list[dict]]:
        """The answer `text` split into its content and the calls it makes.

        The OPTIONAL verb behind `tool_calls` (sdk.TOOL_CALLS): how a package
        writes a call is its template's business, so it reads them back too.
        """
        self._tool_gate()
        return self.runtime.tool_calls(text, tools, final)

    def _tool_gate(self) -> None:
        if TOOL_CALLS not in self.facts.capabilities:
            raise ContractError(
                "function tools were asked for, but this package declares no "
                f"tool_calls capability (it declares: {self.facts.capabilities or 'nothing'})")
        if not callable(getattr(self.runtime, "tool_calls", None)):
            raise ContractError(
                "this package declares the tool_calls capability but its "
                "runtime exposes no tool_calls verb")

    def decode(self, tokens: list[int]) -> str:
        return self.runtime.decode(tokens)

    def generate_tokens(
        self, tokens: list[int], max_tokens: int = DEFAULT_MAX_TOKENS,
        on_token=None, sampling: SamplingSpec | None = None,
        on_step=None, ctx: StepContext | None = None, loop_guard: bool | str = "exact",
        images: list[ImageSpan] = (),
    ) -> GenerationResult:
        """One request: prefill `tokens`, then decode up to `max_tokens` tokens.

        `on_token` is called with every token that joins the output. `on_step`
        is called after every step with that step's `StepResult` and the timing
        so far — including steps that sample nothing and stop tokens the
        engine's own rules keep out of the output — and returning `True` from it
        ends the generation: the caller's stop policy on top of the engine's.
        `ctx` is the step context, checked at step boundaries; a cancelled one
        ends the request at the next step.

        `loop_guard` picks the engine's repetition rule (LOOP_GUARD_MODES):
        output that ends in one span repeated past the threshold ends the
        request and is cut back to a single copy, reported as
        `GenerationResult.looped`. The default, "exact", acts on VERBATIM
        copies only (`loop_cut_exact`); "full" adds the fuzzy passes, which
        read tokens alone and cannot tell a runaway from a table of
        iterations -- a false positive there costs the rest of a healthy
        answer, so they run only when asked. "off" lets a collapse run to
        `max_tokens`. `True` / `False` mean "full" / "off".

        The request runs alone: `sequence` driven by a `Batcher` of its own.
        """
        return Batcher(self).run(self.sequence(
            tokens, max_tokens, on_token, sampling, on_step, ctx, loop_guard, images))

    def sequence(
        self, tokens: list[int], max_tokens: int = DEFAULT_MAX_TOKENS,
        on_token=None, sampling: SamplingSpec | None = None,
        on_step=None, ctx: StepContext | None = None, loop_guard: bool | str = "exact",
        images: list[ImageSpan] = (),
    ):
        """`generate_tokens` as a generator a `Batcher` drives beside others.

        It yields what it needs -- `Admit` first (room for a state), any
        `Wait` (a saved state loading) before its first `Want`, then one
        `Want` a step -- and is sent each step's `(StepResult, seconds)`; it
        returns the `GenerationResult`. A `Preempted` thrown in at a step
        boundary parks it: its state goes to the prefix cache, keyed on what
        it holds, and once admitted again it continues from there -- the
        prompt and the output so far, re-forwarding only what the cache lost.
        """
        mode = ("full" if loop_guard else "off") if isinstance(loop_guard, bool) \
            else loop_guard
        if mode not in LOOP_GUARD_MODES:
            raise ValueError(f"loop_guard {loop_guard!r}: one of {LOOP_GUARD_MODES}")
        rule = {"exact": loop_cut_exact, "full": loop_cut}.get(mode)
        if len(tokens) > self.facts.max_context:
            raise ContractError(
                f"prompt is {len(tokens)} tokens, max_context is {self.facts.max_context}"
            )
        images = list(images)
        if images and not self.vision_on:
            raise ContractError("the prompt carries images, but image input is not on "
                                "(the operator turns it on: --vision)")
        # the prefix cache's key: an image's tokens are its placeholders, the
        # same for every image -- each is keyed on the image's digest instead,
        # so a cached state is reused only for the same images
        keyed = list(tokens)
        for span in images:
            for j in range(span.length):
                keyed[span.start + j] = (span.image.digest, j)
        # a prompt whose cache alone cannot fit the whole budget is refused now,
        # not after the minutes of prefill it takes to reach the limit (the
        # state's fixed part is the model's and not counted: this is the floor)
        need = self.facts.cache_bytes_per_token * len(tokens)
        budget = getattr(self.services.accounting, "budget", None)
        if budget is not None and need > budget:
            raise BudgetRefused(
                f"prompt is {len(tokens)} tokens: its cache needs {need} B, more than "
                f"the whole budget of {budget} B")
        ctx = ctx if ctx is not None else StepContext()
        # what a request does not pin runs under the package's own defaults
        # (Facts.default_sampling -- its generation config), and only where a
        # package declares none under the spec's own (CONTRACT.md)
        spec = request_spec(sampling if sampling is not None
                            else self.facts.default_sampling)
        timing = Timing()
        t_start = time.perf_counter()   # the request, not any one step
        generated: list[int] = []
        finished = False
        cut: int | None = None  # where the repetition guard trimmed the output

        def absorb(step_result):
            nonlocal finished, cut
            before = len(generated)
            stop = self._absorb(step_result, generated)
            finished = step_result.finished or stop
            if len(generated) > before:
                if cut is None and rule is not None:
                    keep = rule(generated)
                    if keep is not None:
                        # the loop's tail is trimmed away at the end; what has
                        # already streamed cannot be recalled, and `timing`
                        # still counts everything the model really produced
                        cut, finished = keep, True
                if not timing.first_token_seconds:
                    # the first token to join the output, measured from the
                    # start of the request and nothing else
                    timing.first_token_seconds = time.perf_counter() - t_start
                if on_token is not None:
                    on_token(generated[-1], timing)
            if on_step is not None and on_step(step_result, timing):
                finished = True

        consumed = 0                    # prompt tokens the first admission forwarded
        replayed = 0                    # ...and the matched ones its truncate forwarded again
        first = True
        while True:
            # what the state must hold before the next sample: the prompt, then
            # (after a preemption) the output so far
            context = keyed + generated
            yield Admit(self.facts.cache_bytes_per_token * (len(context) + ADMIT_HEADROOM))
            state, matched = None, 0
            if not generated and self.prompt_cache is not None:
                # a whole prompt seen before: past its prefill already
                ticket = self.prompt_cache.load_async(keyed)
                while not ticket.ready():
                    yield Wait(ticket)
                state = ticket.result()
                if state is not None:
                    matched = len(keyed)
            source = None
            if state is None and first and not generated:
                # another request is reading this prompt's head right now:
                # its state there, once it gets there, instead of a second read
                share = self._share(keyed)
                while share is not None and not share.ready():
                    yield Wait(share)
            if state is None:
                ticket = self.cache.begin_lookup(context)
                while not ticket.ready():
                    yield Wait(ticket)
                state, matched = ticket.result()
                source = ticket.source
            if self.cache.files is not None and 0 < matched < len(context) \
                    and not self.cache.files.has(tuple(context[:matched])):
                # what this hit was cut back to: the next turn usually
                # diverges at exactly this cut point, and a state saved there
                # starts it instead of re-forwarding the kept tail. The state
                # IS context[:matched] now (the lookup truncated it); the copy
                # is the cache's to save and free. No room for the copy, and
                # the cut point is simply not learned.
                try:
                    kept = self.runtime.clone_state(state)
                except BudgetRefused:
                    kept = None
                if kept is not None:
                    self.cache.files.save(context[:matched], kept, owned=True)
            fed = (list(tokens) + generated)[matched:]
            taken = len(generated)          # output this admission adds is past here
            at = 0                          # of `fed`, forwarded so far
            producer = None
            if first and fed:
                producer = _Producer(_pack(keyed), matched)
                self._producers.append(producer)
            try:
                while at < len(fed):
                    # the Batcher's size: smaller beside other requests, and
                    # ending where another request waits for this state
                    size = self._chunk
                    if producer is not None and producer.cuts:
                        size = min([size] + [c - matched - at for c in producer.cuts if c > matched + at])
                    chunk = fed[at:at + size]
                    last = at + len(chunk) == len(fed)
                    # a step with no sampling spec samples nothing: max_tokens = 0
                    # is exactly zero output, and chunks before the last one never
                    # sample
                    sample = spec if last and max_tokens > len(generated) else None
                    o = matched + at
                    spans = [ImageSpan(sp.start - o, sp.length, sp.image) for sp in images
                             if sp.start < o + len(chunk) and sp.start + sp.length > o]
                    result, elapsed = yield Want(StepItem(state, chunk, sample, spans), ctx,
                                                 left=len(fed) - at - len(chunk))
                    absorb(result)
                    # book the tokens that ran, not the tokens the request fed: a
                    # truncate can queue the kept prefix for re-forwarding inside
                    # this step, and that work must not be read as the model
                    # slowing down
                    ran = result.forwarded_tokens or len(chunk)
                    if first and at == 0:   # a truncate's replay runs in the first step
                        replayed = max(0, ran - len(chunk))
                    timing.prefill_rates.add(ran, elapsed)
                    timing.prefill_seconds += elapsed
                    timing.prefill_tokens += ran
                    at += len(chunk)
                    if first:
                        consumed = at
                    if producer is not None:
                        self._publish(producer, matched + at, state, keyed, sample is not None)
                    if finished or (max_tokens > 0 and len(generated) >= max_tokens):
                        break
                self._unproduce(producer)
                producer = None

                if first and keyed and at == len(fed) \
                        and self.prompt_cache is not None and not self.prompt_cache.has(keyed):
                    # the prompt as it stood before its first token: what a
                    # repeat of this whole prompt starts from. The state has
                    # already sampled that token (the last prefill chunk
                    # does), so the saved one is a clone cut back to the
                    # prompt; no room for the copy, and the prompt is simply
                    # not saved
                    try:
                        kept = self.runtime.clone_state(state)
                        self.runtime.truncate_state(kept, len(keyed))
                        self.prompt_cache.save(keyed, kept, owned=True)
                    except BudgetRefused:
                        pass

                decoded = 0
                while not finished and len(generated) < max_tokens:
                    before = len(generated)
                    result, elapsed = yield Want(StepItem(state, [], spec), ctx)
                    absorb(result)
                    if first and not fed and not decoded:   # the replay, beside the one token
                        replayed = max(0, result.forwarded_tokens - 1)
                    decoded += 1
                    produced = len(generated) - before
                    if produced:  # a step that sampled no output is not a decode speed
                        timing.decode_tokens += produced
                        timing.decode_seconds += elapsed
                        # ...and neither is a step that returned a token it had
                        # already computed. A model that verifies a draft hands
                        # the second token out on the next step with no forward
                        # at all; that step takes microseconds, so as a rate
                        # sample it reads in the tens of thousands and takes the
                        # phase's mean and max with it. The token and the time
                        # still count -- the aggregate is right -- but it is not
                        # a sample of how fast a forward runs.
                        if result.forwarded:
                            timing.decode_rates.add(produced, elapsed)
            except Preempted:
                # the batch needs this request's room: its state waits in the
                # prefix cache (evictable like any entry) for its next turn
                self.cache.store(context[:matched + at] + generated[taken:], state, len(keyed),
                                 supersedes(source, matched))
                first = False
                continue
            except BaseException:
                # a refused allocation, a cancelled or failed step: the state is
                # this request's alone, and nothing else would ever free it
                self.runtime.free_state(state)
                raise
            finally:
                self._unproduce(producer)
            break
        timing.total_seconds = time.perf_counter() - t_start
        result_tokens = generated if cut is None else generated[:cut]
        text = self.runtime.decode(result_tokens)
        # the cache entry is keyed on what the state actually saw — including
        # tokens the guard later trimmed, which the state absorbed all the
        # same. An early end may also have left part of the prompt unfed.
        # The request is done with its state: the cache keeps it as is (a copy
        # would need the room of a second one, which a long conversation's
        # state does not leave)
        self.cache.store(context[:matched + at] + generated[taken:], state, len(keyed),
                         supersedes(source, matched))
        # the prompt tokens this request ran, a replay included: the rest
        # is what the cache really saved (usage's cached_tokens)
        fed_prompt = min(len(tokens), consumed + replayed)
        return GenerationResult(text, result_tokens, fed_prompt, finished, timing,
                                looped=cut is not None, absorbed_tokens=generated)

    def _share(self, keyed: list) -> "Ticket | None":
        """The point of another request's prompt this one should wait for
        (SHARE_MIN more tokens than any cache gives, not read yet): its
        ticket, or None."""
        if not self._producers:
            return None
        key = _pack(keyed)
        have = None
        best, best_len = None, 0
        for p in self._producers:
            n = _common(p.key, key)
            if n <= best_len or n <= p.pos:
                continue
            if have is None:
                have = self.cache.match_len(key)
            if n - have >= SHARE_MIN:
                best, best_len = p, n
        if best is None:
            return None
        return best.cuts.setdefault(best_len, Ticket())

    def _publish(self, producer: _Producer, pos: int, state, keyed: list, sampled: bool) -> None:
        """The producer's state is at `pos`: a copy of it there joins the
        prefix cache for the requests waiting at `pos` (one past it is no
        longer reachable: they go on without it)."""
        producer.pos = pos
        for cut in sorted(c for c in producer.cuts if c <= pos):
            ticket = producer.cuts.pop(cut)
            if cut == pos:
                try:
                    kept = self.runtime.clone_state(state)
                    if sampled:            # the step drew the first token: back to the prompt
                        self.runtime.truncate_state(kept, pos)
                    self.cache.store(keyed[:pos], kept, pos)
                except BudgetRefused:
                    pass
            ticket.finish(None)

    def _unproduce(self, producer: "_Producer | None") -> None:
        if producer is None:
            return
        if producer in self._producers:
            self._producers.remove(producer)
        for ticket in producer.cuts.values():
            ticket.finish(None)
        producer.cuts.clear()

    def close(self) -> None:
        if self.prompt_cache is not None:
            self.prompt_cache.close()
        if self.cache.files is not None:
            self.cache.files.close()
        self.cache.free_all()
        self.runtime.close()

    def _absorb(self, result: StepResult, generated: list[int]) -> bool:
        if result.token is None:
            return False
        if result.token in self.facts.stop_tokens:
            return True
        generated.append(result.token)
        return False


@dataclass
class Admit:
    """A sequence asks to be admitted: room for `nbytes` of state."""

    nbytes: int


@dataclass
class Wait:
    """A sequence waits for work outside the model (a saved state loading)."""

    ticket: object


@dataclass
class Want:
    """A sequence's item for the next step, and the context that may cancel
    it; `left`: the prompt tokens still to feed after this item's."""

    item: StepItem
    ctx: StepContext
    left: int = 0


class Preempted(Exception):
    """Thrown into a sequence the step has no room for (`Engine.sequence`)."""


class _Task:
    """One driven generator: what it waits for, and how it ended."""

    __slots__ = ("gen", "want", "admitted", "done", "result", "error", "order", "since")

    def __init__(self, gen, order: int) -> None:
        self.gen = gen
        self.want = None
        self.admitted = False
        self.done = False
        self.result = None
        self.error: BaseException | None = None
        self.order = order
        self.since = 0.0        # when its last step landed (prefill aging)


class _Flight:
    """A prefill step running on a thread of its own, beside the decode steps
    (`concurrent_prefill`): the runtime's step for these tasks' items; its
    results, or its error, once `done` is set."""

    def __init__(self, engine: "Engine", tasks: list[_Task]) -> None:
        self.tasks = tasks
        self.done = threading.Event()
        self.results = None
        self.error: BaseException | None = None
        self.elapsed = 0.0
        items = [t.want.item for t in tasks]
        threading.Thread(target=self._run, args=(engine.runtime, items),
                         name="footless-prefill", daemon=True).start()

    def _run(self, runtime, items) -> None:
        try:
            started = time.perf_counter()
            self.results = runtime.step(items, StepContext())
            self.elapsed = time.perf_counter() - started
        except BaseException as exc:  # noqa: BLE001 - handed to the tasks when it lands
            self.error = exc
        finally:
            self.done.set()


class Batcher:
    """Several requests' steps as one: the engine's scheduling policy.

    A task is a generator that yields what `Engine.sequence` yields (it may be
    one, or one that delegates to it with `yield from`). Tasks are admitted in
    arrival order while fewer than the package's `max_batch` are, and -- beside
    others -- while the budget has room for what they ask (`Admit`: the ledger's
    free bytes and what the prefix cache could give back). Each `step` runs ONE
    runtime step over the admitted tasks: every decode, and at most one prefill
    chunk -- the one with the fewest prompt tokens left, unless one has waited
    PREFILL_AGING since its last chunk (then the oldest such), with the short
    prompts (PACK_TOKENS) that fit beside it whole. A package that declares `concurrent_prefill`
    runs that chunk on a thread of its own instead (`_Flight`) while the
    decode steps go on: one prefill step in flight at a time, the decodes
    never waiting for it. A task waiting on work outside the model (`Wait`: a
    saved state loading) is admitted but out of the ready and stepping sets
    until its wait is over; a step with nothing else to do blocks briefly on
    one waiting ticket or the prefill in flight instead of spinning. A task
    cancelled at a step boundary is ended there without the model. A step the
    budget refuses runs again without its youngest task, which is preempted
    and admitted again later; alone, a refusal is the task's error, as it
    always was.
    """

    def __init__(self, engine: "Engine") -> None:
        self.engine = engine
        self.tasks: list[_Task] = []
        self._order = 0
        self._overlap = CONCURRENT_PREFILL in engine.facts.capabilities
        self._flight: _Flight | None = None

    def add(self, gen) -> _Task:
        task = _Task(gen, self._order)
        self._order += 1
        self.tasks.append(task)
        self._advance(task)
        return task

    def busy(self) -> bool:
        return bool(self.tasks)

    def run(self, gen):
        """One task, alone, to its end: its return value, or its error raised."""
        task = self.add(gen)
        while not task.done:
            self.step()
        if task.error is not None:
            raise task.error
        return task.result

    def _advance(self, task: _Task, value=None, throw: BaseException | None = None) -> None:
        try:
            if throw is not None:
                task.want = task.gen.throw(throw)
            elif task.want is None:
                task.want = next(task.gen)
            else:
                task.want = task.gen.send(value)
        except StopIteration as stop:
            task.done, task.result = True, stop.value
        except Exception as exc:  # noqa: BLE001 - the task's own failure, reported
            task.done, task.error = True, exc
        else:
            # admission is a sequence's: a task that runs several in turn (a
            # request's choices) asks again for each
            if isinstance(task.want, Admit):
                task.admitted = False
        task.since = time.perf_counter()

    def _room(self) -> int:
        return self.engine.services.accounting.available() + self.engine.cache.nbytes()

    def _chunk(self, live: list[_Task]) -> int:
        """The prefill chunk a sequence cuts next: CHUNK alone; beside
        others SHARED_CHUNK -- or, when the package runs prefill beside the
        decodes, OVERLAP_CHUNK unless another request waits to prefill."""
        if len(live) <= 1:
            return CHUNK
        if not self._overlap:
            return SHARED_CHUNK
        prefilling = sum(1 for t in live if isinstance(t.want, Want) and t.want.item.input_tokens)
        waiting = any(not t.admitted and not t.done for t in self.tasks)
        return SHARED_CHUNK if prefilling > 1 or waiting else OVERLAP_CHUNK

    def _next_prefill(self, prefills: list[_Task]) -> list[_Task]:
        """The prefills a step runs: the one with the fewest tokens left (the
        starved first), and the short prompts that fit beside it whole."""
        if not prefills:
            return []
        now = time.perf_counter()
        starved = [t for t in prefills if now - t.since >= PREFILL_AGING]
        remaining = lambda t: (len(t.want.item.input_tokens) + t.want.left, t.order)  # noqa: E731
        first = min(starved, key=lambda t: t.since) if starved else min(prefills, key=remaining)
        chosen, rows = [first], len(first.want.item.input_tokens)
        for task in sorted(prefills, key=remaining):
            n = len(task.want.item.input_tokens)
            if task is first or task.want.left or n > PACK_TOKENS or rows + n > CHUNK \
                    or len(chosen) >= self.engine.facts.max_batch:
                continue
            chosen.append(task)
            rows += n
        return chosen

    def _land(self, flight: _Flight) -> None:
        """The prefill step in flight is done: its tasks go on, or take its
        error (a refused claim beside other tasks preempts them instead)."""
        self._flight = None
        if flight.error is None:
            for task, result in zip(flight.tasks, flight.results):
                self._advance(task, (result, flight.elapsed))
            return
        others = [t for t in self.tasks if t.admitted and not t.done and t not in flight.tasks]
        for task in flight.tasks:
            if isinstance(flight.error, BudgetRefused) and others:
                self._advance(task, throw=Preempted())
                self.engine.services.log("info", "budget refused a prefill step: "
                                                 "that request waits for room")
            else:
                self._advance(task, throw=flight.error)

    def step(self) -> list[_Task]:
        """Admit what fits, run one step, and return the tasks that ended."""
        engine = self.engine
        live = [t for t in self.tasks if t.admitted and not t.done]
        for task in self.tasks:
            if task.done or task.admitted or not isinstance(task.want, Admit):
                continue
            if len(live) >= engine.facts.max_batch or \
                    (live and task.want.nbytes > self._room()):
                break                   # arrival order: nobody overtakes the head
            task.admitted = True
            live.append(task)
            engine._chunk = self._chunk(live)
            self._advance(task)
        engine._chunk = self._chunk(live)
        if self._flight is not None and self._flight.done.is_set():
            self._land(self._flight)
        for task in live:
            # a wait that is over: resume the sequence here, so its want is
            # settled before the ready set is taken (a still-waiting task
            # keeps its `Wait` and steps nothing this round)
            if not task.done and isinstance(task.want, Wait) \
                    and task.want.ticket.ready():
                self._advance(task)
        flying = self._flight.tasks if self._flight is not None else []
        ready = [t for t in live if not t.done and isinstance(t.want, Want) and t not in flying]
        stepping = []
        for task in ready:
            if task.want.ctx.cancelled():
                # the runtime's own answer to a cancelled item, without the model
                self._advance(task, (StepResult(None, True), 0.0))
            else:
                stepping.append(task)
        decodes = [t for t in stepping if not t.want.item.input_tokens]
        prefills = [t for t in stepping if t.want.item.input_tokens]
        if self._overlap:
            if prefills and self._flight is None:
                self._flight = _Flight(engine, self._next_prefill(prefills))
            batch = decodes
        else:
            batch = decodes + self._next_prefill(prefills)
        if not batch:
            if self._flight is not None:
                # nothing to step beside it: wait for the prefill in flight
                # (briefly -- a request arriving meanwhile is admitted then)
                if self._flight.done.wait(0.05):
                    self._land(self._flight)
            else:
                for task in live:
                    if isinstance(task.want, Wait):
                        # nothing to step and work in flight: block briefly on
                        # one ticket rather than spin -- a request arriving
                        # during the wait is never held longer than this
                        task.want.ticket.wait(0.05)
                        break
        while batch:
            try:
                started = time.perf_counter()
                results = engine.runtime.step([t.want.item for t in batch], StepContext())
                elapsed = time.perf_counter() - started
            except BudgetRefused as exc:
                if len(batch) == 1 and self._flight is None:
                    self._advance(batch[0], throw=exc)
                    break
                victim = max(batch, key=lambda t: t.order)
                batch.remove(victim)
                self._advance(victim, throw=Preempted())
                engine.services.log("info", "budget refused a step: the youngest "
                                            "request waits for room")
                continue
            except Exception as exc:  # noqa: BLE001 - whose item failed is unknown
                for task in batch:
                    self._advance(task, throw=exc)
                break
            for task, result in zip(batch, results):
                self._advance(task, (result, elapsed))
            break
        ended = [t for t in self.tasks if t.done]
        self.tasks = [t for t in self.tasks if not t.done]
        return ended


def load_runtime(path: Path) -> Runtime:
    module_name = f"footless_runtime_{abs(hash(path.resolve()))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ContractError(f"cannot load runtime: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    create = getattr(module, "create", None)
    if not callable(create):
        raise ContractError(f"{path} does not define create()")
    runtime = create()
    missing = [name for name in RUNTIME_METHODS if not callable(getattr(runtime, name, None))]
    if missing:
        raise ContractError(f"{path} runtime is missing methods: {missing}")
    return runtime
