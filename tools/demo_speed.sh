#!/usr/bin/env bash
# A live speed demo against a running `footless serve`, made to be recorded:
#
#   ① prompt processing (prefill) at a few prompt sizes, read from scratch
#   ② the caches: the same prompt again, a conversation's next turn, and
#      several requests arriving together on one long document
#   ③ generation (decode) with 1 to N requests at once
#   ④ a real job: agents working through multi-turn conversations at once
#      (a shared system prompt, tool results fed back each turn) while a
#      long document arrives in the middle
#   ⑤ /v1/decisions and /v1/systemone: typed questions scored at the answer,
#      cold, warm, several at once, and beside running generations
#   ⑥ a summary of all of it
#
#   tools/demo_speed.sh [--url http://localhost:8080] [--api-key KEY]
#                       [--prefill 1000,4000,8000,16000] [--tokens 512] [--max 4]
#                       [--cache-tokens 8000] [--agents 3] [--turns 3]
#                       [--agent-tokens 256] [--doc-tokens 12000]
#                       [--skip prefill,caches,decode,job,decisions]
#
# Needs only bash and python3 (standard library). Every prompt starts with a
# fresh tag, so nothing comes from an earlier run's caches: what the cache
# section shows is what the caches save within this run.
exec python3 - "$@" <<'PY'
import argparse, json, random, shutil, subprocess, sys, threading, time, urllib.error, urllib.request

ap = argparse.ArgumentParser(prog="demo_speed.sh")
ap.add_argument("--url", default="http://localhost:8080")
ap.add_argument("--api-key")
ap.add_argument("--prefill", default="1000,4000,8000,16000", help="prompt sizes, in tokens (approximate)")
ap.add_argument("--tokens", type=int, default=512, help="tokens each request generates (③)")
ap.add_argument("--max", type=int, default=4, help="most requests at once (③)")
ap.add_argument("--cache-tokens", type=int, default=8000, help="the cache section's document size")
ap.add_argument("--agents", type=int, default=3, help="agents working at once (④)")
ap.add_argument("--turns", type=int, default=3, help="turns each agent takes (④)")
ap.add_argument("--agent-tokens", type=int, default=256, help="tokens an agent's answer may take (④)")
ap.add_argument("--doc-tokens", type=int, default=12000, help="the document arriving mid-job (④)")
ap.add_argument("--skip", default="", help="sections to leave out: prefill,caches,decode,job,decisions")
args = ap.parse_args()
SKIP = {s.strip() for s in args.skip.split(",") if s.strip()}

# ---------------------------------------------------------------- the look
E = "\x1b["
RESET, BOLD, DIM = E + "0m", E + "1m", E + "2m"
def fg(n): return f"{E}38;5;{n}m"
GRAD = [51, 45, 39, 33, 63, 99, 135, 171, 207, 213]          # cyan -> magenta
CYAN, MAG, GREEN, YELLOW, GREY, WHITE, RED = fg(51), fg(207), fg(84), fg(221), fg(244), fg(255), fg(203)
TTY = sys.stdout.isatty()
WIDTH = min(shutil.get_terminal_size((100, 30)).columns, 120)
SPIN = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

def out(s=""):
    sys.stdout.write(s + "\n"); sys.stdout.flush()

def gradient(text):
    n = max(len(text) - 1, 1)
    return "".join(fg(GRAD[round(i * (len(GRAD) - 1) / n)]) + c for i, c in enumerate(text)) + RESET

def bar(frac, width, colors=GRAD):
    frac = max(0.0, min(1.0, frac))
    full = frac * width
    cells = []
    for i in range(width):
        c = fg(colors[min(len(colors) - 1, i * len(colors) // max(width, 1))])
        if i + 1 <= full:
            cells.append(c + "█")
        elif i < full:
            cells.append(c + "▏▎▍▌▋▊▉"[int((full - i) * 7)])
        else:
            cells.append(fg(237) + "·")
    return "".join(cells) + RESET

def section(title, subtitle=""):
    out()
    out(f"  {gradient('━' * (WIDTH - 4))}")
    out(f"  {BOLD}{WHITE}{title}{RESET}  {GREY}{subtitle}{RESET}")
    out(f"  {gradient('━' * (WIDTH - 4))}")

class Live:
    """Redraws a block of lines in place (or prints the last frame without a tty)."""
    def __init__(self): self.n = 0
    def draw(self, lines):
        if not TTY: return
        buf = (f"{E}{self.n}A" if self.n else "")
        buf += "".join(f"{E}2K{line}\n" for line in lines)
        sys.stdout.write(buf); sys.stdout.flush()
        self.n = len(lines)
    def done(self, lines):
        if TTY: self.draw(lines)
        else:
            for line in lines: out(line)

def watch(threads, frame, every=0.1):
    """Redraw frame(i) while the threads run; the last frame stays."""
    live, i = Live(), 0
    while any(t.is_alive() for t in threads):
        live.draw(frame(i)); i += 1; time.sleep(every)
    live.done(frame(i))

def start(fn, *a):
    t = threading.Thread(target=fn, args=a, daemon=True); t.start(); return t

# ---------------------------------------------------------------- the API
def request(path, body=None):
    req = urllib.request.Request(args.url.rstrip("/") + path,
                                 json.dumps(body).encode() if body is not None else None,
                                 {"Content-Type": "application/json"})
    if args.api_key:
        req.add_header("Authorization", f"Bearer {args.api_key}")
    try:
        return urllib.request.urlopen(req, timeout=3600)
    except urllib.error.HTTPError as exc:          # the server's own reason
        try:
            reason = json.loads(exc.read())["error"]["message"]
        except Exception:
            reason = exc.reason
        raise RuntimeError(f"HTTP {exc.code} on {path}: {reason}") from None

def chat(messages, max_tokens, on_piece=None, temperature=0.7):
    """Stream one chat answer (on_piece(text) per delta): its text, usage,
    and when it started, gave its first token and ended."""
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    body = {"model": MODEL, "messages": messages, "max_tokens": max_tokens, "stream": True,
            "reasoning_effort": "none", "temperature": temperature,
            "stream_options": {"include_usage": True}}
    r = {"t0": time.perf_counter(), "first": None, "end": None, "text": "", "usage": {}}
    with request("/v1/chat/completions", body) as resp:
        for line in resp:
            if not line.startswith(b"data: {"):
                continue
            chunk = json.loads(line[6:])
            if chunk.get("usage"):
                r["usage"] = chunk["usage"]
            for choice in chunk.get("choices", []):
                piece = choice.get("delta", {}).get("content")
                if piece:
                    r["first"] = r["first"] or time.perf_counter()
                    r["text"] += piece
                    if on_piece:
                        on_piece(piece)
    r["end"] = time.perf_counter()
    r["first"] = r["first"] or r["end"]
    return r

def cached(usage):
    return (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)

def post(path, body):
    t0 = time.perf_counter()
    with request(path, body) as resp:
        payload = json.loads(resp.read())
    return payload, time.perf_counter() - t0

def machine_lines():
    """The GPU (nvidia-smi) or, on a Mac, the chip, its memory and the GPU's wired limit."""
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=name,temperature.gpu,clocks.sm,memory.used,memory.total",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
        name, temp, clk, used, total = [x.strip() for x in q.stdout.splitlines()[0].split(",")]
        return name, f"{temp}°C · {clk} MHz · {int(used) / 1024:.1f}/{int(total) / 1024:.0f} GiB"
    except Exception:
        pass
    try:
        sysctl = lambda k: subprocess.run(["sysctl", "-n", k], capture_output=True, text=True, timeout=5).stdout.strip()
        chip, ram = sysctl("machdep.cpu.brand_string"), int(sysctl("hw.memsize") or 0)
        wired = sysctl("iogpu.wired_limit_mb")
        mp = subprocess.run(["memory_pressure"], capture_output=True, text=True, timeout=10).stdout
        free = next((l.split(":")[1].strip() for l in mp.splitlines() if "free percentage" in l), "?")
        state = f"{ram / 2**30:.0f} GiB unified memory · GPU wired limit {int(wired or 0) / 1024:.0f} GiB · {free} free"
        return chip or None, state
    except Exception:
        return None, None

# ---------------------------------------------------------------- the text it reads
WORDS = {
    "who": ["the engineer", "a young botanist", "the old captain", "our neighbour", "the committee",
            "a curious child", "the night guard", "the orchestra", "a travelling merchant", "the archivist"],
    "did": ["carefully measured", "slowly rebuilt", "quietly studied", "patiently mapped", "boldly questioned",
            "secretly admired", "loudly debated", "gently repaired", "methodically catalogued", "briefly visited"],
    "what": ["the copper bridge", "every northern valley", "a forgotten library", "the river delta",
             "the clock tower", "an abandoned observatory", "the spice market", "the coastal railway",
             "a field of sunflowers", "the glass greenhouse"],
    "when": ["before dawn", "during the long winter", "after the harvest", "on a rainy Tuesday",
             "in the spring of that year", "while the town slept", "at the turn of the century",
             "between two storms", "as the festival began", "long after midnight"],
    "why": ["to understand its history", "because nobody else would", "for the coming generations",
            "to settle an old argument", "in search of a lost map", "to test a new theory",
            "out of simple curiosity", "to prepare for the journey", "as the mayor had asked",
            "to write the final report"],
}

def long_text(n_tokens, rng):
    sents, words = [], 0
    while words * 1.12 < n_tokens:
        s = f"{rng.choice(WORDS['who']).capitalize()} {rng.choice(WORDS['did'])} " \
            f"{rng.choice(WORDS['what'])} {rng.choice(WORDS['when'])} {rng.choice(WORDS['why'])}."
        sents.append(s)
        words += len(s.split())
    return " ".join(sents)

TOPICS = ["Write a vivid short story about a lighthouse keeper who finds a message in a bottle.",
          "Explain, in detail and with examples, how a refrigerator moves heat out of its inside.",
          "Describe an imaginary city built inside a giant tree: its streets, people and festivals.",
          "Write the opening chapter of a science-fiction novel set on a frozen moon.",
          "Explain how the internet routes a message from one computer to another, step by step.",
          "Tell the history of the bicycle as an exciting adventure, from the first draisine on."]

TASKS = ["Draft a project timeline from the notes: phases, owners and risks.",
         "Find every place the notes mention and say what happened there, as a list.",
         "Write a short report for the mayor: what was done, what is left, what it needs.",
         "List the open questions the notes raise and who should answer each one.",
         "Turn the notes into a checklist for the next field trip.",
         "Summarize who did what, grouped by person."]

def tag(kind):
    return f"[{kind} {rng.randrange(1 << 40):x}] "

def row(mark, label, n_tok, ttft, rate=None, extra=""):
    rate_s = f"  {BOLD}{MAG}{rate:6.0f} tok/s{RESET}" if rate is not None else ""
    return f"  {mark} {BOLD}{label:<26}{RESET} {WHITE}{n_tok:>7,}{RESET} {GREY}tokens{RESET}  " \
           f"{YELLOW}{ttft:6.2f} s{RESET}{rate_s}  {extra}"

def first_token_live(label, size_hint, fn):
    """Run fn() on a thread with a scanning bar; its result (or its error raised)."""
    box = {}
    th = start(lambda: box.__setitem__("r", _try(fn)))
    t0 = time.perf_counter()
    def frame(i):
        pos = (i % 40) / 39
        scan = "".join((fg(GRAD[min(9, abs(int((k / 39 - pos) * 30)))]) + "█") if abs(k / 39 - pos) < 0.12
                       else fg(237) + "·" for k in range(40))
        return [f"  {CYAN}{SPIN[i % 10]}{RESET} {BOLD}{label:<26}{RESET} ~{size_hint:>6,} tokens  {scan}{RESET}  "
                f"{YELLOW}{time.perf_counter() - t0:6.1f} s{RESET}"]
    live, i = Live(), 0
    while th.is_alive():
        live.draw(frame(i)); i += 1; time.sleep(0.08)
    if TTY and live.n:
        sys.stdout.write(f"{E}{live.n}A{E}2K"); sys.stdout.flush()
    r = box["r"]
    if isinstance(r, BaseException):
        raise r
    return r

def _try(fn):
    try:
        return fn()
    except BaseException as exc:  # noqa: BLE001 - re-raised by the caller
        return exc

# ---------------------------------------------------------------- start
try:
    with request("/v1/models") as r:
        MODEL = json.loads(r.read())["data"][0]["id"]
except Exception as exc:
    out(f"\n  {RED}✗ no footless server at {args.url}{RESET}  {GREY}({exc}){RESET}")
    out(f"  {GREY}start it first, e.g.  ./serve.sh   or   python3 -m footless serve ...{RESET}\n")
    sys.exit(1)

if TTY:
    sys.stdout.write(E + "?25l")       # hide the cursor while drawing
hw, hw_state = machine_lines()
out()
for line in ("  ███████╗ ██████╗  ██████╗ ████████╗██╗     ███████╗███████╗███████╗",
             "  ██╔════╝██╔═══██╗██╔═══██╗╚══██╔══╝██║     ██╔════╝██╔════╝██╔════╝",
             "  █████╗  ██║   ██║██║   ██║   ██║   ██║     █████╗  ███████╗███████╗",
             "  ██╔══╝  ██║   ██║██║   ██║   ██║   ██║     ██╔══╝  ╚════██║╚════██║",
             "  ██║     ╚██████╔╝╚██████╔╝   ██║   ███████╗███████╗███████║███████║",
             "  ╚═╝      ╚═════╝  ╚═════╝    ╚═╝   ╚══════╝╚══════╝╚══════╝╚══════╝"):
    out(gradient(line))
out()
out(f"  {BOLD}{WHITE}{MODEL}{RESET}  {GREY}·{RESET}  {CYAN}{hw or 'GPU'}{RESET}  {GREY}·{RESET}  "
    f"{GREY}{args.url}{RESET}")
if hw_state:
    out(f"  {GREY}{hw_state}{RESET}")

rng = random.Random(time.time())
R = {"prefill": [], "caches": [], "decode": [], "job": None, "decisions": []}

try:
    # ------------------------------------------------------------ ① prefill
    sizes = [int(x) for x in args.prefill.split(",") if x.strip()]
    if sizes and "prefill" not in SKIP:
        section("① PROMPT PROCESSING", "a fresh prompt read from scratch · time to first token")
        out()
        best = 1000.0
        for size in sizes:
            prompt = tag("run") + "Read the following notes, then answer with one word: OK.\n\n" + long_text(size, rng)
            r = first_token_live("from scratch", size, lambda: chat(prompt, 1))
            n = r["usage"].get("prompt_tokens", size)
            ttft = r["first"] - r["t0"]
            R["prefill"].append((n, ttft, n / ttft))
            best = max(best, n / ttft)
            out(f"  {GREEN}✔{RESET} {BOLD}{n:>7,} tokens{RESET}  {bar((n / ttft) / best, 40)}  "
                f"{YELLOW}{ttft:6.1f} s{RESET}  {BOLD}{MAG}{n / ttft:6.0f} tok/s{RESET}")

    # ------------------------------------------------------------ ② caches
    if "caches" not in SKIP:
        section("② CACHES", "the same work again · what the prompt and prefix caches save")
        out()
        size = args.cache_tokens
        doc = tag("cache") + "Read the following notes.\n\n" + long_text(size, rng)
        ask = doc + "\n\nAnswer with one word: OK."
        cold = first_token_live("cold: read from scratch", size, lambda: chat(ask, 1))
        base = cold["first"] - cold["t0"]
        n = cold["usage"].get("prompt_tokens", 0)
        R["caches"].append(("cold", n, cached(cold["usage"]), base))
        out(row(f"{GREEN}✔{RESET}", "cold: read from scratch", n, base, n / base))

        def show(label, r):
            ttft = r["first"] - r["t0"]
            c, n = cached(r["usage"]), r["usage"].get("prompt_tokens", 0)
            R["caches"].append((label, n, c, ttft))
            out(row(f"{GREEN}✔{RESET}", label, n, ttft,
                    extra=f"{GREY}cached{RESET} {WHITE}{c:,}{RESET}{GREY} · {RESET}{GREEN}×{base / ttft:.0f} faster{RESET}"))
        show("the same prompt again", first_token_live("the same prompt again", size, lambda: chat(ask, 1)))
        turn = [{"role": "user", "content": ask}, {"role": "assistant", "content": "OK"},
                {"role": "user", "content": "Now name one place the notes mention, in a few words."}]
        show("the conversation's next turn", first_token_live("the conversation's next turn", size,
                                                            lambda: chat(turn, 1)))
        # several requests on one new document at once: it is read once
        shared = tag("shared") + "Read the following notes.\n\n" + long_text(size, rng)
        qs = ["Which person appears most often?", "Name one place.", "What season is mentioned?"]
        res = [None] * len(qs)
        t0 = time.perf_counter()
        ths = [start(lambda k=k: res.__setitem__(k, _try(lambda: chat(shared + "\n\n" + qs[k], 1))))
               for k in range(len(qs))]
        watch(ths, lambda i: [f"  {CYAN}{SPIN[i % 10]}{RESET} {BOLD}{len(qs)} requests on one new document{RESET}  "
                              f"{YELLOW}{time.perf_counter() - t0:6.1f} s{RESET}"])
        if TTY:
            sys.stdout.write(f"{E}1A{E}2K"); sys.stdout.flush()
        for r in res:
            if isinstance(r, BaseException):
                raise r
        last = max(r["first"] for r in res) - t0
        tot_c = sum(cached(r["usage"]) for r in res)
        n_all = sum(r["usage"].get("prompt_tokens", 0) for r in res)
        R["caches"].append((f"{len(qs)} at once, one document", n_all, tot_c, last))
        out(row(f"{GREEN}✔{RESET}", f"{len(qs)} at once, one document", n_all, last,
                extra=f"{GREY}all answering after {last:.1f} s (one cold read {base:.1f} s) · cached{RESET} "
                      f"{WHITE}{tot_c:,}{RESET}"))

    # ------------------------------------------------------------ ③ decode
    if "decode" not in SKIP:
        section("③ GENERATION", f"{args.tokens} tokens per request · 1 to {args.max} requests at once")
        for n in range(1, args.max + 1):
            out()
            out(f"  {BOLD}{WHITE}▶ {n} {'request' if n == 1 else 'requests'} at once{RESET}")
            reqs = [{"tokens": 0, "text": "", "first": None, "end": None, "err": None} for _ in range(n)]
            lock = threading.Lock()
            def run(k):
                r = reqs[k]
                def piece(p):
                    with lock:
                        r["tokens"] += 1
                        r["text"] = (r["text"] + p)[-400:]
                        r["first"] = r["first"] or time.perf_counter()
                try:
                    got = chat(TOPICS[k % len(TOPICS)], args.tokens, piece)
                    if got["usage"]:
                        r["tokens"] = got["usage"]["completion_tokens"]
                except Exception as exc:
                    r["err"] = exc
                r["end"] = time.perf_counter()
            ths = [start(run, k) for k in range(n)]
            state = {}
            def frame(i):
                now = time.perf_counter()
                lines, total = [], 0
                snip_w = max(10, WIDTH - 66)
                for k, r in enumerate(reqs):
                    total += r["tokens"]
                    end = r["end"] or now
                    rate = r["tokens"] / (end - r["first"]) if r["first"] and end > r["first"] else 0.0
                    text = " ".join(r["text"].split())[-snip_w:]
                    mark = f"{GREEN}✔{RESET}" if r["end"] else f"{CYAN}{SPIN[(i + k) % 10]}{RESET}"
                    lines.append(f"  {mark} {GREY}#{k + 1}{RESET} {bar(r['tokens'] / args.tokens, 24)} "
                                 f"{WHITE}{r['tokens']:>4}{RESET}{GREY}/{args.tokens}{RESET} "
                                 f"{YELLOW}{rate:5.1f} tok/s{RESET}  {DIM}{text:<{snip_w}}{RESET}")
                firsts = [r["first"] for r in reqs if r["first"]]
                el = max(r["end"] or now for r in reqs) - min(firsts) if firsts else 0.0
                state["agg"] = total / el if el > 0 else 0.0
                lines.append(f"    {GREY}total{RESET} {BOLD}{MAG}{state['agg']:6.1f} tok/s{RESET}   "
                             f"{GREY}{total} tokens in {el:4.1f} s{RESET}")
                return lines
            watch(ths, frame)
            for r in reqs:
                if r["err"]:
                    raise r["err"]
            slowest = min(r["tokens"] / (r["end"] - r["first"]) for r in reqs if r["first"])
            R["decode"].append((n, state["agg"], slowest))

    # ------------------------------------------------------------ ④ a real job
    if "job" not in SKIP and args.agents > 0:
        section("④ A REAL JOB", f"{args.agents} agents · {args.turns} turns each · one shared system prompt · "
                                f"tool results fed back · a {args.doc_tokens:,}-token document arrives mid-job")
        out()
        head = tag("job") + "You are a careful assistant working through a project's field notes. " \
            "Answer the task, then wait for tool results and continue.\n\nFIELD NOTES\n" + long_text(3000, rng)
        agents = [{"turn": 0, "phase": "waiting", "tokens": 0, "first": None, "gen_t": 0.0, "gen_n": 0,
                   "ttfts": [], "cached": [], "prompt": [], "err": None, "end": None}
                  for _ in range(args.agents)]
        doc = {"phase": "waiting", "ttft": None, "n": 0, "err": None, "end": None}
        lock = threading.Lock()
        t_job = time.perf_counter()

        def agent(k):
            a = agents[k]
            msgs = [{"role": "system", "content": head}, {"role": "user", "content": TASKS[k % len(TASKS)]}]
            time.sleep(0.4 * k)                       # they arrive one after another
            try:
                for t in range(args.turns):
                    with lock:
                        a["turn"], a["phase"], a["tokens"] = t + 1, "reading", 0
                    def piece(p):
                        with lock:
                            a["phase"] = "answering"
                            a["tokens"] += 1
                    r = chat(msgs, args.agent_tokens, piece)
                    with lock:
                        a["ttfts"].append(r["first"] - r["t0"])
                        a["cached"].append(cached(r["usage"]))
                        a["prompt"].append(r["usage"].get("prompt_tokens", 0))
                        n_out = r["usage"].get("completion_tokens", a["tokens"])
                        a["gen_n"] += n_out
                        a["gen_t"] += r["end"] - r["first"]
                        a["phase"] = "tool"
                    msgs += [{"role": "assistant", "content": r["text"]},
                             {"role": "user", "content": f"Tool result {t + 1}:\n" + long_text(600, rng)
                                                         + "\n\nContinue the task with this result."}]
            except Exception as exc:
                a["err"] = exc
            with lock:
                a["phase"], a["end"] = "done", time.perf_counter()

        def document():
            time.sleep(4.0)
            try:
                with lock:
                    doc["phase"] = "reading"
                text = tag("doc") + "Summarize the following report in one sentence.\n\n" + long_text(args.doc_tokens, rng)
                r = chat(text, 32)
                with lock:
                    doc["ttft"], doc["n"] = r["first"] - r["t0"], r["usage"].get("prompt_tokens", 0)
            except Exception as exc:
                doc["err"] = exc
            with lock:
                doc["phase"], doc["end"] = "done", time.perf_counter()

        ths = [start(agent, k) for k in range(args.agents)] + [start(document)]
        colors = {"waiting": GREY, "reading": CYAN, "answering": MAG, "tool": YELLOW, "done": GREEN}

        def frame(i):
            now = time.perf_counter()
            lines = []
            with lock:
                for k, a in enumerate(agents):
                    mark = f"{GREEN}✔{RESET}" if a["phase"] == "done" else f"{CYAN}{SPIN[(i + k) % 10]}{RESET}"
                    rate = a["gen_n"] / a["gen_t"] if a["gen_t"] else 0.0
                    last = f"{a['ttfts'][-1]:5.2f} s" if a["ttfts"] else "   — "
                    hit = (f"{100 * a['cached'][-1] / a['prompt'][-1]:3.0f} %" if a["prompt"] and a["prompt"][-1]
                           else "  — ")
                    lines.append(f"  {mark} {BOLD}agent {k + 1}{RESET}  {GREY}turn{RESET} {WHITE}{a['turn']}/{args.turns}{RESET}  "
                                 f"{colors[a['phase']]}{a['phase']:<9}{RESET} {bar(a['tokens'] / args.agent_tokens, 16)} "
                                 f"{GREY}first token{RESET} {YELLOW}{last}{RESET}  {GREY}cached{RESET} {WHITE}{hit}{RESET}  "
                                 f"{YELLOW}{rate:5.1f} tok/s{RESET}")
                mark = f"{GREEN}✔{RESET}" if doc["phase"] == "done" else f"{CYAN}{SPIN[i % 10]}{RESET}"
                info = f"{GREY}first token{RESET} {YELLOW}{doc['ttft']:5.1f} s{RESET}" if doc["ttft"] else ""
                lines.append(f"  {mark} {BOLD}document{RESET} {GREY}~{args.doc_tokens:,} tokens{RESET}  "
                             f"{colors[doc['phase']]}{doc['phase']:<9}{RESET} {info}")
                gen = sum(a["gen_n"] for a in agents)
                lines.append(f"    {GREY}elapsed{RESET} {YELLOW}{now - t_job:5.1f} s{RESET}   {GREY}generated{RESET} "
                             f"{WHITE}{gen}{RESET} {GREY}tokens ·{RESET} {BOLD}{MAG}{gen / max(now - t_job, 1e-6):5.1f} tok/s{RESET} "
                             f"{GREY}in all{RESET}")
            return lines
        watch(ths, frame, 0.15)
        for a in agents:
            if a["err"]:
                raise a["err"]
        if doc["err"]:
            raise doc["err"]
        wall = max(a["end"] for a in agents + [doc]) - t_job
        firsts = [a["ttfts"][0] for a in agents]
        later = [x for a in agents for x in a["ttfts"][1:]]
        hits = [c / p for a in agents for c, p in zip(a["cached"][1:], a["prompt"][1:]) if p]
        gen = sum(a["gen_n"] for a in agents)
        R["job"] = dict(wall=wall, gen=gen, first=sum(firsts) / len(firsts),
                        later=sum(later) / len(later) if later else None,
                        hit=sum(hits) / len(hits) if hits else None, doc=doc["ttft"], doc_n=doc["n"],
                        each=[a["gen_n"] / a["gen_t"] for a in agents if a["gen_t"]])

    # ------------------------------------------------------------ ⑤ decisions
    if "decisions" not in SKIP:
        section("⑤ DECISIONS", "/v1/decisions and /v1/systemone · typed questions scored at the answer, nothing generated")
        out()
        def ticket():
            return (tag("ticket") + "I have been trying to connect my Stripe account for 3 days and the "
                    "integration keeps failing. I am losing sales.\n\nEarlier in this thread:\n" + long_text(1200, rng))
        def decisions_body(text):
            return {"input": text, "questions": [
                {"id": "team", "type": "choice", "question": "Which team should handle this ticket?",
                 "options": [{"name": "billing", "description": "Payment or subscription issues"},
                             {"name": "technical", "description": "Bugs or integration problems"},
                             {"name": "sales", "description": "Pricing or account questions"}]},
                {"id": "frustration", "type": "score", "question": "How frustrated is the customer?",
                 "levels": ["Calm", "Frustrated but civil", "Very angry"]},
                {"id": "urgent", "type": "yes_no", "question": "The customer needs an answer today."}]}
        def systemone_body(text):
            return {"state": text, "model": MODEL, "questions": {
                "urgent": {"type": "noul", "instructions": "The customer needs an answer today."},
                "team": {"type": "choice", "instructions": "Which team should handle this ticket?",
                         "criteria": {"billing": None, "technical": "Bugs or integration problems", "sales": None}},
                "frustration": {"type": "score", "instructions": "How frustrated is the customer?",
                                "criteria": ["Calm", "Frustrated but civil", "Very angry"]}}}
        def summary(route, p):
            """(input tokens, a short answer line, well formed?)"""
            a = p.get("answers") or {}
            ok = True
            for name in ("team", "frustration") + (("urgent",) if route == "decisions" else ()):
                probs = (a.get(name) or {}).get("probabilities") or {}
                ok = ok and abs(sum(probs.values()) - 1.0) < 1e-3
            if route == "decisions":
                n = (p.get("usage") or {}).get("prompt_tokens", 0)
                yes = (a.get("urgent", {}).get("probabilities") or {}).get("yes", 0.0)
                line = (f"team {WHITE}{a.get('team', {}).get('choice')}{RESET} · frustration "
                        f"{WHITE}{a.get('frustration', {}).get('score', 0):.2f}{RESET} · urgent {WHITE}{yes:.0%}{RESET}")
            else:
                n = (p.get("usage") or {}).get("input_tokens", 0)
                t = a.get("team", {})
                line = (f"team {WHITE}{t.get('choice')}{RESET} ({t.get('confidence', 0):.0%}) · frustration "
                        f"{WHITE}{a.get('frustration', {}).get('score', 0):.2f}{RESET} · urgent "
                        f"{WHITE}{a.get('urgent', {}).get('noul', 0):.0%}{RESET}")
                ok = ok and 0 <= a.get("urgent", {}).get("noul", -1) <= 1
            return n, line, ok
        def show(label, route, payload, secs):
            n, line, ok = summary(route, payload)
            R["decisions"].append((label, n, secs, ok))
            mark = f"{GREEN}✔{RESET}" if ok else f"{RED}✗{RESET}"
            out(f"  {mark} {BOLD}{label:<30}{RESET} {WHITE}{n:>6,}{RESET} {GREY}tokens{RESET}  "
                f"{YELLOW}{secs * 1e3:7.0f} ms{RESET}  {BOLD}{MAG}{n / secs:6.0f} tok/s{RESET}  {GREY}{line}{RESET}")
        for route, body in (("decisions", decisions_body(ticket())), ("systemone", systemone_body(ticket()))):
            path = f"/v1/{route}"
            p, s = first_token_live(f"{path} cold", 1200, lambda: post(path, body))
            show(f"{path} · cold", route, p, s)
            p, s = first_token_live(f"{path} warm", 1200, lambda: post(path, body))
            show(f"{path} · the same again", route, p, s)
        # four tickets at once
        bodies = [decisions_body(ticket()) for _ in range(4)]
        res = [None] * 4
        t0 = time.perf_counter()
        ths = [start(lambda k=k: res.__setitem__(k, _try(lambda: post("/v1/decisions", bodies[k]))))
               for k in range(4)]
        watch(ths, lambda i: [f"  {CYAN}{SPIN[i % 10]}{RESET} {BOLD}4 tickets at once{RESET}  "
                              f"{YELLOW}{time.perf_counter() - t0:6.1f} s{RESET}"])
        if TTY:
            sys.stdout.write(f"{E}1A{E}2K"); sys.stdout.flush()
        for r in res:
            if isinstance(r, BaseException):
                raise r
        wall = time.perf_counter() - t0
        n_all = sum((p.get("usage") or {}).get("prompt_tokens", 0) for p, _ in res)
        ok = all(summary("decisions", p)[2] for p, _ in res)
        R["decisions"].append(("4 tickets at once", n_all, wall, ok))
        out(f"  {GREEN if ok else RED}{'✔' if ok else '✗'}{RESET} {BOLD}{'/v1/decisions · 4 at once':<30}{RESET} "
            f"{WHITE}{n_all:>6,}{RESET} {GREY}tokens{RESET}  {YELLOW}{wall * 1e3:7.0f} ms{RESET}  "
            f"{BOLD}{MAG}{n_all / wall:6.0f} tok/s{RESET}  {GREY}all four answered{RESET}")
        # beside two running generations
        gens = [start(lambda k=k: _try(lambda: chat(TOPICS[k], 256))) for k in range(2)]
        time.sleep(2.0)
        p, s = first_token_live("/v1/decisions beside", 1200, lambda: post("/v1/decisions", decisions_body(ticket())))
        show("/v1/decisions · beside 2 answers", "decisions", p, s)
        for g in gens:
            g.join()

    # ------------------------------------------------------------ ⑥ summary
    section("⑥ SUMMARY", MODEL)
    if R["prefill"]:
        out()
        out(f"  {BOLD}{WHITE}Prompt processing{RESET}")
        top = max(r for _, _, r in R["prefill"])
        for n_tok, ttft, rate in R["prefill"]:
            out(f"    {n_tok:>7,} tokens  {bar(rate / top, 34)}  {BOLD}{MAG}{rate:5.0f} tok/s{RESET}  "
                f"{GREY}first token in {ttft:.1f} s{RESET}")
    if R["caches"]:
        out()
        out(f"  {BOLD}{WHITE}Caches{RESET}  {GREY}(time to first token){RESET}")
        base = R["caches"][0][3]
        for label, n_tok, c, ttft in R["caches"]:
            # the document read by several at once: against reading it that many times
            ref = base * (int(label.split()[0]) if "at once" in label else 1)
            gain = f"{GREEN}×{ref / ttft:.1f}{RESET}" if label != "cold" else f"{GREY} — {RESET}"
            out(f"    {label:<32} {bar(1 - ttft / ref if label != 'cold' else 0, 20)}  {YELLOW}{ttft:6.2f} s{RESET}  "
                f"{gain}  {GREY}{c:,} of {n_tok:,} cached{RESET}")
    if R["decode"]:
        out()
        out(f"  {BOLD}{WHITE}Generation, all requests together{RESET}  {GREY}({args.tokens} tokens each){RESET}")
        top = max(a for _, a, _ in R["decode"])
        base = R["decode"][0][1]
        for n, agg, slowest in R["decode"]:
            gain = f"{GREEN}×{agg / base:.2f}{RESET}" if n > 1 else f"{GREY}  —  {RESET}"
            out(f"    {n} at once  {bar(agg / top, 34)}  {BOLD}{MAG}{agg:5.1f} tok/s{RESET}  {gain}  "
                f"{GREY}each ≥ {slowest:.1f} tok/s{RESET}")
    if R["job"]:
        j = R["job"]
        out()
        out(f"  {BOLD}{WHITE}A real job{RESET}  {GREY}({args.agents} agents × {args.turns} turns, a "
            f"{j['doc_n']:,}-token document mid-job){RESET}")
        out(f"    {GREY}done in{RESET} {YELLOW}{j['wall']:.1f} s{RESET}   {GREY}generated{RESET} {WHITE}{j['gen']}{RESET} "
            f"{GREY}tokens ·{RESET} {BOLD}{MAG}{j['gen'] / j['wall']:.1f} tok/s{RESET} {GREY}in all · each agent "
            f"{min(j['each']):.1f}-{max(j['each']):.1f} tok/s while answering{RESET}")
        later = f"{j['later']:.2f} s" if j["later"] is not None else "—"
        hit = f"{j['hit']:.0%}" if j["hit"] is not None else "—"
        out(f"    {GREY}first token: first turn{RESET} {YELLOW}{j['first']:.2f} s{RESET}{GREY}, later turns{RESET} "
            f"{YELLOW}{later}{RESET} {GREY}({hit} of their prompt cached) · the document{RESET} "
            f"{YELLOW}{j['doc']:.1f} s{RESET}")
    if R["decisions"]:
        out()
        out(f"  {BOLD}{WHITE}Decisions{RESET}  {GREY}(questions scored, nothing generated){RESET}")
        for label, n_tok, secs, ok in R["decisions"]:
            mark = f"{GREEN}✔{RESET}" if ok else f"{RED}✗{RESET}"
            out(f"    {mark} {label:<32} {YELLOW}{secs * 1e3:7.0f} ms{RESET}  {GREY}{n_tok:,} tokens{RESET}")
    _, hw_state = machine_lines()
    out()
    if hw_state:
        out(f"  {GREY}now: {hw_state}{RESET}")
    out(f"  {gradient('━' * (WIDTH - 4))}")
    out()
except KeyboardInterrupt:
    out(f"\n  {GREY}stopped{RESET}")
except Exception as exc:
    out(f"\n  {RED}✗ {exc}{RESET}")
    sys.exit(1)
finally:
    if TTY:
        sys.stdout.write(E + "?25h" + RESET); sys.stdout.flush()
PY
