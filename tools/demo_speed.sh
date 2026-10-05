#!/usr/bin/env bash
# A live speed demo against a running `footless serve`, made to be recorded:
# prompt processing (prefill) at a few prompt sizes, then generation (decode)
# with 1, 2, 3 and 4 requests at once, and a summary of both.
#
#   tools/demo_speed.sh [--url http://localhost:8080] [--api-key KEY]
#                       [--prefill 1000,4000,8000] [--tokens 192] [--max 4]
#
# Needs only bash and python3 (standard library). Every prompt starts with a
# fresh tag, so nothing comes from the server's prefix cache: what you see is
# the model working.
exec python3 - "$@" <<'PY'
import argparse, json, random, shutil, subprocess, sys, threading, time, urllib.error, urllib.request

ap = argparse.ArgumentParser(prog="demo_speed.sh")
ap.add_argument("--url", default="http://localhost:8080")
ap.add_argument("--api-key")
ap.add_argument("--prefill", default="1000,4000,8000", help="prompt sizes, in tokens (approximate)")
ap.add_argument("--tokens", type=int, default=192, help="tokens each request generates")
ap.add_argument("--max", type=int, default=4, help="most requests at once")
args = ap.parse_args()

# ---------------------------------------------------------------- the look
E = "\x1b["
RESET, BOLD, DIM = E + "0m", E + "1m", E + "2m"
def fg(n): return f"{E}38;5;{n}m"
GRAD = [51, 45, 39, 33, 63, 99, 135, 171, 207, 213]          # cyan -> magenta
CYAN, MAG, GREEN, YELLOW, GREY, WHITE = fg(51), fg(207), fg(84), fg(221), fg(244), fg(255)
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

def visible_len(s):
    n, i = 0, 0
    while i < len(s):
        if s[i] == "\x1b":
            while s[i] != "m": i += 1
        else:
            n += 1
        i += 1
    return n

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

# ---------------------------------------------------------------- the API
def request(path, body=None, stream=False):
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
        raise RuntimeError(f"HTTP {exc.code}: {reason}") from None

def chat_stream(prompt, max_tokens, on_piece):
    """Stream one chat answer; on_piece(text) per delta. Returns usage."""
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "stream": True, "reasoning_effort": "none",
            "temperature": 0.7, "stream_options": {"include_usage": True}}
    usage = None
    with request("/v1/chat/completions", body) as resp:
        for line in resp:
            if not line.startswith(b"data: {"):
                continue
            chunk = json.loads(line[6:])
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices", []):
                piece = choice.get("delta", {}).get("content")
                if piece:
                    on_piece(piece)
    return usage

def gpu_line():
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=name,temperature.gpu,clocks.sm,memory.used,memory.total",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
        name, temp, clk, used, total = [x.strip() for x in q.stdout.splitlines()[0].split(",")]
        return name, f"{temp}°C · {clk} MHz · {int(used) / 1024:.1f}/{int(total) / 1024:.0f} GiB"
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

# ---------------------------------------------------------------- start
try:
    with request("/v1/models") as r:
        MODEL = json.loads(r.read())["data"][0]["id"]
except Exception as exc:
    out(f"\n  {fg(203)}✗ no footless server at {args.url}{RESET}  {GREY}({exc}){RESET}")
    out(f"  {GREY}start it first, e.g.  docker compose up -d   or   python3 -m footless serve ...{RESET}\n")
    sys.exit(1)

if TTY:
    sys.stdout.write(E + "?25l")       # hide the cursor while drawing
gpu, gpu_state = gpu_line()
out()
for row in ("  ███████╗ ██████╗  ██████╗ ████████╗██╗     ███████╗███████╗███████╗",
            "  ██╔════╝██╔═══██╗██╔═══██╗╚══██╔══╝██║     ██╔════╝██╔════╝██╔════╝",
            "  █████╗  ██║   ██║██║   ██║   ██║   ██║     █████╗  ███████╗███████╗",
            "  ██╔══╝  ██║   ██║██║   ██║   ██║   ██║     ██╔══╝  ╚════██║╚════██║",
            "  ██║     ╚██████╔╝╚██████╔╝   ██║   ███████╗███████╗███████║███████║",
            "  ╚═╝      ╚═════╝  ╚═════╝    ╚═╝   ╚══════╝╚══════╝╚══════╝╚══════╝"):
    out(gradient(row))
out()
out(f"  {BOLD}{WHITE}{MODEL}{RESET}  {GREY}·{RESET}  {CYAN}{gpu or 'GPU'}{RESET}  {GREY}·{RESET}  "
    f"{GREY}{args.url}{RESET}")
if gpu_state:
    out(f"  {GREY}{gpu_state}{RESET}")

results_prefill, results_decode = [], []
rng = random.Random(time.time())

try:
    # ------------------------------------------------------------ prefill
    sizes = [int(x) for x in args.prefill.split(",") if x.strip()]
    if sizes:
        section("① PROMPT PROCESSING", "a fresh prompt read from scratch · time to first token")
        out()
        best = 260.0
        for size in sizes:
            tag = f"[run {rng.randrange(1 << 40):x}] "
            prompt = tag + "Read the following notes, then answer with one word: OK.\n\n" + long_text(size, rng)
            live, t0, state = Live(), time.perf_counter(), {"first": None, "usage": None, "err": None}
            def run():
                try:
                    state["usage"] = chat_stream(prompt, 1, lambda p: state.__setitem__("first", state["first"] or time.perf_counter()))
                except Exception as exc:
                    state["err"] = exc
            th = threading.Thread(target=run, daemon=True); th.start()
            i = 0
            while th.is_alive():
                el = time.perf_counter() - t0
                pos = (i % 40) / 39
                scan = "".join((fg(GRAD[min(9, abs(int((k / 39 - pos) * 30)))]) + "█") if abs(k / 39 - pos) < 0.12
                               else fg(237) + "·" for k in range(40))
                live.draw([f"  {CYAN}{SPIN[i % 10]}{RESET} {BOLD}~{size:>6,} tokens{RESET}  {scan}{RESET}  "
                           f"{YELLOW}{el:6.1f} s{RESET}"])
                i += 1; time.sleep(0.08)
            if state["err"]:
                raise state["err"]
            n = (state["usage"] or {}).get("prompt_tokens", size)
            ttft = (state["first"] or time.perf_counter()) - t0
            rate = n / ttft
            results_prefill.append((n, ttft, rate))
            live.done([f"  {GREEN}✔{RESET} {BOLD}{n:>7,} tokens{RESET}  {bar(rate / best, 40)}  "
                       f"{YELLOW}{ttft:6.1f} s{RESET}  {BOLD}{MAG}{rate:6.0f} tok/s{RESET}"])

    # ------------------------------------------------------------ decode
    section("② GENERATION", f"{args.tokens} tokens per request · 1 to {args.max} requests at once")
    for n in range(1, args.max + 1):
        out()
        out(f"  {BOLD}{WHITE}▶ {n} {'request' if n == 1 else 'requests'} at once{RESET}")
        reqs = [{"tokens": 0, "text": "", "first": None, "end": None, "err": None} for _ in range(n)]
        lock = threading.Lock()
        t0 = time.perf_counter()
        def run(k):
            r = reqs[k]
            def piece(p):
                with lock:
                    r["tokens"] += 1
                    r["text"] = (r["text"] + p)[-400:]
                    r["first"] = r["first"] or time.perf_counter()
            try:
                usage = chat_stream(TOPICS[k % len(TOPICS)], args.tokens, piece)
                if usage:
                    r["tokens"] = usage["completion_tokens"]
            except Exception as exc:
                r["err"] = exc
            r["end"] = time.perf_counter()
        ths = [threading.Thread(target=run, args=(k,), daemon=True) for k in range(n)]
        for th in ths:
            th.start()
        live, i = Live(), 0
        def frame(final=False):
            now = time.perf_counter()
            lines = []
            snip_w = max(10, WIDTH - 66)
            total = 0
            for k, r in enumerate(reqs):
                total += r["tokens"]
                end = r["end"] or now
                rate = r["tokens"] / (end - r["first"]) if r["first"] and end > r["first"] else 0.0
                text = " ".join(r["text"].split())[-snip_w:]
                mark = f"{GREEN}✔{RESET}" if r["end"] else f"{CYAN}{SPIN[(i + k) % 10]}{RESET}"
                lines.append(f"  {mark} {GREY}#{k + 1}{RESET} {bar(r['tokens'] / args.tokens, 24)} "
                             f"{WHITE}{r['tokens']:>4}{RESET}{GREY}/{args.tokens}{RESET} "
                             f"{YELLOW}{rate:5.1f} tok/s{RESET}  {DIM}{text:<{snip_w}}{RESET}")
            # the decode window: from the first token any request got
            firsts = [r["first"] for r in reqs if r["first"]]
            el = max(r["end"] or now for r in reqs) - min(firsts) if firsts else 0.0
            agg = total / el if el > 0 else 0.0
            lines.append(f"    {GREY}total{RESET} {BOLD}{MAG}{agg:6.1f} tok/s{RESET}   "
                         f"{GREY}{total} tokens in {el:4.1f} s{RESET}")
            return lines, agg
        while any(th.is_alive() for th in ths):
            live.draw(frame()[0]); i += 1; time.sleep(0.1)
        for r in reqs:
            if r["err"]:
                raise r["err"]
        lines, agg = frame(final=True)
        live.done(lines)
        slowest = min(r["tokens"] / (r["end"] - r["first"]) for r in reqs if r["first"])
        results_decode.append((n, agg, slowest))

    # ------------------------------------------------------------ summary
    section("③ SUMMARY", MODEL)
    if results_prefill:
        out()
        out(f"  {BOLD}{WHITE}Prompt processing{RESET}")
        top = max(r for _, _, r in results_prefill)
        for n_tok, ttft, rate in results_prefill:
            out(f"    {n_tok:>7,} tokens  {bar(rate / top, 34)}  {BOLD}{MAG}{rate:5.0f} tok/s{RESET}  "
                f"{GREY}first token in {ttft:.1f} s{RESET}")
    out()
    out(f"  {BOLD}{WHITE}Generation, all requests together{RESET}")
    top = max(a for _, a, _ in results_decode)
    base = results_decode[0][1]
    for n, agg, slowest in results_decode:
        gain = f"{GREEN}×{agg / base:.2f}{RESET}" if n > 1 else f"{GREY}  —  {RESET}"
        out(f"    {n} at once  {bar(agg / top, 34)}  {BOLD}{MAG}{agg:5.1f} tok/s{RESET}  {gain}  "
            f"{GREY}each ≥ {slowest:.1f} tok/s{RESET}")
    _, gpu_state = gpu_line()
    out()
    if gpu_state:
        out(f"  {GREY}GPU now: {gpu_state}{RESET}")
    out(f"  {gradient('━' * (WIDTH - 4))}")
    out()
except KeyboardInterrupt:
    out(f"\n  {GREY}stopped{RESET}")
except Exception as exc:
    out(f"\n  {fg(203)}✗ {exc}{RESET}")
    sys.exit(1)
finally:
    if TTY:
        sys.stdout.write(E + "?25h" + RESET); sys.stdout.flush()
PY
