"""The typed decision routes over the scripted mocks: POST /v1/decisions and
POST /v1/systemone.

Every request here is a real HTTP round trip against `footless serve`'s server,
on the mock packages, so the suite runs on nothing but the standard library.
The expected probabilities, masses, and confidences are computed in the test
from the raw math — never through `footless.decisions` — so the port is pinned
rather than echoed; the expected prompt texts are written out for the same
reason.
"""

from __future__ import annotations

import json
import math
import socket
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit

from footless import serve as serving
from footless.engine import Engine

ROOT = Path(__file__).resolve().parent
MOCKUP = ROOT / "mockup"
THINKING = ROOT / "mockup-thinking"


class Api:
    """One serving front end on an ephemeral port, over one mock package."""

    def __init__(self, package: Path, model_name: str | None = None,
                 api_key: str | None = None) -> None:
        self.engine = Engine.open(package)
        self.front = serving.FrontEnd(
            self.engine, model_name or self.engine.model.name,
            root=str(package), api_key=api_key,
        )
        self.httpd = serving.ThreadingHTTPServer(("127.0.0.1", 0), serving._Handler)
        self.httpd.daemon_threads = True
        self.httpd.front = self.front
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.front.start()  # the mock cares nothing about thread affinity
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.front.close()
        self.engine.close()

    def request(self, method: str, path: str, body=None, headers=None):
        if isinstance(body, (bytes, bytearray)):
            data = bytes(body)
        else:
            data = json.dumps(body).encode() if body is not None else None
        head = dict(headers or {})
        if body is not None:
            head.setdefault("Content-Type", "application/json")
        request = urllib.request.Request(self.base + path, data=data, headers=head,
                                         method=method)
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return response.status, response.read()
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read()

    def post(self, path: str, body, headers=None):
        status, raw = self.request("POST", path, body, headers)
        return status, json.loads(raw) if raw else None


# the scripted label log-probabilities: disjoint label sets per question type,
# so one script serves the choice, score, and yes or no questions at once
SCRIPTED = {"A": -0.5, "B": -1.5, "C": -2.5,
            "0": -3.0, "1": -1.0, "2": -2.0,
            "yes": -0.2, "no": -1.8}

INPUT = "S: it rains"

# what one rendered question must look like, spelled out (prompt wording v1)
CHOICE_TEXT = (
    "S: it rains\n"
    "\n"
    "Question: Which team?\n"
    "A: billing - money issues\n"
    "B: tech - broken service\n"
    "C: other - general\n"
    "Answer with the letter of one option only."
)
SCORE_TEXT = (
    "S: it rains\n"
    "\n"
    "Question: How urgent?\n"
    "0: low\n"
    "1: high\n"
    "2: critical\n"
    "Answer with the number of one level only."
)
YESNO_TEXT = (
    "S: it rains\n"
    "\n"
    "Is the following true? Is this spam?\n"
    "yes: unsolicited\n"
    "no: expected\n"
    "Answer with yes or no only."
)

REQUEST = {
    "input": INPUT,
    "questions": [
        {"id": "route", "type": "choice", "question": "Which team?",
         "options": [{"name": "billing", "description": "money issues"},
                     {"name": "tech", "description": "broken service"},
                     {"name": "other", "description": "general"}]},
        {"id": "urgency", "type": "score", "question": "How urgent?",
         "levels": ["low", "high", "critical"]},
        {"id": "spam", "type": "yes_no", "question": "Is this spam?",
         "yes": "unsolicited", "no": "expected"},
    ],
}

# the System One shapes: a noul, a two-option choice, and the same score
SO_NAUL_TEXT = YESNO_TEXT
SO_CHOICE_TEXT = (
    "S: it rains\n"
    "\n"
    "Question: Which team?\n"
    "A: billing - money issues\n"
    "B: tech - broken service\n"
    "Answer with the letter of one option only."
)
SO_SCORE_TEXT = SCORE_TEXT

SYSTEMONE_REQUEST = {
    "state": INPUT,
    "model": "jev-latest",
    "questions": {
        "spam": {"type": "noul", "instructions": "Is this spam?",
                 "criteria": {"true": "unsolicited", "false": "expected"}},
        "route": {"type": "choice", "instructions": "Which team?",
                  "criteria": {"billing": "money issues",
                               "tech": "broken service"}},
        "urgency": {"type": "score", "instructions": "How urgent?",
                    "criteria": ["low", "high", "critical"]},
    },
}


def rendered_text(api: Api, content: str) -> str:
    """The mock's chat render of one user message with thinking off."""
    runtime = api.engine.runtime
    return ("user: " + content + "\nassistant: "
            + runtime.decode(api.engine.facts.thinking_markers))


def rendered_tokens(api: Api, content: str) -> int:
    """How many tokens that render takes: one per byte, plus the markers."""
    runtime = api.engine.runtime
    return (len(runtime.encode("user: " + content + "\nassistant: "))
            + len(api.engine.facts.thinking_markers))


class DecisionsTest(unittest.TestCase):
    """POST /v1/decisions on the mock that declares label_logprobs."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.api = Api(THINKING)
        cls.api.engine.runtime.scripted_label_logprobs = dict(SCRIPTED)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.api.close()

    def test_one_request_answers_choice_score_and_yes_no(self):
        status, body = self.api.post("/v1/decisions", REQUEST)
        self.assertEqual(status, 200)
        self.assertEqual(set(body), {"object", "model", "prompt_format_version",
                                     "answers", "usage"})
        self.assertEqual(body["object"], "decisions")
        self.assertEqual(body["model"], "default")  # no model was named
        self.assertEqual(body["prompt_format_version"], 1)
        self.assertEqual(list(body["answers"]), ["route", "urgency", "spam"])

        # the math, from the raw log-probs: softmax over the labels only
        choice = [math.exp(lp) for lp in (-0.5, -1.5, -2.5)]
        choice_total = sum(choice)
        route = body["answers"]["route"]
        self.assertEqual(set(route),
                         {"type", "probabilities", "label_mass", "choice"})
        self.assertEqual(route["type"], "choice")
        self.assertEqual(route["choice"], "billing")
        self.assertAlmostEqual(route["probabilities"]["billing"],
                               choice[0] / choice_total)
        self.assertAlmostEqual(route["probabilities"]["tech"],
                               choice[1] / choice_total)
        self.assertAlmostEqual(route["probabilities"]["other"],
                               choice[2] / choice_total)
        self.assertAlmostEqual(route["label_mass"], choice_total)

        score = [math.exp(lp) for lp in (-3.0, -1.0, -2.0)]
        score_total = sum(score)
        urgency = body["answers"]["urgency"]
        self.assertEqual(set(urgency),
                         {"type", "probabilities", "label_mass", "score"})
        self.assertEqual(urgency["type"], "score")
        self.assertAlmostEqual(urgency["probabilities"]["0"],
                               score[0] / score_total)
        self.assertAlmostEqual(urgency["probabilities"]["1"],
                               score[1] / score_total)
        self.assertAlmostEqual(urgency["probabilities"]["2"],
                               score[2] / score_total)
        self.assertAlmostEqual(urgency["score"],
                               (score[1] + 2 * score[2]) / score_total)
        self.assertAlmostEqual(urgency["label_mass"], score_total)

        yes_no = [math.exp(lp) for lp in (-0.2, -1.8)]
        yes_total = sum(yes_no)
        spam = body["answers"]["spam"]
        self.assertEqual(set(spam), {"type", "probabilities", "label_mass"})
        self.assertEqual(spam["type"], "yes_no")
        self.assertAlmostEqual(spam["probabilities"]["yes"],
                               yes_no[0] / yes_total)
        self.assertAlmostEqual(spam["probabilities"]["no"],
                               yes_no[1] / yes_total)
        self.assertAlmostEqual(spam["label_mass"], yes_total)

        expected = sum(rendered_tokens(self.api, text)
                       for text in (CHOICE_TEXT, SCORE_TEXT, YESNO_TEXT))
        self.assertEqual(body["usage"], {"prompt_tokens": expected,
                                         "completion_tokens": 0,
                                         "total_tokens": expected})

    def test_return_prompt_token_ids_reports_the_rendered_questions(self):
        runtime = self.api.engine.runtime
        before = len(runtime.chats)
        status, body = self.api.post("/v1/decisions",
                                     {**REQUEST, "return_prompt_token_ids": True})
        self.assertEqual(status, 200)
        chats = runtime.chats[before:]
        # the three questions, then the input alone (the head the cache keys on)
        self.assertEqual([chat["thinking"] for chat in chats], [False] * 4,
                         "every question renders with thinking forced off")
        self.assertEqual([chat["level"] for chat in chats], [None] * 4)
        for qid, text in (("route", CHOICE_TEXT), ("urgency", SCORE_TEXT),
                          ("spam", YESNO_TEXT)):
            with self.subTest(question=qid):
                answer = body["answers"][qid]
                self.assertIn("prompt_token_ids", answer)
                self.assertIn("label_token_ids", answer)
                rendered = runtime.decode(answer["prompt_token_ids"])
                self.assertEqual(rendered, rendered_text(self.api, text))
                for line in text.splitlines():  # the wording lines, one by one
                    self.assertIn(line, rendered)
        self.assertEqual(body["answers"]["route"]["label_token_ids"],
                         [ord("A"), ord("B"), ord("C")])
        self.assertEqual(body["answers"]["urgency"]["label_token_ids"],
                         [ord("0"), ord("1"), ord("2")])
        # the mock's one-token yes and no ids (LABEL_TOKENS)
        self.assertEqual(body["answers"]["spam"]["label_token_ids"], [300, 301])

    def test_temperature_scales_the_probabilities_only(self):
        status, body = self.api.post("/v1/decisions", {**REQUEST, "temperature": 2})
        self.assertEqual(status, 200)
        warm = [math.exp(lp / 2) for lp in (-0.5, -1.5, -2.5)]
        warm_total = sum(warm)
        route = body["answers"]["route"]
        self.assertAlmostEqual(route["probabilities"]["billing"],
                               warm[0] / warm_total)
        self.assertAlmostEqual(route["probabilities"]["tech"],
                               warm[1] / warm_total)
        self.assertAlmostEqual(route["probabilities"]["other"],
                               warm[2] / warm_total)
        # label_mass is the labels' full-vocabulary mass: temperature-proof
        self.assertAlmostEqual(route["label_mass"],
                               sum(math.exp(lp) for lp in (-0.5, -1.5, -2.5)))
        # and the probabilities did move from the temperature-1 shape
        cold = [math.exp(lp) for lp in (-0.5, -1.5, -2.5)]
        self.assertNotAlmostEqual(route["probabilities"]["billing"],
                                  cold[0] / sum(cold))

    def test_prompt_format_version_is_pinned(self):
        status, _ = self.api.post("/v1/decisions",
                                  {**REQUEST, "prompt_format_version": 1})
        self.assertEqual(status, 200)
        status, body = self.api.post("/v1/decisions",
                                     {**REQUEST, "prompt_format_version": 2})
        self.assertEqual(status, 400)
        message = body["error"]["message"]
        self.assertIn("prompt_format_version 2", message)
        self.assertIn("uses version 1", message)

    def test_the_strict_route_refuses_what_it_cannot_serve(self):
        wide = {"id": "wide", "type": "choice", "question": "Which one?",
                "options": [{"name": f"opt{i}"} for i in range(27)]}
        cases = {
            "unknown field": ({**REQUEST, "bogus": 1}, "bogus"),
            "blank input": ({**REQUEST, "input": " "}, "input"),
            "repeated question id": (
                {**REQUEST, "questions": [REQUEST["questions"][0],
                                          dict(REQUEST["questions"][0])]},
                "repeats"),
            "27 options": ({"input": INPUT, "questions": [wide]}, "at most 26"),
        }
        for name, (payload, fragment) in cases.items():
            with self.subTest(case=name):
                status, body = self.api.post("/v1/decisions", payload)
                self.assertEqual(status, 400)
                self.assertEqual(set(body), {"error"})
                self.assertEqual(body["error"]["code"], 400)
                self.assertIn(fragment, body["error"]["message"])

    def test_the_adapter_model_form_is_refused_on_both_routes(self):
        cases = (
            ("/v1/decisions", {**REQUEST, "model": "base:adapter"}),
            ("/v1/systemone",
             {"state": "S", "model": "base:adapter",
              "questions": {"q": {"type": "noul", "instructions": "Is this ok?"}}}),
        )
        for path, payload in cases:
            with self.subTest(route=path):
                status, body = self.api.post(path, payload)
                self.assertEqual(status, 400)
                self.assertEqual(set(body), {"error"})
                self.assertIn("model names the LoRA adapter 'adapter'",
                              body["error"]["message"])


class SystemOneTest(unittest.TestCase):
    """POST /v1/systemone: the published shapes, on the same mock."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.api = Api(THINKING)
        cls.api.engine.runtime.scripted_label_logprobs = dict(SCRIPTED)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.api.close()

    def test_the_served_model_answers_under_its_own_name(self):
        status, body = self.api.post("/v1/systemone", SYSTEMONE_REQUEST)
        self.assertEqual(status, 200)
        self.assertEqual(set(body), {"model", "answers", "usage"})
        self.assertEqual(body["model"], "mockup-thinking")  # not "jev-latest"
        self.assertEqual(list(body["answers"]), ["spam", "route", "urgency"])

        yes_no = [math.exp(lp) for lp in (-0.2, -1.8)]
        yes_total = sum(yes_no)
        spam = body["answers"]["spam"]
        self.assertEqual(set(spam), {"type", "noul", "x_label_mass"})
        self.assertEqual(spam["type"], "noul")
        self.assertAlmostEqual(spam["noul"], yes_no[0] / yes_total)
        self.assertAlmostEqual(spam["x_label_mass"], yes_total)

        choice = [math.exp(lp) for lp in (-0.5, -1.5)]
        choice_total = sum(choice)
        route = body["answers"]["route"]
        self.assertEqual(set(route), {"type", "choice", "confidence",
                                      "probabilities", "x_label_mass"})
        self.assertEqual(route["type"], "choice")
        self.assertEqual(route["choice"], "billing")
        self.assertAlmostEqual(route["probabilities"]["billing"],
                               choice[0] / choice_total)
        self.assertAlmostEqual(route["probabilities"]["tech"],
                               choice[1] / choice_total)
        self.assertAlmostEqual(route["x_label_mass"], choice_total)
        # confidence over the renormalized labels: (n*max(q) - 1) / (n - 1)
        q = [weight / choice_total for weight in choice]
        self.assertAlmostEqual(route["confidence"],
                               min(1.0, max(0.0, (2 * max(q) - 1) / 1)))

        score = [math.exp(lp) for lp in (-3.0, -1.0, -2.0)]
        score_total = sum(score)
        urgency = body["answers"]["urgency"]
        self.assertEqual(set(urgency), {"type", "score", "confidence", "legend",
                                        "probabilities", "x_label_mass"})
        self.assertEqual(urgency["type"], "score")
        self.assertEqual(urgency["legend"],
                         {"0": "low", "1": "high", "2": "critical"})
        self.assertAlmostEqual(urgency["x_label_mass"], score_total)
        q = [weight / score_total for weight in score]
        self.assertAlmostEqual(urgency["score"],
                               math.fsum(i * p for i, p in enumerate(q)))
        # confidence from the spread around the top level, against a uniform
        top = q.index(max(q))
        spread = math.fsum(p * abs(i - top) for i, p in enumerate(q))
        uniform = math.fsum(abs(i - (3 - 1) / 2) for i in range(3)) / 3
        self.assertAlmostEqual(urgency["confidence"],
                               max(0.0, 1 - spread / uniform))

        expected = sum(rendered_tokens(self.api, text) for text in
                       (SO_NAUL_TEXT, SO_CHOICE_TEXT, SO_SCORE_TEXT))
        self.assertEqual(body["usage"], {"input_tokens": expected,
                                         "output_tokens": 0})

    def test_shape_errors_come_back_as_detail_lists(self):
        questions = {"q": {"type": "noul", "instructions": "Is this ok?"}}
        cases = {
            "temperature": (
                {"state": "S", "model": "m", "questions": questions,
                 "temperature": 0.5},
                "temperature"),
            "unknown key in a question": (
                {"state": "S", "model": "m", "questions": {
                    "q": {"type": "noul", "instructions": "Is this ok?",
                          "extra": 1}}},
                "extra"),
            "noul with nothing to decide": (
                {"state": "S", "model": "m", "questions": {"q": {"type": "noul"}}},
                "instructions"),
        }
        for name, (payload, fragment) in cases.items():
            with self.subTest(case=name):
                status, body = self.api.post("/v1/systemone", payload)
                self.assertEqual(status, 422)
                self.assertEqual(set(body), {"detail"})
                self.assertTrue(body["detail"])
                for entry in body["detail"]:
                    self.assertEqual(set(entry), {"loc", "msg", "type"})
                self.assertIn(fragment, json.dumps(body["detail"]))

    def test_an_empty_state_and_a_blank_question_id_are_served(self):
        status, body = self.api.post("/v1/systemone", {
            "state": "", "model": "jev-latest",
            "questions": {"": {"type": "noul", "instructions": "Is this ok?"}},
        })
        self.assertEqual(status, 200)
        self.assertEqual(list(body["answers"]), [""])
        answer = body["answers"][""]
        self.assertEqual(set(answer), {"type", "noul", "noul", "x_label_mass"})
        self.assertEqual(body["usage"]["output_tokens"], 0)

    def test_an_unknown_top_level_field_is_dropped(self):
        plain = {"state": "S", "model": "jev-latest",
                 "questions": {"q": {"type": "noul", "instructions": "Is this ok?"}}}
        status, first = self.api.post("/v1/systemone", plain)
        self.assertEqual(status, 200)
        status, second = self.api.post("/v1/systemone",
                                       {**plain, "mystery": {"x": 1}})
        self.assertEqual(status, 200)
        self.assertEqual(second["answers"], first["answers"])
        self.assertEqual(second["usage"], first["usage"])
        self.assertEqual(second["model"], first["model"])

    def test_levels_the_legend_cannot_return_are_refused_before_scoring(self):
        runtime = self.api.engine.runtime
        before = len(runtime.chats)
        status, body = self.api.post("/v1/systemone", {
            "state": "S", "model": "jev-latest",
            "questions": {"mood": {
                "type": "score", "instructions": "Mood?",
                # the NaN rides in as JSON's own NaN literal
                "criteria": ["Calm", {"label": float("nan")}]}},
        })
        self.assertEqual(status, 400)
        self.assertIn("question 'mood': a level cannot be returned in the legend",
                      body["error"]["message"])
        self.assertEqual(len(runtime.chats), before,
                         "the refusal comes before any render or scoring")


class CapabilityTest(unittest.TestCase):
    """Both routes refuse a package that declares no label_logprobs."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.api = Api(MOCKUP)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.api.close()

    def test_both_routes_name_the_capability_and_what_is_declared(self):
        cases = (
            ("/v1/decisions",
             {"input": "S", "questions": [
                 {"id": "q", "type": "yes_no", "question": "Is this ok?"}]}),
            ("/v1/systemone",
             {"state": "S", "model": "jev-latest",
              "questions": {"q": {"type": "noul", "instructions": "Is this ok?"}}}),
        )
        for path, payload in cases:
            with self.subTest(route=path):
                status, body = self.api.post(path, payload)
                self.assertEqual(status, 400)
                self.assertEqual(set(body), {"error"})
                message = body["error"]["message"]
                self.assertIn("label_logprobs", message)
                self.assertIn("nothing", message)


class GateTest(unittest.TestCase):
    """The per-request gates: reasoning block, context fit, labels, finiteness."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.api = Api(THINKING)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.api.close()

    def test_a_thinking_block_left_open_is_refused(self):
        runtime = self.api.engine.runtime
        original = runtime.chat

        def chat(messages, thinking=True, level=None):
            # a thinking-off render that leaves the open marker at the tail
            return original(messages, thinking, level=level)[:-1]

        runtime.chat = chat
        self.addCleanup(setattr, runtime, "chat", original)
        status, body = self.api.post("/v1/decisions", {
            "input": INPUT,
            "questions": [{"id": "first", "type": "yes_no",
                           "question": "Is this spam?"}]})
        self.assertEqual(status, 400)
        message = body["error"]["message"]
        self.assertIn("question 'first'", message)
        self.assertIn("leaves a reasoning block open", message)

    def test_a_prompt_over_the_context_is_refused_whole(self):
        status, body = self.api.post("/v1/decisions", {
            "input": "x" * 300,
            "questions": [{"id": "big", "type": "yes_no",
                           "question": "too long?"}]})
        self.assertEqual(status, 400)
        message = body["error"]["message"]
        content = ("x" * 300 + "\n\nIs the following true? too long?"
                   "\nAnswer with yes or no only.")
        self.assertIn(f"question 'big': the prompt has "
                      f"{rendered_tokens(self.api, content)} tokens", message)
        self.assertIn(
            f"context length of {self.api.engine.facts.max_context} tokens", message)

    def test_a_label_the_verb_refuses_names_the_question(self):
        runtime = self.api.engine.runtime
        original = runtime.label_logprobs

        def label_logprobs(input_tokens, labels):
            raise ValueError(
                "label 'Z' is not one distinct token at the answer position")

        runtime.label_logprobs = label_logprobs
        self.addCleanup(setattr, runtime, "label_logprobs", original)
        status, body = self.api.post("/v1/decisions", REQUEST)
        self.assertEqual(status, 400)
        message = body["error"]["message"]
        self.assertIn("question 'route'", message)
        self.assertIn("label 'Z' is not one distinct token", message)

    def test_a_refusal_in_a_later_question_names_that_question(self):
        """The questions are scored in one batch call; a label error there
        still names the question it came from (`ValueError.item`)."""
        runtime = self.api.engine.runtime
        original = runtime.label_logprobs
        calls = []

        def label_logprobs(input_tokens, labels):
            calls.append(1)
            if len(calls) == 2:
                raise ValueError("label 'Z' is not one distinct token at the answer position")
            return original(input_tokens, labels)

        runtime.label_logprobs = label_logprobs
        self.addCleanup(setattr, runtime, "label_logprobs", original)
        status, body = self.api.post("/v1/decisions", REQUEST)
        self.assertEqual(status, 400)
        self.assertIn("question 'urgency'", body["error"]["message"])

    def test_non_finite_scores_are_a_server_fault(self):
        runtime = self.api.engine.runtime
        original = runtime.label_logprobs

        def label_logprobs(input_tokens, labels):
            return [(300 + index, float("nan")) for index, _ in enumerate(labels)]

        runtime.label_logprobs = label_logprobs
        self.addCleanup(setattr, runtime, "label_logprobs", original)
        status, body = self.api.post("/v1/systemone", {
            "state": "S", "model": "jev-latest",
            "questions": {"q": {"type": "noul", "instructions": "Is this ok?"}}})
        self.assertEqual(status, 500)
        self.assertEqual(body["error"]["type"], "InternalServerError")
        self.assertIn("non-finite", body["error"]["message"])


class HeadCacheTest(unittest.TestCase):
    """With the batch verb, the head that depends on the input text alone goes
    through the prefix cache: a later request about the same text does not
    forward it again."""

    def setUp(self) -> None:
        self.api = Api(THINKING)
        self.addCleanup(self.api.close)
        runtime = self.api.engine.runtime
        self.baseline = self.api.post("/v1/decisions", REQUEST)[1]  # no batch verb yet
        self.fed, self.heads = [], []
        step = runtime.step

        def counted(batch, ctx):
            self.fed.append(sum(len(item.input_tokens) for item in batch))
            return step(batch, ctx)

        def label_logprobs_batch(items, state=None):
            self.heads.append(None if state is None else list(state.tokens))
            if state is not None:
                for input_tokens, _ in items:
                    self.assertEqual(list(input_tokens[:len(state.tokens)]), state.tokens)
            return [runtime.label_logprobs(t, labels) for t, labels in items]

        runtime.step = counted
        runtime.label_logprobs_batch = label_logprobs_batch
        self.addCleanup(delattr, runtime, "step")
        self.addCleanup(delattr, runtime, "label_logprobs_batch")
        self.head = self.api.engine.runtime.encode("user: " + INPUT + "\n")

    def test_the_text_head_is_forwarded_once_across_requests(self):
        status, body = self.api.post("/v1/decisions", REQUEST)
        self.assertEqual(status, 200)
        self.assertEqual(body["answers"], self.baseline["answers"])
        self.assertEqual(self.heads, [self.head])
        self.assertEqual(sum(self.fed), len(self.head))
        # other questions about the same input: the cached head, nothing forwarded
        self.fed.clear()
        status, _ = self.api.post("/v1/decisions", {
            "input": INPUT,
            "questions": [{"id": "rude", "type": "yes_no", "question": "Is this rude?"}]})
        self.assertEqual(status, 200)
        self.assertEqual(self.heads, [self.head, self.head])
        self.assertEqual(sum(self.fed), 0)

    def test_the_head_is_clamped_to_what_every_pair_shares(self):
        engine = self.api.engine
        a, b = [1, 2, 3, 4, 5], [1, 2, 9, 9]
        engine.label_logprobs_batch([(a, ["yes"]), (b, ["yes"])], shared=99)
        self.assertEqual(self.heads, [[1, 2]])
        engine.label_logprobs_batch([(a, ["yes"])], shared=99)
        self.assertEqual(self.heads[-1], [1, 2, 3, 4])   # its answer token stays a tail


class CancellationTest(unittest.TestCase):
    """A request cancelled between questions abandons quietly."""

    def test_a_cancelled_request_stops_between_questions(self):
        api = Api(THINKING)
        self.addCleanup(api.close)
        runtime = api.engine.runtime
        jobs: list = []
        submit = api.front.submit

        def capture(job):
            jobs.append(job)
            submit(job)

        api.front.submit = capture
        self.addCleanup(setattr, api.front, "submit", submit)
        calls: list = []
        original = runtime.label_logprobs

        def label_logprobs(input_tokens, labels):
            calls.append(labels)
            jobs[0].cancelled.set()  # the client is gone before the next question
            return original(input_tokens, labels)

        runtime.label_logprobs = label_logprobs
        self.addCleanup(setattr, runtime, "label_logprobs", original)
        payload = json.dumps({
            "input": INPUT,
            "questions": [
                {"id": "first", "type": "yes_no", "question": "Is this spam?"},
                {"id": "second", "type": "yes_no", "question": "Is this rude?"},
            ],
        }).encode()
        split = urlsplit(api.base)
        sock = socket.create_connection((split.hostname, split.port))
        self.addCleanup(sock.close)
        sock.sendall(b"POST /v1/decisions HTTP/1.1\r\n"
                     b"Host: localhost\r\nContent-Type: application/json\r\n"
                     + f"Content-Length: {len(payload)}\r\n\r\n".encode() + payload)
        deadline = time.time() + 5  # the first question must run before the leave
        while not calls and time.time() < deadline:
            time.sleep(0.01)
        self.assertTrue(calls, "the first question was scored")
        sock.close()
        # requests run one at a time: this one answers only once the cancelled
        # request is off the queue, so the counts below are settled
        status, _ = api.post("/v1/chat/completions",
                             {"messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(status, 200)
        self.assertEqual(len(calls), 1, "the second question is never scored")
        self.assertTrue(jobs[0].events.empty(),
                        "an abandoned request pushes nothing")


if __name__ == "__main__":
    unittest.main()
