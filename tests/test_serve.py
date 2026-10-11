"""The serving API over the scripted mocks: routes, wire shapes, streams.

Every request here is a real HTTP round trip against `footless serve`'s server,
on the mock packages, so the suite runs on nothing but the standard library.
"""

from __future__ import annotations

import json
import threading
import unittest
import urllib.error
import urllib.request
from dataclasses import replace
from pathlib import Path

from footless import serve as serving
from footless import wire
from footless.engine import Engine
from footless.sdk import ContractError, SamplingSpec

ROOT = Path(__file__).resolve().parent
MOCKUP = ROOT / "mockup"
THINKING = ROOT / "mockup-thinking"

# the mocks script their output: byte tokens over a fixed sentence
SCRIPT_TEXT = "the mock model speaks"
CHAT_PROMPT_TOKENS = len("user: hi\nassistant: ")


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

    def get(self, path: str, headers=None):
        status, raw = self.request("GET", path, headers=headers)
        return status, json.loads(raw) if raw else None

    def post(self, path: str, body, headers=None):
        status, raw = self.request("POST", path, body, headers)
        return status, json.loads(raw) if raw else None

    def stream(self, path: str, body, headers=None):
        """The SSE payloads in order, with None where [DONE] closes the stream."""
        status, raw = self.request("POST", path, body, headers)
        events = []
        for line in raw.decode().splitlines():
            if line.startswith("data: "):
                payload = line[6:]
                events.append(None if payload == "[DONE]" else json.loads(payload))
        return status, events


class ServeTest(unittest.TestCase):
    """POST /v1/chat/completions and /v1/completions on the plain mock."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.api = Api(MOCKUP)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.api.close()

    def chat(self, **body):
        body.setdefault("messages", [{"role": "user", "content": "hi"}])
        return self.api.post("/v1/chat/completions", body)

    # -- routes without generation ----------------------------------------

    def test_health(self):
        status, _ = self.api.request("GET", "/health")
        self.assertEqual(status, 200)

    def test_models_list_the_loaded_package(self):
        status, body = self.api.get("/v1/models")
        self.assertEqual(status, 200)
        self.assertEqual(body["object"], "list")
        (card,) = body["data"]
        self.assertEqual(card["id"], "mockup")
        self.assertEqual(card["object"], "model")
        self.assertEqual(card["owned_by"], "footless")
        self.assertEqual(card["max_model_len"], 256)
        self.assertTrue(card["permission"][0]["allow_sampling"])

    def test_a_model_card_comes_back_by_name(self):
        status, card = self.api.get("/v1/models/mockup")
        self.assertEqual(status, 200)
        self.assertEqual(card["id"], "mockup")

    def test_an_unknown_model_is_a_404_with_the_openai_error_shape(self):
        status, body = self.api.get("/v1/models/nope")
        self.assertEqual(status, 404)
        self.assertEqual(body["error"]["type"], "NotFoundError")
        status, body = self.chat(model="nope")
        self.assertEqual(status, 404)
        self.assertIn("nope", body["error"]["message"])

    def test_an_unknown_path_is_a_404(self):
        status, body = self.api.post("/v1/embeddings", {"input": "x"})
        self.assertEqual(status, 404)
        self.assertEqual(body["error"]["type"], "NotFoundError")

    # -- chat -------------------------------------------------------------

    def test_chat_completions(self):
        status, body = self.chat()
        self.assertEqual(status, 200)
        self.assertEqual(body["object"], "chat.completion")
        self.assertTrue(body["id"].startswith("chatcmpl-"))
        self.assertEqual(body["model"], "mockup")
        (choice,) = body["choices"]
        self.assertEqual(choice["index"], 0)
        self.assertEqual(choice["message"]["role"], "assistant")
        self.assertEqual(choice["message"]["content"], SCRIPT_TEXT)
        self.assertEqual(choice["finish_reason"], "stop")
        self.assertNotIn("stop_reason", choice)
        usage = body["usage"]
        self.assertEqual(usage["prompt_tokens"], CHAT_PROMPT_TOKENS)
        self.assertEqual(usage["completion_tokens"], len(SCRIPT_TEXT))
        self.assertEqual(usage["total_tokens"],
                         CHAT_PROMPT_TOKENS + len(SCRIPT_TEXT))
        self.assertEqual(usage["completion_tokens_details"], {"reasoning_tokens": 0})
        self.assertIn("cached_tokens", usage["prompt_tokens_details"])

    def test_chat_takes_text_content_parts(self):
        status, body = self.chat(messages=[{"role": "user", "content": [
            {"type": "text", "text": "h"}, {"type": "text", "text": "i"}]}])
        self.assertEqual(status, 200)
        self.assertEqual(body["usage"]["prompt_tokens"], CHAT_PROMPT_TOKENS)

    def test_max_completion_tokens_caps_the_output(self):
        status, body = self.chat(max_completion_tokens=5)
        choice = body["choices"][0]
        self.assertEqual(choice["message"]["content"], "the m")
        self.assertEqual(choice["finish_reason"], "length")

    def test_max_tokens_zero_generates_nothing(self):
        status, body = self.chat(max_tokens=0)
        choice = body["choices"][0]
        self.assertEqual(choice["message"]["content"], "")
        self.assertEqual(choice["finish_reason"], "length")
        self.assertEqual(body["usage"]["completion_tokens"], 0)

    def test_n_choices_are_generated_independently(self):
        status, body = self.chat(n=2)
        self.assertEqual([c["index"] for c in body["choices"]], [0, 1])
        self.assertEqual(body["choices"][1]["message"]["content"], SCRIPT_TEXT)
        self.assertEqual(body["usage"]["completion_tokens"], 2 * len(SCRIPT_TEXT))

    def test_a_stop_string_trims_and_is_reported(self):
        status, body = self.chat(stop="mock")
        choice = body["choices"][0]
        self.assertEqual(choice["message"]["content"], "the ")
        self.assertEqual(choice["finish_reason"], "stop")
        self.assertEqual(choice["stop_reason"], "mock")

    def test_a_stop_string_can_stay_in_the_output(self):
        status, body = self.chat(stop="mock", include_stop_str_in_output=True)
        self.assertEqual(body["choices"][0]["message"]["content"], "the mock")

    def test_stop_token_ids_trim_and_are_reported(self):
        status, body = self.chat(stop_token_ids=[ord("m")])
        choice = body["choices"][0]
        self.assertEqual(choice["message"]["content"], "the ")
        self.assertEqual(choice["stop_reason"], ord("m"))

    def test_a_reused_prompt_is_reported_as_cached(self):
        self.chat()
        status, body = self.chat()
        self.assertEqual(body["usage"]["prompt_tokens"], CHAT_PROMPT_TOKENS)
        self.assertEqual(body["usage"]["prompt_tokens_details"]["cached_tokens"],
                         CHAT_PROMPT_TOKENS)

    def test_return_token_ids_reports_the_ids(self):
        status, body = self.chat(return_token_ids=True)
        choice = body["choices"][0]
        self.assertEqual(choice["token_ids"],
                         self.api.engine.runtime.encode(SCRIPT_TEXT))
        self.assertEqual(body["prompt_token_ids"],
                         self.api.engine.runtime.encode("user: hi\nassistant: "))

    # -- completions ------------------------------------------------------

    def test_completions(self):
        status, body = self.api.post("/v1/completions", {"prompt": "hello",
                                                        "max_tokens": 64})
        self.assertEqual(status, 200)
        self.assertEqual(body["object"], "text_completion")
        self.assertTrue(body["id"].startswith("cmpl-"))
        (choice,) = body["choices"]
        self.assertEqual(choice["text"], SCRIPT_TEXT)
        self.assertEqual(choice["finish_reason"], "stop")
        self.assertEqual(body["usage"]["prompt_tokens"], 5)

    def test_completions_default_is_sixteen_tokens(self):
        status, body = self.api.post("/v1/completions", {"prompt": "hello"})
        self.assertEqual(len(body["choices"][0]["text"]), 16)
        self.assertEqual(body["choices"][0]["finish_reason"], "length")

    def test_completions_echo_prepends_the_prompt(self):
        status, body = self.api.post("/v1/completions", {"prompt": "hello",
                                                        "echo": True, "max_tokens": 64})
        self.assertEqual(body["choices"][0]["text"], "hello" + SCRIPT_TEXT)

    def test_completions_accept_token_ids(self):
        status, body = self.api.post("/v1/completions", {"prompt": [1, 2, 3]})
        self.assertEqual(body["usage"]["prompt_tokens"], 3)

    def test_completions_accept_several_prompts_times_n(self):
        status, body = self.api.post("/v1/completions", {"prompt": ["a", "b"], "n": 2,
                                                        "max_tokens": 64})
        self.assertEqual([c["index"] for c in body["choices"]], [0, 1, 2, 3])
        self.assertEqual(body["usage"]["prompt_tokens"], 2)
        self.assertEqual(body["usage"]["completion_tokens"], 4 * len(SCRIPT_TEXT))

    def test_completions_accept_token_id_prompts(self):
        status, body = self.api.post("/v1/completions", {"prompt": [[1], [2, 3]]})
        self.assertEqual(len(body["choices"]), 2)
        self.assertEqual(body["usage"]["prompt_tokens"], 3)

    # -- streaming --------------------------------------------------------

    def test_streaming_chat(self):
        status, events = self.api.stream("/v1/chat/completions",
                                        {"messages": [{"role": "user", "content": "hi"}],
                                         "stream": True})
        self.assertEqual(status, 200)
        self.assertIsNone(events[-1], "the stream closes with [DONE]")
        chunks = events[:-1]
        self.assertEqual(chunks[0]["choices"][0]["delta"],
                         {"role": "assistant", "content": ""})
        self.assertEqual(chunks[0]["object"], "chat.completion.chunk")
        text = "".join(chunk["choices"][0]["delta"].get("content") or ""
                       for chunk in chunks)
        self.assertEqual(text, SCRIPT_TEXT)
        final = chunks[-1]["choices"][0]
        self.assertEqual(final["finish_reason"], "stop")
        self.assertEqual(final["delta"], {})
        self.assertNotIn("usage", chunks[0])

    def test_streaming_can_carry_usage(self):
        status, events = self.api.stream(
            "/v1/chat/completions",
            {"messages": [{"role": "user", "content": "hi"}], "stream": True,
             "stream_options": {"include_usage": True}})
        chunks = events[:-1]
        last = chunks[-1]
        self.assertEqual(last["choices"], [])
        self.assertEqual(last["usage"]["completion_tokens"], len(SCRIPT_TEXT))
        self.assertEqual(last["usage"]["prompt_tokens"], CHAT_PROMPT_TOKENS)

    def test_continuous_usage_stats_ride_every_chunk(self):
        status, events = self.api.stream(
            "/v1/chat/completions",
            {"messages": [{"role": "user", "content": "hi"}], "stream": True,
             "stream_options": {"include_usage": True,
                                "continuous_usage_stats": True}})
        chunks = events[:-1]
        self.assertTrue(all("usage" in chunk for chunk in chunks))
        self.assertEqual(chunks[-1]["usage"]["completion_tokens"], len(SCRIPT_TEXT))

    def test_streaming_completions(self):
        status, events = self.api.stream("/v1/completions",
                                        {"prompt": "hi", "stream": True,
                                         "max_tokens": 64})
        chunks = events[:-1]
        self.assertEqual(chunks[0]["choices"][0]["text"], "")
        text = "".join(chunk["choices"][0]["text"] for chunk in chunks)
        self.assertEqual(text, SCRIPT_TEXT)
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "stop")
        self.assertIsNone(events[-1])

    def test_streaming_completions_echo_rides_the_first_chunk(self):
        status, events = self.api.stream("/v1/completions",
                                        {"prompt": "hi", "stream": True, "echo": True})
        chunks = events[:-1]
        self.assertEqual(chunks[0]["choices"][0]["text"], "hi")

    def test_a_streaming_error_is_answered_before_the_stream_starts(self):
        status, body = self.api.post("/v1/chat/completions",
                                    {"messages": [{"role": "user", "content": "x" * 300}],
                                     "stream": True})
        self.assertEqual(status, 400)
        self.assertIn("Input length", body["error"]["message"])

    def test_a_client_that_leaves_mid_stream_does_not_break_the_server(self):
        import socket
        from urllib.parse import urlsplit

        payload = json.dumps({"messages": [{"role": "user", "content": "hi"}],
                              "stream": True, "max_tokens": 64}).encode()
        split = urlsplit(self.api.base)
        sock = socket.create_connection((split.hostname, split.port))
        sock.sendall(b"POST /v1/chat/completions HTTP/1.1\r\n"
                     b"Host: localhost\r\nContent-Type: application/json\r\n"
                     + f"Content-Length: {len(payload)}\r\n\r\n".encode() + payload)
        sock.close()
        status, body = self.chat()
        self.assertEqual(status, 200)
        self.assertEqual(body["choices"][0]["message"]["content"], SCRIPT_TEXT)

    def test_a_refused_post_leaves_its_connection_clean(self):
        """A 404 on a kept-alive connection reads its body: the next request
        on that connection is read from its own first line."""
        import http.client
        from urllib.parse import urlsplit

        split = urlsplit(self.api.base)
        conn = http.client.HTTPConnection(split.hostname, split.port, timeout=30)
        self.addCleanup(conn.close)
        head = {"Content-Type": "application/json"}
        conn.request("POST", "/v1/embeddings", json.dumps({"input": "test"}), head)
        response = conn.getresponse()
        response.read()
        self.assertEqual(response.status, 404)
        conn.request("POST", "/v1/chat/completions",
                     json.dumps({"messages": [{"role": "user", "content": "hi"}]}), head)
        response = conn.getresponse()
        body = json.loads(response.read())
        self.assertEqual(response.status, 200)
        self.assertEqual(body["choices"][0]["message"]["content"], SCRIPT_TEXT)

    # -- what the engine refuses ------------------------------------------

    def test_a_prompt_over_the_context_is_a_400(self):
        status, body = self.api.post("/v1/completions", {"prompt": "x" * 300})
        self.assertEqual(status, 400)
        self.assertIn("Input length (300)", body["error"]["message"])

    def test_truncate_prompt_tokens_fits_the_prompt(self):
        status, body = self.api.post("/v1/completions",
                                    {"prompt": "x" * 300, "truncate_prompt_tokens": 10})
        self.assertEqual(status, 200)
        self.assertEqual(body["usage"]["prompt_tokens"], 10)
        status, body = self.api.post(
            "/v1/completions",
            {"prompt": "x" * 300, "truncate_prompt_tokens": -1, "max_tokens": 1})
        self.assertEqual(status, 200)
        self.assertEqual(body["usage"]["prompt_tokens"], 256)

    def test_truncation_side_left_keeps_the_tail(self):
        status, body = self.api.post(
            "/v1/completions",
            {"prompt": "abcdef", "truncate_prompt_tokens": 2, "truncation_side": "left",
             "echo": True, "max_tokens": 0})
        self.assertEqual(status, 200)
        self.assertEqual(body["choices"][0]["text"], "ef")

    def test_max_tokens_is_capped_by_the_room_left(self):
        status, body = self.api.post("/v1/completions",
                                     {"prompt": "x" * 250, "max_tokens": 100})
        self.assertEqual(body["usage"]["completion_tokens"], 6)
        self.assertEqual(body["choices"][0]["finish_reason"], "length")

    # -- what the wire cannot serve is dropped, as in vLLM -----------------

    def test_unsupported_fields_are_ignored(self):
        cases = [
            ("tools", [{"type": "function", "function": {"name": "f"}}]),
            ("tool_choice", "auto"),
            ("logit_bias", {"5": 2.0}),
            ("response_format", {"type": "json_object"}),
            ("structured_outputs", {"json": "{}"}),
            ("reasoning_effort", "low"),
            ("thinking_token_budget", 128),
            ("add_special_tokens", True),
            ("skip_special_tokens", False),
            ("chat_template", "x"),
            ("min_tokens", 4),
            ("ignore_eos", True),
            ("use_beam_search", True),
            ("watermarking", True),
            ("best_of", 3),
            ("prompt_logprobs", 1),
            ("suffix", "x"),
            ("documents", [{"title": "t", "text": "x"}]),
            ("stream_interval", 2),
            ("priority", 5),
        ]
        for name, value in cases:
            with self.subTest(field=name):
                status, body = self.chat(max_tokens=4, **{name: value})
                self.assertEqual(status, 200, f"{name} refused")
                self.assertEqual(body["choices"][0]["message"]["content"], "the ")

    def test_neutral_values_of_unsupported_fields_pass(self):
        status, body = self.chat(frequency_penalty=0.0, tools=[], tool_choice="none",
                                 skip_special_tokens=True, watermarking=False,
                                 best_of=1, min_tokens=0)
        self.assertEqual(status, 200)

    def test_unknown_fields_are_ignored(self):
        status, body = self.chat(some_new_field={"x": 1})
        self.assertEqual(status, 200)

    def test_non_text_content_parts_are_dropped(self):
        status, body = self.chat(messages=[{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "http://x"}},
            {"type": "text", "text": "hi"}]}])
        self.assertEqual(status, 200)
        self.assertEqual(body["usage"]["prompt_tokens"], CHAT_PROMPT_TOKENS)

    def test_any_message_role_reaches_the_template(self):
        status, body = self.chat(messages=[{"role": "oracle", "content": "hi"}])
        self.assertEqual(status, 200)
        self.assertEqual(body["usage"]["prompt_tokens"],
                         len("oracle: hi\nassistant: "))

    def test_a_conversation_the_template_refuses_is_a_400(self):
        """The package's template raising ValueError on the messages is the
        request's fault: 400, as vLLM answers it, not a server error."""
        runtime = self.api.engine.runtime
        chat = runtime.chat

        def refusing(messages, *args, **kwargs):
            raise ValueError("System message must be at the beginning.")
        runtime.chat = refusing
        try:
            status, body = self.chat()
        finally:
            runtime.chat = chat
        self.assertEqual(status, 400)
        self.assertIn("System message", body["error"]["message"])

    def test_developer_messages_are_system_messages(self):
        status, body = self.chat(messages=[
            {"role": "developer", "content": "sys"},
            {"role": "user", "content": "hi"}])
        self.assertEqual(status, 200)
        self.assertEqual(body["usage"]["prompt_tokens"],
                         CHAT_PROMPT_TOKENS + len("system: sys\n"))

    def test_seed_is_dropped_when_there_is_more_than_one_choice(self):
        """One seed is one draw stream; several choices get fresh ones."""
        status, body = self.chat(seed=7, n=2)
        self.assertEqual(status, 200)
        self.assertEqual(len(body["choices"]), 2)
        status, body = self.api.post("/v1/completions",
                                     {"prompt": ["a", "b"], "seed": 7})
        self.assertEqual(status, 200)
        self.assertEqual(len(body["choices"]), 2)
        status, body = self.chat(seed=7)  # one choice keeps its stream
        self.assertEqual(status, 200)

    def test_stream_options_need_stream(self):
        status, body = self.chat(stream_options={"include_usage": True})
        self.assertEqual(status, 400)
        self.assertEqual(body["error"]["param"], "stream_options")

    def test_only_json_bodies_are_accepted(self):
        status, raw = self.api.request(
            "POST", "/v1/chat/completions", b"not json",
            {"Content-Type": "text/plain"})
        self.assertEqual(status, 400)
        self.assertIn("application/json", json.loads(raw)["error"]["message"])

    def test_a_broken_json_body_is_a_400(self):
        status, raw = self.api.request(
            "POST", "/v1/chat/completions", b"{oops",
            {"Content-Type": "application/json"})
        self.assertEqual(status, 400)
        self.assertIn("Invalid JSON", json.loads(raw)["error"]["message"])

    def test_logprobs_come_back_empty_without_the_capability(self):
        """The package declares no logprobs capability: the shape is served,
        the steps report nothing, and the fields say so."""
        status, body = self.chat(logprobs=True, top_logprobs=2, max_tokens=3)
        self.assertEqual(status, 200)
        rows = body["choices"][0]["logprobs"]["content"]
        self.assertEqual([row["token"] for row in rows], ["t", "h", "e"])
        self.assertEqual(rows[0]["logprob"], -9999.0)
        self.assertEqual(rows[0]["top_logprobs"], [])
        status, body = self.api.post("/v1/completions",
                                     {"prompt": "hi", "logprobs": 2, "max_tokens": 2})
        logprobs = body["choices"][0]["logprobs"]
        self.assertEqual(logprobs["tokens"], ["t", "h"])
        self.assertEqual(logprobs["token_logprobs"], [None, None])
        self.assertEqual(logprobs["top_logprobs"], [None, None])


class ApiKeyTest(unittest.TestCase):
    def test_the_api_key_guards_the_v1_routes(self):
        api = Api(MOCKUP, api_key="sesame")
        try:
            status, body = api.get("/v1/models")
            self.assertEqual(status, 401)
            self.assertEqual(body, {"error": "Unauthorized"})
            status, _ = api.get("/health")
            self.assertEqual(status, 200, "health stays open")
            status, body = api.get("/v1/models", headers={"Authorization": "Bearer x"})
            self.assertEqual(status, 401)
            status, body = api.get("/v1/models",
                                   headers={"Authorization": "Bearer sesame"})
            self.assertEqual(status, 200)
        finally:
            api.close()

    def test_served_model_name_answers_for_a_foreign_name(self):
        api = Api(MOCKUP, model_name="gpt-something")
        try:
            status, body = api.get("/v1/models")
            self.assertEqual(body["data"][0]["id"], "gpt-something")
            status, body = api.post("/v1/chat/completions",
                                    {"model": "gpt-something",
                                     "messages": [{"role": "user", "content": "hi"}]})
            self.assertEqual(status, 200)
        finally:
            api.close()


class ThinkingTest(unittest.TestCase):
    """Thinking blocks and logprobs, on the mock that offers both."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.api = Api(THINKING)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.api.close()

    def chat(self, **body):
        body.setdefault("messages", [{"role": "user", "content": "hi"}])
        return self.api.post("/v1/chat/completions", body)

    def test_thinking_and_answer_come_back_separately(self):
        status, body = self.chat()
        message = body["choices"][0]["message"]
        self.assertEqual(message["reasoning"], "thinking hard")
        self.assertEqual(message["content"], "the answer")
        self.assertNotIn("<think", message["reasoning"])
        self.assertNotIn("<think", message["content"])
        usage = body["usage"]
        self.assertEqual(usage["completion_tokens"], 24)  # the closing marker counts
        self.assertEqual(usage["completion_tokens_details"]["reasoning_tokens"], 13)

    def test_include_reasoning_false_hides_the_thinking(self):
        status, body = self.chat(include_reasoning=False)
        message = body["choices"][0]["message"]
        self.assertNotIn("reasoning", message)
        self.assertEqual(message["content"], "the answer")
        self.assertEqual(body["usage"]["completion_tokens_details"]["reasoning_tokens"],
                         13)

    def test_reasoning_effort_none_turns_thinking_off(self):
        """With thinking off the template closes the block, so the answer is
        content: a front end that trusted the thinking mode's fact would call
        it reasoning and leave `content` empty."""
        status, body = self.chat(reasoning_effort="none")
        self.assertEqual(status, 200)
        message = body["choices"][0]["message"]
        self.assertEqual(message["content"], "the answer")
        self.assertNotIn("reasoning", message)
        self.assertEqual(body["usage"]["prompt_tokens"],
                         CHAT_PROMPT_TOKENS + 2)  # the closed, empty block
        self.assertEqual(body["usage"]["completion_tokens_details"]["reasoning_tokens"],
                         0)
        self.assertEqual(self.api.engine.runtime.chats[-1]["thinking"], False)

    def test_a_reasoning_level_reaches_the_template(self):
        for level in ("xhigh", "medium", "low"):
            with self.subTest(level=level):
                status, body = self.chat(reasoning_effort=level, max_tokens=2)
                self.assertEqual(status, 200)
                self.assertEqual(self.api.engine.runtime.chats[-1]["level"], level)
                self.assertEqual(body["usage"]["prompt_tokens"],
                                 CHAT_PROMPT_TOKENS + 1 + len(f"[{level}]"))

    def test_no_level_leaves_the_package_its_default(self):
        status, body = self.chat()
        self.assertIsNone(self.api.engine.runtime.chats[-1]["level"])
        self.assertEqual(body["usage"]["prompt_tokens"], CHAT_PROMPT_TOKENS + 1)

    def test_a_level_the_package_does_not_offer_is_a_400(self):
        status, body = self.chat(reasoning_effort="high")
        self.assertEqual(status, 400)
        self.assertEqual(body["error"]["param"], "reasoning_effort")
        self.assertIn("xhigh", body["error"]["message"])

    def test_chat_logprobs_follow_the_generated_tokens(self):
        status, body = self.chat(logprobs=True, top_logprobs=2)
        rows = body["choices"][0]["logprobs"]["content"]
        self.assertEqual(len(rows), 23)  # every rendered token but the markers
        self.assertEqual(rows[0]["token"], "t")
        self.assertEqual(rows[0]["bytes"], [ord("t")])
        self.assertEqual(rows[0]["logprob"], -0.25)
        self.assertEqual(len(rows[0]["top_logprobs"]), 2)
        self.assertEqual(rows[0]["top_logprobs"][0]["token"], "t")
        self.assertEqual(rows[-1]["token"], "r")  # the answer's last character

    def test_completions_logprobs_use_the_offsets_of_the_text(self):
        status, body = self.api.post("/v1/completions",
                                     {"prompt": "go", "logprobs": 1, "max_tokens": 3})
        choice = body["choices"][0]
        self.assertEqual(choice["text"], "the")  # a raw prompt starts outside a block
        self.assertEqual(choice["logprobs"]["tokens"], ["t", "h", "e"])
        self.assertEqual(choice["logprobs"]["text_offset"], [0, 1, 2])
        self.assertEqual(choice["logprobs"]["token_logprobs"], [-0.25, -0.25, -0.25])
        self.assertEqual(choice["logprobs"]["top_logprobs"], [{"t": -0.25},
                                                              {"h": -0.25},
                                                              {"e": -0.25}])

    def test_streaming_separates_the_thinking_from_the_answer(self):
        status, events = self.api.stream("/v1/chat/completions",
                                        {"messages": [{"role": "user", "content": "hi"}],
                                         "stream": True})
        chunks = events[:-1]
        reasoning = "".join(chunk["choices"][0]["delta"].get("reasoning") or ""
                            for chunk in chunks)
        content = "".join(chunk["choices"][0]["delta"].get("content") or ""
                          for chunk in chunks)
        self.assertEqual(reasoning, "thinking hard")
        self.assertEqual(content, "the answer")

    def test_streaming_chunks_carry_logprobs(self):
        status, events = self.api.stream(
            "/v1/chat/completions",
            {"messages": [{"role": "user", "content": "hi"}], "stream": True,
             "logprobs": True, "top_logprobs": 1, "max_completion_tokens": 4})
        chunks = events[:-1]
        rows = [row for chunk in chunks
                for row in (chunk["choices"][0].get("logprobs") or {}).get("content", [])]
        self.assertEqual([row["token"] for row in rows], ["t", "h", "i", "n"])
        self.assertEqual(rows[0]["top_logprobs"][0]["token"], "t")

    def test_a_stop_string_trims_inside_the_thinking(self):
        status, body = self.chat(stop="hard", include_stop_str_in_output=True)
        message = body["choices"][0]["message"]
        self.assertEqual(message["reasoning"], "thinking hard")
        self.assertEqual(message["content"], "")
        self.assertEqual(body["choices"][0]["stop_reason"], "hard")


class ThinkingLevelsTest(unittest.TestCase):
    """What a package declares about reasoning levels is verified, not assumed."""

    def test_a_package_declares_the_levels_its_template_offers(self):
        self.assertEqual(Engine.open(THINKING).model.thinking_levels,
                         ["xhigh", "medium", "low"])
        engine = Engine.open(THINKING)
        try:
            self.assertEqual(engine.facts.thinking_levels, ["xhigh", "medium", "low"])
        finally:
            engine.close()

    def test_a_package_without_levels_declares_none(self):
        engine = Engine.open(MOCKUP)
        try:
            self.assertEqual(engine.model.thinking_levels, [])
            self.assertEqual(engine.facts.thinking_levels, [])
        finally:
            engine.close()

    def test_declared_and_reported_levels_must_agree(self):
        from footless import manifest as manifest_module

        engine = Engine.open(THINKING)
        self.addCleanup(engine.close)
        declared = engine.model
        bad = manifest_module.Manifest(
            name=declared.name, backends=declared.backends,
            max_batch=declared.max_batch, max_context=declared.max_context,
            cache_bytes_per_token=declared.cache_bytes_per_token,
            capabilities=declared.capabilities, thinking_levels=["low"],
        )
        self.assertRaises(manifest_module.ContractError, manifest_module.verify,
                          bad, engine.facts)

    def test_the_cli_flag_takes_a_declared_level_only(self):
        from footless import cli

        engine = Engine.open(THINKING)
        self.addCleanup(engine.close)
        self.assertEqual(cli._reasoning_level(engine, "low", True), "low")
        self.assertIsNone(cli._reasoning_level(engine, None, True))
        with self.assertRaises(ContractError):
            cli._reasoning_level(engine, "high", True)
        with self.assertRaises(ContractError):
            cli._reasoning_level(engine, "low", False)  # --no-thinking disagrees


class StepHookTest(unittest.TestCase):
    """The engine's front-end hooks, on the plain mock."""

    def setUp(self) -> None:
        self.engine = Engine.open(MOCKUP)
        self.addCleanup(self.engine.close)

    def test_on_step_can_end_the_generation(self):
        seen = []

        def on_step(step, timing):
            seen.append(step.token)
            return len(seen) >= 3

        result = self.engine.generate_tokens(
            self.engine.runtime.encode("hello"), 64, on_step=on_step)
        self.assertEqual(len(result.tokens), 3)
        self.assertIs(result.finished, True)

    def test_a_cancelled_context_stops_the_request(self):
        cancelled = threading.Event()

        class Ctx:
            def cancelled(self) -> bool:
                return cancelled.is_set()

        def on_step(step, timing):
            cancelled.set()
            return False

        result = self.engine.generate_tokens(
            self.engine.runtime.encode("hello"), 64, on_step=on_step, ctx=Ctx())
        self.assertEqual(len(result.tokens), 1, "the model sees the cancellation")

    def test_max_tokens_zero_still_feeds_the_whole_prompt(self):
        """An early end leaves no half-fed state behind in the prefix cache."""
        prompt = self.engine.runtime.encode("hello")
        self.engine.generate_tokens(prompt, 0)
        result = self.engine.generate_tokens(prompt, 4)
        self.assertLess(result.prompt_tokens_fed, len(prompt),
                        "the cached prefix is reused")


class DefaultSamplingTest(unittest.TestCase):
    """A field the request omits takes the package's own defaults.

    `_spec` is where a serving request's pins land, and they land over
    `Facts.default_sampling` (the checkpoint's generation config) -- over the
    engine's own spec only where a package declares none.
    """

    def setUp(self) -> None:
        self.api = Api(MOCKUP)
        self.addCleanup(self.api.close)
        self.api.engine.facts = replace(
            self.api.engine.facts,
            default_sampling=SamplingSpec(temperature=1.0, top_k=20, top_p=0.95),
        )

    def test_pinned_fields_go_over_the_package_defaults(self):
        """Pinning `temperature` must not smuggle in the engine's `top_k`
        where the package declares its own."""
        spec = self.api.front._spec(wire.Sampling(temperature=0.5))
        self.assertEqual((spec.temperature, spec.top_k, spec.top_p),
                         (0.5, 20, 0.95))

    def test_a_request_that_pins_nothing_keeps_them_all(self):
        spec = self.api.front._spec(wire.Sampling())
        self.assertEqual((spec.temperature, spec.top_k, spec.top_p),
                         (1.0, 20, 0.95))

    def test_a_request_pins_the_penalties_over_the_package(self):
        """A coding client turns the package's presence penalty off, as on vLLM."""
        self.api.engine.facts = replace(self.api.engine.facts, default_sampling=SamplingSpec(
            temperature=1.0, top_k=20, top_p=0.95, presence_penalty=1.5))
        request = wire.parse_chat({"messages": [{"role": "user", "content": "x"}],
                                   "presence_penalty": 0, "frequency_penalty": 0.25,
                                   "repetition_penalty": 1.1}, {"mockup"}, [])
        self.assertEqual(request.dropped, [])
        spec = self.api.front._spec(request.sampling)
        self.assertEqual((spec.presence_penalty, spec.frequency_penalty,
                          spec.repetition_penalty, spec.top_k), (0.0, 0.25, 1.1, 20))
        self.assertEqual(self.api.front._spec(wire.Sampling()).presence_penalty, 1.5)

    def test_penalties_out_of_vllms_ranges_are_refused(self):
        for field, value in (("presence_penalty", 2.5), ("frequency_penalty", -3),
                             ("repetition_penalty", 0)):
            with self.subTest(field=field):
                status, _ = self.api.post("/v1/chat/completions", {
                    "messages": [{"role": "user", "content": "x"}], field: value})
                self.assertEqual(status, 400)

    def test_a_package_that_declares_none_falls_back_to_the_engine(self):
        self.api.engine.facts = replace(self.api.engine.facts,
                                        default_sampling=None)
        self.assertEqual(self.api.front._spec(wire.Sampling()), SamplingSpec())


TOOLS = ROOT / "mockup-tools"
WEATHER = {"type": "function", "function": {
    "name": "get_weather", "parameters": {"type": "object",
                                          "properties": {"city": {"type": "string"}}}}}


class ToolsTest(unittest.TestCase):
    """Function tools, as vLLM serves them with a tool-call parser, on the mock
    that declares `tool_calls` (its calls are `[[NAME ARGS]]`)."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.api = Api(TOOLS)
        cls.runtime = cls.api.engine.runtime

    @classmethod
    def tearDownClass(cls) -> None:
        cls.api.close()

    def chat(self, script: str, **body):
        self.runtime.script = script
        body.setdefault("messages", [{"role": "user", "content": "hi"}])
        body.setdefault("tools", [WEATHER])
        return self.api.post("/v1/chat/completions", body)

    def stream(self, script: str, **body):
        self.runtime.script = script
        body.setdefault("messages", [{"role": "user", "content": "hi"}])
        body.setdefault("tools", [WEATHER])
        status, events = self.api.stream("/v1/chat/completions", dict(body, stream=True))
        self.assertEqual(status, 200)
        deltas = [e["choices"][0] for e in events if e and e["choices"]]
        content = "".join(d["delta"].get("content") or "" for d in deltas)
        calls = []  # put together as a client does: a call's first delta, then its arguments
        for piece in (c for d in deltas for c in d["delta"].get("tool_calls", [])):
            if piece["index"] == len(calls):
                calls.append(json.loads(json.dumps(piece)))
            else:
                self.assertEqual(set(piece), {"index", "function"})
                calls[piece["index"]]["function"]["arguments"] += piece["function"]["arguments"]
        return content, calls, deltas[-1]["finish_reason"]

    def test_a_call_comes_back_as_tool_calls(self):
        status, body = self.chat('Let me look.\n\n[[get_weather {"city": "Lugo"}]]')
        self.assertEqual(status, 200)
        choice = body["choices"][0]
        self.assertEqual(choice["finish_reason"], "tool_calls")
        message = choice["message"]
        self.assertEqual(message["content"], "Let me look.")  # trimmed around the call
        (call,) = message["tool_calls"]
        self.assertTrue(call["id"].startswith("chatcmpl-tool-"))
        self.assertEqual(call["type"], "function")
        self.assertEqual(call["function"], {"name": "get_weather",
                                            "arguments": '{"city": "Lugo"}'})

    def test_the_tools_reach_the_template_as_vllm_hands_them_over(self):
        self.chat("plain")
        tool = self.runtime.chats[-1]["tools"][0]
        self.assertEqual(tool, {"type": "function", "function": {
            "name": "get_weather", "description": None,
            "parameters": WEATHER["function"]["parameters"]}})

    def test_an_answer_without_calls_is_untouched(self):
        status, body = self.chat("  just text  ")
        choice = body["choices"][0]
        self.assertEqual(choice["finish_reason"], "stop")
        self.assertEqual(choice["message"]["content"], "  just text  ")
        self.assertNotIn("tool_calls", choice["message"])

    def test_only_whitespace_around_calls_leaves_no_content(self):
        status, body = self.chat("\n\n[[get_weather {}]]")
        self.assertIsNone(body["choices"][0]["message"]["content"])
        content, calls, finish = self.stream("\n\n[[get_weather {}]]")
        self.assertEqual(content, "")
        self.assertEqual(len(calls), 1)
        content, calls, finish = self.stream("\n\n[[get_weather {}]]after")
        self.assertEqual(content, "after", "the dropped whitespace stays dropped")

    def test_a_stream_carries_content_then_whole_calls(self):
        content, calls, finish = self.stream(
            'Let me look.\n\n[[get_weather {"city": "Lugo"}]][[get_weather {}]]')
        self.assertEqual(content, "Let me look.\n\n")
        self.assertNotIn("[", content)
        self.assertEqual([c["index"] for c in calls], [0, 1])
        self.assertEqual(calls[0]["function"], {"name": "get_weather",
                                                "arguments": '{"city": "Lugo"}'})
        self.assertTrue(calls[0]["id"].startswith("chatcmpl-tool-"))
        self.assertEqual(finish, "tool_calls")

    def test_a_call_streams_as_vllm_streams_one(self):
        """Its id and name with empty arguments first, then the arguments in
        pieces as the answer writes them; no content alongside."""
        self.runtime.script = '[[get_weather {"city": "Lugo", "days": 3}]]'
        status, events = self.api.stream("/v1/chat/completions", {
            "messages": [{"role": "user", "content": "hi"}], "tools": [WEATHER], "stream": True})
        pieces = [d["tool_calls"][0] for e in events if e and e["choices"]
                  for d in [e["choices"][0]["delta"]] if "tool_calls" in d]
        self.assertEqual(set(pieces[0]), {"id", "type", "index", "function"})
        self.assertEqual(pieces[0]["function"], {"name": "get_weather", "arguments": ""})
        self.assertGreater(len(pieces), 3, "the arguments arrive in pieces")
        self.assertEqual("".join(p["function"]["arguments"] for p in pieces[1:]),
                         '{"city": "Lugo", "days": 3}')

    def test_a_stream_cut_inside_a_call_ends_with_length(self):
        """max_tokens inside a call: what was read of it went out, the
        stream ends with "length" and no error (as vLLM's)."""
        script = '[[get_weather {"city": "Lugo", "days": 3}]]'
        content, calls, finish = self.stream(script, max_tokens=len(script) - 8)
        self.assertEqual(finish, "length")
        self.assertEqual(calls[0]["function"]["name"], "get_weather")
        self.assertTrue('{"city": "Lugo", "days": 3}'.startswith(calls[0]["function"]["arguments"]))

    def test_parallel_tool_calls_false_keeps_the_first(self):
        script = "[[get_weather {}]][[get_time {}]]"
        status, body = self.chat(script, parallel_tool_calls=False)
        self.assertEqual([c["function"]["name"] for c in
                          body["choices"][0]["message"]["tool_calls"]], ["get_weather"])
        content, calls, finish = self.stream(script, parallel_tool_calls=False)
        self.assertEqual([c["function"]["name"] for c in calls], ["get_weather"])

    def test_tool_choice_none_renders_the_tools_and_reads_nothing_back(self):
        status, body = self.chat("[[get_weather {}]]", tool_choice="none")
        message = body["choices"][0]["message"]
        self.assertEqual(message["content"], "[[get_weather {}]]")
        self.assertNotIn("tool_calls", message)
        self.assertEqual(body["choices"][0]["finish_reason"], "stop")
        self.assertIsNotNone(self.runtime.chats[-1]["tools"])

    def test_required_and_named_choices_are_served_as_auto(self):
        for choice in ("required", {"type": "function", "function": {"name": "get_weather"}}):
            with self.subTest(choice=choice):
                status, body = self.chat("[[get_weather {}]]", tool_choice=choice)
                self.assertEqual(status, 200)
                self.assertEqual(len(body["choices"][0]["message"]["tool_calls"]), 1)

    def test_bad_tool_requests_are_refused_as_vllm_refuses_them(self):
        cases = [
            {"tools": []},
            {"tools": [WEATHER], "tool_choice": "sometimes"},
            {"tools": [WEATHER], "tool_choice": {"type": "function",
                                                 "function": {"name": "nope"}}},
            {"tools": [{"type": "code_interpreter"}]},
            {"tools": [{"type": "function", "function": {"description": "no name"}}]},
        ]
        for case in cases:
            with self.subTest(case=case):
                status, body = self.chat("x", **case)
                self.assertEqual(status, 400)
        self.runtime.script = "x"
        status, _ = self.api.post("/v1/chat/completions", {
            "messages": [{"role": "user", "content": "hi"}], "tool_choice": "auto"})
        self.assertEqual(status, 400, "tool_choice without tools")

    def test_an_earlier_call_reaches_the_template_with_mapped_arguments(self):
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": None, "reasoning": "hmm", "tool_calls": [
                {"id": "a", "type": "function",
                 "function": {"name": "get_weather", "arguments": '{"city": "Lugo"}'}},
                {"id": "b", "type": "function",
                 "function": {"name": "get_weather", "arguments": "not json"}}]},
            {"role": "tool", "tool_call_id": "a", "content": "rain"},
        ]
        status, body = self.chat("ok", messages=messages)
        self.assertEqual(status, 200)
        turn = self.runtime.chats[-1]["messages"][1]
        self.assertEqual(turn["reasoning_content"], "hmm")
        self.assertEqual([c["function"]["arguments"] for c in turn["tool_calls"]],
                         [{"city": "Lugo"}, {}])


if __name__ == "__main__":
    unittest.main()
