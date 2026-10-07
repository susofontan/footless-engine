"""Image input: the `vision` capability, its gate, the images' way to the
steps, the prefix cache keyed on the images, and the serving API's image
parts. `mockup-tools` declares the capability (an image takes three
placeholder tokens there, and each step records the spans it was given);
`mockup` does not.
"""

from __future__ import annotations

import base64
import unittest
from pathlib import Path

from footless.engine import Engine
from footless.sdk import VISION, ContractError, Image, SamplingSpec

from tests.test_serve import Api

ROOT = Path(__file__).resolve().parent
MOCKUP, TOOLS = ROOT / "mockup", ROOT / "mockup-tools"
RED, BLUE = Image.of(b"\x89PNG red"), Image.of(b"\x89PNG blue")
GREEDY = SamplingSpec(temperature=0.0)


def ask(image, text="what is it?"):
    return [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": text}]}]


class VisionGateTest(unittest.TestCase):
    def test_a_package_without_the_capability_is_refused(self):
        engine = Engine.open(MOCKUP)
        self.addCleanup(engine.close)
        with self.assertRaises(ContractError) as caught:
            engine.vision(True)
        self.assertIn("vision", str(caught.exception))
        engine.vision(False)                      # asking for nothing is fine

    def test_images_need_vision_on(self):
        engine = Engine.open(TOOLS)
        self.addCleanup(engine.close)
        self.assertIn(VISION, engine.facts.capabilities)
        with self.assertRaises(ContractError):
            engine.chat_prompt(ask(RED))
        with self.assertRaises(ValueError):
            engine.chat_tokens(ask(RED))
        engine.vision(True)
        self.assertTrue(engine.runtime.vision)


class VisionEngineTest(unittest.TestCase):
    def setUp(self):
        self.engine = Engine.open(TOOLS)
        self.addCleanup(self.engine.close)
        self.engine.vision(True)
        self.runtime = self.engine.runtime

    def run_chat(self, messages):
        prompt = self.engine.chat_prompt(messages)
        self.runtime.steps_images.clear()
        result = self.engine.generate_tokens(prompt.tokens, 8, sampling=GREEDY, images=prompt.images)
        return prompt, result

    def test_the_steps_get_the_images_where_they_sit(self):
        prompt, result = self.run_chat(ask(RED))
        self.assertEqual(len(prompt.images), 1)
        span = prompt.images[0]
        self.assertEqual(prompt.tokens[span.start:span.start + span.length], [300] * 3)
        self.assertEqual(self.runtime.steps_images, [(0, span.start, 3, RED.digest)])
        self.assertEqual(result.text, "the mock")

    def test_a_cached_state_is_reused_only_for_the_same_image(self):
        first, _ = self.run_chat(ask(RED))
        start = first.images[0].start
        # the same text with another image: the cached state holds RED's
        # tokens from `start` on, so it is reused up to there and BLUE is fed
        _, other = self.run_chat(ask(BLUE))
        self.assertGreaterEqual(other.prompt_tokens_fed, len(first.tokens) - start)
        self.assertEqual([d for *_, d in self.runtime.steps_images], [BLUE.digest])
        before, rel = self.runtime.steps_images[0][:2]
        self.assertEqual(before + rel, start)                 # where the image sits, from the reused head
        # the same image again: no image left to feed
        _, same = self.run_chat(ask(BLUE))
        self.assertLess(same.prompt_tokens_fed, len(first.tokens) - start)
        self.assertEqual(self.runtime.steps_images, [])


def data_url(raw: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(raw).decode()


class VisionServeTest(unittest.TestCase):
    def setUp(self):
        self.api = Api(TOOLS)
        self.addCleanup(self.api.close)
        self.runtime = self.api.engine.runtime

    def chat(self, part):
        return self.api.post("/v1/chat/completions", {
            "model": self.api.engine.model.name, "max_tokens": 4, "temperature": 0,
            "messages": [{"role": "user", "content": [part, {"type": "text", "text": "hi"}]}]})

    def test_without_vision_image_parts_are_dropped(self):
        status, body = self.chat({"type": "image_url", "image_url": {"url": data_url(b"x")}})
        self.assertEqual(status, 200)
        self.assertEqual(self.runtime.steps_images, [])

    def test_with_vision_a_data_url_reaches_the_steps(self):
        self.api.engine.vision(True)
        status, body = self.chat({"type": "image_url", "image_url": {"url": data_url(b"some png")}})
        self.assertEqual(status, 200, body)
        self.assertEqual([d for *_, d in self.runtime.steps_images], [Image.of(b"some png").digest])
        self.assertEqual(body["usage"]["prompt_tokens"], len("user: ") + 3 + len("hi\nassistant: "))

    def test_with_vision_urls_are_not_fetched(self):
        self.api.engine.vision(True)
        for url in ("http://example.com/a.png", "data:image/png,raw", "data:image/png;base64,!!"):
            status, body = self.chat({"type": "image_url", "image_url": {"url": url}})
            self.assertEqual(status, 400, url)


if __name__ == "__main__":
    unittest.main()
