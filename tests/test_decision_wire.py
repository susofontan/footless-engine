"""The typed decision wire schema: parsing, views, answers, and responses."""

from __future__ import annotations

import math
import unittest

from footless import wire
from footless.decisions import QuestionView, choice_confidence, score_confidence

NAMES = {"served-model"}
YES_NO = {"id": "q1", "type": "yes_no", "question": "Urgent"}


def decisions_body(**extra) -> dict:
    body = {"input": "s", "questions": [dict(YES_NO)]}
    body.update(extra)
    return body


def systemone_body(**extra) -> dict:
    body = {
        "state": "s",
        "model": "jev-latest",
        "questions": {"q": {"type": "noul", "instructions": "x"}},
    }
    body.update(extra)
    return body


class DecisionsParseTest(unittest.TestCase):
    def refused(self, body, message, param):
        with self.assertRaises(wire.ApiError) as caught:
            wire.parse_decisions(body, NAMES)
        error = caught.exception
        self.assertEqual(
            (error.code, error.err_type, error.message, error.param),
            (400, "BadRequestError", message, param))

    def test_defaults(self):
        request = wire.parse_decisions(decisions_body(), NAMES)
        self.assertEqual(request.input, "s")
        self.assertEqual(request.temperature, 1.0)
        self.assertEqual(request.chat_template_kwargs, {})
        self.assertIsNone(request.prompt_format_version)
        self.assertFalse(request.return_prompt_token_ids)
        self.assertEqual(request.model, "default")
        self.assertIsNone(request.request_id)
        self.assertEqual(request.dropped, [])

    def test_fields_are_parsed(self):
        request = wire.parse_decisions({
            "input": {"ticket": "Refund"},
            "temperature": 0.5,
            "chat_template_kwargs": {},
            "prompt_format_version": 2,
            "return_prompt_token_ids": True,
            "model": "any-name",
            "questions": [
                {"id": "team", "type": "choice", "question": {"q": 1},
                 "options": [{"name": "a", "description": "A"}, {"name": "b"}]},
                {"id": "mood", "type": "score", "question": "Mood?",
                 "levels": ["Calm", {"label": "Angry"}]},
                {"id": "urgent", "type": "yes_no", "question": "Urgent",
                 "yes": "Today", "no": None},
            ],
        }, NAMES)
        self.assertEqual(request.input, {"ticket": "Refund"})
        self.assertEqual(request.temperature, 0.5)
        self.assertEqual(request.prompt_format_version, 2)  # the serve layer compares
        self.assertTrue(request.return_prompt_token_ids)
        self.assertEqual(request.model, "any-name")
        team, mood, urgent = request.questions
        self.assertEqual((team.id, team.type, team.question),
                         ("team", "choice", {"q": 1}))
        self.assertEqual(team.options, [("a", "A"), ("b", None)])
        self.assertEqual((mood.id, mood.levels), ("mood", ["Calm", {"label": "Angry"}]))
        self.assertEqual((urgent.id, urgent.yes, urgent.no),
                         ("urgent", "Today", None))

    def test_unknown_fields_are_refused(self):
        cases = {
            "top level": (decisions_body(top_p=0.5),
                          "unknown field 'top_p'", "top_p"),
            "a near miss": (decisions_body(temprature=1),
                            "unknown field 'temprature'", "temprature"),
            "request_id": (decisions_body(request_id="r"),
                           "unknown field 'request_id'", "request_id"),
            "question": ({**decisions_body(),
                          "questions": [{**YES_NO, "frob": 1}]},
                         "unknown field 'frob'", "questions[0].frob"),
            "option": (decisions_body(questions=[{
                "id": "q1", "type": "choice", "question": "Q",
                "options": [{"name": "a"}, {"name": "b", "frob": 1}]}]),
                "unknown field 'frob'", "questions[0].options[1].frob"),
            "field of another type": ({**decisions_body(),
                                       "questions": [{**YES_NO, "levels": ["x"]}]},
                                      "unknown field 'levels'", "questions[0].levels"),
        }
        for case, (body, message, param) in cases.items():
            with self.subTest(case):
                self.refused(body, message, param)

    def test_input_rules(self):
        cases = {
            "missing": ({"questions": [dict(YES_NO)]},
                        "input is required", "input"),
            "null": (decisions_body(input=None), "input is required", "input"),
            "number": (decisions_body(input=5),
                       "input must be a string, an object, or a list", "input"),
            "blank string": (decisions_body(input=" "),
                             "input must not be blank", "input"),
            "empty list": (decisions_body(input=[]),
                           "input must not be blank", "input"),
            "empty object": (decisions_body(input={}),
                             "input must not be blank", "input"),
        }
        for case, (body, message, param) in cases.items():
            with self.subTest(case):
                self.refused(body, message, param)

    def test_question_rules(self):
        cases = {
            "missing": ({"input": "s"}, "questions must be a non-empty list",
                        "questions"),
            "not a list": (decisions_body(questions={}),
                           "questions must be a non-empty list", "questions"),
            "empty": (decisions_body(questions=[]),
                      "questions must be a non-empty list", "questions"),
            "item not an object": (decisions_body(questions=[5]),
                                   "questions[0] must be an object", "questions[0]"),
            "id missing": (decisions_body(questions=[{"type": "yes_no",
                                                      "question": "Q"}]),
                           "questions[0].id is required", "questions[0].id"),
            "id not a string": (decisions_body(questions=[{**YES_NO, "id": 5}]),
                                "questions[0].id must be a string",
                                "questions[0].id"),
            "id blank": (decisions_body(questions=[{**YES_NO, "id": " "}]),
                         "questions[0].id must not be blank", "questions[0].id"),
            "id repeated": (decisions_body(questions=[dict(YES_NO), dict(YES_NO)]),
                            "question id 'q1' repeats another question",
                            "questions[1].id"),
            "type missing": (decisions_body(questions=[{"id": "q1",
                                                        "question": "Q"}]),
                             "questions[0].type is required", "questions[0].type"),
            "type unknown": (decisions_body(questions=[{**YES_NO, "type": "rank"}]),
                             "questions[0].type must be 'choice', 'score', "
                             "or 'yes_no'", "questions[0].type"),
            "question missing": (decisions_body(questions=[{"id": "q1",
                                                            "type": "yes_no"}]),
                                 "questions[0].question is required",
                                 "questions[0].question"),
            "question blank": (decisions_body(questions=[{**YES_NO, "question": {}}]),
                               "questions[0].question must not be blank",
                               "questions[0].question"),
            "question not a text": (decisions_body(
                questions=[{**YES_NO, "question": 5}]),
                "questions[0].question must be a string, an object, or a list",
                "questions[0].question"),
        }
        for case, (body, message, param) in cases.items():
            with self.subTest(case):
                self.refused(body, message, param)

    def choice_body(self, options):
        return decisions_body(questions=[{
            "id": "q1", "type": "choice", "question": "Q", "options": options}])

    def test_option_rules(self):
        two = [{"name": "a"}, {"name": "b"}]
        line_break = "a\nb"
        cases = {
            "missing": (decisions_body(questions=[{
                "id": "q1", "type": "choice", "question": "Q"}]),
                "questions[0].options is required", "questions[0].options"),
            "not a list": (self.choice_body({}),
                           "questions[0].options must be a list",
                           "questions[0].options"),
            "one option": (self.choice_body([{"name": "a"}]),
                           "questions[0].options must hold at least 2 options",
                           "questions[0].options"),
            "27 options": (self.choice_body(
                [{"name": f"o{i}"} for i in range(27)]),
                "questions[0].options must hold at most 26 options",
                "questions[0].options"),
            "item not an object": (self.choice_body([{"name": "a"}, 5]),
                                   "questions[0].options[1] must be an object",
                                   "questions[0].options[1]"),
            "name missing": (self.choice_body([{"name": "a"}, {}]),
                             "questions[0].options[1].name is required",
                             "questions[0].options[1].name"),
            "name not a string": (self.choice_body([{"name": "a"}, {"name": 5}]),
                                  "questions[0].options[1].name must be a string",
                                  "questions[0].options[1].name"),
            "name blank": (self.choice_body([{"name": "a"}, {"name": " "}]),
                           "question 'q1': option names must be nonempty",
                           "questions[0].options"),
            "name with a line break": (self.choice_body(
                [{"name": "a"}, {"name": line_break}]),
                f"question 'q1': option name {line_break!r} must not contain "
                "control or line break characters", "questions[0].options"),
            "name repeated": (self.choice_body([{"name": "a"}, {"name": "A "}]),
                              "question 'q1': option name 'A ' repeats another "
                              "option", "questions[0].options"),
            "description not a text": (self.choice_body(
                [{"name": "a", "description": 5}, {"name": "b"}]),
                "questions[0].options[0].description must be a string, an "
                "object, or a list", "questions[0].options[0].description"),
        }
        for case, (body, message, param) in cases.items():
            with self.subTest(case):
                self.refused(body, message, param)

    def score_body(self, levels):
        return decisions_body(questions=[{
            "id": "q1", "type": "score", "question": "Q", "levels": levels}])

    def test_level_rules(self):
        cases = {
            "missing": (decisions_body(questions=[{
                "id": "q1", "type": "score", "question": "Q"}]),
                "questions[0].levels is required", "questions[0].levels"),
            "not a list": (self.score_body({}),
                           "questions[0].levels must be a list",
                           "questions[0].levels"),
            "one level": (self.score_body(["low"]),
                          "questions[0].levels must hold at least 2 levels",
                          "questions[0].levels"),
            "11 levels": (self.score_body([str(i) for i in range(11)]),
                          "questions[0].levels must hold at most 10 levels",
                          "questions[0].levels"),
            "blank level": (self.score_body(["low", " "]),
                            "questions[0].levels[1] must not be blank",
                            "questions[0].levels[1]"),
            "level not a text": (self.score_body(["low", 5]),
                                 "questions[0].levels[1] must be a string, an "
                                 "object, or a list", "questions[0].levels[1]"),
        }
        for case, (body, message, param) in cases.items():
            with self.subTest(case):
                self.refused(body, message, param)

    def test_yes_and_no_rules(self):
        for name in ("yes", "no"):
            with self.subTest(name):
                self.refused(decisions_body(questions=[{**YES_NO, name: 5}]),
                             f"questions[0].{name} must be a string, an object, "
                             f"or a list", f"questions[0].{name}")

    def test_temperature_rules(self):
        cases = {
            "zero": (0, "temperature must be > 0.0"),
            "negative": (-1, "temperature must be > 0.0"),
            "string": ("0.5", "temperature must be a number"),
            "boolean": (True, "temperature must be a number"),
            "not a number": (float("nan"), "temperature must be finite"),
            "infinite": (float("inf"), "temperature must be finite"),
        }
        for case, (value, message) in cases.items():
            with self.subTest(case):
                self.refused(decisions_body(temperature=value), message,
                             "temperature")
        self.assertEqual(
            wire.parse_decisions(decisions_body(temperature=2), NAMES).temperature,
            2.0)

    def test_chat_template_kwargs_must_be_empty(self):
        self.refused(decisions_body(chat_template_kwargs={"k": 1}),
                     "chat_template_kwargs must be empty", "chat_template_kwargs")
        self.refused(decisions_body(chat_template_kwargs=[]),
                     "chat_template_kwargs must be an object",
                     "chat_template_kwargs")
        for value in ({}, None):
            with self.subTest(value):
                wire.parse_decisions(decisions_body(chat_template_kwargs=value),
                                     NAMES)

    def test_version_and_token_id_rules(self):
        for value in ("1", 1.5, True):
            with self.subTest(value):
                self.refused(decisions_body(prompt_format_version=value),
                             "prompt_format_version must be an integer",
                             "prompt_format_version")
        self.refused(decisions_body(return_prompt_token_ids=1),
                     "return_prompt_token_ids must be a boolean",
                     "return_prompt_token_ids")
        self.assertFalse(wire.parse_decisions(
            decisions_body(return_prompt_token_ids=None), NAMES
        ).return_prompt_token_ids)

    def test_model_rules(self):
        self.refused(decisions_body(model=5), "model must be a string", "model")
        self.refused(
            decisions_body(model="base:adapter"),
            "model names the LoRA adapter 'adapter', which /v1/decisions "
            "does not support", "model")
        for model, want in ((None, "default"), ("", ""), ("jev-latest", "jev-latest")):
            with self.subTest(model):
                request = wire.parse_decisions(decisions_body(model=model), NAMES)
                self.assertEqual(request.model, want)


class SystemOneParseTest(unittest.TestCase):
    def refused(self, body, loc, msg, kind):
        with self.assertRaises(wire.ApiError) as caught:
            wire.parse_systemone(body, NAMES)
        error = caught.exception
        self.assertEqual((error.code, error.err_type),
                         (422, "ValidationError"))
        self.assertEqual(error.body(),
                         {"detail": [{"loc": loc, "msg": msg, "type": kind}]})

    def test_empty_state_and_blank_ids_are_allowed(self):
        for state in ("", "s", {}, {"a": 1}, [], [1, 2]):
            with self.subTest(state):
                self.assertEqual(
                    wire.parse_systemone(systemone_body(state=state), NAMES).state,
                    state)
        request = wire.parse_systemone(systemone_body(questions={
            "": {"type": "choice", "criteria": {"only": None}},
            " ": {"type": "score", "criteria": ["one"]},
        }), NAMES)
        self.assertEqual(list(request.questions), ["", " "])

    def test_fields_are_parsed(self):
        request = wire.parse_systemone(systemone_body(questions={
            "urgent": {"type": "noul", "instructions": "Needs an answer today",
                       "criteria": {"true": "Reply today", "false": "Can wait"}},
            "team": {"type": "choice", "instructions": "Which team?",
                     "criteria": {"billing": "Payments", "sales": None}},
            "mood": {"type": "score", "criteria": ["Calm", {"label": "Angry"}]},
        }), NAMES)
        self.assertEqual(list(request.questions), ["urgent", "team", "mood"])
        urgent, team, mood = request.questions.values()
        self.assertEqual(urgent.criteria, {"true": "Reply today", "false": "Can wait"})
        self.assertEqual(team.criteria, {"billing": "Payments", "sales": None})
        self.assertEqual(mood.criteria, ["Calm", {"label": "Angry"}])
        self.assertIsNone(mood.instructions)

    def test_option_and_level_limits(self):
        wire.parse_systemone(systemone_body(questions={
            "q": {"type": "choice",
                  "criteria": {f"o{i}": None for i in range(255)}}}), NAMES)
        self.refused(systemone_body(questions={
            "q": {"type": "choice",
                  "criteria": {f"o{i}": None for i in range(256)}}}),
            ["body", "questions", "q", "criteria"],
            "Dictionary should have at most 255 items", "too_long")
        wire.parse_systemone(systemone_body(questions={
            "q": {"type": "score", "criteria": ["one"]}}), NAMES)
        wire.parse_systemone(systemone_body(questions={
            "q": {"type": "score", "criteria": [str(i) for i in range(10)]}}), NAMES)
        self.refused(systemone_body(questions={
            "q": {"type": "score", "criteria": [str(i) for i in range(11)]}}),
            ["body", "questions", "q", "criteria"],
            "List should have at most 10 items", "too_long")

    def test_request_rules(self):
        questions = {"q": {"type": "noul", "instructions": "x"}}
        cases = {
            "state missing": ({"model": "m", "questions": questions},
                              ["body", "state"], "Field required", "missing"),
            "state not a text": (systemone_body(state=5),
                                 ["body", "state"],
                                 "Input should be a valid string, dictionary "
                                 "or list", "value_error"),
            "model missing": ({"state": "s", "questions": questions},
                              ["body", "model"], "Field required", "missing"),
            "model not a string": (systemone_body(model=5), ["body", "model"],
                                   "Input should be a valid string", "string_type"),
            "questions missing": ({"state": "s", "model": "m"},
                                  ["body", "questions"], "Field required",
                                  "missing"),
            "questions not a map": (systemone_body(questions=[]),
                                    ["body", "questions"],
                                    "Input should be a valid dictionary",
                                    "dict_type"),
            "no questions": (systemone_body(questions={}), ["body", "questions"],
                             "Dictionary should have at least 1 item", "too_short"),
            "chat_template_kwargs not an object": (
                systemone_body(chat_template_kwargs=[]),
                ["body", "chat_template_kwargs"],
                "Input should be a valid dictionary", "dict_type"),
            "chat_template_kwargs not empty": (
                systemone_body(chat_template_kwargs={"k": 1}),
                ["body", "chat_template_kwargs"],
                "Value error, chat_template_kwargs must be empty", "value_error"),
        }
        for case, (body, loc, msg, kind) in cases.items():
            with self.subTest(case):
                self.refused(body, loc, msg, kind)

    def test_decisions_only_fields_are_refused_by_name(self):
        for name in ("temperature", "prompt_format_version",
                     "return_prompt_token_ids"):
            for value in (0.5, 1, False):
                with self.subTest(name=name, value=value):
                    self.refused(systemone_body(**{name: value}), ["body", name],
                                 f"Value error, {name} is not part of this API, "
                                 "use /v1/decisions for it", "value_error")
            with self.subTest(name=name):
                wire.parse_systemone(systemone_body(**{name: None}), NAMES)

    def test_question_rules(self):
        cases = {
            "question not an object": (
                systemone_body(questions={"q": 5}), ["body", "questions", "q"],
                "Input should be a valid dictionary", "dict_type"),
            "type missing": (
                systemone_body(questions={"q": {"instructions": "x"}}),
                ["body", "questions", "q", "type"], "Field required", "missing"),
            "unknown type": (
                systemone_body(questions={"q": {"type": "yes_no"}}),
                ["body", "questions", "q", "type"],
                "Input should be 'noul', 'choice' or 'score'", "literal_error"),
            "unknown key in a question": (
                systemone_body(questions={"q": {"type": "noul",
                                                "instruction": "x"}}),
                ["body", "questions", "q", "instruction"],
                "Extra inputs are not permitted", "extra_forbidden"),
            "instructions not a text": (
                systemone_body(questions={"q": {"type": "noul",
                                                "instructions": 5}}),
                ["body", "questions", "q", "instructions"],
                "Input should be a valid string, dictionary or list",
                "value_error"),
            "noul that asks nothing": (
                systemone_body(questions={"q": {"type": "noul"}}),
                ["body", "questions", "q"],
                "Value error, a noul question needs instructions or a true or "
                "false description to decide on", "value_error"),
            "noul with blank criteria": (
                systemone_body(questions={"q": {
                    "type": "noul", "instructions": None,
                    "criteria": {"true": " "}}}),
                ["body", "questions", "q"],
                "Value error, a noul question needs instructions or a true or "
                "false description to decide on", "value_error"),
            "unknown key in noul criteria": (
                systemone_body(questions={"q": {"type": "noul",
                                                "criteria": {"yes": "x"}}}),
                ["body", "questions", "q", "criteria", "yes"],
                "Extra inputs are not permitted", "extra_forbidden"),
            "noul criteria not an object": (
                systemone_body(questions={"q": {"type": "noul",
                                                "criteria": ["x"]}}),
                ["body", "questions", "q", "criteria"],
                "Input should be a valid dictionary", "dict_type"),
            "noul description not a text": (
                systemone_body(questions={"q": {
                    "type": "noul", "criteria": {"true": 5}}}),
                ["body", "questions", "q", "criteria", "true"],
                "Input should be a valid string, dictionary or list",
                "value_error"),
            "no options": (
                systemone_body(questions={"q": {"type": "choice"}}),
                ["body", "questions", "q", "criteria"], "Field required",
                "missing"),
            "empty options": (
                systemone_body(questions={"q": {"type": "choice",
                                                "criteria": {}}}),
                ["body", "questions", "q", "criteria"],
                "Dictionary should have at least 1 item", "too_short"),
            "repeated option name": (
                systemone_body(questions={"q": {
                    "type": "choice", "criteria": {"a": None, "A ": None}}}),
                ["body", "questions", "q", "criteria"],
                "Value error, option name 'A ' repeats another option",
                "value_error"),
            "option description not a text": (
                systemone_body(questions={"q": {
                    "type": "choice", "criteria": {"a": 5, "b": None}}}),
                ["body", "questions", "q", "criteria", "a"],
                "Input should be a valid string, dictionary or list",
                "value_error"),
            "no levels": (
                systemone_body(questions={"q": {"type": "score"}}),
                ["body", "questions", "q", "criteria"], "Field required",
                "missing"),
            "levels not a list": (
                systemone_body(questions={"q": {"type": "score",
                                                "criteria": {"a": None}}}),
                ["body", "questions", "q", "criteria"],
                "Input should be a valid list", "list_type"),
            "empty levels": (
                systemone_body(questions={"q": {"type": "score",
                                                "criteria": []}}),
                ["body", "questions", "q", "criteria"],
                "List should have at least 1 item", "too_short"),
            "null level": (
                systemone_body(questions={"q": {"type": "score",
                                                "criteria": ["low", None]}}),
                ["body", "questions", "q", "criteria", 1],
                "Input should be a valid string, dictionary or list",
                "value_error"),
            "blank level": (
                systemone_body(questions={"q": {"type": "score",
                                                "criteria": ["low", " "]}}),
                ["body", "questions", "q", "criteria", 1],
                "Value error, must not be blank", "value_error"),
        }
        for case, (body, loc, msg, kind) in cases.items():
            with self.subTest(case):
                self.refused(body, loc, msg, kind)

    def test_model_adapter_is_a_plain_400(self):
        with self.assertRaises(wire.ApiError) as caught:
            wire.parse_systemone(systemone_body(model="base:adapter"), NAMES)
        error = caught.exception
        self.assertEqual(error.code, 400)
        self.assertEqual(
            error.message,
            "model names the LoRA adapter 'adapter', which /v1/systemone "
            "does not support")
        self.assertIn("error", error.body())


class StrictAndPermissiveTest(unittest.TestCase):
    def test_decisions_refuses_what_systemone_drops(self):
        with self.assertRaises(wire.ApiError) as caught:
            wire.parse_decisions(decisions_body(top_p=0.5), NAMES)
        self.assertEqual(caught.exception.message, "unknown field 'top_p'")
        request = wire.parse_systemone(systemone_body(top_p=0.5, zeta=1), NAMES)
        self.assertEqual(request.dropped, ["top_p", "zeta"])

    def test_no_behaviour_fields_are_not_worth_a_line(self):
        # As on the other routes, fields that carry no behaviour here are
        # accepted and ignored without a dropped entry.
        request = wire.parse_systemone(systemone_body(user="u"), NAMES)
        self.assertEqual(request.dropped, [])

    def test_request_id_is_serve_bookkeeping(self):
        # The request id comes from the X-Request-Id header; as a body field
        # it is unknown: refused on /v1/decisions, dropped on /v1/systemone.
        with self.assertRaises(wire.ApiError):
            wire.parse_decisions(decisions_body(request_id="r"), NAMES)
        request = wire.parse_systemone(systemone_body(request_id="r"), NAMES)
        self.assertEqual(request.dropped, ["request_id"])
        self.assertIsNone(request.request_id)


class ViewTest(unittest.TestCase):
    def test_decision_view(self):
        request = wire.parse_decisions({
            "input": "s",
            "questions": [
                {"id": "team", "type": "choice", "question": "Which team?",
                 "options": [{"name": "billing", "description": "Payments"},
                             {"name": "sales"}]},
                {"id": "mood", "type": "score", "question": "Mood?",
                 "levels": ["Calm", "Angry", "Furious"]},
                {"id": "urgent", "type": "yes_no", "question": "Urgent",
                 "no": "Can wait"},
            ],
        }, NAMES)
        team, mood, urgent = (wire.decision_view(q) for q in request.questions)
        self.assertEqual(team, QuestionView(
            "choice", "Which team?", ["billing", "sales"], ["Payments", None]))
        self.assertEqual(mood, QuestionView(
            "score", "Mood?", ["0", "1", "2"], ["Calm", "Angry", "Furious"]))
        self.assertEqual(urgent, QuestionView(
            "yes_no", "Urgent", ["yes", "no"], [None, "Can wait"]))

    def test_systemone_view(self):
        request = wire.parse_systemone(systemone_body(questions={
            "urgent": {"type": "noul", "instructions": "Today",
                       "criteria": {"true": "Reply today"}},
            "team": {"type": "choice", "criteria": {"billing": None}},
            "mood": {"type": "score", "criteria": ["Calm", "Angry"]},
            "bare": {"type": "noul", "criteria": {"false": "Can wait"}},
        }), NAMES)
        urgent, team, mood, bare = (wire.systemone_view(q)
                                    for q in request.questions.values())
        self.assertEqual(urgent, QuestionView(
            "yes_no", "Today", ["yes", "no"], ["Reply today", None]))
        self.assertEqual(team, QuestionView("choice", None, ["billing"], [None]))
        self.assertEqual(mood, QuestionView("score", None, ["0", "1"],
                                            ["Calm", "Angry"]))
        self.assertEqual(bare, QuestionView("yes_no", None, ["yes", "no"],
                                            [None, "Can wait"]))


class AnswerTest(unittest.TestCase):
    def test_decisions_answer(self):
        choice = QuestionView("choice", "Q", ["billing", "technical", "sales"], [])
        answer = wire.decisions_answer(choice, [0.2, 0.5, 0.3], 0.6)
        self.assertEqual(answer, {
            "type": "choice",
            "probabilities": {"billing": 0.2, "technical": 0.5, "sales": 0.3},
            "label_mass": 0.6,
            "choice": "technical",
        })
        score = QuestionView("score", "Q", ["0", "1", "2"], [])
        answer = wire.decisions_answer(score, [0.2, 0.5, 0.3], 0.6)
        self.assertEqual(set(answer), {"type", "probabilities", "label_mass",
                                       "score"})
        self.assertAlmostEqual(answer["score"], math.fsum([0, 0.5, 0.6]))
        yes_no = QuestionView("yes_no", "Q", ["yes", "no"], [])
        answer = wire.decisions_answer(yes_no, [0.3, 0.7], 0.6)
        self.assertEqual(set(answer), {"type", "probabilities", "label_mass"})
        self.assertEqual(answer["probabilities"], {"yes": 0.3, "no": 0.7})

    def test_returned_ids_ride_the_answer(self):
        view = QuestionView("yes_no", "Q", ["yes", "no"], [])
        answer = wire.decisions_answer(view, [0.3, 0.7], 0.6,
                                       prompt_token_ids=[1, 2], label_token_ids=[3])
        self.assertEqual(answer["prompt_token_ids"], [1, 2])
        self.assertEqual(answer["label_token_ids"], [3])
        self.assertNotIn("prompt_token_ids",
                         wire.decisions_answer(view, [0.3, 0.7], 0.6))

    def test_systemone_answer(self):
        yes_no = QuestionView("yes_no", None, ["yes", "no"], [])
        answer = wire.systemone_answer(yes_no, [0.3, 0.7], 0.9)
        self.assertEqual(answer, {"type": "noul", "noul": 0.3, "x_label_mass": 0.9})

        choice = QuestionView("choice", None, ["a", "b", "c"], [])
        answer = wire.systemone_answer(choice, [0.3, 0.3, 0.3], 0.9)
        self.assertEqual(set(answer), {"type", "choice", "confidence",
                                       "probabilities", "x_label_mass"})
        # Reported as scored — a softmax can miss a sum of 1 by a few ulp.
        self.assertEqual(list(answer["probabilities"].values()), [0.3] * 3)
        self.assertEqual(answer["choice"], "a")
        self.assertAlmostEqual(answer["confidence"],
                               choice_confidence([1 / 3] * 3))

        score = QuestionView("score", None, ["0", "1", "2"], ["Calm", "Civil",
                                                             "Angry"])
        answer = wire.systemone_answer(score, [0.3, 0.3, 0.3], 0.9)
        self.assertEqual(set(answer), {"type", "score", "confidence", "legend",
                                       "probabilities", "x_label_mass"})
        self.assertAlmostEqual(answer["score"], math.fsum([0, 0.3, 0.6]))
        self.assertAlmostEqual(answer["confidence"],
                               score_confidence([1 / 3] * 3))
        self.assertEqual(answer["legend"], {"0": "Calm", "1": "Civil",
                                            "2": "Angry"})
        self.assertEqual(answer["x_label_mass"], 0.9)

    def test_non_finite_values_are_a_server_fault(self):
        views = [
            QuestionView("yes_no", None, ["yes", "no"], []),
            QuestionView("choice", None, ["a", "b"], []),
            QuestionView("score", None, ["0", "1"], ["low", "high"]),
        ]
        cases = [
            ([float("nan"), 0.5], 1.0),
            ([0.5, float("inf")], 1.0),
            ([0.5, 0.5], float("nan")),
        ]
        for view in views:
            for probs, mass in cases:
                with self.subTest(view.kind, probs=probs, mass=mass):
                    with self.assertRaises(RuntimeError):
                        wire.systemone_answer(view, probs, mass)


class ResponseTest(unittest.TestCase):
    def test_decisions_response(self):
        answers = {"q": {"type": "yes_no"}}
        self.assertEqual(wire.decisions_response("m", 1, answers, 7), {
            "object": "decisions",
            "model": "m",
            "prompt_format_version": 1,
            "answers": answers,
            "usage": {"prompt_tokens": 7, "completion_tokens": 0,
                      "total_tokens": 7},
        })

    def test_systemone_response(self):
        answers = {"q": {"type": "noul"}}
        self.assertEqual(wire.systemone_response("served-model", answers, 7), {
            "model": "served-model",
            "answers": answers,
            "usage": {"input_tokens": 7, "output_tokens": 0},
        })


class ApiErrorTest(unittest.TestCase):
    def test_body_without_a_detail_list_is_the_error_envelope(self):
        error = wire.ApiError("boom", 400, "BadRequestError", "field")
        self.assertEqual(error.body(),
                         wire.error("boom", 400, "BadRequestError", "field"))

    def test_detail_errors_are_422_with_a_detail_list(self):
        entries = [{"loc": ["body", "state"], "msg": "Field required",
                    "type": "missing"}]
        error = wire.detail_error(entries)
        self.assertEqual((error.code, error.err_type),
                         (422, "ValidationError"))
        self.assertEqual(error.body(), {"detail": entries})
        # The optional input echo is left out: each entry is loc, msg, type.
        for entry in error.body()["detail"]:
            self.assertEqual(set(entry), {"loc", "msg", "type"})


if __name__ == "__main__":
    unittest.main()
