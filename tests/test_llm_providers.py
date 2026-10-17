import json

import pytest

from unseen.llm import openai_compat
from unseen.llm.base import LLMError
from unseen.llm.fallback import FallbackProvider
from unseen.story.config import load_story_config
from unseen.story.engine import make_provider

KEYS = ("GEMINI_API_KEY", "GROQ_API_KEY", "CEREBRAS_API_KEY", "OPENROUTER_API_KEY")


class Resp:
    def __init__(self, status: int, body: dict | None = None, text: str = ""):
        self.status_code = status
        self._body = body or {}
        self.text = text or json.dumps(self._body)

    def json(self):
        return self._body


def ok_body(content: str) -> dict:
    return {"choices": [{"message": {"content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 120, "completion_tokens": 30}}


@pytest.fixture
def no_keys(monkeypatch):
    for k in KEYS:
        monkeypatch.delenv(k, raising=False)


def test_openai_compat_parses_and_falls_back_from_json_mode(monkeypatch, no_keys):
    monkeypatch.setenv("GROQ_API_KEY", "k")
    sent = []

    def fake_post(url, json=None, headers=None, timeout=None):
        sent.append(dict(json))          # copy: the provider reuses and mutates the body
        if "response_format" in json:
            return Resp(400, text="response_format json_object is not supported by this model")
        return Resp(200, ok_body('{"title": "x", "lines": ["a"]}'))

    monkeypatch.setattr(openai_compat.httpx, "post", fake_post)
    p = openai_compat.OpenAICompatProvider("groq", "https://api.example/v1", "m", "GROQ_API_KEY")
    text, usage = p.complete_json("sys", "user", {"type": "object", "properties": {}}, 0.5, 100)
    assert text.startswith('{"title"') and usage["output_tokens"] == 30 and usage["done_reason"] == "stop"
    assert "response_format" in sent[0] and "response_format" not in sent[1] and not p.json_mode
    assert "JSON fields to fill" in sent[1]["messages"][1]["content"]


def test_openai_compat_errors_are_clear(monkeypatch, no_keys):
    with pytest.raises(LLMError, match="set GROQ_API_KEY"):
        openai_compat.OpenAICompatProvider("groq", "https://x/v1", "m", "GROQ_API_KEY")
    monkeypatch.setenv("GROQ_API_KEY", "k")
    p = openai_compat.OpenAICompatProvider("groq", "https://x/v1", "m", "GROQ_API_KEY")
    for status, pattern in ((429, "free quota"), (401, "key rejected"), (404, "llm-models")):
        monkeypatch.setattr(openai_compat.httpx, "post", lambda *a, s=status, **k: Resp(s, text="nope"))
        with pytest.raises(LLMError, match=pattern):
            p.complete_json("s", "u", {}, 0.5, 10)


class Stub:
    def __init__(self, name, fail=False):
        self.name, self.model, self.fail, self.calls = name, f"{name}-model", fail, 0

    def complete_json(self, *a):
        self.calls += 1
        if self.fail:
            raise LLMError(f"{self.name} quota")
        return "{}", {}


def test_fallback_switches_and_sticks():
    a, b, c = Stub("gemini", fail=True), Stub("groq"), Stub("ollama")
    fb = FallbackProvider([a, b, c], log=lambda m: None)
    assert fb.complete_json("s", "u", {}, 0.5, 10) == ("{}", {})
    assert fb.name == "groq" and fb.model == "groq-model"
    fb.complete_json("s", "u", {}, 0.5, 10)
    assert a.calls == 1 and b.calls == 2 and c.calls == 0     # stays on the provider that worked
    with pytest.raises(LLMError, match="every LLM provider failed"):
        FallbackProvider([Stub("x", True), Stub("y", True)], log=lambda m: None).complete_json("s", "u", {}, 0, 1)


def test_chain_skips_providers_without_keys(monkeypatch, no_keys):
    cfg = load_story_config().llm
    only_local = make_provider(cfg, log=lambda m: None)
    assert only_local.name == "ollama" and not isinstance(only_local, FallbackProvider)
    monkeypatch.setenv("GROQ_API_KEY", "k")
    chain = make_provider(cfg, log=lambda m: None)
    assert isinstance(chain, FallbackProvider) and [p.name for p in chain.providers] == ["groq", "ollama"]
    assert make_provider(cfg, "ollama").name == "ollama"
    with pytest.raises(ValueError, match="unknown LLM provider"):
        make_provider(cfg, "nonexistent")
