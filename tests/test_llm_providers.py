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
        self.headers = {}
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


def test_truncated_output_retries_with_a_bigger_budget():
    from pydantic import BaseModel
    from unseen.llm.base import generate, looks_truncated

    class Small(BaseModel):
        title: str
        lines: list[str]

    budgets = []

    class Cutoff:
        name, model = "gemini", "g"

        def complete_json(self, system, user, schema, temperature, max_tokens):
            budgets.append(max_tokens)
            if len(budgets) == 1:   # cut mid-way, API gives no finish reason
                return '{"title": "x", "lines": ["a", "b"', {"done_reason": None}
            return '{"title": "x", "lines": ["a"]}', {"done_reason": "STOP"}

    obj, stats, _ = generate(Cutoff(), Small, system="s", user="u", name="script", max_tokens=5000,
                             log=lambda m: None)
    assert obj.title == "x" and budgets == [5000, 10000] and stats.attempts == 2
    import json as _json
    try:
        _json.loads('{"a": [1, 2')
    except _json.JSONDecodeError as exc:
        assert looks_truncated('{"a": [1, 2', exc, None)
    assert looks_truncated("{}", ValueError("x"), "MAX_TOKENS")
    assert not looks_truncated("{}", ValueError("schema mismatch"), "STOP")


def test_gemini_caps_thinking_and_drops_it_for_models_without(monkeypatch, no_keys):
    from unseen.llm import gemini
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    sent = []
    good = {"candidates": [{"content": {"parts": [{"text": "thinking...", "thought": True}, {"text": "{}"}]},
                            "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 2, "thoughtsTokenCount": 900}}

    def fake_post(url, json=None, headers=None, timeout=None):
        sent.append(_copy(json))
        if "thinkingConfig" in json["generationConfig"] and len(sent) == 1:
            return Resp(400, text="thinking budget is not supported for this model")
        return Resp(200, good)

    monkeypatch.setattr(gemini.httpx, "post", fake_post)
    p = gemini.GeminiProvider("https://g/v1beta", "m", thinking_budget=1024, max_output_tokens=16384)
    text, usage = p.complete_json("s", "u", {}, 0.5, 5000)
    assert sent[0]["generationConfig"]["thinkingConfig"] == {"thinkingBudget": 1024}
    assert sent[0]["generationConfig"]["maxOutputTokens"] == 16384
    assert "thinkingConfig" not in sent[1]["generationConfig"] and p.thinking_budget is None
    assert text == "{}" and usage["thinking_tokens"] == 900 and usage["done_reason"] == "STOP"


def _copy(body):
    import copy
    return copy.deepcopy(body)


def test_critic_is_a_different_hosted_model(tmp_path, monkeypatch):
    from storyfakes import CONCEPTS, CRITIQUE_PASS, RATINGS, FakeEmbedder, FakeProvider, draft_dict
    import unseen.story.render as render
    from unseen.state.db import StateDB
    from unseen.story.engine import StoryEngine

    monkeypatch.setattr(render, "STORIES_DIR", tmp_path / "stories")
    writer = FakeProvider({"ConceptList": [CONCEPTS], "StoryDraft": [draft_dict()], "Critique": [CRITIQUE_PASS]})
    writer.name = "gemini"
    critic = FakeProvider({"ConceptRatings": [RATINGS], "Critique": [CRITIQUE_PASS]})
    critic.name = "groq"
    local = FakeProvider({})
    local.name = "ollama"
    engine = StoryEngine(provider=FallbackProvider([writer, critic, local], log=lambda m: None),
                         db=StateDB(tmp_path / "t.db"), embedder=FakeEmbedder(), seed=5, log=lambda m: None)
    record, _ = engine.new_story(category="fiction_horror", target_seconds=30)
    assert writer.calls == ["ConceptList", "StoryDraft"] and local.calls == []
    assert critic.calls == ["ConceptRatings", "Critique"]   # the critic rates concepts and the script
    assert [s["provider"] for s in record.llm] == ["gemini", "groq", "gemini", "groq"]


def test_transient_errors_retry_then_cool_down_then_recover():
    from unseen.llm.base import TransientLLMError

    now = [0.0]
    slept = []

    class Flaky:
        def __init__(self, name, failures):
            self.name, self.model, self.failures, self.calls = name, name, failures, 0

        def complete_json(self, *a):
            self.calls += 1
            if self.failures > 0:
                self.failures -= 1
                raise TransientLLMError(f"{self.name} 503 high demand")
            return f"from {self.name}", {}

    gemini, groq = Flaky("gemini", failures=1), Flaky("groq", failures=0)
    fb = FallbackProvider([gemini, groq], log=lambda m: None, clock=lambda: now[0], sleep=slept.append)
    assert fb.complete_json("s", "u", {}, 0, 1)[0] == "from gemini"     # one short retry was enough
    assert slept == [5.0] and gemini.calls == 2 and groq.calls == 0

    gemini.failures = 2                                                    # fails twice: cool down
    assert fb.complete_json("s", "u", {}, 0, 1)[0] == "from groq"
    now[0] = 30.0
    assert fb.complete_json("s", "u", {}, 0, 1)[0] == "from groq"         # still cooling down
    assert gemini.calls == 4
    now[0] = 61.0
    assert fb.complete_json("s", "u", {}, 0, 1)[0] == "from gemini"       # preferred again after cool-down
    assert fb.name == "gemini"


def test_critic_view_shares_cooldowns():
    a, b, c = Stub("gemini"), Stub("groq"), Stub("ollama")
    fb = FallbackProvider([a, b, c], log=lambda m: None)
    view = fb.excluding({"gemini", "ollama"})
    assert [p.name for p in view.providers] == ["groq"]
    fb._cool_down("groq")
    assert view._cool is fb._cool and "groq" in view._cool
    assert fb.excluding({"gemini", "groq", "ollama"}) is None
