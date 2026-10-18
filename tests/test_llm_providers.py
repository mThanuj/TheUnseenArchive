import json
import re

import pytest

from unseen.llm import openai_compat
from unseen.llm.base import LLMError, ProviderLLMError, SkipCallLLMError, TransientLLMError
from unseen.llm.fallback import FallbackProvider
from unseen.story.config import load_story_config
from unseen.story.engine import make_provider

_LLM = load_story_config().llm
KEYS = ("GEMINI_API_KEY", *sorted({c.key_env for c in _LLM.openai_compat.values()}))


class Resp:
    def __init__(self, status: int, body: dict | None = None, text: str = "", headers: dict | None = None):
        self.status_code = status
        self.headers = headers or {}
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


def _raised(monkeypatch, provider, resp):
    monkeypatch.setattr(openai_compat.httpx, "post", lambda *a, **k: resp)
    with pytest.raises(LLMError) as info:
        provider.complete_json("s", "u", {}, 0.5, 10)
    return info.value


def test_rate_limit_errors_are_classified(monkeypatch, no_keys):
    monkeypatch.setenv("GROQ_API_KEY", "k")
    p = openai_compat.OpenAICompatProvider("groq", "https://x/v1", "m", "GROQ_API_KEY")
    groq = lambda msg: Resp(429, {"error": {"message": msg}})   # noqa: E731
    tpm = _raised(monkeypatch, p, groq("Rate limit reached for model `m` on tokens per minute (TPM): Limit 8000, "
                                      "Used 7000, Requested 3000. Please try again in 7.5s."))
    assert isinstance(tpm, TransientLLMError) and tpm.retry_after == 7.5 and not tpm.daily
    tpd = _raised(monkeypatch, p, groq("Rate limit reached on tokens per day (TPD): Limit 200000. Please try "
                                      "again in 2m3s."))
    assert tpd.daily and tpd.retry_after == 123
    big = _raised(monkeypatch, p, groq("Request too large for model `m` on tokens per minute (TPM): Limit 6000, "
                                      "Requested 7525"))
    assert isinstance(big, SkipCallLLMError)
    assert isinstance(_raised(monkeypatch, p, Resp(400, text="json_validate_failed")), SkipCallLLMError)
    assert isinstance(_raised(monkeypatch, p, Resp(401, text="bad key")), ProviderLLMError)
    header = _raised(monkeypatch, p, Resp(429, text="slow down", headers={"retry-after": "4"}))
    assert header.retry_after == 4.0
    free = _raised(monkeypatch, p, Resp(429, {"error": {"message": "Rate limit exceeded: free-models-per-day"}}))
    assert free.family and free.daily                         # OpenRouter: every free model on the account
    neurons = _raised(monkeypatch, p, Resp(429, {"error": {"message": "You have used up your daily free "
                                                                      "allocation of 10,000 neurons"}}))
    assert neurons.family and neurons.daily                   # Cloudflare: one budget for all models
    upstream = _raised(monkeypatch, p, Resp(200, {"error": {"code": 429, "message": "Provider returned error",
                                                            "metadata": {"raw": "x:free is rate-limited upstream"}}}))
    assert isinstance(upstream, TransientLLMError) and not upstream.family and "upstream" in str(upstream)
    # 403: a bad key/token disables the provider; anything else only this model (with the API's reason)
    cf = _raised(monkeypatch, p, Resp(403, {"success": False, "errors": [{"code": 10000,
                                                                          "message": "Authentication error"}]}))
    assert isinstance(cf, ProviderLLMError) and "10000 Authentication error" in str(cf)
    plan = _raised(monkeypatch, p, Resp(403, {"message": "Model not available on your current plan"}))
    assert type(plan) is LLMError and "not available on your current plan" in str(plan)


def test_gemini_quota_errors_say_which_limit(monkeypatch, no_keys):
    from unseen.llm import gemini
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    p = gemini.GeminiProvider("https://g/v1beta", "gemini-3.8-flash")

    def body(quota_id, delay="37s", message="Quota exceeded"):
        return {"error": {"code": 429, "message": message, "details": [
            {"@type": "type.googleapis.com/google.rpc.QuotaFailure",
             "violations": [{"quotaId": quota_id, "quotaValue": "5"}]},
            {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": delay}]}}

    for quota_id, daily in (("GenerateRequestsPerMinutePerProjectPerModel-FreeTier", False),
                            ("GenerateRequestsPerDayPerProjectPerModel-FreeTier", True)):
        monkeypatch.setattr(gemini.httpx, "post", lambda *a, b=body(quota_id), **k: Resp(429, b))
        with pytest.raises(TransientLLMError) as info:
            p.complete_json("s", "u", {}, 0.5, 10)
        assert info.value.retry_after == 37.0 and info.value.daily is daily and quota_id in str(info.value)
    no_tier = body("GenerateRequestsPerDayPerProjectPerModel-FreeTier", message="Quota exceeded, limit: 0")
    monkeypatch.setattr(gemini.httpx, "post", lambda *a, **k: Resp(429, no_tier))
    with pytest.raises(TransientLLMError) as info:
        p.complete_json("s", "u", {}, 0.5, 10)
    assert info.value.daily
    monkeypatch.setattr(gemini.httpx, "post", lambda *a, **k: Resp(200, {"promptFeedback": {"blockReason": "X"}}))
    with pytest.raises(SkipCallLLMError):                      # blocked prompt: this call only
        p.complete_json("s", "u", {}, 0.5, 10)


def test_model_patterns_and_retry_times():
    from unseen.llm.base import expand_model_pattern, try_again_seconds
    listed = ["gemini-2.5-flash", "gemini-3.8-flash", "gemini-3.7-flash", "gemini-3-flash-preview",
              "gemini-3.1-flash-lite", "gemini-3.1-flash-image", "gemini-2.5-pro", "gemma-3-27b-it"]
    assert expand_model_pattern("gemini-*-flash", listed) == ["gemini-3.8-flash", "gemini-3.7-flash", "gemini-2.5-flash"]
    assert expand_model_pattern("gemini-*-flash-lite", listed) == ["gemini-3.1-flash-lite"]
    assert expand_model_pattern("gemini-*-ultra", listed) == []
    assert try_again_seconds("Please try again in 1m2.5s.") == 62.5
    assert try_again_seconds("try again in 450ms") == 0.45
    assert try_again_seconds("try again later") is None


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
    groq_models = [e.id for e in cfg.openai_compat["groq"].models]
    assert isinstance(chain, FallbackProvider) and len(groq_models) >= 2
    assert [p.name for p in chain.providers] == ["groq"] * len(groq_models) + ["ollama"]   # every model, then local
    assert make_provider(cfg, "ollama").name == "ollama"
    with pytest.raises(ValueError, match="unknown LLM provider"):
        make_provider(cfg, "nonexistent")


def test_best_models_of_every_provider_come_before_normal_ones(monkeypatch, no_keys):
    from unseen.llm.gemini import GeminiProvider
    cfg = load_story_config().llm
    for k in KEYS:
        monkeypatch.setenv(k, "k")
    for c in cfg.openai_compat.values():   # URL placeholders such as ${ACCOUNT_ID}
        for var in re.findall(r"\$\{(\w+)\}", c.url):
            monkeypatch.setenv(var, "acct")
    monkeypatch.setattr(GeminiProvider, "list_models", lambda self: [
        "gemini-2.5-pro", "gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.1-flash-lite", "gemini-3-flash-preview",
        "gemma-4-31b-it"])
    chain = make_provider(cfg, log=lambda m: None)
    slots = chain.providers
    tiers = [p.tier for p in slots]
    assert tiers == sorted(tiers) and slots[-1].name == "ollama"
    gemini = [p.model for p in slots if p.name == "gemini"]
    # configured names the key doesn't list (gemma-4-26b-a4b-it here) are dropped; pro is not configured
    assert gemini == ["gemini-3.8-flash", "gemini-3.7-flash", "gemma-4-31b-it", "gemini-3.1-flash-lite"]
    best = [p for p in slots if p.tier == 0]
    order = [cfg.chain.index(p.name) for p in best]
    assert order == sorted(order)          # within a tier: provider by provider, in chain order
    assert {p.name for p in best} >= {"gemini", "groq"}
    forced = make_provider(cfg, "groq", log=lambda m: None)   # --provider groq still rotates its models
    assert isinstance(forced, FallbackProvider) and {p.name for p in forced.providers} == {"groq"}


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


def test_gemini_drops_json_mode_for_models_without_it(monkeypatch, no_keys):
    from unseen.llm import gemini
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    sent = []
    good = {"candidates": [{"content": {"parts": [{"text": "{}"}]}, "finishReason": "STOP"}]}

    def fake_post(url, json=None, headers=None, timeout=None):
        sent.append(_copy(json))
        if "responseMimeType" in json["generationConfig"]:
            return Resp(400, text="JSON mode is not enabled for models/gemma-4-31b-it")
        return Resp(200, good)

    monkeypatch.setattr(gemini.httpx, "post", fake_post)
    p = gemini.GeminiProvider("https://g/v1beta", "gemma-4-31b-it", thinking_budget=None)
    assert p.complete_json("s", "u", {}, 0.5, 100)[0] == "{}" and not p.json_mode
    p.complete_json("s", "u", {}, 0.5, 100)
    assert len(sent) == 3 and "responseMimeType" not in sent[2]["generationConfig"]   # remembered


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
    engine.cfg.quality.drafts = 1
    record, _ = engine.new_story(category="fiction_horror", target_seconds=30)
    assert writer.calls == ["ConceptList", "StoryDraft"] and local.calls == []
    assert critic.calls == ["ConceptRatings", "Critique"]   # the critic rates concepts and the script
    assert [s["provider"] for s in record.llm] == ["gemini", "groq", "gemini", "groq"]


class Slot:
    """A provider slot that raises the queued errors in order, then answers."""

    def __init__(self, name, model, *errors, tier=0):
        self.name, self.model, self.tier, self.errors, self.calls = name, model, tier, list(errors), 0

    def complete_json(self, *a):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return f"from {self.name}:{self.model}", {}


def fake_time():
    now = [0.0]
    slept = []

    def sleep(s):
        slept.append(s)
        now[0] += s
    return now, slept, (lambda: now[0]), sleep


def test_every_model_of_a_provider_before_the_next_provider():
    now, slept, clock, sleep = fake_time()
    g1 = Slot("gemini", "g-3.8", TransientLLMError("per-minute", retry_after=30))
    g2 = Slot("gemini", "g-3.7", TransientLLMError("daily", daily=True))
    q1 = Slot("groq", "oss-120b")
    local = Slot("ollama", "gemma", tier=1)
    fb = FallbackProvider([g1, g2, q1, local], log=lambda m: None, clock=clock, sleep=sleep)
    # call 1: g1 limited for 30 s -> g2's daily quota is gone -> only then the next provider
    assert fb.complete_json("s", "u", {}, 0, 1)[0] == "from groq:oss-120b"
    assert fb.name == "groq" and g1.calls == 1 and g2.calls == 1 and q1.calls == 1
    assert fb.complete_json("s", "u", {}, 0, 1)[0] == "from groq:oss-120b"      # g1 still cooling, g2 gone
    assert g1.calls == 1 and g2.calls == 1
    now[0] = 31.0
    assert fb.complete_json("s", "u", {}, 0, 1)[0] == "from gemini:g-3.8"       # preferred again
    assert local.calls == 0 and slept == []


def test_short_retry_then_wait_for_a_hosted_model_before_the_local_one():
    now, slept, clock, sleep = fake_time()
    g1 = Slot("gemini", "g", TransientLLMError("busy", retry_after=3), TransientLLMError("busy", retry_after=40))
    q1 = Slot("groq", "q", TransientLLMError("busy"))
    local = Slot("ollama", "gemma", tier=1)
    fb = FallbackProvider([g1, q1, local], log=lambda m: None, clock=clock, sleep=sleep, max_wait_s=300)
    # g1: retry in 3 s -> then busy for 40 s; q1: busy (cools 60 s); both cooling -> wait for g1, not ollama
    assert fb.complete_json("s", "u", {}, 0, 1)[0] == "from gemini:g"
    assert slept == [3.0, 40.0] and local.calls == 0 and g1.calls == 3
    # a wait longer than max_wait_s: the local model answers instead
    g1.errors, q1.errors = [TransientLLMError("busy", retry_after=600)], [TransientLLMError("busy", retry_after=600)]
    fb.max_wait_s = 60
    assert fb.complete_json("s", "u", {}, 0, 1)[0] == "from ollama:gemma" and fb.name == "ollama"


def test_provider_wide_limits_and_per_call_skips():
    now, slept, clock, sleep = fake_time()
    o1 = Slot("openrouter", "a:free", TransientLLMError("free-models-per-day", daily=True, family=True))
    o2 = Slot("openrouter", "b:free")
    q1 = Slot("groq", "big", SkipCallLLMError("request too large"))
    q2 = Slot("groq", "small")
    fb = FallbackProvider([o1, o2, q1, q2], log=lambda m: None, clock=clock, sleep=sleep)
    assert fb.complete_json("s", "u", {}, 0, 1)[0] == "from groq:small"   # every openrouter model skipped
    assert o2.calls == 0 and q1.calls == 1
    assert fb.complete_json("s", "u", {}, 0, 1)[0] == "from groq:big"     # the skip was for that call only
    k1 = Slot("gemini", "x", ProviderLLMError("API key rejected"))
    k2 = Slot("gemini", "y")
    fb2 = FallbackProvider([k1, k2, Slot("groq", "z")], log=lambda m: None, clock=clock, sleep=sleep)
    assert fb2.complete_json("s", "u", {}, 0, 1)[0] == "from groq:z" and k2.calls == 0


def test_critic_prefers_other_providers_within_each_tier():
    g1, g2 = Slot("gemini", "pro"), Slot("gemini", "flash")
    q1, q2 = Slot("groq", "oss-120b"), Slot("groq", "qwen", tier=1)
    g3, local = Slot("gemini", "lite", tier=1), Slot("ollama", "gemma", tier=1)
    fb = FallbackProvider([g1, g2, q1, q2, g3, local], log=lambda m: None)
    fb.complete_json("s", "u", {}, 0, 1)                  # g1 wrote
    view = fb.critic_view()
    assert [FallbackProvider.key(p) for p in view.providers] == [
        "groq:oss-120b", "gemini:flash", "groq:qwen", "gemini:lite"]
    minor = fb.minor_view()                               # repairs: normal tier first, best quota saved
    assert [FallbackProvider.key(p) for p in minor.providers] == [
        "groq:qwen", "gemini:lite", "ollama:gemma", "gemini:pro", "gemini:flash", "groq:oss-120b"]
    assert minor.complete_json("s", "u", {}, 0, 1)[0] == "from groq:qwen" and g1.calls == 1
    assert minor._cool is fb._cool


def test_gemini_per_model_settings_override_the_defaults(monkeypatch, no_keys):
    from unseen.llm import gemini
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    sent = []
    good = {"candidates": [{"content": {"parts": [{"text": "{}"}]}, "finishReason": "STOP"}]}
    monkeypatch.setattr(gemini.httpx, "post", lambda url, json=None, **k: sent.append(_copy(json)) or Resp(200, good))
    p = gemini.GeminiProvider("https://g/v1beta", "gemma-4-31b-it", thinking_budget=1024,
                              generation={"thinkingConfig": {"thinkingLevel": "minimal"}})
    p.complete_json("s", "u", {}, 0.5, 100)
    assert sent[0]["generationConfig"]["thinkingConfig"] == {"thinkingLevel": "minimal"}


def test_critic_view_shares_cooldowns():
    a, b, c = Stub("gemini"), Stub("groq"), Stub("ollama")
    fb = FallbackProvider([a, b, c], log=lambda m: None)
    view = fb.excluding({"gemini", "ollama"})
    assert [p.name for p in view.providers] == ["groq"]
    fb._cool_down("groq")
    assert view._cool is fb._cool and "groq" in view._cool
    assert fb.excluding({"gemini", "groq", "ollama"}) is None


def test_gemini_sends_images_and_only_seeing_models_get_image_calls(monkeypatch, no_keys):
    import base64
    from unseen.llm import gemini
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    sent = []
    good = {"candidates": [{"content": {"parts": [{"text": "{}"}]}, "finishReason": "STOP"}]}
    monkeypatch.setattr(gemini.httpx, "post", lambda url, json=None, **k: sent.append(_copy(json)) or Resp(200, good))
    p = gemini.GeminiProvider("https://g/v1beta", "gemini-3.8-flash")
    p.complete_json("s", "u", {}, 0.5, 100, images=[("Scene 1, frame at 0.0 s:", b"\xff\xd8jpeg")])
    parts = sent[0]["contents"][0]["parts"]
    assert parts[1] == {"text": "Scene 1, frame at 0.0 s:"}
    assert parts[2]["inline_data"] == {"mime_type": "image/jpeg", "data": base64.b64encode(b"\xff\xd8jpeg").decode()}

    class Seeing(Slot):
        supports_images = True

        def complete_json(self, *a, images=None):
            self.got = images
            return super().complete_json(*a)

    blind, seeing = Slot("groq", "oss"), Seeing("gemini", "flash")
    fb = FallbackProvider([blind, seeing], log=lambda m: None)
    assert fb.complete_json("s", "u", {}, 0, 1, images=[("x", b"y")])[0] == "from gemini:flash"
    assert blind.calls == 0 and seeing.got == [("x", b"y")] and fb.name == "gemini"
    assert fb.complete_json("s", "u", {}, 0, 1)[0] == "from groq:oss"          # text calls: normal order
    with pytest.raises(LLMError, match="accepts images"):
        FallbackProvider([Slot("groq", "a"), Slot("zai", "b")], log=lambda m: None).complete_json(
            "s", "u", {}, 0, 1, images=[("x", b"y")])
