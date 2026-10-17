import json

import pytest

import unseen.story.render as render
from storyfakes import CONCEPTS, CRITIQUE_PASS, CRITIQUE_WEAK, RATINGS, FakeEmbedder, FakeProvider, draft_dict
from unseen.state.db import StateDB
from unseen.story.config import Topic
from unseen.story.engine import StoryEngine
from unseen.story.models import Source
from unseen.story.research import ResearchError, parse_article, trim_source

SOURCE = ("The Example lights are unexplained lights seen in the Example valley since 1981. Observers "
          "reported white and yellow lights hovering above the ground for up to two hours. A research "
          "station was set up in 1983 to record them automatically. Some researchers proposed that dust "
          "burning in the air explains the glow. Others suggested piezoelectric effects in the rocks. "
          "No explanation has been confirmed. ") * 6


def fake_fetch(title):
    return SOURCE, Source(title=f"Wikipedia: {title}", url="https://en.wikipedia.org/wiki/Example",
                          license="CC BY-SA 4.0 (Wikipedia)", revision_id=1, retrieved_at="2026-10-17T00:00:00Z")


CLAIMS = {"claims": [
    {"id": 7, "text": "Lights have been seen since 1981.", "quote": "seen in the Example valley since 1981", "tag": "fact"},
    {"id": 8, "text": "They hover for up to two hours.", "quote": "hovering above the ground for up to two hours", "tag": "fact"},
    {"id": 9, "text": "A station records them.", "quote": "A research station was set up in 1983", "tag": "fact"},
    {"id": 10, "text": "Burning dust may explain it.", "quote": "dust burning in the air explains the glow", "tag": "theory"},
    {"id": 11, "text": "Invented detail.", "quote": "the lights spelled out a warning", "tag": "legend"},
]}


def single_draft(engine, rounds: int = 2):
    """Pin the revision-loop semantics these tests were written for (one draft, two rewrite rounds)."""
    engine.cfg.quality.drafts = 1
    engine.cfg.quality.revision_rounds = rounds
    return engine


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(render, "STORIES_DIR", tmp_path / "stories")
    return tmp_path


def test_fiction_pipeline_writes_files_and_db(env):
    provider = FakeProvider({"ConceptList": [CONCEPTS], "ConceptRatings": [RATINGS], "StoryDraft": [draft_dict()],
                             "Critique": [CRITIQUE_PASS]})
    db = StateDB(env / "t.db")
    engine = StoryEngine(provider=provider, db=db, embedder=FakeEmbedder(), seed=1, log=lambda m: None)
    single_draft(engine)
    record, out = engine.new_story(category="fiction_horror", target_seconds=30)
    assert record.problems == [] and not record.revised
    assert record.concept.title == "Second Shift" and record.concept_ratings["Second Shift"] == 33
    assert {"device", "location", "anomaly"} <= set(record.ingredients)
    assert "structure" not in record.ingredients            # only for Reels of 40 s or more
    assert (out / "story.json").is_file() and (out / "story.md").is_file()
    prompts = json.loads((out / "prompts.json").read_text(encoding="utf-8"))
    assert len(prompts) == len(record.draft.scenes) and all(p["text"] for p in prompts)
    assert len({p["id"] for p in prompts}) == len(prompts)
    assert db.get(record.id)["title"] == "Room Nine" and len(db.embeddings()) == 1
    assert provider.calls == ["ConceptList", "ConceptRatings", "StoryDraft", "Critique"] and provider.unloaded
    assert "AI-generated" in record.ai_disclosure
    assert "Editor notes" in (out / "story.md").read_text(encoding="utf-8")


def test_local_model_gets_one_patch_then_a_second_critique(env):
    patch = {"narration": [{"id": 6, "text": "The camera in the footage is still recording.", "claim_ids": []}],
             "scenes": [{"id": 7, "video_prompt": ""}], "ending_note": "Sharper ending."}
    provider = FakeProvider({"ConceptList": [CONCEPTS], "StoryDraft": [draft_dict()],
                             "Critique": [CRITIQUE_WEAK, CRITIQUE_PASS], "StoryPatch": [patch]})
    provider.name = "ollama"
    engine = StoryEngine(provider=provider, db=StateDB(env / "t.db"), embedder=FakeEmbedder(), seed=2,
                         log=lambda m: None)
    single_draft(engine)
    record, _ = engine.new_story(category="psychological_mystery", target_seconds=30)
    assert record.revised and record.draft.ending_note == "Sharper ending."
    assert record.draft.narration[5].text == "The camera in the footage is still recording."
    assert record.draft.scenes[6].video_prompt            # empty edit keeps the original
    assert provider.calls == ["ConceptList", "StoryDraft", "Critique", "StoryPatch", "Critique"]
    assert [h["avg"] for h in record.critique_history] == [7.25, 8.0]
    assert all(h["kept"] for h in record.critique_history)


def test_hosted_writer_rewrites_and_keeps_the_better_version(env):
    improved = draft_dict(ending_note="Sharper ending.")
    provider = FakeProvider({"ConceptList": [CONCEPTS], "StoryDraft": [draft_dict(), improved],
                             "Critique": [CRITIQUE_WEAK, CRITIQUE_PASS]})
    engine = StoryEngine(provider=provider, db=StateDB(env / "t.db"), embedder=FakeEmbedder(), seed=3,
                         log=lambda m: None)
    single_draft(engine)
    record, _ = engine.new_story(category="fiction_horror", target_seconds=30)
    assert record.revised and record.draft.ending_note == "Sharper ending."
    assert provider.calls.count("StoryDraft") == 2 and "StoryPatch" not in provider.calls
    assert record.critique.scores()["ending"] == 8


def test_rewrite_that_scores_lower_is_discarded(env):
    worse = dict(CRITIQUE_WEAK, hook=3)
    provider = FakeProvider({"ConceptList": [CONCEPTS],
                             "StoryDraft": [draft_dict(), draft_dict(ending_note="Worse 1."),
                                            draft_dict(ending_note="Worse 2.")],
                             "Critique": [CRITIQUE_WEAK, worse]})
    engine = StoryEngine(provider=provider, db=StateDB(env / "t.db"), embedder=FakeEmbedder(), seed=6,
                         log=lambda m: None)
    single_draft(engine)
    record, _ = engine.new_story(category="fiction_horror", target_seconds=30)
    assert not record.revised and record.draft.ending_note == "The camera is not hers."
    assert [h["kept"] for h in record.critique_history] == [True, False, False]
    assert provider.calls.count("StoryDraft") == 3          # script + revision_rounds rewrites


def test_rule_breaking_lines_are_rewritten_not_the_whole_script(env):
    d = draft_dict()
    d["narration"][0]["text"] = "This hook line is far far too long to be a hook for anyone at all"
    fixes = {"lines": [{"id": 1, "text": "The tape starts mid-step.", "claim_ids": []}]}
    provider = FakeProvider({"ConceptList": [CONCEPTS], "StoryDraft": [d], "LineFixes": [fixes],
                             "Critique": [CRITIQUE_PASS]})
    engine = StoryEngine(provider=provider, db=StateDB(env / "t.db"), embedder=FakeEmbedder(), seed=4,
                         log=lambda m: None)
    single_draft(engine)
    record, _ = engine.new_story(category="fiction_horror", target_seconds=30)
    assert record.draft.narration[0].text == "The tape starts mid-step."
    assert record.problems == []
    # ConceptRatings has no fake answer here: rating is skipped and the most novel concept is used
    assert provider.calls == ["ConceptList", "ConceptRatings", "StoryDraft", "LineFixes", "Critique"]


def test_real_mystery_uses_only_verified_claims(env):
    d = draft_dict()
    for n in d["narration"]:
        n["claim_ids"] = [1]
    d["narration"][3]["text"] = "Some believe burning dust explains it."
    d["narration"][3]["claim_ids"] = [4]
    d["recorded_year"] = 1985          # period footage of a 1981 case, on a 1983+ camcorder
    check = {"lines": [{"id": 1, "verdict": "supported"},
                       {"id": 2, "verdict": "contradicted", "corrected": "The lights were first seen in 1981."}]}
    provider = FakeProvider({"ClaimList": [CLAIMS], "ConceptList": [CONCEPTS], "StoryDraft": [d],
                             "Critique": [CRITIQUE_PASS], "FactCheck": [check]})
    topic = Topic(id="example", wikipedia="Example lights", year=1981, kind="lights", sensitivity="low")
    engine = StoryEngine(provider=provider, db=StateDB(env / "t.db"), embedder=FakeEmbedder(), seed=3,
                         fetch=fake_fetch, topics=[topic], log=lambda m: None)
    single_draft(engine)
    record, out = engine.new_story(category="real_mystery", target_seconds=30)
    assert [c.id for c in record.claims] == [1, 2, 3, 4]               # renumbered, invented one dropped
    assert all("warning" not in c.quote for c in record.claims)
    assert record.topic_id == "example" and record.sources[0].url.startswith("https://en.wikipedia.org/")
    assert record.draft.archive_entry.startswith("RECONSTRUCTION - ")   # never posing as real evidence
    assert record.problems == []
    assert "Sources: Wikipedia: Example lights" in record.ai_disclosure
    assert "quote-verified" in (out / "story.md").read_text(encoding="utf-8")
    assert record.draft.narration[1].text == "The lights were first seen in 1981."   # fact-check correction
    assert {"id": 2, "verdict": "contradicted", "original": "The corridor light was already flickering.",
            "corrected": "The lights were first seen in 1981."} in record.fact_check
    # the topic is now used: the next pick avoids it unless nothing else is left
    assert engine.db.used_topics() == {"example"}


def test_research_refuses_when_too_few_claims_verify(env):
    junk = {"claims": [{"id": i, "text": "x", "quote": f"made up quote number {i} here", "tag": "fact"}
                       for i in range(1, 6)]}
    provider = FakeProvider({"ClaimList": [junk]})
    topic = Topic(id="example", wikipedia="Example lights", year=1981, kind="lights", sensitivity="low")
    engine = StoryEngine(provider=provider, db=StateDB(env / "t.db"), embedder=FakeEmbedder(),
                         fetch=fake_fetch, topics=[topic], log=lambda m: None)
    single_draft(engine)
    with pytest.raises(ResearchError):
        engine.new_story(category="real_mystery", target_seconds=30)


def test_topic_eligibility_respects_sensitivity(env):
    topics = [Topic(id="recent_deaths", wikipedia="X", year=2015, kind="deaths", sensitivity="medium"),
              Topic(id="old_deaths", wikipedia="Y", year=1900, kind="deaths", sensitivity="medium"),
              Topic(id="recent_lights", wikipedia="Z", year=2015, kind="lights", sensitivity="low")]
    engine = StoryEngine(provider=FakeProvider({}), db=StateDB(env / "t.db"), embedder=FakeEmbedder(),
                         topics=topics, log=lambda m: None)
    single_draft(engine)
    assert {t.id for t in engine.eligible_topics()} == {"old_deaths", "recent_lights"}


def test_failed_topic_falls_back_to_another(env):
    d = draft_dict()
    for n in d["narration"]:
        n["claim_ids"] = [1]

    def flaky_fetch(title):
        if title == "Broken title":
            raise ResearchError("Wikipedia has no article 'Broken title'")
        return fake_fetch(title)

    topics = [Topic(id="broken", wikipedia="Broken title", year=1900, kind="x", sensitivity="low"),
              Topic(id="good", wikipedia="Example lights", year=1981, kind="lights", sensitivity="low")]
    provider = FakeProvider({"ClaimList": [CLAIMS], "ConceptList": [CONCEPTS], "StoryDraft": [d],
                             "Critique": [CRITIQUE_PASS]})
    for seed in range(4):   # whichever topic is picked first, the story must end up on "good"
        engine = StoryEngine(provider=provider, db=StateDB(env / f"t{seed}.db"), embedder=FakeEmbedder(),
                             seed=seed, fetch=flaky_fetch, topics=topics, log=lambda m: None)
        single_draft(engine)
        record, _ = engine.new_story(category="real_mystery", target_seconds=30)
        assert record.topic_id == "good"


def test_wikipedia_search_fallback(monkeypatch):
    import unseen.story.research as research
    article = {"query": {"pages": [{"title": "Gef", "extract": "Gef was a talking mongoose. " * 40,
                                    "fullurl": "https://en.wikipedia.org/wiki/Gef", "revisions": [{"revid": 5}]}]}}
    calls = []

    def fake_api(params, timeout=30):
        calls.append(params.get("titles") or params.get("srsearch"))
        if params.get("list") == "search":
            return {"query": {"search": [{"title": "Gef"}]}}
        if params.get("titles") == "Gef (mongoose)":
            return {"query": {"pages": [{"title": "Gef (mongoose)", "missing": True}]}}
        return article

    monkeypatch.setattr(research, "_api", fake_api)
    text, src = research.fetch_article("Gef (mongoose)")
    assert src.title == "Wikipedia: Gef" and calls == ["Gef (mongoose)", "Gef (mongoose)", "Gef"]

    monkeypatch.setattr(research, "_api", lambda params, timeout=30: (
        {"query": {"search": [{"title": "Completely unrelated page"}]}} if params.get("list")
        else {"query": {"pages": [{"title": "Nope", "missing": True}]}}))
    with pytest.raises(ResearchError):
        research.fetch_article("Gef (mongoose)")


def test_wikipedia_parsing_and_trimming():
    data = {"query": {"pages": [{"pageid": 1, "title": "Mary Celeste", "extract": "Lead. " * 200 + "\nReferences\njunk",
                                 "fullurl": "https://en.wikipedia.org/wiki/Mary_Celeste",
                                 "extlinks": [{"url": "https://example.org/a"}],
                                 "revisions": [{"revid": 123, "timestamp": "2026-01-01T00:00:00Z"}]}]}}
    text, src = parse_article(data)
    assert src.revision_id == 123 and src.external_links == ["https://example.org/a"]
    assert "junk" not in trim_source(text, 10_000)
    with pytest.raises(ResearchError):
        parse_article({"query": {"pages": [{"title": "Nope", "missing": True}]}})


def test_best_of_two_drafts_keeps_the_better_one(env):
    provider = FakeProvider({"ConceptList": [CONCEPTS], "ConceptRatings": [RATINGS],
                             "StoryDraft": [draft_dict(ending_note="first"), draft_dict(ending_note="second")],
                             "Critique": [CRITIQUE_WEAK, CRITIQUE_PASS]})
    engine = StoryEngine(provider=provider, db=StateDB(env / "t.db"), embedder=FakeEmbedder(), seed=7,
                         log=lambda m: None)
    record, _ = engine.new_story(category="fiction_horror", target_seconds=30)
    # ranked: Second Shift (33) then Low Tide (26); the Low Tide draft scored higher
    assert record.concept.title == "Low Tide" and record.draft.ending_note == "second"
    assert provider.calls == ["ConceptList", "ConceptRatings", "StoryDraft", "Critique", "StoryDraft", "Critique"]
    assert not record.revised


def test_ingredients_keep_concepts_simple(env):
    engine = StoryEngine(provider=FakeProvider({}), db=StateDB(env / "t.db"), embedder=FakeEmbedder(), seed=11,
                         log=lambda m: None)
    short = [engine.sample_ingredients(20) for _ in range(40)]
    long = [engine.sample_ingredients(50) for _ in range(200)]
    assert all({"device", "location", "anomaly"} <= set(i) for i in short + long)
    assert not any("structure" in i for i in short)
    assert any("structure" in i for i in long) and not all("structure" in i for i in long)
    assert any("twist" in i for i in short) and not all("twist" in i for i in short)
