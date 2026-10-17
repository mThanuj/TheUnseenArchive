import pytest

from storyfakes import CRITIQUE_PASS, FakeProvider, draft, draft_dict
from unseen.llm.base import LLMError, extract_json, generate, inline_refs
from unseen.story.config import load_story_config, load_topics
from unseen.story.models import Claim, Critique, StoryDraft
from unseen.story.validate import (banned_in, quote_in_source, sanitize_video_prompt, validate_story,
                                   verify_claims)


@pytest.fixture(scope="module")
def cfg():
    return load_story_config()


def test_configs_load(cfg):
    assert set(cfg.categories) == {"fiction_horror", "psychological_mystery", "real_mystery"}
    topics = load_topics()
    assert len({t.id for t in topics}) == len(topics) >= 20
    assert all(t.sensitivity in {"low", "medium"} for t in topics)


def test_schema_is_self_contained():
    schema = inline_refs(StoryDraft.model_json_schema())
    assert "$ref" not in str(schema) and "$defs" not in schema
    assert schema["properties"]["scenes"]["items"]["properties"]["device"]["enum"]


def test_schema_guide_lists_nested_fields_with_descriptions():
    from unseen.llm.base import schema_guide
    guide = schema_guide(inline_refs(StoryDraft.model_json_schema()))
    assert "- scenes: list of objects (minItems=3/maxItems=10)" in guide
    assert "- scenes[].video_prompt (string): final text-to-video prompt" in guide
    assert "- scenes[].device (string; one of: camcorder" in guide
    assert "- narration[].pause_after_s (number; min 0.0; max 3.0)" in guide
    assert len(guide) < 6000          # compact enough for CPU prompt processing


def test_word_count_handles_curly_apostrophes_and_accents():
    from unseen.story.validate import words
    assert len(words("Don’t open the café door")) == 5
    assert quote_in_source("the cafe was closed", "In 1972 the café was closed for good.")


def test_extract_json_handles_fences_and_chatter():
    assert extract_json('Sure!\n```json\n{"a": 1}\n```') == '{"a": 1}'
    assert extract_json('noise {"a": {"b": 2}} trailing') == '{"a": {"b": 2}}'
    with pytest.raises(ValueError):
        extract_json("no json here")


def test_valid_draft_has_no_problems(cfg):
    assert validate_story(draft(), cfg, 30, "fiction_horror") == []


def test_validator_catches_each_rule(cfg):
    d = draft_dict()
    d["narration"][0]["text"] = "This hook line is far far too long to be a hook for anyone at all"
    assert any("hook" in p for p in validate_story(StoryDraft.model_validate(d), cfg, 30, "fiction_horror"))

    d = draft_dict()
    d["scenes"][0]["narration_ids"] = []          # line 1 unassigned
    assert any("line 1 is not assigned" in p for p in validate_story(StoryDraft.model_validate(d), cfg, 30, "fiction_horror"))

    assert any("add up to" in p for p in validate_story(draft(), cfg, 60, "fiction_horror"))

    d = draft_dict()
    d["scenes"][2]["video_prompt"] = d["scenes"][2]["video_prompt"] + " Heavy film grain and VHS glitch."
    probs = validate_story(StoryDraft.model_validate(d), cfg, 30, "fiction_horror")
    assert any("scene 3 video_prompt contains" in p for p in probs)

    d = draft_dict(logline="Little did they know the motel was haunted.")
    assert any("cliche" in p for p in validate_story(StoryDraft.model_validate(d), cfg, 30, "fiction_horror"))

    d = draft_dict()
    d["scenes"][3]["video_prompt"] = "A door opens."
    assert any("scene 4 video_prompt has 3 words" in p for p in validate_story(StoryDraft.model_validate(d), cfg, 30, "fiction_horror"))


def test_real_mystery_grounding_rules(cfg):
    claims = [Claim(id=1, text="The ship was found adrift.", quote="was found adrift", tag="fact"),
              Claim(id=2, text="A waterspout may have caused panic.", quote="a waterspout", tag="theory")]
    d = draft_dict()
    for n in d["narration"]:
        n["claim_ids"] = [1]
    d["narration"][2]["claim_ids"] = [2]                     # theory stated without hedge
    d["narration"][3]["claim_ids"] = [9]                     # unknown claim
    probs = validate_story(StoryDraft.model_validate(d), cfg, 30, "real_mystery", claims)
    assert any("line 3 states a theory claim as fact" in p for p in probs)
    assert any("unknown claim ids [9]" in p for p in probs)

    d["narration"][2]["text"] = "Some believe a waterspout drove them off."
    d["narration"][3]["claim_ids"] = [1]
    assert validate_story(StoryDraft.model_validate(d), cfg, 30, "real_mystery", claims) == []

    for n in d["narration"]:
        n["claim_ids"] = []
    assert any("cite claims" in p for p in validate_story(StoryDraft.model_validate(d), cfg, 30, "real_mystery", claims))


def test_quote_verification_tolerates_punctuation_but_not_invention():
    source = "On 4 December 1872, the ship was sighted by the Dei Gratia — drifting, with no one aboard."
    assert quote_in_source("the ship was sighted by the Dei Gratia, drifting", source)
    assert quote_in_source("THE SHIP WAS SIGHTED BY THE \"DEI GRATIA\"", source)
    assert not quote_in_source("the crew was taken by a sea monster", source)
    assert not quote_in_source("the ship", source)          # too short to count as evidence
    ok, bad = verify_claims([Claim(id=1, text="x", quote="sighted by the Dei Gratia", tag="fact"),
                             Claim(id=2, text="y", quote="taken by pirates at night", tag="theory")], source)
    assert [c.id for c in ok] == [1] and [c.id for c in bad] == [2]


def test_banned_words_whole_word_and_sanitizer(cfg):
    banned = cfg.rules.video_prompt_banned_words
    assert banned_in("rough stone texture on the wall", banned) == []      # 'text' is not 'texture'
    assert banned_in("Grainy VHS footage", banned) == ["grainy", "vhs"]
    clean = sanitize_video_prompt("A door opens, grainy, VHS footage with text on the wall.", banned)
    assert not banned_in(clean, banned) and "door opens" in clean


def test_generate_retries_invalid_json_then_semantic_problems(cfg):
    good = draft_dict()
    bad_hook = draft_dict()
    bad_hook["narration"][0]["text"] = "This hook line is far far too long to be a hook for anyone at all"
    provider = FakeProvider({"StoryDraft": ["not json", bad_hook, good]})
    obj, stats, problems = generate(provider, StoryDraft, system="s", user="u", name="script", max_retries=2,
                                    check=lambda d: validate_story(d, cfg, 30, "fiction_horror"), log=lambda m: None)
    assert stats.attempts == 3 and problems == [] and obj.narration[0].text.startswith("Entry")


def test_generate_gives_up_with_clear_error():
    provider = FakeProvider({"Critique": ["nope"]})
    with pytest.raises(LLMError, match="output starts: 'nope'"):
        generate(provider, Critique, system="s", user="u", name="critique", max_retries=1, log=lambda m: None)
    provider = FakeProvider({"Critique": [CRITIQUE_PASS]})
    obj, _, _ = generate(provider, Critique, system="s", user="u", name="critique", log=lambda m: None)
    assert obj.verdict == "pass"
