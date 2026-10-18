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
    assert "- scenes: list of objects (minItems=3/maxItems=12)" in guide
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


def test_first_person_slips_are_flagged_but_quotes_allowed(cfg):
    from unseen.story.validate import first_person_slip
    assert first_person_slip("Footsteps echo from above, but the shadow just passed me.")
    assert first_person_slip("My lamp was out.")
    assert not first_person_slip('The last log reads: "I am opening the door."')
    assert not first_person_slip("Nobody at the motel owns a camera like hers.")
    d = draft_dict()
    d["narration"][1]["text"] = "I watched the corridor light flicker."
    probs = validate_story(StoryDraft.model_validate(d), cfg, 30, "fiction_horror")
    assert any("line 2 uses I/me/my" in p for p in probs)


def test_script_prompt_asks_for_an_exact_scene_count(cfg):
    from storyfakes import CONCEPTS
    from unseen.story.models import Concept
    from unseen.story.prompts import scene_count, script_prompt
    assert [scene_count(cfg, s) for s in (20, 30, 40, 50, 90)] == [5, 8, 10, 12, 12]
    p = script_prompt(cfg, "fiction_horror", 40, Concept.model_validate(CONCEPTS["concepts"][0]), {}, {}, None)
    assert "EXACTLY 10 scenes" in p and "at most 11 spoken words" in p


def test_llm_near_miss_values_are_coerced():
    from unseen.story.models import NarrationLine, Scene, SfxCue
    base = dict(draft_dict()["scenes"][0])
    assert Scene.model_validate({**base, "device": "Smartphone"}).device == "phone"
    assert Scene.model_validate({**base, "device": "security camera"}).device == "cctv"
    assert Scene.model_validate({**base, "device": "hologram"}).device == "camcorder"
    assert SfxCue.model_validate({"at": 0.3, "sound": "click"}).sound == "machine"
    assert NarrationLine.model_validate({"id": 1, "text": "x", "delivery": "precise"}).delivery == "flat"
    assert Critique.model_validate({k: 7 for k in ("hook", "escalation", "coherence", "originality", "ending",
                                                   "visual_feasibility", "visual_variety", "brand_fit")}
                                   | {"issues": []}).verdict == "revise"


def test_era_and_readable_text_checks(cfg):
    d = draft_dict(recorded_year=1977)
    d["scenes"][1]["device"] = "phone"
    d["scenes"][2]["video_prompt"] = d["scenes"][2]["video_prompt"] + ' A printout reads "6EQUJ5" in red ink.'
    d["narration"][0]["text"] = "An eerie tape from the motel."
    probs = validate_story(StoryDraft.model_validate(d), cfg, 30, "fiction_horror")
    assert any("scene 2: a phone could not record footage in 1977" in p for p in probs)
    assert any("scene 3 video_prompt asks for readable text" in p for p in probs)
    assert any("line 1 uses ['eerie']" in p for p in probs)
    # real cases: period footage or a present-day revisit, nothing in between
    ok_period = validate_story(StoryDraft.model_validate(draft_dict(recorded_year=1985)), cfg, 30,
                               "real_mystery", [], 1981)
    assert not any("recorded_year" in p for p in ok_period)
    mid = validate_story(StoryDraft.model_validate(draft_dict(recorded_year=2004)), cfg, 30, "real_mystery", [], 1981)
    assert any("recorded_year 2004 fits neither" in p for p in mid)


def test_osd_dates_are_stripped_and_scene_issues_found(cfg):
    from unseen.story.repair import apply_scene_fixes, clean_osd, scene_issues
    from unseen.story.models import SceneFixes
    d = draft(recorded_year=1977)
    d.scenes[0].osd = "CAM 01  10-17-2026  20:03:12"
    d.scenes[1].osd = "03:14:12 PM  REC"
    d.scenes[2].osd = "REEL 04 - 1974"
    d.scenes[3].device = "film_16mm"
    d.scenes[3].osd = "REC"
    clean_osd(d)
    assert [s.osd for s in d.scenes[:4]] == ["CAM 01", "REC", "REEL 04", ""]    # film has no overlay
    issues = scene_issues(d, cfg, "fiction_horror", None)
    assert set(issues) == {s.id for s in d.scenes if s.device == "camcorder"}   # camcorder (1983+) in 1977
    fixes = SceneFixes.model_validate({"scenes": [{"id": 1, "device": "16mm", "video_prompt": "A new prompt."}]})
    assert apply_scene_fixes(d, fixes, {1}) == ["scene 1 rewritten"]
    assert d.scenes[0].device == "film_16mm" and d.scenes[0].video_prompt == "A new prompt."


def test_too_few_scenes_and_uncited_specifics_are_flagged(cfg):
    d = draft_dict()
    d["scenes"] = d["scenes"][:3]
    probs = validate_story(StoryDraft.model_validate(d), cfg, 20, "fiction_horror")
    assert any("only 3 scenes; a 20 s Reel needs 5" in p for p in probs)
    claims = [Claim(id=1, text="Lights were seen.", quote="lights were seen", tag="fact")]
    d = draft_dict()
    for n in d["narration"]:
        n["claim_ids"] = [1]
    d["narration"][1]["text"] = "The dashcam logged it at 14:03."      # invented time, no claim behind it
    d["narration"][1]["claim_ids"] = []
    probs = validate_story(StoryDraft.model_validate(d), cfg, 30, "real_mystery", claims)
    assert any("line 2 states a specific detail without citing a claim" in p for p in probs)
    d["narration"][1]["text"] = "No one came back for it."             # atmospheric: fine uncited
    assert validate_story(StoryDraft.model_validate(d), cfg, 30, "real_mystery", claims) == []


def test_unhedged_theory_lines_get_a_hedge_as_last_resort(cfg):
    from unseen.story.repair import hedge_lines
    claims = [Claim(id=1, text="Dust may burn.", quote="dust burning", tag="theory"),
              Claim(id=2, text="Seen since 1981.", quote="since 1981", tag="fact")]
    d = draft()
    d.narration[0].text, d.narration[0].claim_ids = "Dust analysis shows titanium.", [1]
    d.narration[1].text, d.narration[1].claim_ids = "Some believe the dust burns.", [1]
    d.narration[2].claim_ids = [2]
    assert hedge_lines(d, cfg, claims) == ["line 1 -> Dust analysis shows titanium, according to one theory."]
    assert d.narration[1].text == "Some believe the dust burns."        # already hedged: untouched


def test_a_legend_is_framed_once_but_theories_every_time(cfg):
    from unseen.story.repair import hedge_lines
    from unseen.story.validate import unhedged_lines
    claims = [Claim(id=1, text="Two green children appeared.", quote="two green children", tag="legend"),
              Claim(id=2, text="They came from underground.", quote="from underground", tag="theory")]
    d = draft()
    for n in d.narration[:4]:
        n.claim_ids = [1]
    d.narration[4].claim_ids = [2]
    assert unhedged_lines(d, cfg, claims) == {1: ["legend"], 2: ["legend"], 3: ["legend"], 4: ["legend"],
                                              5: ["theory"]}
    d.narration[0].text = "According to two medieval chroniclers, the children walked out of a pit."
    assert unhedged_lines(d, cfg, claims) == {5: ["theory"]}     # framed once: lines 2-4 can be plain
    d = draft()
    for n in d.narration[:3]:
        n.claim_ids = [1]
    assert hedge_lines(d, cfg, claims) == ["line 1 -> Entry four twelve. The tape starts mid-step, reportedly."]
    assert d.narration[1].text == "The corridor light was already flickering."   # inside the frame


def test_real_cases_keep_one_device_look_and_clock_times_count_as_text(cfg):
    from unseen.story.repair import scene_issues
    from unseen.story.validate import device_mix_issues
    d = draft()
    d.scenes[2].device = "phone"
    assert device_mix_issues(d, cfg, "real_mystery", 1150) == {
        3: "use camcorder like the rest of this reconstruction (one device for the whole reconstruction)"}
    assert device_mix_issues(d, cfg, "real_mystery", 1981) == {}   # two looks are fine after cameras existed
    d.scenes[3].device = "cctv"
    assert set(device_mix_issues(d, cfg, "real_mystery", 1981)) == {4}   # a third look is not
    assert device_mix_issues(d, cfg, "fiction_horror", None) == {}
    assert 3 in scene_issues(d, cfg, "real_mystery", 1150)
    from unseen.story.validate import readable_text_risk
    assert readable_text_risk("A digital readout flashes 07:45 above the door.",
                              cfg.rules.video_prompt_text_risk) == ["readout", "07:45"]


def test_real_victims_are_never_shown_and_banned_sentences_are_dropped(cfg):
    from unseen.story.repair import scene_issues
    from unseen.story.validate import victim_shots
    d = draft()
    d.scenes[5].video_prompt += " Three bodies lie half buried in the snow by the creek."
    assert victim_shots(d, cfg, "real_mystery") == {6: ["bodies"]}
    assert victim_shots(d, cfg, "fiction_horror") == {}
    assert "never show real victims" in scene_issues(d, cfg, "real_mystery", 1959)[6]
    assert any("shows real victims" in p for p in validate_story(d, cfg, 30, "real_mystery", [], 1959))
    clean = sanitize_video_prompt("The sensor renders the scene in grainy, muted tones. A light blinks.",
                                  cfg.rules.video_prompt_banned_words)
    assert clean == "A light blinks."                        # the whole sentence goes, no "in, muted"


def test_rewrite_prompt_carries_an_outline_not_the_full_draft(cfg):
    from storyfakes import CONCEPTS, CRITIQUE_WEAK, PROMPT
    from unseen.story.models import Concept
    from unseen.story.prompts import revise_prompt, script_prompt
    base = script_prompt(cfg, "fiction_horror", 30, Concept.model_validate(CONCEPTS["concepts"][0]), {}, {}, None)
    p = revise_prompt(base, draft(), Critique.model_validate(CRITIQUE_WEAK))
    assert PROMPT not in p and "[6] (calm, pause 0.6s) Nobody at the motel owns a camera like hers." in p
    assert "scene 7: ending is flat" in p


def test_repeated_narration_lines_are_flagged(cfg):
    from unseen.story.repair import line_issues
    from unseen.story.validate import repeated_lines
    assert repeated_lines(draft()) == {}
    d = draft()
    d.narration[1].text = "The 2019 inspection found the vault's interior brickwork intact."
    d.narration[4].text = "2019 inspection found brickwork intact, no impact signs."
    assert repeated_lines(d) == {5: 2}
    assert "repeats line 2" in line_issues(d, cfg, 30, "fiction_horror", None)[5]
    assert any("line 5 repeats line 2" in p for p in validate_story(d, cfg, 30, "fiction_horror"))
