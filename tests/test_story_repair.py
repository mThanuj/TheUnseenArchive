from collections import Counter

from storyfakes import PROMPT, draft
from unseen.story.config import load_story_config
from unseen.story.models import StoryPatch
from unseen.story.repair import apply_patch, auto_fix, fit_video_prompts, line_issues, place_lines
from unseen.story.validate import banned_in, words

CFG = load_story_config()


def test_lines_are_placed_once_in_order_and_unknown_ids_dropped():
    d = draft()
    d.scenes[0].narration_ids = []          # line 1 orphaned
    d.scenes[1].narration_ids = [2, 99]     # unknown id
    d.scenes[2].narration_ids = [2, 3]      # line 2 duplicated
    place_lines(d)
    counts = Counter(i for s in d.scenes for i in s.narration_ids)
    assert all(counts[n.id] == 1 for n in d.narration) and 99 not in counts
    assert d.scenes[0].narration_ids == [1] and d.scenes[2].narration_ids == [3]


def test_durations_fit_target_and_narration():
    d = draft()
    auto_fix(d, CFG, 30)
    assert abs(sum(s.duration_s for s in d.scenes) - 30) <= 0.5
    d = draft()
    auto_fix(d, CFG, 60)                    # 7 scenes x 5 s max = 35 s: clamp, don't explode
    assert all(s.duration_s == 5.0 for s in d.scenes)
    d = draft()
    d.scenes[0].duration_s = 2.0            # too short for its narration line
    auto_fix(d, CFG, 28)
    assert d.scenes[0].duration_s >= 3.7


def test_video_prompts_are_cleaned_enriched_and_trimmed():
    d = draft()
    d.scenes[0].video_prompt = "A door opens slowly. Grainy VHS look."
    d.scenes[1].video_prompt = " ".join([PROMPT] * 3)
    fit_video_prompts(d, CFG)
    assert not banned_in(d.scenes[0].video_prompt, CFG.rules.video_prompt_banned_words)
    assert "The camera is handheld, walking." in d.scenes[0].video_prompt
    assert len(words(d.scenes[1].video_prompt)) <= CFG.rules.video_prompt_words.max
    assert d.scenes[1].video_prompt.endswith(".")


def test_line_issues_target_only_offending_lines():
    d = draft()
    d.narration[0].text = "This hook line is far far too long to be a hook for anyone at all"
    d.narration[3].text = "Little did they know the room was waiting."
    issues = line_issues(d, CFG, 30, "fiction_horror", None)
    assert set(issues) == {1, 4}
    assert "hook" in issues[1] and "cliche" in issues[4]
    assert line_issues(draft(), CFG, 30, "fiction_horror", None) == {}


def test_patch_edits_only_named_fields():
    d = draft()
    patch = StoryPatch.model_validate({"narration": [{"id": 2, "text": "The light had stopped."}],
                                       "scenes": [{"id": 3, "lighting": "a single red exit sign"}, {"id": 42}],
                                       "ending_note": ""})
    changed = apply_patch(d, patch, "fiction_horror")
    assert changed == ["line 2", "scene 3.lighting"]
    assert d.narration[1].text == "The light had stopped." and d.scenes[2].lighting == "a single red exit sign"
    assert d.scenes[2].video_prompt == PROMPT and d.ending_note == "The camera is not hers."
