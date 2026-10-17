"""Prompt builders for each LLM step. The schemas carry field-level instructions; these carry
the brand, the craft rules, and the constraints of the video model."""

from __future__ import annotations

import json

from .config import StoryConfig, Topic
from .models import Claim, Concept, Critique, StoryDraft

CATEGORY_BRIEF = {
    "fiction_horror": "an original fictional horror story presented as a recovered recording",
    "psychological_mystery": ("an original psychological mystery presented as a recovered recording: "
                              "unreliable perception, behaviour that doesn't add up, conflicting evidence, "
                              "a revelation that reframes everything"),
    "real_mystery": ("a dramatized reconstruction of a documented real-world mystery, strictly faithful "
                     "to the provided claims"),
}


def system_prompt(cfg: StoryConfig) -> str:
    b, r = cfg.brand, cfg.rules
    return f"""You are the head writer of "{b.name}" ("{b.tagline}"), an Instagram Reels channel of short,
unsettling stories presented as fragments of a mysterious archive: recovered camcorder tapes,
security footage, night-vision recordings and other evidence that was never meant to be seen.

Narrator: {b.narrator}

Craft rules:
- The first spoken line is the hook: max {r.hook_max_words} words, starts mid-situation, no greeting.
- Escalate every scene; no filler, no exposition dumps; specific concrete details over adjectives.
- Implication beats explanation. Silence before a reveal. The last line lands on a concrete,
  disturbing or thought-provoking detail, never a question and never "or is it".
- Believable and documentary-like, never Hollywood. Ordinary places, ordinary people, one wrong detail.
- Avoid these cliches: {", ".join(r.banned_cliches)}.
- Content limits: {r.content_limits}

Always answer with a single JSON object that matches the requested schema."""


def concepts_prompt(cfg: StoryConfig, category: str, target_seconds: int, ingredients: dict,
                    recent_titles: list[str], topic: Topic | None, claims: list[Claim] | None) -> str:
    parts = [f"Pitch 5 different concepts for {CATEGORY_BRIEF[category]}. Finished length: about "
             f"{target_seconds} seconds (one Reel)."]
    if category == "real_mystery" and topic:
        parts.append(f"The case: {topic.wikipedia} ({topic.year}). Every concept must be built only from "
                     f"these sourced claims (no invented facts):\n{_claims_block(claims or [])}")
    else:
        parts.append("Use these ingredients as a starting point (you may twist them):\n"
                     + "\n".join(f"- {k}: {v}" for k, v in ingredients.items()))
    if recent_titles:
        parts.append("Recent stories on the channel (do NOT repeat their premise, setting or twist):\n"
                     + "\n".join(f"- {t}" for t in recent_titles))
    parts.append("Make the 5 concepts genuinely different from each other in premise, device and ending.")
    return "\n\n".join(parts)


def script_prompt(cfg: StoryConfig, category: str, target_seconds: int, concept: Concept,
                  ingredients: dict, devices: dict[str, str], claims: list[Claim] | None) -> str:
    r, ln = cfg.rules, cfg.length
    words_target = round(target_seconds * ln.words_per_second * 0.85)
    device_lines = "\n".join(f"- {k}: {v}" for k, v in devices.items())
    parts = [
        f"Write the full Reel script for this concept ({CATEGORY_BRIEF[category]}):\n"
        f"{concept.model_dump_json(indent=2)}",
        f"Timing: scenes add up to about {target_seconds} s; each scene is ONE generated video clip of "
        f"{ln.scene_seconds.min:g}-{ln.scene_seconds.max:g} s (usually {ln.scene_seconds.typical:g} s); at most "
        f"{ln.max_scenes} scenes. Narration totals about {words_target} spoken words (slow delivery, "
        f"{ln.words_per_second} words per second, with pauses). A scene may have no narration (silence is "
        f"powerful). Every narration line belongs to exactly one scene, in order.",
        f"Recording devices (scene.device) and how they look:\n{device_lines}\n"
        f"Ingredients: {json.dumps(ingredients)}",
        "video_prompt rules (a text-to-video model renders each scene from this paragraph alone, so it "
        "must be self-contained and repeat key details like place and lighting in every scene):\n"
        f"- One literal paragraph, present tense, {r.video_prompt_words.min}-{r.video_prompt_words.max} words, "
        "written like a cinematographer describing a single continuous shot, in chronological order.\n"
        "- Start with the main action; then specific movements; what is visible; the environment; the camera "
        "(the device, and how it moves or that it is fixed); lighting and colours; any change during the shot.\n"
        "- Something must visibly move or change for the whole shot (a light, a door, a figure, the camera).\n"
        "- Prefer distant or partly hidden figures, silhouettes, doorways, corridors, weather, light.\n"
        "- Avoid: readable text or signs, close-ups of hands handling objects, crowds, faces in close-up, gore.\n"
        f"- Never use these words: {', '.join(r.video_prompt_banned_words)} (grain, tape damage and the "
        "on-screen overlay are added in post-production). Never describe sound.",
        "sfx: 1-3 cues per scene placed where the action happens (a door opening, footsteps, a hum that "
        "stops). Use 'silence' right before the biggest reveal.",
        "osd: realistic camera overlay text for the device (timestamps, CAM numbers, REC) matching the era.",
    ]
    if category == "real_mystery":
        parts.append("Real case rules: every factual narration line lists the claim_ids it relies on. "
                     "Claims tagged disputed/theory/legend must be hedged in the line (reportedly, some believe, "
                     "according to...). Never add facts that are not in the claims. Visuals are a dramatized "
                     "reconstruction.\nClaims:\n" + _claims_block(claims or []))
    return "\n\n".join(parts)


def critique_prompt(cfg: StoryConfig, draft: StoryDraft, category: str) -> str:
    return (f"Critique this Reel script ({CATEGORY_BRIEF[category]}) as a demanding horror editor and as a "
            f"text-to-video specialist. Score each dimension 1-10 honestly (7 = publishable). In issues, list "
            f"concrete, actionable fixes with line/scene ids: weak hook, predictable turn, filler, confusing "
            f"continuity, shots a video model will render badly (text, hands, crowds, close-up faces, too "
            f"much happening), an ending that does not land. verdict = 'revise' if any score is below 7.\n\n"
            f"{draft.model_dump_json()}")


def revise_prompt(base_prompt: str, draft: StoryDraft, critique: Critique) -> str:
    return (f"{base_prompt}\n\nA previous draft:\n{draft.model_dump_json()}\n\nThe editor's critique "
            f"(scores {critique.scores()}):\n- " + "\n- ".join(critique.issues)
            + "\n\nWrite the improved, complete script. Keep what works; fix every issue.")


def patch_prompt(cfg: StoryConfig, draft: StoryDraft, critique: Critique, category: str,
                 claims: list[Claim] | None) -> str:
    r = cfg.rules
    parts = [f"Here is a Reel script ({CATEGORY_BRIEF[category]}):\n{draft.model_dump_json()}",
             f"The editor's critique (scores {critique.scores()}):\n- " + "\n- ".join(critique.issues),
             "Fix every issue with targeted edits only: rewrite narration lines by id and/or edit scene "
             "fields by id. Do not add, remove or renumber lines or scenes. Leave fields empty to keep them. "
             f"Narration lines max {r.narration_line_max_words} words (the first line max {r.hook_max_words}). "
             f"Video prompts follow the same rules as before ({r.video_prompt_words.min}-"
             f"{r.video_prompt_words.max} words, literal, chronological, visible motion, no "
             f"{', '.join(r.video_prompt_banned_words[:8])}...)."]
    if category == "real_mystery":
        parts.append("Use only these claims; cite claim_ids on factual lines; hedge theory/legend/disputed "
                     "claims:\n" + _claims_block(claims or []))
    return "\n\n".join(parts)


def line_repair_prompt(cfg: StoryConfig, draft: StoryDraft, issues: dict[int, str], category: str,
                       claims: list[Claim] | None) -> str:
    current = "\n".join(f"[{n.id}] {n.text}" for n in draft.narration)
    todo = "\n".join(f"[{i}] {next(n.text for n in draft.narration if n.id == i)}\n    problem: {why}"
                     for i, why in sorted(issues.items()))
    parts = [f"Story: {draft.title} - {draft.logline}\nFull narration for context:\n{current}",
             f"Rewrite ONLY these lines to fix their problems, keeping the meaning and the Archivist's voice "
             f"(max {cfg.rules.narration_line_max_words} words per line):\n{todo}",
             "Return one entry per listed line id."]
    if category == "real_mystery":
        parts.append("Claims you may cite (claim_ids); never add facts beyond them:\n" + _claims_block(claims or []))
    return "\n\n".join(parts)


def claims_prompt(topic: Topic, source_text: str) -> str:
    return (f"Below is source text about '{topic.wikipedia}'. Extract 8-14 atomic claims that a short, "
            f"eerie but faithful retelling would need: what happened, when, where, what was found, what "
            f"remains unexplained, and the main explanations proposed. Use ONLY this text. For each claim, "
            f"copy a supporting quote VERBATIM from the text (max 30 words) and tag it.\n\n"
            f"SOURCE TEXT:\n{source_text}")


def _claims_block(claims: list[Claim]) -> str:
    return "\n".join(f"[{c.id}] ({c.tag}) {c.text}" for c in claims)
