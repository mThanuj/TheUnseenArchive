"""Prompt builders for each LLM step. The schemas carry field-level instructions; these carry
the brand, the craft rules, and the constraints of the video model."""

from __future__ import annotations

import json

from .config import StoryConfig, Topic
from .models import Claim, Concept, Critique, StoryDraft
from .validate import expected_scenes as scene_count

CATEGORY_BRIEF = {
    "fiction_horror": "an original fictional horror story presented as a recovered recording",
    "psychological_mystery": ("an original psychological mystery presented as a recovered recording: "
                              "unreliable perception, behaviour that doesn't add up, conflicting evidence, "
                              "a revelation that reframes everything"),
    "real_mystery": ("a dramatized reconstruction of a documented real-world mystery, strictly faithful "
                     "to the provided claims"),
}

ENDING_RULE = {
    "fiction": ("- The first scene shows something wrong within the first second; the last line points the viewer at a\n"
                "  detail they missed, so they rewatch (e.g. \"Watch the mirror, not the bed.\")."),
    "real_mystery": ("- The first scene shows the most arresting documented sight within the first second; the last line\n"
                     "  lands on what remains unexplained, stated plainly. Times, numbers and names only as they\n"
                     "  appear in the claims. Never invent a clue, sign, symbol, reading, log or event to make the\n"
                     "  case more dramatic: the documented facts are the story."),
}


def system_prompt(cfg: StoryConfig, category: str | None = None) -> str:
    b, r = cfg.brand, cfg.rules
    ending = ENDING_RULE["real_mystery" if category == "real_mystery" else "fiction"]
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
- The Archivist states facts: times, counts, objects, names of places. Never adjectives that tell the
  viewer how to feel (eerie, mysterious, chilling, sinister...). Short sentences. Precise numbers.
{ending}

Voice examples (style only; never reuse their content):
- "Tape four. The family checked out at 9:14. The camera kept recording the empty room. At 9:31 the
  bed is unmade again. Housekeeping logged no entry. Watch the mirror, not the bed."
- "The trail camera fired forty times between two and three a.m. Thirty-nine frames are empty. In
  frame thirty-eight, it is close enough to see that it is wearing his coat."
- Avoid these cliches: {", ".join(r.banned_cliches)}.
- Content limits: {r.content_limits}

Always answer with a single JSON object that matches the requested schema."""


def concepts_prompt(cfg: StoryConfig, category: str, target_seconds: int, ingredients: dict,
                    recent_titles: list[str], topic: Topic | None, claims: list[Claim] | None) -> str:
    parts = [f"Pitch 5 different concepts for {CATEGORY_BRIEF[category]}. Finished length: about "
             f"{target_seconds} seconds (one Reel)."]
    if category == "real_mystery" and topic:
        parts.append(f"The case: {topic.wikipedia} ({topic.year}). Every concept must be built only from "
                     f"these sourced claims (no invented facts). Refer to private people by role, not name, "
                     f"unless they are long-dead historical figures:\n{_claims_block(claims or [])}")
    else:
        parts.append("Use these ingredients as a starting point (you may twist them):\n"
                     + "\n".join(f"- {k}: {v}" for k, v in ingredients.items()))
    if recent_titles:
        parts.append("Recent stories on the channel (do NOT repeat their premise, setting or twist):\n"
                     + "\n".join(f"- {t}" for t in recent_titles))
    if category == "real_mystery":
        parts.append("Make the 5 concepts 5 different ANGLES on this one documented case: which documented moment "
                     "or sight opens it and what it builds to. Never a new event, an invented investigation or an "
                     "invented clue. The premise is what was documented; a theory may appear only as a hedged "
                     "possibility near the end, never as the premise or the title. The title names the case, the "
                     "place or a documented detail. visual_hook = a documented sight from the claims, shown in the "
                     "first second; rewatch_clue = a documented visual detail from the claims (never an invented "
                     "sign, symbol, reading, log, timestamp or object). Keep it understandable on one watch.")
        return "\n\n".join(parts)
    parts.append("Make the 5 concepts genuinely different from each other in premise, device and ending. "
                 "Each anomaly must be something a camera can SHOW, not only something heard or felt. Keep each "
                 "concept SIMPLE: one place, one anomaly, one reveal, understandable on the first watch (a "
                 "30-second Reel cannot carry several mysteries). visual_hook = what we see in the very first "
                 "second; rewatch_clue = a visible detail planted early that the ending makes the viewer look for.")
    return "\n\n".join(parts)


def script_prompt(cfg: StoryConfig, category: str, target_seconds: int, concept: Concept,
                  ingredients: dict, devices: dict[str, str], claims: list[Claim] | None) -> str:
    r, ln = cfg.rules, cfg.length
    words_target = round(target_seconds * ln.words_per_second * 0.75)
    n_scenes = scene_count(cfg, target_seconds)
    per_scene = int(ln.scene_seconds.max * ln.words_per_second)
    device_lines = "\n".join(f"- {k}: {v}" for k, v in devices.items())
    if category == "real_mystery":
        shape = (f"Scene 1 must SHOW the visual hook ({concept.visual_hook}) within its first second. The documented "
                 f"detail ({concept.rewatch_clue}) is visible in an early scene. End on what remains unexplained, "
                 f"stated plainly. Nothing in the narration or the footage may suggest evidence the claims don't contain.")
    else:
        shape = (f"Scene 1 must SHOW the visual hook ({concept.visual_hook}) within its first second. Plant the rewatch "
                 f"clue ({concept.rewatch_clue}) visibly in an early scene; the final narration line points the viewer "
                 f"back to it. One place, one anomaly, one reveal: cut anything that needs explaining.")
    parts = [
        f"Write the full Reel script for this concept ({CATEGORY_BRIEF[category]}):\n"
        f"{concept.model_dump_json(indent=2)}",
        shape,
        f"Timing: write EXACTLY {n_scenes} scenes (about {target_seconds} s in total); each scene is ONE "
        f"generated video clip of {ln.scene_seconds.min:g}-{ln.scene_seconds.max:g} s (usually "
        f"{ln.scene_seconds.typical:g} s). Narration totals about {words_target} spoken words, delivered slowly "
        f"({ln.words_per_second} words per second). A scene holds at most {per_scene} spoken words; several "
        f"scenes should be silent (silence is powerful). Every narration line belongs to exactly one scene, in order.",
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
        "- Avoid: readable text or signs, numbers, clock times, digital readouts and nameplates (the narration "
        "can say the number; the shot shows the object), close-ups of hands handling objects, crowds, faces in "
        "close-up, gore.\n"
        f"- Never use these words: {', '.join(r.video_prompt_banned_words)} (grain, tape damage and the "
        "on-screen overlay are added in post-production). Never describe sound.",
        "sfx: 1-3 cues per scene placed where the action happens (a door opening, footsteps, a hum that "
        "stops). Use 'silence' right before the biggest reveal.",
        "osd: realistic camera overlay label for the device (CAM numbers, REC; dates/times are added later). "
        "Film (film_8mm, film_16mm) has no overlay: leave osd empty.",
        "Visual storytelling (the footage must carry the story on its own, even muted):\n"
        "- No two consecutive scenes may repeat the same framing and action; use at least 4 distinct setups "
        "(e.g. wide establishing, detail, reveal, aftermath) across the Reel.\n"
        "- The anomaly must be VISIBLE on screen in at least two scenes and escalate (subtle -> undeniable); "
        "sound and narration support it, they do not replace it.\n"
        "- If the twist depends on something (time running backwards, a second camera, a missing minute), "
        "plant a visual clue for it before the reveal.\n"
        "- No readable text anywhere in a shot (no printouts, screens with words, handwriting, signs, labels, "
        "quoted words): video models render it as gibberish. Show the object, not the writing.\n"
        "- recorded_year is when the footage was shot; every scene's device must exist by then (devices list "
        "shows the first year). Keep the place, era and clothing consistent with it.\n"
        "- The Archivist narrates in the third person about the recording and the people in it (never 'I' or "
        "'me' as the person filming), except when quoting a log or a voice heard on the tape.",
    ]
    if category == "real_mystery":
        parts.append("Real case framing: this is a RECONSTRUCTION. archive_entry starts with 'RECONSTRUCTION -' "
                     "and names the case and date; never present any tape, reel, film, recording or document as "
                     "real evidence, and never invent investigations, simulations or files. Open with the most "
                     "arresting documented detail from the claims (a last message, a number, what was found); "
                     "exact quoted words are allowed only if they appear in that claim's source quote. Tell it in "
                     "order: what happened, what was found, what remains unexplained; hedged theories come last.")
        parts.append("Real case look: ONE consistent look for the whole Reel. If the event happened after cameras "
                     "existed, the footage is EITHER period footage shot around the event (only devices that existed "
                     "then, at most two device types) OR a present-day revisit of the place (recorded_year = this "
                     "year). If it happened before cameras existed (before ~1925), it is a present-day dramatized "
                     "reconstruction filmed in ONE device style (film_16mm documentary look) for every scene, or a "
                     "present-day revisit showing the place as it is today without period people; never a phone or "
                     "camcorder inside a historical scene. Visual details of real places and objects must match the "
                     "claims marked [visual]; never invent their appearance.")
        parts.append("Real case rules: every factual narration line lists the claim_ids it relies on (at least two "
                     "thirds of the lines cite claims; the rest are short scene-setting with no facts in them). "
                     "Claims tagged theory/disputed are hedged in every line that uses them (reportedly, some "
                     "believe, according to...). A legend is framed ONCE, in the first line that tells it (e.g. "
                     "'According to two medieval chroniclers, ...'); after that, tell it plainly in the Archivist's "
                     "voice and do not repeat 'according to legend'. Never add facts that are not in the claims: no "
                     "invented times, dates, counts, measurements, readings, logs, vehicles, witnesses, signs or "
                     "objects. Never show real victims: no bodies, remains or injuries on screen; mention deaths "
                     "and injuries only in general terms (e.g. 'three died of injuries no one could explain') and "
                     "let the footage show places, objects and traces (an empty tent, footprints, a creek). "
                     "Refer to private people by "
                     "their role (the photographer, his daughter, a crew member), not by name, unless they are "
                     "long-dead historical figures; never speculate about anyone's guilt or character. Visuals are "
                     "a dramatized reconstruction.\nClaims:\n" + _claims_block(claims or []))
    return "\n\n".join(parts)


def critique_prompt(cfg: StoryConfig, draft: StoryDraft, category: str) -> str:
    real = (" This is a REAL case retold from sourced claims: hedges on theories and disputed points, one "
            "framing of a legend, and the absence of invented clues are requirements, not flaws (flag only "
            "repetitive hedging). Judge how gripping the documented facts are made, not whether more drama "
            "could be invented." if category == "real_mystery" else "")
    return (f"Critique this Reel script ({CATEGORY_BRIEF[category]}) as a demanding horror editor and as a "
            f"text-to-video specialist. Be harsh and calibrated: typical first drafts score 5-7; 8 means "
            f"strong; 9-10 only for work you would publish today unchanged. In issues, list concrete, "
            f"actionable fixes with line/scene ids: weak hook, predictable turn, filler, repetitive shots "
            f"(the same action in several scenes), an anomaly that is only heard or narrated but never seen, "
            f"a twist without a visual clue, narrator perspective slips, confusing continuity, shots a video "
            f"model will render badly (text, hands, crowds, close-up faces, too much happening), an ending that "
            f"does not land. At most 6 issues, most important first, each under 30 words. "
            f"verdict = 'revise' if any score is below 7.{real}\n\n{draft.model_dump_json()}")


def draft_outline(d: StoryDraft) -> str:
    """A draft without video prompts and sound cues: enough for a rewrite to keep what works, and
    small enough that the whole rewrite request fits Groq's 8k tokens-per-minute cap."""
    lines = [f"title: {d.title}", f"logline: {d.logline}", f"archive_entry: {d.archive_entry}",
             f"recorded_year: {d.recorded_year}", "narration:"]
    lines += [f"  [{n.id}] ({n.delivery}, pause {n.pause_after_s:g}s) {n.text}"
              + (f"  claims {n.claim_ids}" if n.claim_ids else "") for n in d.narration]
    lines.append("scenes:")
    lines += [f"  [{s.id}] {s.duration_s:g} s, {s.device}, lines {s.narration_ids}: {s.visual} | motion: "
              f"{s.motion} | camera: {s.camera} | light: {s.lighting}" for s in d.scenes]
    lines.append(f"ending: {d.ending_note}")
    return "\n".join(lines)


def revise_prompt(base_prompt: str, draft: StoryDraft, critique: Critique, checks: list[str] | None = None) -> str:
    extra = ("\n\nAutomatic checks also failed (fix these too):\n- " + "\n- ".join(checks)) if checks else ""
    return (f"{base_prompt}\n\nA previous draft (outline; write every field, including new video prompts):\n"
            f"{draft_outline(draft)}{extra}\n\nThe editor's critique "
            f"(scores {critique.scores()}):\n- " + "\n- ".join(critique.issues)
            + "\n\nWrite the improved, complete script. Keep what works; fix every issue. If hook, "
              "escalation, originality, ending or visual variety scored below 7, rework the scene structure "
              "and the ending itself, not just the wording.")


def concept_rating_prompt(concepts: list[Concept], category: str, recent_titles: list[str]) -> str:
    listed = "\n".join(f"{i}. {c.title}: {c.logline} First second: {c.visual_hook}. Twist: {c.twist}"
                       for i, c in enumerate(concepts, 1))
    recent = ("\nRecent stories on the channel:\n" + "\n".join(f"- {t}" for t in recent_titles)) if recent_titles else ""
    return (f"Rate these concepts for {CATEGORY_BRIEF[category]} (an Instagram Reel of about 30 seconds). Be "
            f"harsh and calibrated: typical concepts score 5-7, 9-10 only for exceptional ones. Penalize "
            f"concepts that are too complicated to understand on one watch. Rate every "
            f"concept by its number.{recent}\n\n{listed}")


def patch_prompt(cfg: StoryConfig, draft: StoryDraft, critique: Critique, category: str,
                 claims: list[Claim] | None) -> str:
    r = cfg.rules
    parts = [f"Here is a Reel script ({CATEGORY_BRIEF[category]}):\n{draft.model_dump_json()}",
             f"The editor's critique (scores {critique.scores()}):\n- " + "\n- ".join(critique.issues),
             "Fix the issues with the FEWEST targeted edits: at most 5 narration lines and 4 scenes, "
             "only those the issues name. Do not repeat unchanged lines or scenes. Do not add, remove or "
             "renumber lines or scenes. Leave fields empty to keep them. "
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
        parts.append("Keep any hedge a line already has (reportedly, according to, theory...); a legend only needs "
                     "its framing once, so don't add 'according to legend' to lines that already sit inside it.")
        parts.append("Claims you may cite (claim_ids); never add facts beyond them:\n" + _claims_block(claims or []))
    return "\n\n".join(parts)


def claims_prompt(topic: Topic, source_text: str) -> str:
    return (f"Below is source text about '{topic.wikipedia}'. Extract 8-14 atomic claims that a short, "
            f"eerie but faithful retelling would need: what happened, when, where, what was found, what "
            f"remains unexplained, and the main explanations proposed. Also include 1-3 claims describing what "
            f"the key place or object physically looked like, if the text says (mark them visual=true). Use "
            f"ONLY this text. For each claim, "
            f"copy a supporting quote VERBATIM from the text (max 30 words) and tag it.\n\n"
            f"SOURCE TEXT:\n{source_text}")


def _claims_block(claims: list[Claim]) -> str:
    return "\n".join(f"[{c.id}] ({c.tag}{', visual' if c.visual else ''}) {c.text}" for c in claims)


def fact_check_prompt(draft: StoryDraft, claims: list[Claim]) -> str:
    sourced = "\n".join(f"[{c.id}] ({c.tag}) {c.text}\n    source: \"{c.quote}\"" for c in claims)
    lines = "\n".join(f"[{n.id}] {n.text}  (cites {n.claim_ids or 'nothing'})" for n in draft.narration)
    return ("You are a strict fact-checker for a dramatized documentary Reel. For every narration line, judge "
            "whether it says ONLY what its cited claims (and their source quotes) support. Hedged wording of "
            "theories/legends is fine; adding specifics, picking one side of an open question, or contradicting "
            "a claim is not. A line that cites nothing but states specifics (a time, count, log, reading, "
            "measurement, sign, object or event) is 'unsupported', and so is a line pointing the viewer at a clue "
            "the claims don't mention. Lines with no factual assertion are 'atmospheric'. For unsupported or "
            "contradicted lines, write a corrected line that keeps the tone but states only what the claims "
            "support.\n\n"
            f"CLAIMS:\n{sourced}\n\nNARRATION:\n{lines}")


def scene_repair_prompt(cfg: StoryConfig, draft: StoryDraft, issues: dict[int, str]) -> str:
    devices = "\n".join(f"- {d.id} (from {d.since}): {d.desc}" for d in cfg.ingredients.device)
    todo = "\n".join(f"[scene {i}] device={next(s.device for s in draft.scenes if s.id == i)}\n"
                     f"    prompt: {next(s.video_prompt for s in draft.scenes if s.id == i)}\n    problem: {why}"
                     for i, why in sorted(issues.items()))
    return (f"Story: {draft.title} (footage recorded in {draft.recorded_year}). Rewrite ONLY these scenes so "
            f"they keep their purpose in the story but fix the problems. Video prompts: one literal paragraph, "
            f"{cfg.rules.video_prompt_words.min}-{cfg.rules.video_prompt_words.max} words, visible motion, no "
            f"readable text.\n\nDevices (first year available):\n{devices}\n\n{todo}")
