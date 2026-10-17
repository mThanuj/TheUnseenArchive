"""Data shapes for the story engine.

The `*Draft`, `ConceptList`, `ClaimList` and `Critique` models double as the JSON schemas the LLM
must fill (field descriptions are part of the schema the model sees). `StoryRecord` is what is
saved to disk and SQLite.
"""

from __future__ import annotations

from typing import Annotated, Callable, Literal, get_args

from pydantic import BaseModel, BeforeValidator, Field

Category = Literal["fiction_horror", "psychological_mystery", "real_mystery"]
DeviceName = Literal["camcorder", "cctv", "night_vision", "phone", "bodycam", "dashcam", "trail_cam",
                     "webcam", "film_8mm", "film_16mm"]
DeliveryName = Literal["calm", "hushed", "tense", "flat", "urgent"]
SoundName = Literal["room_tone", "drone", "hum", "electrical_buzz", "footsteps", "door", "knock", "impact",
                    "creak", "wind", "water_drip", "radio_static", "breathing", "whisper", "dropout",
                    "silence", "distant_voice", "machine"]
ClaimTag = Literal["fact", "disputed", "theory", "legend"]


def _lenient(allowed_type, synonyms: dict[str, str], default: str) -> Callable:
    """Map near-miss enum values from the LLM ("click", "precise", "smartphone") onto allowed ones,
    so a single odd word doesn't force regenerating a whole script."""
    allowed = set(get_args(allowed_type))

    def coerce(value):
        if not isinstance(value, str):
            return value
        key = value.strip().lower().replace("-", "_").replace(" ", "_")
        if key in allowed:
            return key
        return synonyms.get(key, default)

    return coerce


Device = Annotated[DeviceName, BeforeValidator(_lenient(DeviceName, {
    "smartphone": "phone", "mobile": "phone", "cellphone": "phone", "cell_phone": "phone", "iphone": "phone",
    "security_camera": "cctv", "surveillance": "cctv", "surveillance_camera": "cctv", "cctv_camera": "cctv",
    "nightvision": "night_vision", "night_vision_camera": "night_vision", "infrared": "night_vision",
    "body_camera": "bodycam", "body_cam": "bodycam", "dash_cam": "dashcam", "dashboard_camera": "dashcam",
    "trailcam": "trail_cam", "trail_camera": "trail_cam", "game_camera": "trail_cam",
    "web_cam": "webcam", "laptop": "webcam", "laptop_webcam": "webcam",
    "8mm": "film_8mm", "super8": "film_8mm", "super_8": "film_8mm", "home_movie": "film_8mm",
    "16mm": "film_16mm", "newsreel": "film_16mm", "film": "film_16mm",
    "vhs": "camcorder", "hi8": "camcorder", "handycam": "camcorder", "video_camera": "camcorder",
    "handheld": "camcorder"}, "camcorder"))]
Delivery = Annotated[DeliveryName, BeforeValidator(_lenient(DeliveryName, {
    "whisper": "hushed", "whispered": "hushed", "quiet": "hushed", "soft": "hushed", "low": "hushed",
    "precise": "flat", "cold": "flat", "neutral": "flat", "clinical": "flat", "matter_of_fact": "flat",
    "measured": "calm", "steady": "calm", "slow": "calm", "anxious": "tense", "nervous": "tense",
    "uneasy": "tense", "fast": "urgent", "panicked": "urgent"}, "calm"))]
Sound = Annotated[SoundName, BeforeValidator(_lenient(SoundName, {
    "click": "machine", "beep": "machine", "whir": "machine", "motor": "machine", "static": "radio_static",
    "hiss": "radio_static", "buzz": "electrical_buzz", "thud": "impact", "bang": "impact", "crash": "impact",
    "crunch": "impact", "slam": "door", "door_slam": "door", "step": "footsteps", "steps": "footsteps",
    "drip": "water_drip", "voice": "distant_voice", "voices": "distant_voice", "whispers": "whisper",
    "breath": "breathing", "ambience": "room_tone", "ambient": "room_tone", "tone": "drone",
    "rumble": "drone", "scrape": "creak", "squeak": "creak", "glitch": "dropout", "cut": "dropout"},
    "room_tone"))]


# ------------------------------------------------------------------ LLM outputs

class Concept(BaseModel):
    title: str = Field(description="2-6 words, evocative, no clickbait punctuation")
    logline: str = Field(description="one sentence (max 35 words): who/what was recorded and what goes wrong")
    hook: str = Field(description="the first spoken line, max 12 words, starts mid-situation")
    twist: str = Field(description="the turn or revelation at the end, one sentence")
    why_original: str = Field(description="one sentence: what makes this unlike typical horror shorts")


class ConceptList(BaseModel):
    concepts: list[Concept] = Field(min_length=3, max_length=6)


class ConceptRating(BaseModel):
    index: int = Field(description="concept number as listed (1-based)")
    hook: int = Field(ge=1, le=10, description="would a scrolling viewer stop within one second")
    dread: int = Field(ge=1, le=10, description="does it get under the skin")
    originality: int = Field(ge=1, le=10)
    visual_potential: int = Field(ge=1, le=10, description="can short AI video clips show the anomaly clearly")
    note: str = Field(default="", description="max 20 words")

    @property
    def total(self) -> int:
        return self.hook + self.dread + self.originality + self.visual_potential


class ConceptRatings(BaseModel):
    ratings: list[ConceptRating]


class Claim(BaseModel):
    id: int
    text: str = Field(description="one atomic statement paraphrased from the source")
    quote: str = Field(description="the exact words from the source text supporting it, copied verbatim, max 30 words")
    tag: ClaimTag = Field(description="fact = source states it as established; disputed = source notes "
                                      "disagreement; theory = proposed explanation; legend = folklore/unverified account")
    visual: bool = Field(default=False, description="true if the claim describes what a place, object or "
                                                    "person physically looked like")


class ClaimList(BaseModel):
    claims: list[Claim] = Field(min_length=4, max_length=16)


class NarrationLine(BaseModel):
    id: int
    text: str = Field(description="spoken words only, max 22 words")
    delivery: Delivery = "calm"
    pause_after_s: float = Field(default=0.4, ge=0.0, le=3.0, description="silence after the line")
    claim_ids: list[int] = Field(default_factory=list, description="real mysteries only: ids of supporting claims")


class SfxCue(BaseModel):
    at: float = Field(ge=0.0, le=1.0, description="position within the scene, 0 = start, 1 = end")
    sound: Sound
    note: str = Field(default="", description="what exactly is heard, e.g. 'three slow knocks'")


class Scene(BaseModel):
    id: int
    narration_ids: list[int] = Field(description="narration line ids spoken during this scene (may be empty)")
    duration_s: float = Field(ge=2.0, le=5.0)
    device: Device
    visual: str = Field(description="what is in frame, max 20 words")
    motion: str = Field(description="what moves or changes during the shot, max 15 words")
    camera: str = Field(description="perspective and camera movement (or 'fixed'), max 12 words")
    lighting: str = Field(description="light sources and colour, max 12 words")
    continuity: str = Field(description="link to the previous shot, max 15 words")
    video_prompt: str = Field(description="final text-to-video prompt, one literal paragraph, 45-130 words")
    osd: str = Field(default="", description="camera label only, e.g. 'CAM 03' or 'REC' (dates and times are "
                                             "added automatically from recorded_year)")
    sfx: list[SfxCue] = Field(default_factory=list, max_length=3)


class StoryDraft(BaseModel):
    title: str
    logline: str
    archive_entry: str = Field(description="catalogue line shown at the start, e.g. "
                                           "'ENTRY 0412 - Hi8 tape - recovered 2004, Ashby, Ohio'")
    recorded_year: int = Field(ge=1890, le=2100, description="year the footage was recorded; every scene's "
                                                             "device must already exist that year")
    narration: list[NarrationLine] = Field(min_length=3, max_length=24)
    scenes: list[Scene] = Field(min_length=3, max_length=12)
    ending_note: str = Field(description="one sentence: why the ending lands")
    tags: list[str] = Field(description="3-8 lowercase tags: setting, device, anomaly, era")
    caption: str = Field(description="Instagram caption draft, 1-3 short sentences, no hashtags")


class Critique(BaseModel):
    hook: int = Field(ge=1, le=10)
    escalation: int = Field(ge=1, le=10)
    coherence: int = Field(ge=1, le=10)
    originality: int = Field(ge=1, le=10)
    ending: int = Field(ge=1, le=10)
    visual_feasibility: int = Field(ge=1, le=10, description="can a text-to-video model render every shot believably")
    visual_variety: int = Field(ge=1, le=10, description="distinct shots, and the anomaly visibly escalates on screen")
    brand_fit: int = Field(ge=1, le=10)
    issues: list[str] = Field(description="concrete problems with line/scene ids and how to fix them")
    verdict: Literal["pass", "revise"] = "revise"   # optional: the scores decide revisions anyway

    def scores(self) -> dict[str, int]:
        return {k: getattr(self, k) for k in ("hook", "escalation", "coherence", "originality",
                                              "ending", "visual_feasibility", "visual_variety", "brand_fit")}


class LineFix(BaseModel):
    id: int = Field(description="narration line id")
    text: str = Field(description="the rewritten line, spoken words only")
    claim_ids: list[int] = Field(default_factory=list, description="real mysteries: supporting claim ids")


class LineFixes(BaseModel):
    lines: list[LineFix]


class SceneFix(BaseModel):
    id: int = Field(description="scene id")
    device: Device = Field(description="a device that existed in the recorded year")
    video_prompt: str = Field(description="rewritten 45-130 word prompt with no readable text")


class SceneFixes(BaseModel):
    scenes: list[SceneFix]


class LineVerdict(BaseModel):
    id: int = Field(description="narration line id")
    verdict: Literal["supported", "atmospheric", "unsupported", "contradicted"] = Field(
        description="supported: says only what its cited claims say; atmospheric: no factual assertion; "
                    "unsupported: adds facts not in the claims; contradicted: conflicts with a claim")
    corrected: str = Field(default="", description="for unsupported/contradicted: the line rewritten to say "
                                                   "only what the cited claims support, max 22 words")


class FactCheck(BaseModel):
    lines: list[LineVerdict]


class SceneEdit(BaseModel):
    id: int = Field(description="scene id")
    visual: str = Field(default="", description="new value, or empty to keep")
    motion: str = Field(default="", description="new value, or empty to keep")
    camera: str = Field(default="", description="new value, or empty to keep")
    lighting: str = Field(default="", description="new value, or empty to keep")
    video_prompt: str = Field(default="", description="new 45-130 word prompt, or empty to keep")


class StoryPatch(BaseModel):
    """A revision as targeted edits: much cheaper than regenerating the whole script."""
    narration: list[LineFix] = Field(default_factory=list, max_length=5,
                                     description="only the narration lines that must change (max 5)")
    scenes: list[SceneEdit] = Field(default_factory=list, max_length=4,
                                    description="only the scenes that must change (max 4)")
    ending_note: str = Field(default="", description="new ending note, or empty to keep")


# ------------------------------------------------------------------ saved record

class Source(BaseModel):
    title: str
    url: str
    license: str
    revision_id: int | None = None
    retrieved_at: str
    external_links: list[str] = Field(default_factory=list)


class StoryRecord(BaseModel):
    id: str
    created_at: str
    category: Category
    target_seconds: int
    ingredients: dict[str, str] = Field(default_factory=dict)
    topic_id: str | None = None
    concept: Concept
    concept_alternatives: list[Concept] = Field(default_factory=list)
    novelty_max_similarity: float | None = None
    draft: StoryDraft
    critique: Critique | None = None
    critique_history: list[dict] = Field(default_factory=list)
    concept_ratings: dict[str, int] = Field(default_factory=dict)
    revised: bool = False
    problems: list[str] = Field(default_factory=list)
    sources: list[Source] = Field(default_factory=list)
    claims: list[Claim] = Field(default_factory=list)
    fact_check: list[dict] = Field(default_factory=list)
    llm: list[dict] = Field(default_factory=list)
    ai_disclosure: str = ""
