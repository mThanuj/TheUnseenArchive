"""Data shapes for the story engine.

The `*Draft`, `ConceptList`, `ClaimList` and `Critique` models double as the JSON schemas the LLM
must fill (field descriptions are part of the schema the model sees). `StoryRecord` is what is
saved to disk and SQLite.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Category = Literal["fiction_horror", "psychological_mystery", "real_mystery"]
Device = Literal["camcorder", "cctv", "night_vision", "phone", "bodycam", "dashcam", "trail_cam",
                 "webcam", "film_8mm"]
Delivery = Literal["calm", "hushed", "tense", "flat", "urgent"]
Sound = Literal["room_tone", "drone", "hum", "electrical_buzz", "footsteps", "door", "knock", "impact",
                "creak", "wind", "water_drip", "radio_static", "breathing", "whisper", "dropout",
                "silence", "distant_voice", "machine"]
ClaimTag = Literal["fact", "disputed", "theory", "legend"]


# ------------------------------------------------------------------ LLM outputs

class Concept(BaseModel):
    title: str = Field(description="2-6 words, evocative, no clickbait punctuation")
    logline: str = Field(description="one sentence (max 35 words): who/what was recorded and what goes wrong")
    hook: str = Field(description="the first spoken line, max 12 words, starts mid-situation")
    twist: str = Field(description="the turn or revelation at the end, one sentence")
    why_original: str = Field(description="one sentence: what makes this unlike typical horror shorts")


class ConceptList(BaseModel):
    concepts: list[Concept] = Field(min_length=3, max_length=6)


class Claim(BaseModel):
    id: int
    text: str = Field(description="one atomic statement paraphrased from the source")
    quote: str = Field(description="the exact words from the source text supporting it, copied verbatim, max 30 words")
    tag: ClaimTag = Field(description="fact = source states it as established; disputed = source notes "
                                      "disagreement; theory = proposed explanation; legend = folklore/unverified account")


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
    osd: str = Field(default="", description="on-screen camera overlay, e.g. 'CAM 03  10-17-1998  02:13:44'")
    sfx: list[SfxCue] = Field(default_factory=list, max_length=3)


class StoryDraft(BaseModel):
    title: str
    logline: str
    archive_entry: str = Field(description="catalogue line shown at the start, e.g. "
                                           "'ENTRY 0412 - Hi8 tape - recovered 2004, Ashby, Ohio'")
    narration: list[NarrationLine] = Field(min_length=3, max_length=24)
    scenes: list[Scene] = Field(min_length=3, max_length=10)
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
    brand_fit: int = Field(ge=1, le=10)
    issues: list[str] = Field(description="concrete problems with line/scene ids and how to fix them")
    verdict: Literal["pass", "revise"]

    def scores(self) -> dict[str, int]:
        return {k: getattr(self, k) for k in ("hook", "escalation", "coherence", "originality",
                                              "ending", "visual_feasibility", "brand_fit")}


class LineFix(BaseModel):
    id: int = Field(description="narration line id")
    text: str = Field(description="the rewritten line, spoken words only")
    claim_ids: list[int] = Field(default_factory=list, description="real mysteries: supporting claim ids")


class LineFixes(BaseModel):
    lines: list[LineFix]


class SceneEdit(BaseModel):
    id: int = Field(description="scene id")
    visual: str = Field(default="", description="new value, or empty to keep")
    motion: str = Field(default="", description="new value, or empty to keep")
    camera: str = Field(default="", description="new value, or empty to keep")
    lighting: str = Field(default="", description="new value, or empty to keep")
    video_prompt: str = Field(default="", description="new 45-130 word prompt, or empty to keep")


class StoryPatch(BaseModel):
    """A revision as targeted edits: much cheaper than regenerating the whole script."""
    narration: list[LineFix] = Field(default_factory=list, description="rewritten narration lines (by id)")
    scenes: list[SceneEdit] = Field(default_factory=list, description="edited scenes (by id)")
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
    revised: bool = False
    problems: list[str] = Field(default_factory=list)
    sources: list[Source] = Field(default_factory=list)
    claims: list[Claim] = Field(default_factory=list)
    llm: list[dict] = Field(default_factory=list)
    ai_disclosure: str = ""
