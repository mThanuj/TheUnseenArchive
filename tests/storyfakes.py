"""Shared builders for story-engine tests (no network, no Ollama, no model downloads)."""

from __future__ import annotations

import hashlib
import json

import numpy as np

from unseen.story.models import StoryDraft

PROMPT = ("A handheld camcorder moves slowly down a narrow motel corridor at night, passing a row of "
          "closed doors with brass numbers. A ceiling light flickers on and off above the camera. At the "
          "far end a door slowly swings open on its own, revealing a dark room. The camera stops and holds "
          "on the doorway. Dim yellow light, deep shadows, muted colors, realistic amateur footage.")

LINES = [
    "Entry four twelve. The tape starts mid-step.",
    "The corridor light was already flickering.",
    "Room nine was empty when she checked it.",
    "She checked it again.",
    "The second time, the bed was made.",
    "Nobody at the motel owns a camera like hers.",
]


def draft_dict(**over) -> dict:
    narration = [{"id": i + 1, "text": t, "delivery": "calm", "pause_after_s": 0.6, "claim_ids": []}
                 for i, t in enumerate(LINES)]
    assign = [[1], [2], [3], [4], [], [5], [6]]
    scenes = [{"id": i + 1, "narration_ids": ids, "duration_s": 4.0, "device": "camcorder",
               "visual": "motel corridor", "motion": "door swings open", "camera": "handheld, walking",
               "lighting": "flickering ceiling light", "continuity": "same corridor", "video_prompt": PROMPT,
               "osd": "REC  10-17-2004  02:13", "sfx": [{"at": 0.5, "sound": "door", "note": "slow creak"}]}
              for i, ids in enumerate(assign)]
    d = {"title": "Room Nine", "logline": "A night clerk films an empty motel room that keeps changing.",
         "archive_entry": "ENTRY 0412 - camcorder tape - recovered 2004, Ashby", "recorded_year": 2004,
         "narration": narration,
         "scenes": scenes, "ending_note": "The camera is not hers.", "tags": ["motel", "camcorder"],
         "caption": "Recovered from a motel lost-and-found."}
    d.update(over)
    return d


def draft(**over) -> StoryDraft:
    return StoryDraft.model_validate(draft_dict(**over))


CRITIQUE_PASS = {"hook": 8, "escalation": 8, "coherence": 8, "originality": 8, "ending": 8,
                 "visual_feasibility": 8, "visual_variety": 8, "brand_fit": 8, "issues": [], "verdict": "pass"}
CRITIQUE_WEAK = dict(CRITIQUE_PASS, ending=5, visual_variety=5, verdict="revise",
                     issues=["scene 7: ending is flat", "scenes 1-4 repeat the same shot"])

RATINGS = {"ratings": [  # concept 2 ("Second Shift") rates best
    {"index": 1, "hook": 6, "dread": 6, "originality": 5, "visual_potential": 7, "note": ""},
    {"index": 2, "hook": 9, "dread": 8, "originality": 8, "visual_potential": 8, "note": ""},
    {"index": 3, "hook": 7, "dread": 7, "originality": 6, "visual_potential": 6, "note": ""},
]}

CONCEPTS = {"concepts": [
    {"title": "Room Nine", "logline": "A night clerk films an empty motel room that keeps changing.",
     "hook": "The tape starts mid-step.", "twist": "The camera is not hers.", "why_original": "quiet"},
    {"title": "Second Shift", "logline": "A guard notices the CCTV shows tomorrow.",
     "hook": "Camera three shows a date that hasn't happened.", "twist": "He is on it.", "why_original": "time"},
    {"title": "Low Tide", "logline": "A ferry camera records a passenger who never boarded.",
     "hook": "Nobody bought ticket forty.", "twist": "The ticket is his.", "why_original": "ferry"},
]}


class FakeProvider:
    """Answers by schema title; each title maps to a list of raw responses consumed in order."""
    name = "fake"
    model = "fake-1"

    def __init__(self, responses: dict[str, list]):
        self.responses = {k: list(v) for k, v in responses.items()}
        self.calls: list[str] = []
        self.unloaded = False

    def complete_json(self, system, user, schema, temperature, max_tokens):
        from unseen.llm.base import LLMError
        title = schema.get("title", "?")
        self.calls.append(title)
        if title not in self.responses:
            raise LLMError(f"fake provider has no response for {title}")
        queue = self.responses[title]
        item = queue.pop(0) if len(queue) > 1 else queue[0]
        return (item if isinstance(item, str) else json.dumps(item)), {"prompt_tokens": 100, "output_tokens": 50}

    def unload(self):
        self.unloaded = True


class FakeEmbedder:
    def embed(self, texts):
        out = []
        for t in texts:
            seed = int(hashlib.sha256(t.encode()).hexdigest()[:8], 16)
            v = np.random.default_rng(seed).normal(size=16).astype(np.float32)
            out.append(v / np.linalg.norm(v))
        return np.stack(out)
