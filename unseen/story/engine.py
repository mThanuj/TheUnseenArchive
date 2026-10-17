"""The story pipeline: category -> (research) -> concepts -> novelty pick -> script -> critique ->
(revise) -> checks -> files + SQLite.

Real mysteries are grounded: Wikipedia text is fetched, the model extracts claims with verbatim
quotes, quotes that don't occur in the source are rejected, and the script may only use the
surviving claims (theories/legends must be hedged). Nothing about real cases is taken from the
model's memory.
"""

from __future__ import annotations

import random
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np

from ..llm.base import LLMError, generate
from ..state.db import StateDB
from .config import LLMCfg, StoryConfig, Topic, load_story_config, load_topics
from .models import Claim, ClaimList, Concept, ConceptList, Critique, Source, StoryDraft, StoryRecord
from .novelty import Embedder, FastEmbedder, concept_text, max_similarity
from .prompts import claims_prompt, concepts_prompt, critique_prompt, revise_prompt, script_prompt, system_prompt
from .render import save_story
from .research import ResearchError, fetch_article, trim_source
from .validate import banned_in, sanitize_video_prompt, validate_story, verify_claims

Log = Callable[[str], None]


def make_provider(cfg: LLMCfg, name: str | None = None):
    name = name or cfg.provider
    if name == "ollama":
        from ..llm.ollama import OllamaProvider
        return OllamaProvider(cfg.ollama.url, cfg.ollama.model, cfg.ollama.num_ctx, cfg.ollama.timeout_s)
    if name == "gemini":
        from ..llm.gemini import GeminiProvider
        return GeminiProvider(cfg.gemini.url, cfg.gemini.model, cfg.gemini.timeout_s)
    raise ValueError(f"unknown LLM provider {name!r} (ollama | gemini)")


def slugify(text: str, n: int = 32) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:n].strip("-") or "story"


def _weighted(rng: random.Random, items: list, weights: list[float]):
    return rng.choices(items, weights=weights, k=1)[0]


class StoryEngine:
    def __init__(self, provider=None, provider_name: str | None = None, log: Log = print,
                 seed: int | None = None, db: StateDB | None = None, embedder: Embedder | None = None,
                 fetch: Callable[[str], tuple[str, Source]] = fetch_article,
                 cfg: StoryConfig | None = None, topics: list[Topic] | None = None):
        self.cfg = cfg or load_story_config()
        self.topics = topics if topics is not None else load_topics()
        self.provider = provider or make_provider(self.cfg.llm, provider_name)
        self.db = db or StateDB()
        self.log = log
        self.rng = random.Random(seed)
        self._embedder = embedder
        self.fetch = fetch
        self.stats: list[dict] = []

    # --------------------------------------------------------------- choices
    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self.log("  loading embedding model (first run downloads ~67 MB)")
            self._embedder = FastEmbedder()
        return self._embedder

    def pick_category(self) -> str:
        cats = dict(self.cfg.categories)
        recent = self.db.recent_categories(2)
        if len(recent) == 2 and recent[0] == recent[1]:
            cats.pop(recent[0], None)  # never three in a row
        keys = list(cats)
        return _weighted(self.rng, keys, [cats[k].weight for k in keys])

    def pick_length(self) -> int:
        ch = self.cfg.length.choices
        return _weighted(self.rng, [c.seconds for c in ch], [c.weight for c in ch])

    def sample_ingredients(self) -> dict[str, str]:
        ing = self.cfg.ingredients
        recent = self.db.recent_ingredients(ing.cooldown_stories)

        def pick(key: str, options: list[str]) -> str:
            used = {r.get(key) for r in recent}
            return self.rng.choice([o for o in options if o not in used] or options)

        return {
            "device": pick("device", [d.id for d in ing.device]),
            "location": pick("location", ing.location),
            "anomaly": pick("anomaly", ing.anomaly),
            "structure": pick("structure", ing.structure),
            "twist": pick("twist", ing.twist),
        }

    def eligible_topics(self) -> list[Topic]:
        year = datetime.now().year
        limit = self.cfg.real_mystery.min_years_ago_sensitive
        return [t for t in self.topics if t.sensitivity == "low" or year - t.year >= limit]

    def pick_topic(self, topic_id: str | None = None) -> Topic:
        eligible = self.eligible_topics()
        if topic_id:
            for t in eligible:
                if t.id == topic_id:
                    return t
            raise ValueError(f"topic {topic_id!r} not found or not eligible; see config/topics.yaml")
        used = self.db.used_topics()
        fresh = [t for t in eligible if t.id not in used]
        if not eligible:
            raise ValueError("no eligible topics in config/topics.yaml")
        return self.rng.choice(fresh or eligible)

    # ------------------------------------------------------------------ steps
    def _gen(self, model_cls, *, user: str, name: str, temperature: float, max_tokens: int, check=None):
        obj, stats, problems = generate(
            self.provider, model_cls, system=system_prompt(self.cfg), user=user, name=name,
            temperature=temperature, max_tokens=max_tokens, max_retries=self.cfg.llm.max_retries,
            check=check, log=self.log)
        self.stats.append(stats.as_dict())
        return obj, problems

    def research(self, topic: Topic) -> tuple[list[Claim], Source]:
        self.log(f"Researching '{topic.wikipedia}' on Wikipedia")
        text, source = self.fetch(topic.wikipedia)
        text = trim_source(text, self.cfg.real_mystery.max_source_chars)
        claims, _ = self._gen(ClaimList, user=claims_prompt(topic, text), name="claims",
                              temperature=0.2, max_tokens=3000)
        ok, bad = verify_claims(claims.claims, text)
        self.log(f"  claims: {len(ok)} verified against the source, {len(bad)} rejected (quote not found)")
        if len(ok) < 4:
            raise ResearchError(f"only {len(ok)} claims could be verified against the source; refusing to "
                                f"write a real-case story on that basis")
        return [c.model_copy(update={"id": i}) for i, c in enumerate(ok, 1)], source

    def choose_concept(self, concepts: list[Concept]) -> tuple[Concept, float, np.ndarray, list[Concept]]:
        history = self.db.embeddings()
        vecs = self.embedder.embed([concept_text(c.title, c.logline, c.twist) for c in concepts])
        sims = [max_similarity(v, history) for v in vecs]
        order = sorted(range(len(concepts)), key=lambda i: sims[i])
        best = order[0]
        if sims[best] > self.cfg.novelty.max_similarity:
            self.log(f"  warning: every concept resembles a past story (best similarity {sims[best]:.2f})")
        self.log("  concepts: " + "; ".join(f"'{concepts[i].title}' sim={sims[i]:.2f}" for i in order))
        return concepts[best], round(sims[best], 3), vecs[best], [concepts[i] for i in order[1:]]

    # --------------------------------------------------------------- pipeline
    def new_story(self, category: str | None = None, topic_id: str | None = None,
                  target_seconds: int | None = None) -> tuple[StoryRecord, Path]:
        t0 = time.monotonic()
        self.stats = []
        category = category or ("real_mystery" if topic_id else self.pick_category())
        if category not in self.cfg.categories:
            raise ValueError(f"unknown category {category!r}; choose from {sorted(self.cfg.categories)}")
        target = target_seconds or self.pick_length()
        ingredients = {} if category == "real_mystery" else self.sample_ingredients()
        topic, claims, sources = None, None, []
        if category == "real_mystery":
            topic = self.pick_topic(topic_id)
            claims, source = self.research(topic)
            sources = [source]
        self.log(f"New story: {category}, ~{target} s" + (f", topic {topic.id}" if topic else
                                                          f", ingredients {ingredients}"))

        recent_titles = [r["title"] for r in self.db.recent(self.cfg.novelty.recent_titles_in_prompt)]
        concepts, _ = self._gen(ConceptList, name="concepts", temperature=self.cfg.llm.temperature.concepts,
                                max_tokens=2500, user=concepts_prompt(self.cfg, category, target, ingredients,
                                                                       recent_titles, topic, claims))
        concept, similarity, vec, alternatives = self.choose_concept(concepts.concepts)
        self.log(f"  chosen: '{concept.title}' - {concept.logline}")

        devices = {d.id: d.desc for d in self.cfg.ingredients.device}
        base = script_prompt(self.cfg, category, target, concept, ingredients, devices, claims)
        check = lambda d: validate_story(d, self.cfg, target, category, claims)  # noqa: E731
        draft, problems = self._gen(StoryDraft, user=base, name="script", check=check,
                                    temperature=self.cfg.llm.temperature.script, max_tokens=6000)

        critique, revised = None, False
        try:
            critique, _ = self._gen(Critique, user=critique_prompt(self.cfg, draft, category), name="critique",
                                    temperature=self.cfg.llm.temperature.critique, max_tokens=1500)
        except LLMError as exc:
            self.log(f"  critique skipped: {exc}")
        if critique and (critique.verdict == "revise"
                         or min(critique.scores().values()) < self.cfg.rules.min_critique_score):
            new, new_problems = self._gen(StoryDraft, user=revise_prompt(base, draft, critique), name="revise",
                                          check=check, temperature=self.cfg.llm.temperature.script,
                                          max_tokens=6000)
            if len(new_problems) <= len(problems):
                draft, problems, revised = new, new_problems, True

        banned = self.cfg.rules.video_prompt_banned_words
        for s in draft.scenes:
            if banned_in(s.video_prompt, banned):
                s.video_prompt = sanitize_video_prompt(s.video_prompt, banned)
        problems = check(draft)

        created = datetime.now()
        story_id = f"{created:%Y%m%d-%H%M%S}-{slugify(draft.title)}"
        disclosure = ("Dramatized reconstruction of a documented case. Footage is AI-generated. Sources: "
                      + "; ".join(s.title for s in sources)) if category == "real_mystery" else \
            "Fiction. Footage and narration are AI-generated."
        record = StoryRecord(
            id=story_id, created_at=created.isoformat(timespec="seconds"), category=category,
            target_seconds=target, ingredients=ingredients, topic_id=topic.id if topic else None,
            concept=concept, concept_alternatives=alternatives, novelty_max_similarity=similarity,
            draft=draft, critique=critique, revised=revised, problems=problems, sources=sources,
            claims=claims or [], llm=self.stats, ai_disclosure=disclosure)
        out = save_story(record)
        scores = critique.scores() if critique else {}
        self.db.add_story(
            id=story_id, created_at=record.created_at, category=category, title=draft.title,
            logline=draft.logline, topic_id=record.topic_id, ingredients=ingredients, tags=draft.tags,
            status="draft" if not problems else "draft_with_problems", dir=str(out),
            provider=self.provider.name, model=self.provider.model,
            critique_avg=round(sum(scores.values()) / len(scores), 2) if scores else None, embedding=vec)
        unload = getattr(self.provider, "unload", None)
        if unload:
            unload()
        self.log(f"Story saved to {out} in {(time.monotonic() - t0) / 60:.1f} min"
                 + (f" ({len(problems)} unresolved check(s))" if problems else ""))
        return record, out
