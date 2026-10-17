"""The story pipeline: category -> (research) -> concepts -> novelty pick -> script -> polish ->
critique -> (patch + polish) -> checks -> files + SQLite.

"Polish" fixes timing/placement/prompt problems deterministically and asks the model to rewrite
only the narration lines that break a rule; the editor's revision is applied as a small patch.
Neither regenerates the whole script (on a CPU that costs 10+ minutes per attempt).

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
from .models import (Claim, ClaimList, Concept, ConceptList, ConceptRatings, Critique, FactCheck, LineFixes,
                     SceneFixes, Source, StoryDraft, StoryPatch, StoryRecord)
from .novelty import Embedder, FastEmbedder, concept_text, max_similarity
from .prompts import (claims_prompt, concept_rating_prompt, concepts_prompt, critique_prompt, fact_check_prompt,
                      line_repair_prompt, patch_prompt, revise_prompt, scene_repair_prompt, script_prompt,
                      system_prompt)
from .render import save_story
from .repair import apply_line_fixes, apply_patch, apply_scene_fixes, auto_fix, line_issues, scene_issues
from .research import ResearchError, fetch_article, trim_source
from .validate import validate_story, verify_claims

Log = Callable[[str], None]


def build_provider(cfg: LLMCfg, name: str):
    if name == "ollama":
        from ..llm.ollama import OllamaProvider
        return OllamaProvider(cfg.ollama.url, cfg.ollama.model, cfg.ollama.num_ctx, cfg.ollama.timeout_s,
                              think=cfg.ollama.think)
    if name == "gemini":
        from ..llm.gemini import GeminiProvider
        return GeminiProvider(cfg.gemini.url, cfg.gemini.model, cfg.gemini.timeout_s,
                              thinking_budget=cfg.gemini.thinking_budget,
                              max_output_tokens=cfg.gemini.max_output_tokens)
    if name in cfg.openai_compat:
        from ..llm.openai_compat import OpenAICompatProvider
        c = cfg.openai_compat[name]
        return OpenAICompatProvider(name, c.url, c.model, c.key_env, c.timeout_s, c.extra_body)
    known = ["ollama", "gemini", *cfg.openai_compat]
    raise ValueError(f"unknown LLM provider {name!r}; configured: {known}")


def make_provider(cfg: LLMCfg, name: str | None = None, log: Log = print):
    """One named provider, or the configured chain minus providers without API keys."""
    if name:
        return build_provider(cfg, name)
    from ..llm.fallback import FallbackProvider
    providers = []
    for n in cfg.chain:
        try:
            providers.append(build_provider(cfg, n))
        except LLMError as exc:
            log(f"  LLM provider {n} skipped: {exc}")
    if len(providers) == 1:
        return providers[0]
    return FallbackProvider(providers, log)


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
        self.provider = provider or make_provider(self.cfg.llm, provider_name, log)
        if hasattr(self.provider, "log"):
            self.provider.log = log   # streaming progress goes to the same log
        self.db = db or StateDB()
        self.log = log
        self.rng = random.Random(seed)
        self._embedder = embedder
        self.fetch = fetch
        self.stats: list[dict] = []
        self.event_year: int | None = None   # real mysteries: the case's year (era checks)

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

    def pick_topic(self, topic_id: str | None = None, exclude: set[str] | None = None) -> Topic:
        eligible = self.eligible_topics()
        if topic_id:
            for t in eligible:
                if t.id == topic_id:
                    return t
            raise ValueError(f"topic {topic_id!r} not found or not eligible; see config/topics.yaml")
        eligible = [t for t in eligible if t.id not in (exclude or set())]
        if not eligible:
            raise ValueError("no eligible topics left in config/topics.yaml")
        used = self.db.used_topics()
        fresh = [t for t in eligible if t.id not in used]
        return self.rng.choice(fresh or eligible)

    def research_some_topic(self, topic_id: str | None, attempts: int = 3) -> tuple[Topic, list[Claim], Source]:
        """Research a topic; if one fails (missing article, too few verifiable claims), try another."""
        tried: set[str] = set()
        last_error: Exception | None = None
        for i in range(1 if topic_id else attempts):
            try:
                topic = self.pick_topic(topic_id, exclude=tried)
            except ValueError:
                if last_error:
                    raise last_error   # ran out of topics: report the real research failure
                raise
            try:
                claims, source = self.research(topic)
                return topic, claims, source
            except (ResearchError, LLMError) as exc:
                tried.add(topic.id)
                last_error = exc
                self.log(f"  research failed for {topic.id}: {exc}")
                if topic_id or i == attempts - 1:
                    raise
        raise ResearchError("no topic could be researched")

    # ------------------------------------------------------------------ steps
    def _gen(self, model_cls, *, user: str, name: str, temperature: float, max_tokens: int, check=None,
             retries: int | None = None, provider=None):
        obj, stats, problems = generate(
            provider or self.provider, model_cls, system=system_prompt(self.cfg), user=user, name=name,
            temperature=temperature, max_tokens=max_tokens,
            max_retries=self.cfg.llm.max_retries if retries is None else retries,
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

    def choose_concept(self, concepts: list[Concept], category: str, recent_titles: list[str]
                       ) -> tuple[Concept, float, np.ndarray, list[Concept], dict[str, int]]:
        """Novel enough first (embedding similarity), then the critic's best-rated among those."""
        history = self.db.embeddings()
        vecs = self.embedder.embed([concept_text(c.title, c.logline, c.twist) for c in concepts])
        sims = [max_similarity(v, history) for v in vecs]
        pool = [i for i in range(len(concepts)) if sims[i] <= self.cfg.novelty.max_similarity]
        if not pool:
            self.log(f"  warning: every concept resembles a past story (best similarity {min(sims):.2f})")
            pool = list(range(len(concepts)))
        ratings = self.rate_concepts(concepts, category, recent_titles)
        best = max(pool, key=lambda i: (ratings.get(i, -1), -sims[i]))
        order = [best] + sorted((i for i in range(len(concepts)) if i != best), key=lambda i: -ratings.get(i, -1))
        self.log("  concepts: " + "; ".join(f"'{concepts[i].title}' sim={sims[i]:.2f}"
                                            + (f" rating={ratings[i]}/40" if i in ratings else "") for i in order))
        named = {concepts[i].title: r for i, r in ratings.items()}
        return concepts[best], round(sims[best], 3), vecs[best], [concepts[i] for i in order[1:]], named

    def rate_concepts(self, concepts: list[Concept], category: str, recent_titles: list[str]) -> dict[int, int]:
        """{concept index: total rating} from the critic model; {} when skipped or failed."""
        critic = self.critic_provider()
        if not self.cfg.quality.rate_concepts or critic.name == "ollama":
            return {}   # rating on the slow local model is not worth the extra minutes
        try:
            ratings, _ = self._gen(ConceptRatings, name="rate-concepts", provider=critic, temperature=0.2,
                                   max_tokens=1200, retries=1,
                                   user=concept_rating_prompt(concepts, category, recent_titles))
        except LLMError as exc:
            self.log(f"  concept rating skipped: {str(exc)[:160]}")
            return {}
        return {r.index - 1: r.total for r in ratings.ratings if 1 <= r.index <= len(concepts)}

    def is_local(self) -> bool:
        return self.provider.name == "ollama"

    def needs_revision(self, c: Critique, problems: list[str] | None = None) -> bool:
        s = c.scores()
        return (bool(problems) or c.verdict == "revise" or min(s.values()) < self.cfg.rules.min_critique_score
                or sum(s.values()) / len(s) < self.cfg.quality.target_avg_score)

    def checks(self, draft: StoryDraft, target: int, category: str, claims: list[Claim] | None) -> list[str]:
        return validate_story(draft, self.cfg, target, category, claims, self.event_year)

    @staticmethod
    def _avg(c: Critique) -> float:
        s = c.scores()
        return round(sum(s.values()) / len(s), 2)

    def improve(self, draft: StoryDraft, critique: Critique | None, base: str, target: int, category: str,
                claims: list[Claim] | None) -> tuple[StoryDraft, Critique | None, bool, list[dict]]:
        """Revise -> re-critique -> keep whichever version scores higher.

        Hosted writer: full rewrites (seconds each, up to `revision_rounds`). Local model: one small
        patch, because a full rewrite on the laptop CPU takes 10+ minutes.
        """
        if critique is None:
            return draft, None, False, []
        history = [{"round": 0, "scores": critique.scores(), "avg": self._avg(critique), "kept": True}]
        best, best_c, revised = draft, critique, False
        best_p = self.checks(best, target, category, claims)
        local = self.is_local()
        for rnd in range(1, (1 if local else self.cfg.quality.revision_rounds) + 1):
            if not self.needs_revision(best_c, best_p):
                break
            try:
                if local:
                    cand = best.model_copy(deep=True)
                    patch, _ = self._gen(StoryPatch, name="revise", temperature=self.cfg.llm.temperature.script,
                                         max_tokens=2000, retries=0,
                                         user=patch_prompt(self.cfg, best, best_c, category, claims))
                    self.log(f"  patch: {', '.join(apply_patch(cand, patch, category)) or 'no changes'}")
                else:
                    cand, _ = self._gen(StoryDraft, name=f"rewrite-{rnd}", temperature=self.cfg.llm.temperature.script,
                                        max_tokens=5000, retries=1, user=revise_prompt(base, best, best_c, best_p))
            except LLMError as exc:
                self.log(f"  revision skipped: {str(exc)[:200]}")
                break
            self.polish(cand, target, category, claims)
            c = self.critique(cand, category)
            if c is None:
                break
            cand_p = self.checks(cand, target, category, claims)
            # each unresolved automatic check costs half a point: an anachronism or an unsupported fact
            # should not win just because the prose scored slightly higher
            better = self._avg(c) - 0.5 * len(cand_p) > self._avg(best_c) - 0.5 * len(best_p)
            history.append({"round": rnd, "scores": c.scores(), "avg": self._avg(c), "checks": len(cand_p),
                            "kept": better})
            self.log(f"  revision {rnd}: avg {self._avg(best_c)} -> {self._avg(c)}, checks {len(best_p)} -> "
                     f"{len(cand_p)}" + ("" if better else " (worse: keeping the previous version)"))
            if better:
                best, best_c, best_p, revised = cand, c, cand_p, True
        return best, best_c, revised, history

    def critic_provider(self):
        """Other hosted models than the writer, in preference order (models grade themselves kindly).

        Returned as a fallback view that shares cool-downs with the writer's chain; the local model
        is left out because an extra pass on the laptop CPU costs minutes."""
        from ..llm.fallback import FallbackProvider
        p = self.provider
        if isinstance(p, FallbackProvider):
            view = p.excluding({p.name, "ollama"})
            if view is not None:
                return view
        return p

    def critique(self, draft: StoryDraft, category: str) -> Critique | None:
        user = critique_prompt(self.cfg, draft, category)
        critic = self.critic_provider()
        for provider in ([critic, self.provider] if critic is not self.provider else [self.provider]):
            try:
                critique, _ = self._gen(Critique, user=user, name="critique", provider=provider,
                                        temperature=self.cfg.llm.temperature.critique, max_tokens=1500)
                return critique
            except LLMError as exc:
                self.log(f"  critique by {provider.name} failed: {str(exc)[:200]}")
        return None

    def fact_check(self, draft: StoryDraft, claims: list[Claim]) -> list[dict]:
        """Real mysteries: an independent model checks each line against the claims it cites and
        corrected wording replaces unsupported/contradicted lines."""
        critic = self.critic_provider()
        result = None
        for provider in ([critic, self.provider] if critic is not self.provider else [self.provider]):
            try:
                result, _ = self._gen(FactCheck, name="fact-check", provider=provider, temperature=0.1,
                                      max_tokens=2000, user=fact_check_prompt(draft, claims))
                break
            except LLMError as exc:
                self.log(f"  fact-check by {provider.name} failed: {str(exc)[:200]}")
        if result is None:
            return [{"id": None, "verdict": "not checked"}]
        by_id = {n.id: n for n in draft.narration}
        report = []
        for v in result.lines:
            line = by_id.get(v.id)
            if line is None:
                continue
            entry = {"id": v.id, "verdict": v.verdict, "original": line.text}
            if v.verdict in ("unsupported", "contradicted") and v.corrected.strip():
                line.text = v.corrected.strip()
                entry["corrected"] = line.text
            report.append(entry)
        bad = [e for e in report if e["verdict"] in ("unsupported", "contradicted")]
        self.log(f"  fact-check: {len(report) - len(bad)} line(s) supported/atmospheric, {len(bad)} corrected"
                 + ("" if all("corrected" in e for e in bad) else " (some without a correction)"))
        return report

    def polish(self, draft: StoryDraft, target: int, category: str, claims: list[Claim] | None,
               rounds: int = 2) -> None:
        """Deterministic fixes, then LLM rewrites of only the offending scenes and narration lines."""
        for _ in range(rounds):
            fixes = auto_fix(draft, self.cfg, target)
            if fixes:
                self.log("  auto-fixed: " + "; ".join(fixes))
            bad_scenes = scene_issues(draft, self.cfg, category, self.event_year)
            if bad_scenes:
                self.log(f"  rewriting {len(bad_scenes)} scene(s): "
                         + "; ".join(f"{i}: {w}" for i, w in sorted(bad_scenes.items())))
                try:
                    scene_fixes, _ = self._gen(SceneFixes, name="scene-repair", temperature=0.5, max_tokens=2500,
                                               user=scene_repair_prompt(self.cfg, draft, bad_scenes))
                    apply_scene_fixes(draft, scene_fixes, set(bad_scenes))
                    auto_fix(draft, self.cfg, target)
                except LLMError as exc:
                    self.log(f"  scene repair failed: {str(exc)[:200]}")
            issues = line_issues(draft, self.cfg, target, category, claims)
            if not issues:
                if not bad_scenes:
                    return
                continue
            self.log(f"  rewriting {len(issues)} line(s): " + "; ".join(f"{i}: {w}" for i, w in sorted(issues.items())))
            try:
                fixes_obj, _ = self._gen(LineFixes, name="repair", temperature=0.5, max_tokens=1200,
                                         user=line_repair_prompt(self.cfg, draft, issues, category, claims))
            except LLMError as exc:
                self.log(f"  line repair failed: {exc}")
                return
            apply_line_fixes(draft, fixes_obj, set(issues), category)
        auto_fix(draft, self.cfg, target)

    # --------------------------------------------------------------- pipeline
    def new_story(self, category: str | None = None, topic_id: str | None = None,
                  target_seconds: int | None = None) -> tuple[StoryRecord, Path]:
        """Write one story. The LLM is unloaded afterwards even on failure (frees RAM/VRAM)."""
        try:
            return self._new_story(category, topic_id, target_seconds)
        finally:
            unload = getattr(self.provider, "unload", None)
            if unload:
                unload()

    def _new_story(self, category: str | None, topic_id: str | None,
                   target_seconds: int | None) -> tuple[StoryRecord, Path]:
        t0 = time.monotonic()
        self.stats = []
        category = category or ("real_mystery" if topic_id else self.pick_category())
        if category not in self.cfg.categories:
            raise ValueError(f"unknown category {category!r}; choose from {sorted(self.cfg.categories)}")
        target = target_seconds or self.pick_length()
        ingredients = {} if category == "real_mystery" else self.sample_ingredients()
        topic, claims, sources = None, None, []
        self.event_year = None
        if category == "real_mystery":
            topic, claims, source = self.research_some_topic(topic_id)
            sources = [source]
            self.event_year = topic.year
        self.log(f"New story: {category}, ~{target} s" + (f", topic {topic.id}" if topic else
                                                          f", ingredients {ingredients}"))

        recent_titles = [r["title"] for r in self.db.recent(self.cfg.novelty.recent_titles_in_prompt)]
        concepts, _ = self._gen(ConceptList, name="concepts", temperature=self.cfg.llm.temperature.concepts,
                                max_tokens=2500, user=concepts_prompt(self.cfg, category, target, ingredients,
                                                                       recent_titles, topic, claims))
        concept, similarity, vec, alternatives, ratings = self.choose_concept(concepts.concepts, category,
                                                                              recent_titles)
        self.log(f"  chosen: '{concept.title}' - {concept.logline}")

        devices = {d.id: f"{d.desc} (exists from {d.since})" for d in self.cfg.ingredients.device}
        base = script_prompt(self.cfg, category, target, concept, ingredients, devices, claims)
        draft, _ = self._gen(StoryDraft, user=base, name="script",
                             temperature=self.cfg.llm.temperature.script, max_tokens=5000)
        self.polish(draft, target, category, claims)
        draft, critique, revised, history = self.improve(draft, self.critique(draft, category), base, target,
                                                         category, claims)
        fact_report: list[dict] = []
        if category == "real_mystery" and claims:
            fact_report = self.fact_check(draft, claims)
            self.polish(draft, target, category, claims)   # corrected lines may need hedging/shortening

        problems = self.checks(draft, target, category, claims)
        unsupported = [e for e in fact_report if e.get("verdict") in ("unsupported", "contradicted")
                       and "corrected" not in e]
        problems += [f"narration line {e['id']} is not supported by its claims (fact-check)" for e in unsupported]

        created = datetime.now()
        story_id = f"{created:%Y%m%d-%H%M%S}-{slugify(draft.title)}"
        disclosure = ("Dramatized reconstruction of a documented case. Footage is AI-generated. Sources: "
                      + "; ".join(s.title for s in sources)) if category == "real_mystery" else \
            "Fiction. Footage and narration are AI-generated."
        record = StoryRecord(
            id=story_id, created_at=created.isoformat(timespec="seconds"), category=category,
            target_seconds=target, ingredients=ingredients, topic_id=topic.id if topic else None,
            concept=concept, concept_alternatives=alternatives, novelty_max_similarity=similarity,
            concept_ratings=ratings, draft=draft, critique=critique, critique_history=history,
            revised=revised, problems=problems, sources=sources, claims=claims or [], fact_check=fact_report,
            llm=self.stats, ai_disclosure=disclosure)
        out = save_story(record)
        scores = critique.scores() if critique else {}
        self.db.add_story(
            id=story_id, created_at=record.created_at, category=category, title=draft.title,
            logline=draft.logline, topic_id=record.topic_id, ingredients=ingredients, tags=draft.tags,
            status="draft" if not problems else "draft_with_problems", dir=str(out),
            provider=self.provider.name, model=self.provider.model,
            critique_avg=round(sum(scores.values()) / len(scores), 2) if scores else None, embedding=vec)
        self.log(f"Story saved to {out} in {(time.monotonic() - t0) / 60:.1f} min"
                 + (f" ({len(problems)} unresolved check(s))" if problems else ""))
        return record, out
