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

from ..llm.base import LLMError, expand_model_pattern, generate
from ..state.db import StateDB
from .config import LLMCfg, ModelEntry, StoryConfig, Topic, load_story_config, load_topics
from .models import (Claim, ClaimList, Concept, ConceptList, ConceptRatings, Critique, FactCheck, LineFixes,
                     SceneFixes, Source, StoryDraft, StoryPatch, StoryRecord)
from .novelty import Embedder, FastEmbedder, concept_text, max_similarity
from .prompts import (claims_prompt, concept_rating_prompt, concepts_prompt, critique_prompt, fact_check_prompt,
                      line_repair_prompt, patch_prompt, revise_prompt, scene_repair_prompt, script_prompt,
                      system_prompt)
from .render import save_story
from .repair import (apply_line_fixes, apply_patch, apply_scene_fixes, auto_fix, hedge_lines, line_issues,
                     scene_issues)
from .research import ResearchError, fetch_article, trim_source
from .validate import validate_story, verify_claims

Log = Callable[[str], None]


def build_provider(cfg: LLMCfg, name: str, entry: ModelEntry | None = None):
    """One provider object for `name`, using `entry` (default: its first configured model)."""
    if name == "ollama":
        from ..llm.ollama import OllamaProvider
        return OllamaProvider(cfg.ollama.url, cfg.ollama.model, cfg.ollama.num_ctx, cfg.ollama.timeout_s,
                              think=cfg.ollama.think)
    if name == "gemini":
        from ..llm.gemini import GeminiProvider
        e = entry or cfg.gemini.models[0]
        return GeminiProvider(cfg.gemini.url, e.id, cfg.gemini.timeout_s,
                              thinking_budget=cfg.gemini.thinking_budget,
                              max_output_tokens=cfg.gemini.max_output_tokens, generation=e.extra_body)
    if name in cfg.openai_compat:
        from ..llm.openai_compat import OpenAICompatProvider
        c = cfg.openai_compat[name]
        e = entry or c.models[0]
        return OpenAICompatProvider(name, c.url, e.id, c.key_env, c.timeout_s, {**c.extra_body, **e.extra_body})
    known = ["ollama", "gemini", *cfg.openai_compat]
    raise ValueError(f"unknown LLM provider {name!r}; configured: {known}")


TIERS = {"best": 0, "normal": 1}


def provider_models(cfg: LLMCfg, name: str) -> list:
    """One provider object per configured model of `name`, in order, each tagged with its tier.

    '*' patterns are expanded against the models the key can use (one listing call); when a listing
    is available, configured names the key doesn't offer are dropped."""
    probe = build_provider(cfg, name)   # raises LLMError when the API key is missing
    if name == "ollama":
        probe.tier = TIERS["normal"]
        return [probe]
    entries = cfg.gemini.models if name == "gemini" else cfg.openai_compat[name].models
    available = probe.list_models() if any("*" in e.id for e in entries) else None
    expanded: list[ModelEntry] = []
    for e in entries:
        ids = expand_model_pattern(e.id, available) if "*" in e.id else [e.id]
        for i in ids:
            if (available is None or i in available) and i not in {x.id for x in expanded}:
                expanded.append(e.model_copy(update={"id": i}))
    if not expanded:
        raise LLMError(f"{name}: none of the configured models ({', '.join(e.id for e in entries)}) is "
                       f"available to this key; see `uv run unseen llm-models --provider {name}`")
    slots = []
    for e in expanded:
        p = build_provider(cfg, name, e)
        p.tier = TIERS[e.tier]
        slots.append(p)
    return slots


def make_provider(cfg: LLMCfg, name: str | None = None, log: Log = print):
    """The configured chain (or one provider's models): best tier first, provider by provider,
    then the normal tier; providers without API keys are skipped."""
    from ..llm.fallback import FallbackProvider
    slots = []
    for n in [name] if name else cfg.chain:
        try:
            slots += provider_models(cfg, n)
        except LLMError as exc:
            if name:
                raise
            log(f"  LLM provider {n} skipped: {exc}")
    if not slots:
        raise LLMError("no LLM provider is available")
    slots.sort(key=lambda p: p.tier)   # stable: chain order and model order kept within a tier
    if len(slots) == 1:
        return slots[0]
    return FallbackProvider(slots, log, cooldown_s=cfg.cooldown_s, max_wait_s=cfg.max_wait_s)


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
        self.category: str | None = None
        self.topic: Topic | None = None

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

    def sample_ingredients(self, target_seconds: int = 30) -> dict[str, str]:
        """Device, location and anomaly always; a twist sometimes; a structure only for longer Reels."""
        ing = self.cfg.ingredients
        recent = self.db.recent_ingredients(ing.cooldown_stories)

        def pick(key: str, options: list[str]) -> str:
            used = {r.get(key) for r in recent}
            return self.rng.choice([o for o in options if o not in used] or options)

        out = {
            "device": pick("device", [d.id for d in ing.device]),
            "location": pick("location", ing.location),
            "anomaly": pick("anomaly", ing.anomaly),
        }
        if self.rng.random() < ing.twist_probability:
            out["twist"] = pick("twist", ing.twist)
        if target_seconds >= ing.structure_min_seconds and self.rng.random() < ing.structure_probability:
            out["structure"] = pick("structure", ing.structure)
        return out

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
            provider or self.provider, model_cls, system=system_prompt(self.cfg, self.category), user=user,
            name=name, temperature=temperature, max_tokens=max_tokens,
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

    def rank_concepts(self, concepts: list[Concept], category: str, recent_titles: list[str]
                      ) -> tuple[list[tuple[Concept, float, np.ndarray]], dict[str, int]]:
        """Concepts best-first: novel enough (embedding similarity) first, then by the critic's rating.

        Returns ([(concept, similarity, embedding), ...], {title: rating})."""
        history = self.db.embeddings()
        vecs = self.embedder.embed([concept_text(c.title, c.logline, c.twist) for c in concepts])
        sims = [max_similarity(v, history) for v in vecs]
        novel = {i for i in range(len(concepts)) if sims[i] <= self.cfg.novelty.max_similarity}
        if not novel:
            self.log(f"  warning: every concept resembles a past story (best similarity {min(sims):.2f})")
            novel = set(range(len(concepts)))
        ratings = self.rate_concepts(concepts, category, recent_titles)
        order = sorted(range(len(concepts)), key=lambda i: (i in novel, ratings.get(i, -1), -sims[i]), reverse=True)
        self.log("  concepts: " + "; ".join(f"'{concepts[i].title}' sim={sims[i]:.2f}"
                                            + (f" rating={ratings[i]}/40" if i in ratings else "") for i in order))
        named = {concepts[i].title: r for i, r in ratings.items()}
        return [(concepts[i], round(sims[i], 3), vecs[i]) for i in order], named

    def rate_concepts(self, concepts: list[Concept], category: str, recent_titles: list[str]) -> dict[int, int]:
        """{concept index: total rating} from the critic model; {} when skipped or failed."""
        critic = self.critic_provider()
        if not self.cfg.quality.rate_concepts or critic.name == "ollama":
            return {}   # rating on the slow local model is not worth the extra minutes
        try:
            ratings, _ = self._gen(ConceptRatings, name="rate-concepts", provider=critic, temperature=0.2,
                                   max_tokens=2000, retries=1,
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
            # should not win just because the prose scored slightly higher; and a rewrite may never add
            # failed checks (e.g. dropping scenes so the Reel falls short of its length)
            better = (self._avg(c) - 0.5 * len(cand_p) > self._avg(best_c) - 0.5 * len(best_p)
                      and len(cand_p) <= len(best_p))
            history.append({"round": rnd, "scores": c.scores(), "avg": self._avg(c), "checks": len(cand_p),
                            "kept": better})
            self.log(f"  revision {rnd}: avg {self._avg(best_c)} -> {self._avg(c)}, checks {len(best_p)} -> "
                     f"{len(cand_p)}" + ("" if better else " (worse: keeping the previous version)"))
            if better:
                best, best_c, best_p, revised = cand, c, cand_p, True
        return best, best_c, revised, history

    def critic_provider(self):
        """Another hosted model than the writer, preferably from another provider (models grade their
        own work kindly).

        Returned as a fallback view that shares cool-downs with the writer's chain; the local model
        is left out because an extra pass on the laptop CPU costs minutes."""
        from ..llm.fallback import FallbackProvider
        p = self.provider
        if isinstance(p, FallbackProvider):
            view = p.critic_view()
            if view is not None:
                return view
        return p

    def minor_provider(self):
        """Small fix-up calls (line and scene repairs) start on the normal tier, so the best models'
        small daily quotas (Gemini: 20 requests per model) go to scripts, critiques and fact-checks."""
        from ..llm.fallback import FallbackProvider
        p = self.provider
        return p.minor_view() if isinstance(p, FallbackProvider) else p

    def critique(self, draft: StoryDraft, category: str) -> Critique | None:
        user = critique_prompt(self.cfg, draft, category)
        critic = self.critic_provider()
        for provider in ([critic, self.provider] if critic is not self.provider else [self.provider]):
            try:
                critique, _ = self._gen(Critique, user=user, name="critique", provider=provider,
                                        temperature=self.cfg.llm.temperature.critique, max_tokens=2500)
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
        if category == "real_mystery":
            # never pose as real evidence, and name the documented case (not a theory or an invented file)
            case = f"{self.topic.wikipedia} ({self.topic.year})" if self.topic else draft.archive_entry
            draft.archive_entry = f"RECONSTRUCTION - {case}"
        for _ in range(rounds):
            fixes = auto_fix(draft, self.cfg, target)
            if fixes:
                self.log("  auto-fixed: " + "; ".join(fixes))
            if category == "real_mystery":   # before the length repairs, so they can shorten around it
                hedged = hedge_lines(draft, self.cfg, claims)
                if hedged:
                    self.log("  hedged: " + "; ".join(hedged))
            bad_scenes = scene_issues(draft, self.cfg, category, self.event_year)
            if bad_scenes:
                self.log(f"  rewriting {len(bad_scenes)} scene(s): "
                         + "; ".join(f"{i}: {w}" for i, w in sorted(bad_scenes.items())))
                try:
                    scene_fixes, _ = self._gen(SceneFixes, name="scene-repair", temperature=0.5, max_tokens=2500,
                                               provider=self.minor_provider(),
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
                                         provider=self.minor_provider(),
                                         user=line_repair_prompt(self.cfg, draft, issues, category, claims))
            except LLMError as exc:
                self.log(f"  line repair failed: {str(exc)[:200]}")
                break
            changed = apply_line_fixes(draft, fixes_obj, set(issues), category)
            self.log("  repaired: " + (", ".join(changed) if changed else
                                       f"nothing usable (returned ids {[f.id for f in fixes_obj.lines]})"))
        if category == "real_mystery":
            hedged = hedge_lines(draft, self.cfg, claims)
            if hedged:
                self.log("  hedged: " + "; ".join(hedged))
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
        self.category = category
        target = target_seconds or self.pick_length()
        ingredients = {} if category == "real_mystery" else self.sample_ingredients(target)
        topic, claims, sources = None, None, []
        self.event_year = None
        self.topic = None
        if category == "real_mystery":
            topic, claims, source = self.research_some_topic(topic_id)
            sources = [source]
            self.event_year = topic.year
            self.topic = topic
        self.log(f"New story: {category}, ~{target} s" + (f", topic {topic.id}" if topic else
                                                          f", ingredients {ingredients}"))

        recent_titles = [r["title"] for r in self.db.recent(self.cfg.novelty.recent_titles_in_prompt)]
        concepts, _ = self._gen(ConceptList, name="concepts", temperature=self.cfg.llm.temperature.concepts,
                                max_tokens=2500, user=concepts_prompt(self.cfg, category, target, ingredients,
                                                                       recent_titles, topic, claims))
        ranked, ratings = self.rank_concepts(concepts.concepts, category, recent_titles)

        devices = {d.id: f"{d.desc} (exists from {d.since})" for d in self.cfg.ingredients.device}
        n_drafts = 1 if self.is_local() else max(1, self.cfg.quality.drafts)
        drafts = []   # (composite score, rank, concept, sim, vec, base prompt, draft, critique)
        for rank, (cand, sim, cvec) in enumerate(ranked[:n_drafts]):
            self.log(f"  draft {rank + 1}/{n_drafts}: '{cand.title}' - {cand.logline}")
            base_c = script_prompt(self.cfg, category, target, cand, ingredients, devices, claims)
            try:
                d, _ = self._gen(StoryDraft, user=base_c, name=f"script-{rank + 1}",
                                 temperature=self.cfg.llm.temperature.script, max_tokens=5000)
            except LLMError as exc:
                if drafts or rank + 1 < n_drafts:
                    self.log(f"  draft {rank + 1} failed: {str(exc)[:200]}")
                    continue
                raise
            self.polish(d, target, category, claims)
            c = self.critique(d, category)
            score = (self._avg(c) if c else 0.0) - 0.5 * len(self.checks(d, target, category, claims))
            drafts.append((score, rank, cand, sim, cvec, base_c, d, c))
            if c:
                self.log(f"  draft {rank + 1} score {score:.2f} (critique avg {self._avg(c)})")
        if not drafts:
            raise LLMError("no draft could be written")
        _, _, concept, similarity, vec, base, draft, first_critique = max(drafts, key=lambda x: (x[0], -x[1]))
        alternatives = [c for c, _, _ in ranked if c is not concept]
        if len(drafts) > 1:
            self.log(f"  chosen: '{concept.title}'")
        draft, critique, revised, history = self.improve(draft, first_critique, base, target, category, claims)
        fact_report: list[dict] = []
        if category == "real_mystery" and claims:
            fact_report = self.fact_check(draft, claims)
            self.polish(draft, target, category, claims)   # corrected lines may need hedging/shortening

        problems = self.checks(draft, target, category, claims)
        unsupported = [e for e in fact_report if e.get("verdict") in ("unsupported", "contradicted")
                       and "corrected" not in e]
        problems += [f"narration line {e['id']} is not supported by its claims (fact-check)" for e in unsupported]
        if any(e.get("verdict") == "not checked" for e in fact_report):
            problems.append("the independent fact-check could not run (no model answered); re-check before use")

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
