"""Typed access to config/story.yaml and config/topics.yaml."""

from __future__ import annotations

from pydantic import BaseModel

from ..config import CONFIG_DIR, load_yaml


class OllamaCfg(BaseModel):
    url: str
    model: str
    num_ctx: int = 8192
    timeout_s: float = 2400
    think: bool | str = False


class GeminiCfg(BaseModel):
    url: str
    model: str
    timeout_s: float = 300
    thinking_budget: int | None = 1024
    max_output_tokens: int = 16384


class Temperatures(BaseModel):
    concepts: float = 1.0
    script: float = 0.8
    critique: float = 0.2


class OpenAICompatCfg(BaseModel):
    url: str
    model: str
    key_env: str
    timeout_s: float = 300


class LLMCfg(BaseModel):
    chain: list[str] = ["ollama"]   # the real order lives in config/story.yaml (llm.chain)
    ollama: OllamaCfg
    gemini: GeminiCfg
    openai_compat: dict[str, OpenAICompatCfg] = {}
    temperature: Temperatures = Temperatures()
    max_retries: int = 2


class Brand(BaseModel):
    name: str
    tagline: str
    narrator: str


class CategoryCfg(BaseModel):
    weight: float
    label: str


class LengthChoice(BaseModel):
    seconds: int
    weight: float


class SceneSeconds(BaseModel):
    min: float
    max: float
    typical: float


class LengthCfg(BaseModel):
    choices: list[LengthChoice]
    words_per_second: float
    scene_seconds: SceneSeconds
    max_scenes: int


class Device(BaseModel):
    id: str
    desc: str


class Ingredients(BaseModel):
    cooldown_stories: int
    device: list[Device]
    location: list[str]
    anomaly: list[str]
    structure: list[str]
    twist: list[str]


class WordRange(BaseModel):
    min: int
    max: int


class Rules(BaseModel):
    min_critique_score: int = 7
    hook_max_words: int
    narration_line_max_words: int
    banned_cliches: list[str]
    video_prompt_banned_words: list[str]
    video_prompt_words: WordRange
    content_limits: str


class NoveltyCfg(BaseModel):
    max_similarity: float = 0.86
    recent_titles_in_prompt: int = 12


class RealMysteryCfg(BaseModel):
    min_years_ago_sensitive: int
    max_source_chars: int
    hedge_words: list[str]


class StoryConfig(BaseModel):
    llm: LLMCfg
    brand: Brand
    categories: dict[str, CategoryCfg]
    length: LengthCfg
    ingredients: Ingredients
    rules: Rules
    novelty: NoveltyCfg = NoveltyCfg()
    real_mystery: RealMysteryCfg


class Topic(BaseModel):
    id: str
    wikipedia: str
    year: int
    kind: str
    sensitivity: str


class Topics(BaseModel):
    topics: list[Topic]


def load_story_config() -> StoryConfig:
    return StoryConfig.model_validate(load_yaml(CONFIG_DIR / "story.yaml"))


def load_topics() -> list[Topic]:
    return Topics.model_validate(load_yaml(CONFIG_DIR / "topics.yaml")).topics
