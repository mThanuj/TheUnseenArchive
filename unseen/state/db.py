"""SQLite state (state/unseen.db). Milestone 2 stores stories; later milestones add runs/posts."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np

from ..config import REPO_ROOT

DEFAULT_DB = REPO_ROOT / "state" / "unseen.db"
SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS stories (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    category TEXT NOT NULL,
    title TEXT NOT NULL,
    logline TEXT NOT NULL,
    topic_id TEXT,
    ingredients TEXT NOT NULL,
    tags TEXT NOT NULL,
    status TEXT NOT NULL,
    dir TEXT NOT NULL,
    provider TEXT,
    model TEXT,
    critique_avg REAL,
    embedding BLOB
);
CREATE INDEX IF NOT EXISTS stories_created ON stories(created_at);
"""


class StateDB:
    def __init__(self, path: Path | None = None):
        self.path = Path(path or DEFAULT_DB)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.execute("INSERT OR IGNORE INTO meta(key, value) VALUES ('schema_version', '1')")
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def add_story(self, *, id: str, created_at: str, category: str, title: str, logline: str,
                  topic_id: str | None, ingredients: dict, tags: list[str], status: str, dir: str,
                  provider: str, model: str, critique_avg: float | None,
                  embedding: np.ndarray | None) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO stories VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (id, created_at, category, title, logline, topic_id, json.dumps(ingredients), json.dumps(tags),
             status, dir, provider, model, critique_avg,
             embedding.astype(np.float32).tobytes() if embedding is not None else None))
        self.conn.commit()

    def recent(self, limit: int = 50) -> list[sqlite3.Row]:
        return list(self.conn.execute("SELECT * FROM stories ORDER BY created_at DESC LIMIT ?", (limit,)))

    def get(self, story_id: str) -> sqlite3.Row | None:
        rows = list(self.conn.execute("SELECT * FROM stories WHERE id = ? OR id LIKE ?",
                                      (story_id, f"{story_id}%")))
        return rows[0] if len(rows) == 1 else None

    def embeddings(self) -> list[np.ndarray]:
        return [np.frombuffer(r["embedding"], dtype=np.float32)
                for r in self.conn.execute("SELECT embedding FROM stories WHERE embedding IS NOT NULL")]

    def used_topics(self) -> set[str]:
        return {r["topic_id"] for r in self.conn.execute("SELECT topic_id FROM stories WHERE topic_id IS NOT NULL")}

    def recent_ingredients(self, n: int) -> list[dict]:
        return [json.loads(r["ingredients"]) for r in self.recent(n)]

    def recent_categories(self, n: int) -> list[str]:
        return [r["category"] for r in self.recent(n)]
