"""Fetch source material for real-world mysteries from Wikipedia (MediaWiki Action API).

Follows Wikimedia's API etiquette (descriptive User-Agent, one request per article). Text is
CC BY-SA 4.0: the engine paraphrases facts, never copies prose into narration, and keeps the
article URL, revision id and license with every story for attribution.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

import httpx

from .models import Source

API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = ("TheUnseenArchive/0.1 (https://github.com/mThanuj/TheUnseenArchive; "
              "research for an AI-disclosed storytelling channel)")
LICENSE = "CC BY-SA 4.0 (Wikipedia)"


class ResearchError(RuntimeError):
    pass


def parse_article(data: dict) -> tuple[str, Source]:
    """(plain text, Source) from a formatversion=2 `action=query` response."""
    pages = (data.get("query") or {}).get("pages") or []
    if not pages or pages[0].get("missing") or pages[0].get("invalid"):
        title = pages[0].get("title") if pages else "?"
        raise ResearchError(f"Wikipedia article not found: '{title}'")
    p = pages[0]
    text = p.get("extract") or ""
    if len(text) < 500:
        raise ResearchError(f"Wikipedia article '{p.get('title')}' has too little text ({len(text)} chars)")
    rev = (p.get("revisions") or [{}])[0]
    src = Source(
        title=f"Wikipedia: {p.get('title')}",
        url=p.get("fullurl") or f"https://en.wikipedia.org/wiki/{p.get('title', '').replace(' ', '_')}",
        license=LICENSE,
        revision_id=rev.get("revid"),
        retrieved_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        external_links=[e.get("url", "") for e in (p.get("extlinks") or [])][:40],
    )
    return text, src


def _api(params: dict, timeout: float = 30) -> dict:
    try:
        r = httpx.get(API, params={"format": "json", "formatversion": "2", **params},
                      headers={"User-Agent": USER_AGENT}, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPError as exc:
        raise ResearchError(f"Wikipedia request failed: {exc!r}") from None


def _significant(text: str) -> set[str]:
    stop = {"the", "of", "a", "an", "and", "in", "on", "incident", "event", "mystery", "case"}
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in stop and len(w) > 1}


def search_title(query: str) -> str | None:
    """Best matching article title, only if it shares a significant word with the query."""
    data = _api({"action": "query", "list": "search", "srsearch": query, "srlimit": "3"})
    for hit in (data.get("query") or {}).get("search") or []:
        if _significant(hit.get("title", "")) & _significant(query):
            return hit["title"]
    return None


def fetch_article(title: str, timeout: float = 30) -> tuple[str, Source]:
    """Article text + Source. A title that doesn't exist falls back to Wikipedia search."""
    params = {
        "action": "query", "redirects": "1", "titles": title,
        "prop": "extracts|info|extlinks|revisions", "explaintext": "1", "exsectionformat": "plain",
        "inprop": "url", "ellimit": "max", "rvprop": "ids|timestamp",
    }
    data = _api(params, timeout)
    pages = (data.get("query") or {}).get("pages") or []
    if pages and pages[0].get("missing"):
        found = search_title(title)
        if not found:
            raise ResearchError(f"Wikipedia has no article '{title}' and search found no close match")
        data = _api({**params, "titles": found}, timeout)
    return parse_article(data)


def trim_source(text: str, max_chars: int) -> str:
    """Keep the lead and early sections, cut at a paragraph boundary, drop reference tails."""
    for tail in ("\nSee also\n", "\nReferences\n", "\nFurther reading\n", "\nExternal links\n", "\nNotes\n"):
        i = text.find(tail)
        if i > 0:
            text = text[:i]
    if len(text) <= max_chars:
        return text
    cut = text.rfind("\n", 0, max_chars)
    return text[:cut if cut > max_chars // 2 else max_chars]
