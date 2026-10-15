"""Minimal field helpers for the Gemini add-on (no BeautifulSoup dependency)."""

from __future__ import annotations

import re

EXPRESSION_FIELD_CANDIDATES = (
    "Word",
    "Expression",
    "Vocabulary",
    "Vocab",
    "Term",
    "Focus",
    "Target",
    "Front",
)


def field_value(meta: object) -> str:
    if isinstance(meta, dict):
        return str(meta.get("value", ""))
    return str(meta or "")


def deck_query_clause(deck: str) -> str:
    deck = deck.strip()
    if not deck:
        return ""
    if any(ch in deck for ch in ' "():'):
        escaped = deck.replace('"', '\\"')
        return f'deck:"{escaped}"'
    return f"deck:{deck}"


def _plain_text(html: str) -> str:
    if not html:
        return ""
    text = re.sub(r"<[^>]+>", " ", html)
    return " ".join(text.split()).strip()


def guess_expression(fields: dict, note_id: object = "") -> str:
    for name in EXPRESSION_FIELD_CANDIDATES:
        if name in fields:
            text = _plain_text(field_value(fields[name]))
            if text:
                return text
    for name, meta in fields.items():
        text = _plain_text(field_value(meta))
        if text and "yomitan-glossary" not in text.lower() and len(text) < 80:
            return text
    return str(note_id)
