"""Gemini API helpers for sentence translation + short grammar notes.

Stdlib only (urllib) so the Anki add-on needs no extra packages.
"""

from __future__ import annotations

import json
import re
import ssl
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash-lite"
# Flash-Lite free tier ~15 RPM — stay under that with headroom
DEFAULT_REQUEST_GAP_SEC = 4.2

# Fallback if the live models.list call fails / no key yet
FALLBACK_GEMINI_MODELS: tuple[str, ...] = (
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
)

DEFAULT_SYSTEM_INSTRUCTIONS = (
    "You help a Japanese language learner study Anki cards.\n"
    "Given the Japanese sentence below, respond with JSON only matching the schema.\n"
    "Never use romaji or Latin phonetic spellings of Japanese. "
    "When citing Japanese, use kanji/kana only; explain in English."
)

DEFAULT_TRANSLATION_INSTRUCTIONS = (
    "natural, conversational English. Translate meaning (do not transliterate). "
    "Keep proper nouns when appropriate. No romaji."
)

DEFAULT_GRAMMAR_INSTRUCTIONS = (
    "2–4 short sentences on the important grammar, particles, and conjugations.\n"
    "Cite Japanese only in kanji/kana — never romaji "
    "(write そちらの方 and ご注文, not sochira no kata / go-chuumon; write は, not wa).\n"
    "Plain language, no fluff, no bullet lists."
)

_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "{model}:generateContent?key={key}"
)
_LIST_MODELS_URL = "https://generativelanguage.googleapis.com/v1beta/models?key={key}&pageSize=100"

_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "translation": {
            "type": "STRING",
            "description": "Natural English translation for a Japanese learner",
        },
        "grammar": {
            "type": "STRING",
            "description": "Short grammar explanation (2-4 sentences)",
        },
    },
    "required": ["translation", "grammar"],
}


@dataclass
class GeminiResult:
    translation: str = ""
    grammar: str = ""
    raw_text: str = ""


class GeminiError(Exception):
    def __init__(
        self,
        message: str,
        status: int | None = None,
        *,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status = status
        self.retryable = retryable

    def __str__(self) -> str:  # noqa: D105
        return self.message


def plain_field_text(html_or_text: str) -> str:
    """Strip light HTML / furigana markup to plain sentence text."""
    text = (html_or_text or "").strip()
    if not text:
        return ""
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.I)
    text = re.sub(r"<[^>]+>", "", text)
    # Common Anki furigana: 漢字[かんじ] or 漢字[かんじ]
    text = re.sub(r"\[[^\]]+\]", "", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def build_prompt(
    sentence_ja: str,
    *,
    want_translation: bool = True,
    want_grammar: bool = True,
    expression: str = "",
    system_instructions: str = "",
    translation_instructions: str = "",
    grammar_instructions: str = "",
) -> str:
    system = (system_instructions or DEFAULT_SYSTEM_INSTRUCTIONS).strip()
    trans_i = (translation_instructions or DEFAULT_TRANSLATION_INSTRUCTIONS).strip()
    gram_i = (grammar_instructions or DEFAULT_GRAMMAR_INSTRUCTIONS).strip()

    parts = [system]
    if want_translation:
        parts.append(f"- translation: {trans_i}")
    else:
        parts.append('- translation: return an empty string ""')
    if want_grammar:
        parts.append(f"- grammar: {gram_i}")
    else:
        parts.append('- grammar: return an empty string ""')
    if expression.strip():
        parts.append(f"Focus word on the card (optional context): {expression.strip()}")
    parts.append("Japanese sentence:")
    parts.append(sentence_ja.strip())
    return "\n".join(parts)


def _extract_candidate_text(payload: dict[str, Any]) -> str:
    candidates = payload.get("candidates") or []
    if not candidates:
        block = payload.get("promptFeedback") or {}
        reason = block.get("blockReason") or payload.get("error") or "no candidates"
        raise GeminiError(f"Gemini returned no text ({reason})")
    content = candidates[0].get("content") or {}
    parts = content.get("parts") or []
    texts = [str(p.get("text", "")) for p in parts if p.get("text")]
    if not texts:
        raise GeminiError("Gemini candidate had no text parts")
    return "\n".join(texts).strip()


def _parse_structured(text: str) -> GeminiResult:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        # Fall back: whole body as translation if JSON failed
        return GeminiResult(translation=cleaned, grammar="", raw_text=text)
    if not isinstance(data, dict):
        return GeminiResult(translation=cleaned, grammar="", raw_text=text)
    return GeminiResult(
        translation=str(data.get("translation") or "").strip(),
        grammar=str(data.get("grammar") or "").strip(),
        raw_text=text,
    )


def _http_generate(
    model_id: str,
    api_key: str,
    prompt: str,
    *,
    timeout_sec: float,
) -> tuple[str, str]:
    """POST generateContent, trying several config shapes until one works.

    Returns (raw_response_text, variant_label).
    Alias models like gemini-flash-latest reject thinkingConfig / some schema combos.
    """
    url = _ENDPOINT.format(model=model_id, key=api_key)
    ctx = ssl.create_default_context()

    variants: list[tuple[str, dict[str, Any]]] = [
        (
            "json+schema",
            {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.2,
                    "maxOutputTokens": 512,
                    "responseMimeType": "application/json",
                    "responseSchema": _RESPONSE_SCHEMA,
                },
            },
        ),
        (
            "json",
            {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.2,
                    "maxOutputTokens": 512,
                    "responseMimeType": "application/json",
                },
            },
        ),
        (
            "plain",
            {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.2,
                    "maxOutputTokens": 512,
                },
            },
        ),
    ]
    # Optional speed hint for true 2.5 IDs only (aliases often 400 on this).
    if re.search(r"gemini-2\.5", model_id):
        variants.insert(
            0,
            (
                "json+schema+no-think",
                {
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {
                        "temperature": 0.2,
                        "maxOutputTokens": 512,
                        "responseMimeType": "application/json",
                        "responseSchema": _RESPONSE_SCHEMA,
                        "thinkingConfig": {"thinkingBudget": 0},
                    },
                },
            ),
        )

    last_detail = ""
    last_code = 400
    for label, body in variants:
        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout_sec, context=ctx) as resp:
                raw = resp.read().decode("utf-8")
            try:
                from .gemini_log import log as _glog

                _glog("HTTP variant ok", model=model_id, variant=label, bytes=len(raw))
            except Exception:  # noqa: BLE001
                pass
            return raw, label
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            last_detail = detail
            last_code = exc.code
            try:
                from .gemini_log import log as _glog

                _glog(
                    "HTTP variant fail",
                    model=model_id,
                    variant=label,
                    status=exc.code,
                    detail=detail[:200],
                )
            except Exception:  # noqa: BLE001
                pass
            # Try next shape only on client/request errors
            if exc.code not in {400, 404}:
                raise GeminiError(
                    f"Gemini HTTP {exc.code}: {detail}",
                    status=exc.code,
                    retryable=exc.code in {429, 500, 502, 503},
                ) from exc
            continue
        except urllib.error.URLError as exc:
            raise GeminiError(
                f"Gemini network error: {exc.reason}", retryable=True
            ) from exc

    raise GeminiError(
        f"Gemini HTTP {last_code}: {last_detail}",
        status=last_code,
        retryable=False,
    )


def generate_for_sentence(
    api_key: str,
    sentence_ja: str,
    *,
    model: str = DEFAULT_GEMINI_MODEL,
    want_translation: bool = True,
    want_grammar: bool = True,
    expression: str = "",
    timeout_sec: float = 45.0,
    system_instructions: str = "",
    translation_instructions: str = "",
    grammar_instructions: str = "",
) -> GeminiResult:
    """Call Gemini once for translation and/or grammar."""
    key = (api_key or "").strip()
    if not key:
        raise GeminiError("Gemini API key is empty")
    sentence = plain_field_text(sentence_ja)
    if not sentence:
        raise GeminiError("Sentence field is empty")
    if not want_translation and not want_grammar:
        raise GeminiError("Nothing to generate (enable translation and/or grammar)")

    prompt = build_prompt(
        sentence,
        want_translation=want_translation,
        want_grammar=want_grammar,
        expression=expression,
        system_instructions=system_instructions,
        translation_instructions=translation_instructions,
        grammar_instructions=grammar_instructions,
    )
    model_id = _model_short_name(model or DEFAULT_GEMINI_MODEL)
    t0 = time.monotonic()
    raw, variant = _http_generate(model_id, key, prompt, timeout_sec=timeout_sec)
    elapsed = time.monotonic() - t0
    try:
        from .gemini_log import log as _glog

        _glog(
            "HTTP generateContent ok",
            model=model_id,
            variant=variant,
            elapsed_s=f"{elapsed:.2f}",
            bytes=len(raw),
        )
    except Exception:  # noqa: BLE001
        pass

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise GeminiError(f"Gemini returned non-JSON: {raw[:200]}") from exc

    if isinstance(payload.get("error"), dict):
        err = payload["error"]
        msg = err.get("message") or str(err)
        code = err.get("code")
        raise GeminiError(f"Gemini error: {msg}", status=code, retryable=code == 429)

    text = _extract_candidate_text(payload)
    return _parse_structured(text)


# Simple inline tags Gemini may emit (e.g. bold grammar points). No attributes.
_SAFE_HTML_TAG_RE = re.compile(
    r"</?(?:b|i|em|strong|u)\s*>|<\s*br\s*/?\s*>",
    re.IGNORECASE,
)


def to_anki_html(text: str) -> str:
    """Store text for Anki fields: keep safe inline tags, escape the rest, <br> for newlines."""
    if not text:
        return ""

    kept: list[str] = []

    def _stash(match: re.Match[str]) -> str:
        raw = match.group(0)
        if re.match(r"<\s*br\s*/?\s*>", raw, re.IGNORECASE):
            kept.append("<br>")
        else:
            # Normalize to lowercase tag name, preserve open vs close
            name = re.sub(r"[^a-zA-Z/]", "", raw).lower()
            if name.startswith("/"):
                kept.append(f"</{name[1:]}>")
            else:
                kept.append(f"<{name}>")
        return f"\0TAG{len(kept) - 1}\0"

    protected = _SAFE_HTML_TAG_RE.sub(_stash, text)
    escaped = (
        protected.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "<br>")
    )
    for i, tag in enumerate(kept):
        escaped = escaped.replace(f"\0TAG{i}\0", tag)
    return escaped


def _model_short_name(name: str) -> str:
    # API returns "models/gemini-2.5-flash-lite"
    return name.split("/", 1)[-1].strip()


def _is_text_generate_model(entry: dict[str, Any]) -> bool:
    methods = entry.get("supportedGenerationMethods") or []
    if "generateContent" not in methods:
        return False
    name = _model_short_name(str(entry.get("name") or "")).lower()
    if not name.startswith("gemini-"):
        return False
    # Skip image/video/audio/TTS/embedding specialists
    skip_bits = (
        "embed",
        "image",
        "imagen",
        "veo",
        "tts",
        "audio",
        "live",
        "robotics",
        "computer-use",
    )
    return not any(bit in name for bit in skip_bits)


def list_gemini_models(api_key: str, *, timeout_sec: float = 30.0) -> list[str]:
    """Return model ids that support generateContent (newest-ish order preserved)."""
    key = (api_key or "").strip()
    if not key:
        raise GeminiError("Gemini API key is empty — paste a key to refresh models")

    url = _LIST_MODELS_URL.format(key=key)
    req = urllib.request.Request(url, method="GET")
    ctx = ssl.create_default_context()
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec, context=ctx) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:400]
        raise GeminiError(f"List models HTTP {exc.code}: {detail}", status=exc.code) from exc
    except urllib.error.URLError as exc:
        raise GeminiError(f"List models network error: {exc.reason}", retryable=True) from exc

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise GeminiError(f"List models returned non-JSON: {raw[:200]}") from exc

    if isinstance(payload.get("error"), dict):
        raise GeminiError(payload["error"].get("message") or str(payload["error"]))

    names: list[str] = []
    seen: set[str] = set()
    for entry in payload.get("models") or []:
        if not isinstance(entry, dict) or not _is_text_generate_model(entry):
            continue
        short = _model_short_name(str(entry.get("name") or ""))
        if not short or short in seen:
            continue
        seen.add(short)
        names.append(short)

    if not names:
        raise GeminiError("No generateContent Gemini text models returned for this key")

    # Prefer lite/flash casually for free tier: keep API order but bubble defaults up
    preferred = [m for m in FALLBACK_GEMINI_MODELS if m in seen]
    rest = [m for m in names if m not in preferred]
    return preferred + rest


def normalize_model_id(model: str, available: list[str] | None = None) -> str:
    """Pick a usable model id; drop unknowns when a live list is provided."""
    mid = (model or "").strip() or DEFAULT_GEMINI_MODEL
    if available:
        if mid in available:
            return mid
        for fallback in FALLBACK_GEMINI_MODELS:
            if fallback in available:
                return fallback
        return available[0]
    if mid in FALLBACK_GEMINI_MODELS:
        return mid
    return DEFAULT_GEMINI_MODEL

