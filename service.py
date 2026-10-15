"""Collection-backed helpers for Gemini Sentence Assist."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

from aqt import mw

from .cleaner.fields import deck_query_clause, guess_expression
from .cleaner.gemini import (
    DEFAULT_GEMINI_MODEL,
    DEFAULT_REQUEST_GAP_SEC,
    GeminiError,
    generate_for_sentence,
    plain_field_text,
    to_anki_html,
)


def list_decks() -> list[str]:
    assert mw.col is not None
    decks = mw.col.decks
    if hasattr(decks, "all_names_and_ids"):
        names = [d.name for d in decks.all_names_and_ids()]
    else:  # older Anki
        names = list(decks.allNames())  # type: ignore[attr-defined]
    return sorted(names, key=str.casefold)


def list_field_names(deck: str = "") -> list[str]:
    """Collect note field names from cards in the deck (or whole collection)."""
    assert mw.col is not None
    query = deck_query_clause(deck) if deck.strip() else ""
    ids = list(mw.col.find_notes(query)) if query else list(mw.col.find_notes(""))
    names: set[str] = set()
    for model in mw.col.models.all():
        names.update(f["name"] for f in model["flds"])
    for nid in ids[:200]:
        note = mw.col.get_note(nid)
        names.update(note.keys())
    return sorted(names, key=str.casefold)


def list_deck_note_field_names(deck: str) -> list[str]:
    """Field names that actually appear on notes in this deck (not every note type)."""
    assert mw.col is not None
    deck_clause = deck_query_clause(deck)
    ids = list(mw.col.find_notes(deck_clause)) if deck_clause else list(mw.col.find_notes(""))
    names: set[str] = set()
    for nid in ids[:400]:
        note = mw.col.get_note(nid)
        names.update(note.keys())
        if len(names) >= 60:
            break
    if not names:
        return list_field_names(deck)
    return sorted(names, key=str.casefold)


@dataclass
class GeminiFillResult:
    note_id: int
    ok: bool
    skipped: bool = False
    error: str | None = None
    wrote_translation: bool = False
    wrote_grammar: bool = False
    elapsed_s: float | None = None
    translation_text: str = ""
    grammar_text: str = ""
    expression: str = ""
    sentence: str = ""


@dataclass
class GeminiJob:
    """Main-thread-prepared work unit; HTTP can run off-thread."""

    note_id: int
    expression: str
    sentence: str
    do_trans: bool
    do_gram: bool


def collect_gemini_jobs(
    note_ids: list[int],
    *,
    sentence_field: str,
    translation_field: str,
    grammar_field: str,
    want_translation: bool = True,
    want_grammar: bool = True,
    overwrite: bool = False,
) -> list[GeminiJob]:
    """Read note fields on the main thread. Safe to call before background HTTP."""
    assert mw.col is not None
    sentence_field = (sentence_field or "").strip()
    translation_field = (translation_field or "").strip()
    grammar_field = (grammar_field or "").strip()
    jobs: list[GeminiJob] = []
    for nid in note_ids:
        note = mw.col.get_note(nid)
        if sentence_field not in note.keys():
            continue
        sentence = _field_plain(note, sentence_field)
        if not sentence:
            continue
        do_trans = bool(
            want_translation
            and translation_field in note.keys()
            and (overwrite or not _field_plain(note, translation_field))
        )
        do_gram = bool(
            want_grammar
            and grammar_field in note.keys()
            and (overwrite or not _field_plain(note, grammar_field))
        )
        if not do_trans and not do_gram:
            continue
        jobs.append(
            GeminiJob(
                note_id=int(nid),
                expression=guess_expression(
                    {name: note[name] for name in note.keys()}, nid
                ),
                sentence=sentence,
                do_trans=do_trans,
                do_gram=do_gram,
            )
        )
    return jobs


def apply_gemini_result(
    job: GeminiJob,
    gem: Any,
    *,
    translation_field: str,
    grammar_field: str,
    undo: Any,
    elapsed_s: float | None = None,
    write: bool = True,
) -> GeminiFillResult:
    """Optionally write Gemini output onto the note (main thread only).

    write=False builds the same result for Preview/Test without touching the collection.
    """
    assert mw.col is not None
    translation_text = str(getattr(gem, "translation", "") or "").strip()
    grammar_text = str(getattr(gem, "grammar", "") or "").strip()
    wrote_t = wrote_g = False
    if write:
        note = mw.col.get_note(job.note_id)
        if job.do_trans and translation_text and translation_field in note.keys():
            note[translation_field] = to_anki_html(translation_text)
            wrote_t = True
        if job.do_gram and grammar_text and grammar_field in note.keys():
            note[grammar_field] = to_anki_html(grammar_text)
            wrote_g = True
        if wrote_t or wrote_g:
            mw.col.update_note(note)
            mw.col.merge_undo_entries(undo)
    else:
        # Preview-only: report what *would* be written
        wrote_t = bool(job.do_trans and translation_text)
        wrote_g = bool(job.do_gram and grammar_text)

    return GeminiFillResult(
        note_id=job.note_id,
        ok=True,
        wrote_translation=wrote_t,
        wrote_grammar=wrote_g,
        skipped=not (wrote_t or wrote_g),
        error=None if (wrote_t or wrote_g) else "Gemini returned empty text",
        elapsed_s=elapsed_s,
        translation_text=translation_text,
        grammar_text=grammar_text,
        expression=job.expression,
        sentence=job.sentence,
    )


def describe_notes_for_gemini(
    note_ids: list[int],
    *,
    sentence_field: str,
    translation_field: str,
    grammar_field: str,
    want_translation: bool = True,
    want_grammar: bool = True,
    overwrite: bool = False,
) -> list[dict[str, Any]]:
    """Build list rows for specific note IDs (e.g. Browser selection)."""
    assert mw.col is not None
    sentence_field = (sentence_field or "").strip()
    translation_field = (translation_field or "").strip()
    grammar_field = (grammar_field or "").strip()
    rows: list[dict[str, Any]] = []
    for nid in note_ids:
        note = mw.col.get_note(int(nid))
        if sentence_field not in note.keys():
            continue
        sentence = _field_plain(note, sentence_field)
        if not sentence:
            continue
        has_to = bool(translation_field) and translation_field in note.keys()
        has_gram = bool(grammar_field) and grammar_field in note.keys()
        need_trans = bool(
            want_translation
            and has_to
            and (overwrite or not _field_plain(note, translation_field))
        )
        need_gram = bool(
            want_grammar
            and has_gram
            and (overwrite or not _field_plain(note, grammar_field))
        )
        rows.append(
            {
                "note_id": int(nid),
                "expression": guess_expression(
                    {name: note[name] for name in note.keys()}, nid
                ),
                "preview": sentence[:240],
                "translation_preview": (
                    _field_plain(note, translation_field)[:240] if has_to else ""
                ),
                "grammar_preview": (
                    _field_plain(note, grammar_field)[:160] if has_gram else ""
                ),
                "need_translation": need_trans,
                "need_grammar": need_gram,
            }
        )
    return rows


@dataclass
class GeminiCandidateResult:
    rows: list[dict[str, Any]]
    scanned: int = 0
    with_sentence: int = 0
    missing_from_field: int = 0
    missing_to_field: int = 0
    missing_grammar_field: int = 0
    already_filled: int = 0
    query: str = ""
    from_field: str = ""
    to_field: str = ""
    grammar_field: str = ""
    want_translation: bool = True
    want_grammar: bool = True
    available_fields: list[str] | None = None

    def __post_init__(self) -> None:
        if self.available_fields is None:
            self.available_fields = []

    def summary_html(self) -> str:
        lines = [
            f"<b>{len(self.rows)} note(s)</b> listed · scanned {self.scanned} · "
            f"had From text: {self.with_sentence}",
            f"From=<code>{_escape_html(self.from_field)}</code> → "
            f"To=<code>{_escape_html(self.to_field)}</code>",
        ]
        if self.missing_from_field:
            lines.append(
                f"⚠ {self.missing_from_field} note(s) have no field named "
                f"<code>{_escape_html(self.from_field)}</code>"
            )
        if self.missing_to_field:
            lines.append(
                f"⚠ {self.missing_to_field} note(s) have no To field "
                f"<code>{_escape_html(self.to_field)}</code> — "
                "pick a field that exists on notes in this deck"
            )
        if self.missing_grammar_field:
            lines.append(
                f"⚠ {self.missing_grammar_field} note(s) have no Grammar field "
                f"<code>{_escape_html(self.grammar_field)}</code>"
            )
        if self.already_filled and not self.rows:
            lines.append(
                f"{self.already_filled} note(s) already have To/Grammar filled — "
                "enable <b>Overwrite</b> or use <b>List all with JP sentence</b>"
            )
        if (
            not self.rows
            and self.with_sentence
            and self.missing_to_field >= self.with_sentence
            and self.want_translation
        ):
            sample = ", ".join(self.available_fields[:14]) or "(none)"
            lines.append(
                "To field is not on these notes. Fields actually used in this deck: "
                f"{sample}"
            )
        if not self.rows and self.with_sentence == 0 and self.scanned:
            sample = ", ".join(self.available_fields[:12]) or "(none)"
            lines.append(f"No non-empty From text found. Fields on deck notes: {sample}")
        if self.rows:
            mins = max(1, int(round(len(self.rows) * DEFAULT_REQUEST_GAP_SEC / 60)))
            lines.append(
                f"Tick notes (or <b>Select all</b>), then <b>Next</b> → Generate · ~{mins} min all"
            )
        return "<br>".join(lines)


def _escape_html(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _field_plain(note: Any, name: str) -> str:
    if not name or name not in note.keys():
        return ""
    return plain_field_text(note[name])


def list_gemini_candidates(
    deck: str,
    sentence_field: str,
    translation_field: str,
    grammar_field: str,
    *,
    want_translation: bool = True,
    want_grammar: bool = True,
    overwrite: bool = False,
    need_only: bool = True,
    require_empty: str | None = None,
    limit: int = 5000,
) -> GeminiCandidateResult:
    """Notes with a JP sentence; optionally only those still needing fill.

    require_empty:
      - None — use need_only / overwrite as usual
      - \"translation\" — only notes whose To (EN) field is empty
      - \"grammar\" — only notes whose Grammar field is empty
    """
    assert mw.col is not None
    sentence_field = (sentence_field or "").strip()
    translation_field = (translation_field or "").strip()
    grammar_field = (grammar_field or "").strip()
    if require_empty not in (None, "translation", "grammar"):
        raise RuntimeError(f"Invalid require_empty={require_empty!r}")
    if not sentence_field:
        raise RuntimeError("Pick a From (JP sentence) field first.")
    if not want_translation and not want_grammar and require_empty is None:
        raise RuntimeError("Enable translation and/or grammar.")
    if (want_translation or require_empty == "translation") and not translation_field:
        raise RuntimeError("Pick a To (EN) field.")
    if (want_grammar or require_empty == "grammar") and not grammar_field:
        raise RuntimeError("Pick a Grammar field (or disable grammar).")

    available = list_deck_note_field_names(deck)
    if (want_translation or require_empty == "translation") and translation_field not in available:
        raise RuntimeError(
            f"To field {translation_field!r} was not found on notes in this deck.\n"
            f"Choose an existing English field in the To dropdown.\n"
            f"Fields on notes in this deck: {', '.join(available) or '(none found)'}"
        )
    if sentence_field not in available:
        raise RuntimeError(
            f"From field {sentence_field!r} was not found on notes in this deck.\n"
            f"Choose your Japanese sentence field in the From dropdown.\n"
            f"Fields on notes in this deck: {', '.join(available) or '(none found)'}"
        )
    if (want_grammar or require_empty == "grammar") and grammar_field not in available:
        raise RuntimeError(
            f"Grammar field {grammar_field!r} was not found on notes in this deck.\n"
            f"Fields on notes in this deck: {', '.join(available) or '(none found)'}"
        )

    # Scan entire deck in Python — Anki field:* search is unreliable across versions.
    deck_clause = deck_query_clause(deck)
    query = deck_clause or ""
    ids = list(mw.col.find_notes(query))

    out = GeminiCandidateResult(
        rows=[],
        scanned=len(ids),
        query=query or "(whole collection)",
        from_field=sentence_field,
        to_field=translation_field,
        grammar_field=grammar_field,
        want_translation=want_translation,
        want_grammar=want_grammar,
        available_fields=available,
    )

    for nid in ids:
        if len(out.rows) >= limit:
            break
        note = mw.col.get_note(nid)
        if sentence_field not in note.keys():
            out.missing_from_field += 1
            continue
        sentence = _field_plain(note, sentence_field)
        if not sentence:
            continue
        out.with_sentence += 1

        has_to = bool(translation_field) and translation_field in note.keys()
        has_gram = bool(grammar_field) and grammar_field in note.keys()
        if want_translation and not has_to:
            out.missing_to_field += 1
        if want_grammar and not has_gram:
            out.missing_grammar_field += 1

        to_empty = has_to and not _field_plain(note, translation_field)
        gram_empty = has_gram and not _field_plain(note, grammar_field)

        if require_empty == "translation":
            if not to_empty:
                if has_to:
                    out.already_filled += 1
                continue
        elif require_empty == "grammar":
            if not gram_empty:
                if has_gram:
                    out.already_filled += 1
                continue

        need_trans = bool(
            (want_translation or require_empty == "translation")
            and has_to
            and (overwrite or to_empty or require_empty == "translation")
        )
        need_gram = bool(
            (want_grammar or require_empty == "grammar")
            and has_gram
            and (overwrite or gram_empty or require_empty == "grammar")
        )

        if require_empty is None and need_only and not need_trans and not need_gram:
            # Distinguish "already filled" from "target field missing on this note"
            targets_exist = (not want_translation or has_to) and (
                not want_grammar or has_gram
            )
            if targets_exist:
                out.already_filled += 1
            continue

        gram_prev = ""
        if has_gram:
            gram_prev = _field_plain(note, grammar_field)[:160]

        out.rows.append(
            {
                "note_id": int(nid),
                "expression": guess_expression(
                    {name: note[name] for name in note.keys()}, nid
                ),
                "preview": sentence[:240],
                "translation_preview": (
                    _field_plain(note, translation_field)[:240] if has_to else ""
                ),
                "grammar_preview": gram_prev,
                "need_translation": need_trans,
                "need_grammar": need_gram,
            }
        )

    return out


def fill_gemini_notes(
    note_ids: list[int],
    *,
    api_key: str,
    sentence_field: str,
    translation_field: str,
    grammar_field: str,
    want_translation: bool = True,
    want_grammar: bool = True,
    overwrite: bool = False,
    model: str = DEFAULT_GEMINI_MODEL,
    gap_sec: float = DEFAULT_REQUEST_GAP_SEC,
    progress: Callable[[int, int, str], bool] | None = None,
) -> list[GeminiFillResult]:
    """Call Gemini and write fields. progress(i, total, label) -> False to cancel."""
    assert mw.col is not None
    sentence_field = (sentence_field or "").strip()
    translation_field = (translation_field or "").strip()
    grammar_field = (grammar_field or "").strip()
    if not sentence_field:
        raise RuntimeError("Pick a Sentence (JP) field first.")
    if want_translation and not translation_field:
        raise RuntimeError("Pick an EN translation target field (or uncheck Generate EN).")
    if want_grammar and not grammar_field:
        raise RuntimeError("Pick a grammar field (or uncheck Generate grammar).")

    results: list[GeminiFillResult] = []
    undo = mw.col.add_custom_undo_entry("Gemini Sentence Assist")
    total = len(note_ids)
    last_call = 0.0

    for idx, nid in enumerate(note_ids):
        label = f"Gemini {idx + 1}/{total}"
        if progress is not None and not progress(idx, total, label):
            results.append(
                GeminiFillResult(note_id=int(nid), ok=False, error="Cancelled")
            )
            break

        try:
            note = mw.col.get_note(nid)
            if sentence_field not in note.keys():
                results.append(
                    GeminiFillResult(
                        note_id=int(nid),
                        ok=False,
                        error=f"Missing field “{sentence_field}”",
                    )
                )
                continue

            sentence = _field_plain(note, sentence_field)
            if not sentence:
                results.append(
                    GeminiFillResult(
                        note_id=int(nid), ok=True, skipped=True, error="Empty sentence"
                    )
                )
                continue

            do_trans = bool(
                want_translation
                and translation_field in note.keys()
                and (overwrite or not _field_plain(note, translation_field))
            )
            do_gram = bool(
                want_grammar
                and grammar_field in note.keys()
                and (overwrite or not _field_plain(note, grammar_field))
            )
            if not do_trans and not do_gram:
                results.append(
                    GeminiFillResult(note_id=int(nid), ok=True, skipped=True)
                )
                continue

            # Rate-limit between successful API calls (free-tier friendly).
            if last_call:
                wait = gap_sec - (time.monotonic() - last_call)
                if wait > 0:
                    time.sleep(wait)

            expression = guess_expression(
                {name: note[name] for name in note.keys()}, nid
            )
            gem = generate_for_sentence(
                api_key,
                sentence,
                model=model,
                want_translation=do_trans,
                want_grammar=do_gram,
                expression=expression,
            )
            last_call = time.monotonic()

            wrote_t = wrote_g = False
            if do_trans and gem.translation:
                note[translation_field] = to_anki_html(gem.translation)
                wrote_t = True
            if do_gram and gem.grammar:
                note[grammar_field] = to_anki_html(gem.grammar)
                wrote_g = True
            if wrote_t or wrote_g:
                mw.col.update_note(note)
                mw.col.merge_undo_entries(undo)
            results.append(
                GeminiFillResult(
                    note_id=int(nid),
                    ok=True,
                    wrote_translation=wrote_t,
                    wrote_grammar=wrote_g,
                    skipped=not (wrote_t or wrote_g),
                    error=None
                    if (wrote_t or wrote_g)
                    else "Gemini returned empty text",
                )
            )
        except GeminiError as exc:
            results.append(
                GeminiFillResult(note_id=int(nid), ok=False, error=str(exc))
            )
            if exc.retryable and exc.status == 429:
                # Back off a bit extra on quota; still continue next notes.
                time.sleep(max(gap_sec, 8.0))
        except Exception as exc:  # noqa: BLE001
            results.append(
                GeminiFillResult(note_id=int(nid), ok=False, error=str(exc))
            )

    return results
