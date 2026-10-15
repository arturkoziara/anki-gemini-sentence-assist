"""Anki add-on entry: Tools → Gemini Sentence Assist… / Browser right-click."""

from __future__ import annotations

from aqt import gui_hooks, mw
from aqt.qt import QAction, qconnect

ADDON_PACKAGE = __name__

from .gemini_dialog import open_gemini_dialog


def _selected_note_ids(browser) -> list[int]:
    try:
        return list(browser.selected_notes())
    except Exception:  # noqa: BLE001 — older Anki API
        try:
            return list(browser.selectedNotes())  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            return []


def _on_gemini() -> None:
    open_gemini_dialog()


def _browser_gemini(browser) -> None:
    """Open Gemini with the notes selected in Browse."""
    ids = _selected_note_ids(browser)
    if not ids:
        from aqt.utils import showInfo

        showInfo("Select one or more notes in the Browser first.")
        return
    open_gemini_dialog(note_ids=ids)


def _on_browser_context_menu(browser, menu) -> None:
    action = menu.addAction("Gemini: assist selected notes…")
    qconnect(action.triggered, lambda _checked=False: _browser_gemini(browser))


action_gemini = QAction("Gemini Sentence Assist…", mw)
qconnect(action_gemini.triggered, _on_gemini)
mw.form.menuTools.addAction(action_gemini)

gui_hooks.browser_will_show_context_menu.append(_on_browser_context_menu)
